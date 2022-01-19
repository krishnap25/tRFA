"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import math
import numpy as np
import random
from sklearn.metrics import pairwise_distances
from wquantiles import median as weighted_median

from baseline_constants import ACCURACY_KEY, OptimLoggingKeys, AGGR_MEAN, AGGR_GEO_MED
from baseline_constants import AGGR_TRIM_MEAN, AGGR_NORM_MEAN, AGGR_CO_MED, AGGR_KRUM

from utils.model_utils import batch_data


class Model(ABC):

    def __init__(self, lr, seed, max_batch_size, optimizer=None):
        self.lr = lr
        self.optimizer = optimizer
        self.rng = random.Random(seed)
        self.size = None

        # largest batch size for which GPU will not run out of memory
        self.max_batch_size = max_batch_size if max_batch_size is not None else 2 ** 14
        print('***** using a max batch size of', self.max_batch_size)
        self.flops = 0

    @abstractmethod
    def create_model(self):
        """Creates the model for the task.

        Returns:
            A 5-tuple consisting of:
                features: A placeholder for the samples' features.
                labels: A placeholder for the samples' labels.
                loss_op: A Tensorflow operation that, when run with the features and
                    the labels, computes the loss function on these features and models.
                train_op: A Tensorflow operation that, when run with the features and
                    the labels, trains the model.
                eval_metric_ops: A Tensorflow operation that, when run with features and labels,
                    returns the accuracy of the model.
        """
        return None, None, None, None, None

    def train(self, data, num_epochs=1, batch_size=10, lr=None):
        """
        Trains the client model.

        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
            averaged_loss: average of stochastic loss in the final epoch
        """
        if lr is None:
            lr = self.lr
        averaged_loss = 0.0

        batched_x, batched_y = batch_data(data, batch_size, rng=self.rng, shuffle=True)
        if self.optimizer.w is None:
            self.optimizer.initialize_w()

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i, raw_x_batch in enumerate(batched_x):
                input_data = self.process_x(raw_x_batch)
                raw_y_batch = batched_y[i]
                target_data = self.process_y(raw_y_batch)

                loss = self.optimizer.run_step(input_data, target_data)
                total_loss += loss
            averaged_loss = total_loss / len(batched_x)
        # print('inner opt:', epoch, averaged_loss)
        #with self.graph.as_default():

            #update = [self.sess.run(v) for v in tf.trainable_variables()]
            #update = [np.subtract(update[i], init_values[i]) for i in range(len(update))] # Should be the gradient here
        self.optimizer.end_local_updates() # required for pytorch models
        update = np.copy(self.optimizer.w - self.optimizer.w_on_last_update)

        self.optimizer.update_w()

        comp = num_epochs * len(batched_y) * batch_size * self.flops
        return comp, update, averaged_loss

    def test(self, eval_data, train_data=None, split_by_user=True, train_users=True):
        # `train_users` is not used for split_by_sample
        """
        Tests the current model on the given data.
        Args:
            eval_data: dict of the form {'x': [list], 'y': [list]}
            train_data: None or same format as eval_data. If None, do not measure statistics on train_data.
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        if split_by_user:
            output = {'eval': [-float('inf'), -float('inf')], 'train': [-float('inf'), -float('inf')]}

            if self.optimizer.w is None:
                self.optimizer.initialize_w()

            total_loss, total_correct, count = 0.0, 0, 0
            batched_x, batched_y = batch_data(eval_data, self.max_batch_size, shuffle=False, eval_mode=True)
            for x, y in zip(batched_x, batched_y):
                x_vecs = self.process_x(x)
                labels = self.process_y(y)

                loss = self.optimizer.loss(x_vecs, labels)
                correct = self.optimizer.correct(x_vecs, labels)

                total_loss += loss * len(y)  # loss returns average over batch
                total_correct += correct  # eval_op returns sum over batch
                count += len(y)
                # counter_1 += 1
            loss = total_loss / count
            acc = total_correct / count
            if train_users:
                output['train'] = [loss, acc]
            else:
                output['eval'] = [loss, acc]

            return {
                    ACCURACY_KEY: output['eval'][1],
                    OptimLoggingKeys.TRAIN_LOSS_KEY: output['train'][0],
                    OptimLoggingKeys.TRAIN_ACCURACY_KEY: output['train'][1],
                    OptimLoggingKeys.EVAL_LOSS_KEY: output['eval'][0],
                    OptimLoggingKeys.EVAL_ACCURACY_KEY: output['eval'][1]
                    }
        else:
            data_lst = [eval_data] if train_data is None else [eval_data, train_data]
            output = {'eval': [-float('inf'), -float('inf')], 'train': [-float('inf'), -float('inf')]}

            if self.optimizer.w is None:
                self.optimizer.initialize_w()
            # counter_0 = 0
            for data, data_type in zip(data_lst, ['eval', 'train']):
                # counter_1 = 0
                total_loss, total_correct, count = 0.0, 0, 0
                batched_x, batched_y = batch_data(data, self.max_batch_size, shuffle=False, eval_mode=True)
                for x, y in zip(batched_x, batched_y):
                    x_vecs = self.process_x(x)
                    labels = self.process_y(y)

                    loss = self.optimizer.loss(x_vecs, labels)
                    correct = self.optimizer.correct(x_vecs, labels)

                    total_loss += loss * len(y)  # loss returns average over batch
                    total_correct += correct  # eval_op returns sum over batch
                    count += len(y)
                    # counter_1 += 1
                loss = total_loss / count
                acc = total_correct / count
                output[data_type] = [loss, acc]
                # counter_1 += 1

            return {ACCURACY_KEY: output['eval'][1],
                    OptimLoggingKeys.TRAIN_LOSS_KEY: output['train'][0],
                    OptimLoggingKeys.TRAIN_ACCURACY_KEY: output['train'][1],
                    OptimLoggingKeys.EVAL_LOSS_KEY: output['eval'][0],
                    OptimLoggingKeys.EVAL_ACCURACY_KEY: output['eval'][1]
                    }

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)


class ServerModel:
    def __init__(self, model):
        self.model = model
        self.rng = model.rng

    @property
    def size(self):
        return self.model.optimizer.size()

    @property
    def cur_model(self):
        return self.model

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        var_vals = {}
        for c in clients:
            c.model.optimizer.reset_w(self.model.optimizer.w)
            c.model.size = self.model.optimizer.size()

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = np.sum(weights)
        weighted_updates = np.zeros_like(points[0])

        for w, p in zip(weights, points):
            weighted_updates += (w / tot_weights) * p

        return weighted_updates

    def update(self, updates, aggregation=AGGR_MEAN, max_update_norm=None, maxiter=4, 
            fraction_to_discard=0.0, norm_bound=None, 
        ):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
            aggregation: Algorithm used for aggregation. Allowed values are:
                [ 'mean', 'geom_median']
            max_update_norm: Reject updates larger than this norm,
            maxiter: maximum number of calls to the Weiszfeld algorithm if using the geometric median
        """
        if len(updates) == 0:
            print('No updates obtained. Continuing without update')
            return 1, False
        def accept_update(u):
            # norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
            norm = np.linalg.norm(u[1])
            return not (np.isinf(norm) or np.isnan(norm))
        all_updates = updates
        updates = [u for u in updates if accept_update(u)]
        if len(updates) < len(all_updates):
            print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
        if len(updates) == 0:
            print('All individual updates rejected. Continuing without update')
            return 1, False

        points = [u[1] for u in updates]  # list of np.ndarray
        alphas = [u[0] for u in updates]  # list of integers
        # print('>>>>>>', np.linalg.norm(points, axis=1).max())
        # Sent 140: linear
        # 90 perc: 0.2157708390519796
        # 95 perc: 0.22640943180791126
        # 99 perc: 0.236319450830711
        # max: 0.27914787017971243
        if aggregation == AGGR_MEAN:
            weighted_updates = self.weighted_average_oracle(points, alphas)
            num_comm_rounds = 1
        elif aggregation == AGGR_GEO_MED:
            weighted_updates, num_comm_rounds, _ = self.geometric_median_update(points, alphas, maxiter=maxiter)
        elif aggregation == AGGR_TRIM_MEAN:
            weighted_updates = self.trimmed_mean_update(points, alphas, fraction_to_discard)
            num_comm_rounds = 1
        elif aggregation == AGGR_NORM_MEAN:
            weighted_updates = self.norm_bounded_mean_update(points, alphas, norm_bound)
            num_comm_rounds = 1
        elif aggregation == AGGR_CO_MED:
            # `weighted_median` computes median along the last axis, so we need to transpose points
            weighted_updates = weighted_median(np.array(points).T, np.array(alphas))
            num_comm_rounds = 1
        elif aggregation == AGGR_KRUM:
            weighted_updates = self.multikrum_update(points, alphas, fraction_to_discard)
            num_comm_rounds = 1
        else:
            raise ValueError('Unknown aggregation strategy: {}'.format(aggregation))

        # update_norm = np.linalg.norm([np.linalg.norm(v) for v in weighted_updates])
        update_norm = np.linalg.norm(weighted_updates)

        if max_update_norm is None or update_norm < max_update_norm:
            self.model.optimizer.w += np.array(weighted_updates)
            self.model.optimizer.reset_w(self.model.optimizer.w)  # update server model
            updated = True
        else:
            print('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            updated = False
        return num_comm_rounds, updated

    @staticmethod
    def multikrum_update(points, alphas, fraction_to_discard, do_averaging=True):
        points = np.asarray(points)  # (n, d)
        alphas = np.asarray(alphas)  # (n, )
        n = points.shape[0]
        # f: expected number of corrupted updates; cap it at n // 2
        num_corrupt = min(len(points) // 2 - 1, math.ceil(len(points) * fraction_to_discard))
        num_good = n - num_corrupt - 2  # n - f - 2
        multikrum_param = n - num_corrupt  # parameter `m` in the paper
        sqdist = pairwise_distances(points) ** 2 
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = np.sort(sqdist[i])[:num_good+1].sum()  # exclude k = i
        if do_averaging: # multi-krum
            good_idxs = np.argsort(scores)[:multikrum_param]
            points = points[good_idxs]
            alphas = alphas[good_idxs]
            return np.average(points, weights=alphas, axis=0)
        else:  # krum
            idx = np.argmin(scores)
            return points[idx]

    @staticmethod
    def norm_bounded_mean_update(points, alphas, norm_bound):
        points = np.array(points)  # (n, d)
        alphas = np.array(alphas)
        norms = np.linalg.norm(points, axis=1)  # (n,)
        multiplier = np.minimum(norm_bound / norms, 1)  # (n,)
        points = points * multiplier[:, None]
        return np.average(points, weights=alphas, axis=0)

    @staticmethod
    def trimmed_mean_update(points, alphas, fraction_to_discard):
        if fraction_to_discard == 0: # no trimming necessary, return simple mean
            return np.average(points, weights=alphas, axis=0)
        points = np.asarray(points)  # (n, d)
        alphas = np.asarray(alphas)
        aggregated_update = np.zeros_like(points[0])
        # discard at least 1 but do not discard too many
        num_points_to_discard = min(len(points) // 2, math.ceil(len(points) * fraction_to_discard))  
        for i in range(aggregated_update.shape[0]):
            values = np.asarray([p[i] for p in points])
            idxs = np.argsort(values)[num_points_to_discard: -num_points_to_discard]
            aggregated_update[i] = np.average(values[idxs], weights=alphas[idxs])
        return aggregated_update
        

    @staticmethod
    def geometric_median_update(points, alphas, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        alphas = np.asarray(alphas, dtype=points[0].dtype) / sum(alphas)
        median = ServerModel.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = ServerModel.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            print('Starting Weiszfeld algorithm')
            print(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = np.asarray([alpha / max(eps, ServerModel.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = ServerModel.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = ServerModel.geometric_median_objective(median, points, alphas)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         ServerModel.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                print(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        return median, num_oracle_calls, logs

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        #return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        return sum([alpha * ServerModel.l2dist(median, p) for alpha, p in zip(alphas, points)])

# TODO : Test this class

class Optimizer(ABC):

    def __init__(self, starting_w=None, loss=None, loss_prime=None):
        self.w = starting_w
        self.w_on_last_update = np.copy(starting_w)
        self.optimizer_model = None

    @abstractmethod
    def loss(self, x, y):
        return None

    @abstractmethod
    def gradient(self, x, y):
        return None

    @abstractmethod
    def run_step(self, batched_x, batched_y): # should run a first order method step and return loss obtained
        return None

    @abstractmethod
    def correct(self, x, y):
        return None

    def end_local_updates(self):
        pass

    def reset_w(self, w):
        self. w = np.copy(w)
        self.w_on_last_update = np.copy(w)

