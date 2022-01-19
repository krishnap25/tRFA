"""Script to run the baselines."""

import argparse
import copy
import importlib
import gc
import math
import pickle as pkl
import random
import os
import sys
import time
import json
from datetime import timedelta

import numpy as np
import pandas as pd
from PIL import Image
import torch

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, \
        SIM_TIMES, AVG_LOSS_KEY, ACCURACY_KEY, TRAINING_KEYS
from baseline_constants import OptimLoggingKeys
from baseline_constants import CORRUPTION_OMNISCIENT_KEY, \
        CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY, CORRUPTION_GAUSS_KEY, CORRUPTION_GAUSS2_KEY
from baseline_constants import (
    AGGR_MEAN, AGGR_GEO_MED,
    AGGR_TRIM_MEAN, AGGR_NORM_MEAN, AGGR_CO_MED, AGGR_KRUM
)
from baseline_constants import REGULARIZATION_PARAMS
from client import Client
from model import ServerModel
from server import Server
from utils.constants import DATASETS
from utils.model_utils import read_data, preprocess_data_x, preprocess_data_y, batch_data
from metrics.writer import writer_print_metrics, writer_get_metrics_names

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'
SUMMARY_METRICS_PATH = 'outputs/summary_metrics'
SUMMARY_METRICS_PATH_REG = 'outputs/reg/'


def main():
    args = parse_args()
    global_start_time = start_time = time.time()

    # set torch seed
    torch_seed = (args.seed + 25) // 2
    torch.random.manual_seed(torch_seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Create 2 different models: one for server and one shared by all clients
    print('Obtained Model Path : ' + str(model_path))
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    else:
        args.lr = model_params[0]
    if args.decay_lr_every is None:
        args.decay_lr_every = 100 if args.dataset == 'femnist' else 50
    #tf.reset_default_graph()
    print(args)
    client_model = ClientModel(*model_params, seed=args.seed)
    server_model = ServerModel(ClientModel(*model_params, seed=args.seed - 1))
    # Set device
    if args.gpu is not None and args.gpu >= 0 and args.gpu <= 4:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    if hasattr(client_model, 'set_device'):
        client_model.set_device(device)
        server_model.model.set_device(device)

    # Create server
    server = Server(server_model)

    # Create clients
    clients, corrupted_client_ids = setup_clients(args.dataset, model_name=args.model,
                                                  model=client_model, validation=args.validation,
                                                  corruption=args.corruption, fraction_corrupt=args.fraction_corrupt,
                                                  seed=args.seed, split_by_user=args.split_by_user,
                                                  data_dir=args.data_dir)

    train_clients = clients['train_clients']
    if args.split_by_user:
        test_clients = clients['test_clients']
        if args.validation:
            validation_clients = clients['validation_clients']
        else:
            validation_clients = []
    else:
        test_clients = clients['train_clients']

    ######### Normalization ########
    if (
            (args.dataset == 'sent140' and args.model in ['erm_lstm_log_reg'])
        or (args.dataset == 'shakespeare' and args.model in
            ['erm_lstm_log_reg'])
    ):
        print('Data is not normalized here')
        # no normalizing performed here
        pass
    else:
        # normalize data:
        print('-------Normalizing the data!')
        dataset_file_constants_train = 'dataset_constants/' + args.dataset + '_train_mean_std'
        with open(dataset_file_constants_train) as json_file:
            data = json.load(json_file)
        mean_train = data['mean']
        std_train = data['mean']

        dataset_file_constants_test = 'dataset_constants/' + args.dataset + '_train_mean_std'
        with open(dataset_file_constants_test) as json_file:
            data = json.load(json_file)
        mean_test = data['mean']
        std_test = data['mean']

        for c in train_clients:
            for i in range(len(c.train_data['x'])):
                for j in range(len(c.train_data['x'][i])):
                    c.train_data['x'][i][j] = (c.train_data['x'][i][j] - mean_train[j])/std_train[j]
        for c in test_clients:
            for i in range(len(c.eval_data['x'])):
                for j in range(len(c.eval_data['x'][i])):
                    c.eval_data['x'][i][j] = (c.eval_data['x'][i][j] - mean_train[j]) / std_train[j]

        if args.validation:
            for c in validation_clients:
                for i in range(len(c.train_data['x'])):
                    for j in range(len(c.train_data['x'][i])):
                        c.train_data['x'][i][j] = (c.train_data['x'][i][j] - mean_train[j]) / std_train[j]
    
    if args.make_iid:
        # TODO: handle validation clients
        print("----> Making the data iid by reshuffing the data!!")
        convert_to_iid(train_clients, seed=0, train=True)
        convert_to_iid(test_clients, seed=1, train=False)
        
    print('#Clients = %d, %d; setup time = %s' % (len(train_clients), len(test_clients),
                                               timedelta(seconds=round(time.time() - start_time))))
    gc.collect()  # free discarded memory in setup_clients

    ### Logging utilities (Needed for split by user)
    # train_ids, train_groups, train_clients_num_train_samples, train_clients_num_test_samples = server.get_clients_info(train_clients)
    # test_ids, test_groups, test_clients_num_train_samples, test_clients_num_test_samples = server.get_clients_info(test_clients)
    # if args.validation:
    #     validation_ids, validation_groups, validation_clients_num_train_samples, validation_clients_num_test_samples = server.get_clients_info(
    #     validation_clients)

    # Logging utilities
    # import pdb
    # pdb.set_trace()
    all_ids, all_groups, all_num_train_samples, all_num_test_samples = server.get_clients_info(train_clients)  # TODO: only works for split by sample
    summary = None

    def log_helper(iteration, path_for_validation, comm_rounds=None):
        if comm_rounds is None:
            comm_rounds = iteration
        nonlocal summary
        start_time = time.time()
        if args.no_logging:
            stat_metrics = None
        else:
            stat_metrics = server.test_model(train_clients, train_and_test=True)

        summary_iter = print_metrics(iteration, comm_rounds, stat_metrics,
                                     all_num_train_samples, all_num_test_samples,
                                     time.time() - start_time)
        if iteration == 0:
            summary = pd.Series(summary_iter).to_frame().T
        else:
            summary = summary.append(summary_iter, ignore_index=True)
            summary.to_csv(path_for_validation, mode='w', header=True, index=False)
        return summary_iter

    # Test untrained model on all clients
    def reinit_clients():
        for c in train_clients:
            c.reinit_model()
        # for c in test_clients:
        #     c.reinit_model()
        # if args.validation:
        #     for c in validation_clients:
        #         c.reinit_model()

    # Simulate training
    def main_training(path_for_validation, regularization_param=None):
        # Required for initial logging to be correct when initializing from non-zero weights
        if regularization_param is not None:
            server.model.model.optimizer.lmbda = regularization_param

        # Initialize diagnostics
        s = log_helper(0, path_for_validation)
        initial_loss = s.get(OptimLoggingKeys.TRAIN_LOSS_KEY[0], None)
        initial_avg_loss = None
        num_no_progress = 0

        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
            sys.stdout.flush()
            start_time = time.time()

            # Select clients to train this round
            server.select_clients(online(train_clients), num_clients=clients_per_round)
            c_ids, c_groups, c_num_train_samples, c_num_test_samples = server.get_clients_info()

            # Simulate server model training on selected clients' data
            sys_metrics, avg_loss, losses = server.train_model(num_epochs=args.num_epochs,
                                                               batch_size=args.batch_size,
                                                               minibatch=args.minibatch,
                                                               lr=args.lr,
                                                               lmbda=regularization_param,
                                                               )
            updates = server.updates
            writer_print_metrics(i, c_ids, sys_metrics, c_groups, c_num_test_samples, args.output_sys_file)

            # Diagnostics - 1
            if initial_avg_loss is None:
                initial_avg_loss = avg_loss
            if avg_loss > 3 * initial_avg_loss and args.corruption != CORRUPTION_OMNISCIENT_KEY:
                print('-'*50)
                print('Abnormal loss encountered. Diagnostics:')
                print('all average loss:', ['{:.3f}'.format(l) for l in losses])
                print('selected_clients:', [c.id for c in server.selected_clients])
                print('corrupted_clients:', corrupted_client_ids)

            # Update server model

            total_num_comm_rounds, is_updated = server.update_model(
                aggregation=args.aggregation,
                corruption=args.corruption,
                corrupted_client_ids=corrupted_client_ids,
                maxiter=args.weiszfeld_maxiter,
                fraction_to_discard=args.fraction_to_discard,
                norm_bound=args.norm_bound
            )

            # Diagnostics - 2
            if is_updated:
                num_no_progress = 0
            else:
                num_no_progress += 1
            if num_no_progress > args.patience_iter:
                print('No progress made in {} iterations. Quitting.'.format(num_no_progress))
                sys.exit(1)

            # Logging
            #norm = _norm(server_model.model)
            norm = np.linalg.norm(server_model.model.optimizer.w)
            print('\t\t\tRound: {} AvgLoss: {:.3f} Norm: {:.2f} Time: {} Tot_time {}'.format(
                i + 1, avg_loss, norm,
                timedelta(seconds=round(time.time() - start_time)),
                timedelta(seconds=round(time.time() - global_start_time))
            ), flush=True)

            # Test model on all clients
            if (
                    (i + 1) in [10, 75]
                    or (i + 1) % eval_every == 0
                    or (i + 1) == num_rounds
                    or (args.corruption == CORRUPTION_OMNISCIENT_KEY and (i + 1) < 10)
            ):
                s = log_helper(i + 1, path_for_validation, total_num_comm_rounds)
                if OptimLoggingKeys.TRAIN_LOSS_KEY in s:
                    if initial_loss is not None and s[OptimLoggingKeys.TRAIN_LOSS_KEY] > 3 * initial_loss:
                        print('Loss > 3 * initial_loss. Exiting')
                        break
                    if math.isnan(s[OptimLoggingKeys.TRAIN_LOSS_KEY]):
                        print('Loss NaN. Exiting')
                        break

            if (i + 1) % args.decay_lr_every == 0:
                args.lr /= args.lr_decay

        # Save server model
        if args.validation:
            summary.to_csv(path_for_validation, mode='w', header=True, index=False)
        else:
            summary.to_csv(path_for_validation, mode='w', header=True, index=False)
        #save_model(server_model, args.dataset, args.model, args.output_summary_file)

    def validation_experiment():
        list_path = [SUMMARY_METRICS_PATH_REG + args.dataset + '_' + args.model + '_reg_' + '_' + str(lmbda)
                     + '.csv' for lmbda in REGULARIZATION_PARAMS]
        for counter_reg in range(len(REGULARIZATION_PARAMS)):
            reinit_clients()
            main_training(list_path[counter_reg], regularization_param=REGULARIZATION_PARAMS[counter_reg])

    if args.old_validation:
        validation_experiment()
    else:
        main_training(args.output_summary_file + '.csv', regularization_param=args.reg_param)

    print('Job complete. Total time taken:', timedelta(seconds=round(time.time() - global_start_time)))

def online(clients):
    """We assume all users are always online."""
    return clients


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=DATASETS,
                        required=True)
    parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('--split_by_user',
                        help='name of model;',
                        type=bool,
                        default=False)
    parser.add_argument('--data_dir', type=str, help='directory where data is stored')
    parser.add_argument('--full_record',
                        help='name of model;',
                        type=bool,
                        default=False)
    parser.add_argument('--num-rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=-1)
    parser.add_argument('--eval-every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=-1)
    parser.add_argument('--make_iid', action='store_true')
    parser.add_argument('--clients-per-round',
                        help='number of clients trained per round;',
                        type=int,
                        default=-1)
    parser.add_argument('--seed',
                        help='random seed for reproducibility;',
                        type=int)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=32)
    parser.add_argument('--minibatch',
                        help='None for FedAvg, else fraction;',
                        type=float,
                        default=None)
    parser.add_argument('--fraction_to_discard', type=float, default=0)
    parser.add_argument('--norm_bound', type=float, default=1e6)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=1)
    parser.add_argument('-t',
                        help='simulation time: small, medium, or large;',
                        type=str,
                        choices=SIM_TIMES,
                        default='large')
    parser.add_argument('-lr',
                        help='learning rate for local optimizers;',
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument('--lr-decay',
                        help='decay in learning rate',
                        type=float,
                        default=1.0)
    parser.add_argument('--decay-lr-every',
                        help='number of iterations to decay learning rate',
                        type=int,
                        default=200)
    parser.add_argument('-reg_param',
                        help='regularization learning parameter',
                        type=float,
                        default=None)
    parser.add_argument('--safety_user_level',
                        help='decay in learning rate',
                        type=float,
                        default=None)
    parser.add_argument('--output_stat_file',
                        help='Filename to log stat metrics in CSV',
                        default=STAT_METRICS_PATH)
    parser.add_argument('--output_sys_file',
                        help='Filename to log system metrics in CSV',
                        default=SYS_METRICS_PATH)
    parser.add_argument('--output_summary_file',
                        help='Filename to log summary of optimization performance in CSV',
                        default=SUMMARY_METRICS_PATH)
    parser.add_argument('--validation',
                        help='If specified, hold out part of training data to use as a dev set for parameter search',
                        type=bool,
                        default=False)
    parser.add_argument('--patience-iter',
                        help='Number of patience rounds of no updates to wait for before giving up',
                        type=int,
                        default=20)
    parser.add_argument('--corruption',
                        help=""""Corrupt data in any clients? If not specified, add no corruptions.
                        Choice '{}' simply flips the labels of corrupt devices, while choice '{}'
                        leads to corrupt devices to propose an update leading to negation of the update.
                        """.format(CORRUPTION_FLIP_KEY, CORRUPTION_OMNISCIENT_KEY),
                        choices=[CORRUPTION_OMNISCIENT_KEY, CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY, CORRUPTION_GAUSS_KEY, CORRUPTION_GAUSS2_KEY],
                        type=str)
    parser.add_argument('--fraction-corrupt',
                        help="""Fraction of data to corrupt.
                        Chooses clients randomly until total fraction of corrupt data has just been exceeded
                        specified fraction""",
                        type=float,
                        default=0.1)
    parser.add_argument('--aggregation',
                        help='Aggregation technique used to combine updates or gradients',
                        choices=[AGGR_MEAN, AGGR_GEO_MED, AGGR_TRIM_MEAN, AGGR_NORM_MEAN, AGGR_CO_MED, AGGR_KRUM],
                        default=AGGR_MEAN)
    parser.add_argument('--weiszfeld-maxiter',
                        type=int,
                        default=4)
    parser.add_argument('--no-logging',
                        help='if specified, do not perform testing. Instead save model to disk.',
                        action='store_true')
    parser.add_argument('--old_validation',
                        help='Performs validation in sequential manner',
                        type=bool,
                        default=False)
    parser.add_argument('--gpu',
                        help='Which gpu to use. Unspecified (=None) means CPU',
                        type=int)

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 2)
        print('Random seed not provided. Using {} as seed'.format(args.seed))
    return args


def setup_clients(dataset, model_name=None, model=None, validation=False, corruption=None, fraction_corrupt=0.1, seed=-1,
                  split_by_user=True, subsample_fraction=0.5, data_dir=None):
    """Instantiates clients based on given train and test data directories.
        If validation is True, use part of training set as validation set

    Return:
        all_clients: list of Client objects.
    """
    if data_dir is None:
        data_dir = '..'
    else:
        print('Loading data from path:', data_dir)
    train_data_dir = os.path.join(data_dir, 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join(data_dir, 'data', dataset, 'data', 'test')

    clients, groups, train_data, test_data = read_data(train_data_dir, test_data_dir, split_by_user=split_by_user,
                                                       dataset=dataset)
    if seed != -1:
        np.random.seed(seed)
    else:
        np.random.seed(42)

    if split_by_user:
        train_users = clients['train_users']
        test_users = clients['test_users']

        if dataset == 'femnist':
            # Only subsample femnist
            indices_train = np.arange(len(train_users))
            indices_test = np.arange(len(test_users))
            fraction_used = int(subsample_fraction * len(train_users))
            subindices_train = np.random.choice(indices_train, fraction_used, replace=False)
            subindices_test = np.random.choice(indices_test, fraction_used, replace=False)

            train_users = [train_users[i] for i in subindices_train]
            test_users = [test_users[i] for i in subindices_test]

        train_groups = [[] for _ in train_users]
        test_groups = [[] for _ in test_users]

        if validation:
            # Fraction 0.5 used by KP for validation for lstm experiments
            fraction = 0.5 if (dataset == 'sent140') else 0.8
            split_point = int(fraction * len(train_users))
            validation_users = train_users[split_point:]
            validation_data = {}
            validation_groups = train_groups[split_point:]
            for u in validation_users:
                validation_data[u] = dict(train_data[u])
                del train_data[u]

            train_users = train_users[:split_point]
            train_groups = train_groups[:split_point]

    else: # split by sample
        users = clients['train_users']
    # subsample
        if dataset == 'femnist':
            rng_sub = random.Random(25)
            users = rng_sub.sample(users, 1000)
            train_data = {u: p for (u, p) in train_data.items() if u in users}
            test_data = {u: p for (u, p) in test_data.items() if u in users}
            print('Subsampled number of users:', len(train_data))
            # note: groups are empty
        elif dataset == 'shakespeare':
            pass
        elif dataset == 'sent140':
            print('Not subsampling data for sent140')
        else:
            print('Not subsampling data')
            sys.exit(-1)

        if validation:  # split training set into train and val in the ratio 80:20
            print('Validation mode, splitting train data into train and val sets...')
            for idx, u in enumerate(users):
                data = list(zip(train_data[u]['x'], train_data[u]['y']))
                rng = random.Random(idx)
                rng.shuffle(data)
                split_point = int(0.8 * len(data))
                x, y = zip(*data[:split_point])
                x1, y1 = zip(*data[split_point:])
                train_data[u] = {'x': list(x), 'y': list(y)}
                test_data[u] = {'x': list(x1), 'y': list(y1)}
        if len(groups) == 0:
            groups = [[] for _ in users]
    if split_by_user:

        print('------>', len(train_users))

        train_clients = []
        test_clients = []

        # if True, do not preprocess data
        do_preprocess = not (
                dataset == 'sent140' and model_name in ['erm_lstm_log_reg', 'rsm_lstm_log_reg']
        )
        print('do_preprocess = ', do_preprocess)

        for u, g in zip(train_users, train_groups):
            train_data_u_x = (
                preprocess_data_x(train_data[u]['x'],
                                  dataset=dataset,
                                  model_name=model_name)
                if do_preprocess else train_data[u]['x']
            )
            train_data_u_y = (
                preprocess_data_y(train_data[u]['y'],
                                  dataset=dataset,
                                  model_name=model_name)
                if do_preprocess else train_data[u]['y']
            )
            train_data_u = {'x': train_data_u_x, 'y': train_data_u_y}
            train_clients.append(Client(u, g, train_data=train_data_u, model=model, dataset=dataset))
            corrupted_clients = apply_corruption_all(train_clients, dataset, corruption, fraction_corrupt, seed)

        for u, g in zip(test_users, test_groups):
            test_data_u_x = (
                preprocess_data_x(test_data[u]['x'],
                                  dataset=dataset,
                                  model_name=model_name)
                if do_preprocess else test_data[u]['x']
            )
            test_data_u_y = (
                preprocess_data_y(test_data[u]['y'],
                                  dataset=dataset,
                                  model_name=model_name)
                if do_preprocess else test_data[u]['y']
            )
            test_data_u = {'x': test_data_u_x, 'y': test_data_u_y}
            test_clients.append(Client(u, g,  eval_data=test_data_u, model=model, dataset=dataset))

        all_clients = {
            'train_clients': train_clients,
            'test_clients': test_clients
        }

        if validation:
            validation_clients = []
            for u, g in zip(validation_users, validation_groups):
                validation_data_u_x = (
                    preprocess_data_x(validation_data[u]['x'],
                                      dataset=dataset,
                                      model_name=model_name)
                    if do_preprocess else validation_data[u]['x']
                )
                validation_data_u_y = (
                    preprocess_data_y(validation_data[u]['y'],
                                      dataset=dataset,
                                      model_name=model_name)
                    if do_preprocess else validation_data[u]['y']
                )
                validation_data_u = {'x': validation_data_u_x, 'y': validation_data_u_y}
                validation_clients.append(Client(u, g, train_data=validation_data_u, model=model, dataset=dataset))

            all_clients['validation_clients'] = validation_clients

        return all_clients, corrupted_clients

    else:
        # every client is used for training when split_by_user is False
        train_clients = []
        for u, g in zip(users, groups):
            # TODO: skip preprocess if necessary
            train_data_u_x = preprocess_data_x(train_data[u]['x'], dataset=dataset, model_name=model_name)
            train_data_u_y = preprocess_data_y(train_data[u]['y'], dataset=dataset, model_name=model_name)
            test_data_u_x = preprocess_data_x(test_data[u]['x'], dataset=dataset, model_name=model_name)
            test_data_u_y = preprocess_data_y(test_data[u]['y'], dataset=dataset, model_name=model_name)

            train_data_u = {'x': train_data_u_x, 'y': train_data_u_y}
            test_data_u = {'x': test_data_u_x, 'y': test_data_u_y}

            train_clients.append(Client(u, g, train_data_u, test_data_u, model, dataset=dataset))
        # all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)] # old code
        corrupted_clients = apply_corruption_all(train_clients, dataset, corruption, fraction_corrupt, seed)

        all_clients = {
            'train_clients':train_clients
        }

        return all_clients, corrupted_clients


def corrupt_one_client_flip(dataset, client, corruption):
    """Flip labels of train data of given client in-place. Note: eval data is unchanged"""
    x = client.train_data['x']
    y = client.train_data['y']
    if dataset == 'femnist':
        if corruption == CORRUPTION_FLIP_KEY:
            client.train_data['y_true'] = copy.deepcopy(y)
            for i in range(len(y)):
                if y[i] < 10:
                    # digit: apply deterministic permutation
                    y[i] = (7 * y[i] + 1) % 10
                elif y[i] < 36:
                    # upper case letter: convert to lower case
                    y[i] += 26
                else:
                    # lower case letter: convert to upper case
                    y[i] -= 26
        elif corruption == CORRUPTION_P_X_KEY:
            x_new = []
            for img in x:
                x_new.append([1-pixel for pixel in img])
            client.train_data['x_true'] = x
            client.train_data['x'] = x_new
        else:
            raise ValueError('Unknown corruption, {}'.format(corruption))
    elif dataset == 'sent140':
        for i in range(len(y)):
            y[i] = 1 - y[i]  # flip 0-1 labels
    elif dataset == 'shakespeare':
        x_new = []
        y_new = []
        for i in range(len(y)):
            # reverse sentence
            s = x[i] + y[i]
            s = s[::-1]
            x_new.append(s[:-1])
            y_new.append(s[-1])
        # modify client in-place
        client.train_data['x_true'] = x
        client.train_data['y_true'] = y
        client.train_data['x'] = x_new
        client.train_data['y'] = y_new


def apply_corruption_all(client_list, dataset, corruption, fraction_corrupt, seed):
    """Apply corruptions to clients so that a total of `fraction_corrupt` fraction of the data is corrupted.
        Return list of corrupt clients and modify client_list inplace.
    """
    if corruption:
        client_dict = {client.id: client for client in client_list}
        rng = random.Random(seed - 1)
        users = [client.id for client in client_list]
        rng.shuffle(users)

        # choose prefix of `users` to corrupt until fraction has just been exceeded
        num_data_pts = [len(client_dict[u].train_data['y']) for u in users]
        total_num_data_pts = sum(num_data_pts)
        target_num_data_pts = fraction_corrupt * total_num_data_pts  # number of data points to corrupt
        num_corrupted_data_pts = 0
        end_idx = 0  # exclusive
        while num_corrupted_data_pts < target_num_data_pts:
            num_corrupted_data_pts += num_data_pts[end_idx]
            end_idx += 1
        corrupted_clients = users[:end_idx]
        print('Corrupting {:0.4f} fraction of data'.format(num_corrupted_data_pts / total_num_data_pts))

        # flip labels if need be
        if corruption in [CORRUPTION_FLIP_KEY, CORRUPTION_P_X_KEY]:
            for u in corrupted_clients:
                corrupt_one_client_flip(dataset, client_dict[u], corruption)

    else:
        corrupted_clients = []

    return frozenset(corrupted_clients)


def get_corrupted_fraction(selected_clients, corrupted_client_ids):
    total_num_pts = sum([len(c.train_data['y']) for c in selected_clients])
    corrupted_lens = [len(c.train_data['y']) for c in selected_clients
                      if c.id in corrupted_client_ids]
    num_corrupted_clients = len(corrupted_lens)
    num_corrupted_pts = sum(corrupted_lens)
    return (num_corrupted_clients, len(selected_clients),
            num_corrupted_pts / total_num_pts)


def save_model(server_model, dataset, model, output_summary_file):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    start_time = time.time()
    ckpt_path = os.path.join('checkpoints', *(output_summary_file.split(os.path.sep)[1:]))
    print(ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save('%s.ckpt' % ckpt_path)
    print('Model saved in path: {} in time {:.2f} sec'.format(save_path, time.time() - start_time))


def print_metrics(iteration, comm_rounds, metrics, train_weights, test_weights, elapsed_time=0):
    """Prints weighted averages of the given metrics.
    Args:
        iteration: current iteration number
        comm_rounds: number of communication rounds
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        train_weights: dict with client ids as keys. Each entry is the weight
            for that client for training metrics.
        test_weights: dict with client ids as keys. Each entry is the weight
            for that client for testing metrics
        elapsed_time: time taken for testing
    """
    output = {'iteration': iteration, 'comm_rounds': comm_rounds}
    if metrics is None:
        print(iteration, comm_rounds)
    else:
        print(iteration, end=', ')
        ordered_tr_weights = [train_weights[c] for c in sorted(train_weights)]
        ordered_te_weights = [test_weights[c] for c in sorted(test_weights)]
        metric_names = get_metrics_names(metrics)
        for metric in metric_names:
            ordered_weights = ordered_tr_weights if metric in TRAINING_KEYS else ordered_te_weights
            ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
            avg_metric = np.average(ordered_metric, weights=ordered_weights)
            output[metric] = avg_metric
            print('%s: %g' % (metric, avg_metric), end=', ')
        print('Time:', timedelta(seconds=round(elapsed_time)))
    sys.stdout.flush()
    return output

def convert_to_iid(client_list, seed, train=True):
    rng = random.Random(seed)
    x, y = [], []
    for c in client_list:
        data = c.train_data if train else c.eval_data
        x.extend(data['x']); y.extend(data['y'])
    # split per-clients
    n = len(x) // len(client_list)  # num data per client
    batched_x, batched_y = batch_data({'x': x, 'y': y}, n, rng)
    for i, c in enumerate(client_list):
        data = c.train_data if train else c.eval_data
        data['x'], data['y'] = batched_x[i], batched_y[i]

def get_metrics_names(metrics):
    """Gets the names of the metrics.
    Args:
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys."""
    if len(metrics) == 0:
        return []
    metrics_dict = next(iter(metrics.values()))
    return list(metrics_dict.keys())


if __name__ == '__main__':
    main()
