from model import Model, Optimizer
import numpy as np

IMAGE_SIZE = 28


class ClientModel(Model):

    def __init__(self, lr, num_classes, max_batch_size=None, seed=None, optimizer=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr, seed, max_batch_size, optimizer=ErmOptimizer())

    def create_model(self):
        """Model function for linear model."""
        #features = tf.placeholder(
        #    tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        #labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        #logits = tf.layers.dense(inputs=features, units=self.num_classes)
        #predictions = {
        #    "classes": tf.argmax(input=logits, axis=1),
        #    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        #}
        #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        #train_op = self.optimizer.minimize(
        #    loss=loss,
        #    global_step=tf.train.get_global_step())
        #eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        #return features, labels, loss, train_op,

    # def process_y(self, raw_y_batch):
    #     """Pre-processes each batch of labels before being fed to the model."""
    #     res = []
    #     for i in range(len(raw_y_batch)):
    #         num = np.zeros(62) # Number of classes
    #         num[raw_y_batch[i]] = 1.0
    #         res.append(num)
    #     return np.asarray(res)

class ErmOptimizer(Optimizer):

    def __init__(self, starting_w=np.zeros((784, 62))):

        super(ErmOptimizer, self).__init__(starting_w)
        self.optimizer_model = None
        # self.learning_rate = 0.0003
        self.learning_rate = 0.017  # used before launched 0.52 on train accuracy
        # self.lmbda = 0.01
        # self.lmbda = 0.1  # used before
        self.lmbda = 0.0
        # print("lol" + str(self.learning_rate))

    def single_loss(self, x, y):
        return 0.5 * np.linalg.norm(y - np.dot(x, self.w))**2 + self.lmbda/2 * np.linalg.norm(self.w)**2

    def loss(self, batched_x, batched_y):
        n = len(batched_y)
        loss = 0.0
        for i in range(n):
            loss += self.single_loss(batched_x[i], batched_y[i])
        averaged_loss = loss / n
        return averaged_loss

    def gradient(self, x, y): # x is only 1 image here
        return -1.0 * np.outer(x, y - np.dot(x, self.w)) + self.lmbda * self.w

    def run_step(self, batched_x, batched_y):
        loss = 0.0
        s = np.zeros(self.w.shape)
        n = len(batched_y)
        for i in range(n):
            s += self.learning_rate * self.gradient(batched_x[i], batched_y[i])
            loss += self.single_loss(batched_x[i], batched_y[i])
        self.w -= s/n
        averaged_loss = loss/n
        # print('learning rate ? ' + str(self.learning_rate))

        return averaged_loss

    def update_w(self):
        self.w_on_last_update = self.w

    def correct_single_label(self, x, y):
        prediction = np.argmax(np.dot(x, self.w))
        ground_value = np.argmax(y)

        return float(prediction == ground_value)

    def initialize_w(self):
        self.w = np.zeros((784, 62))
        self.w_on_last_update = np.zeros((784, 62))

    def correct(self, x, y):
        nb_correct = 0.0
        for i in range(len(y)):
            nb_correct += self.correct_single_label(x[i], y[i])
        return nb_correct

    def size(self):
        return 784*62
