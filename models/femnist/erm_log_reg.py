from model import Model, Optimizer
import numpy as np

IMAGE_SIZE = 28

class ClientModel(Model):

    def __init__(self, lr, num_classes, max_batch_size=None, seed=None, optimizer=None):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(lr, seed, max_batch_size, optimizer=ErmOptimizer())

    def create_model(self):
        """Model function for linear model."""
        pass

class ErmOptimizer(Optimizer):

    def __init__(self, starting_w=np.zeros((784, 62))):

        super(ErmOptimizer, self).__init__(starting_w)
        self.optimizer_model = None
        self.learning_rate = 0.2  # used before launched 0.52 on train accuracy
        self.lmbda = 0.0

    def single_loss(self, x, y):
        y_hat = self.softmax(np.dot(x, self.w))
        return - np.sum(y * np.log(y_hat) + 1e-6) + self.lmbda/2 * np.linalg.norm(self.w)**2

    def loss(self, batched_x, batched_y):
        n = len(batched_y)
        loss = 0.0
        for i in range(n):
            loss += self.single_loss(batched_x[i], batched_y[i])
        averaged_loss = loss / n
        return averaged_loss

    def gradient(self, x, y):  # x is only 1 image here
        p_hat = self.softmax(np.dot(x, self.w))  # vector of probabilities
        a = np.sum(y) * p_hat - y

        return np.outer(x, a) + self.lmbda * self.w

    def run_step(self, batched_x, batched_y):
        loss = 0.0
        s = np.zeros(self.w.shape)
        n = len(batched_y)
        for i in range(n):
            s += self.learning_rate * self.gradient(batched_x[i], batched_y[i])
            loss += self.single_loss(batched_x[i], batched_y[i])
        self.w -= s/n
        averaged_loss = loss/n

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

    def softmax(self, x):

        res = np.exp(x - np.max(x))
        s = np.sum(res)

        return res / s
