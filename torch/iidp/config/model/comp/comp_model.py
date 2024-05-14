import numpy as np
import matplotlib.pyplot as plt


class StreamParallelThroughputModel(object):
    def __init__(self):
        self.a = 0
        self.b = 0
        self.x_data = []
        self.y_data = []

    def train(self, x_data, y_data):
        """
        Arguments:
            x_data: List of number of VSWs
            y_data: List of throughput normalized by 1 VSW
        """
        #print(f'[INFO] x_data: {x_data}')
        #print(f'[INFO] y_data: {y_data}')
        self.x_data = x_data
        self.y_data = y_data
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        xlog_data = np.log(x_data)
        self.a, self.b = np.polyfit(xlog_data, y_data, 1)

        #print('============= Trained Logarithmic Model =============')
        #print(f"slope (a): {self.a}")
        #print(f"intercept (b): {self.b}")
        #print('====================================================')

    def evaluate(self, x):
        """
        Arguments:
            x: Number of VSWs
        Return:
            Normalized throughput
        """
        y = self.a * np.log(x) + self.b
        return y

    def plot(self, x_data, y_data, file_path='comp_model.png', title='Stream parallel throughput model'):
        plt.clf()
        plt.scatter(x_data, y_data, c='tab:red', marker='o', label='Data')
        x_new, y_pred = [], []
        for x in range(1, x_data[-1]+1):
            x_new.append(x)
            y_pred.append(self.evaluate(x))
        plt.plot(x_new, y_pred, c='tab:blue', ls='-', label='Model')
        plt.xlabel('Number of VSWs')
        plt.ylabel('Normalized throughput')
        plt.legend()
        plt.title(title)
        plt.savefig(file_path)

    def __repr__(self):
        pass


class TrueThroughputModel(object):
    def __init__(self):
        self.true_thp_model = []

    def train(self, x_data, y_data):
        #print(f'[INFO] x_data: {x_data}')
        #print(f'[INFO] y_data: {y_data}')
        self.true_thp_model = y_data

    def evaluate(self, x):
        return self.true_thp_model[x-1]