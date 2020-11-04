"""
    date:       2020/10/11 3:46 下午
    written by: neonleexiang
"""
import numpy as np
import matplotlib.pyplot as plt
import random


class LinearRegression:
    def __init__(self, alpha):
        self._alpha = alpha
        # we need not to store the data_x and data_y

        # self._x = 0
        # self._y = 0

        self._theta0 = 0
        self._theta1 = 0

    @staticmethod
    def hypothesis(theta0, theta1, x):
        return theta0 + (theta1 * x)

    # normal derivatives
    def derivatives(self, theta0, theta1, x, y):
        d_theta0 = 0
        d_theta1 = 0
        for (xi, yi) in zip(x, y):
            d_theta0 += self.hypothesis(theta0, theta1, xi) - yi
            d_theta1 += (self.hypothesis(theta0, theta1, xi) - yi) * xi

        d_theta0 /= len(x)
        d_theta1 /= len(x)

        return d_theta0, d_theta1

    # stochastic gradient descent derivatives
    def sgd_derivatives(self, theta0, theta1, x, y):
        d_theta0 = 0
        d_theta1 = 0
        xi, yi = random.choice(list(zip(x, y)))
        d_theta0 += self.hypothesis(theta0, theta1, xi) - yi
        d_theta1 += (self.hypothesis(theta0, theta1, xi) - yi) * xi
        return d_theta0, d_theta1

    '''
        因为我们给的房价数据非常小，没有必要用随机梯度下降算法，直接批量梯度下降能在短期获得最好的拟合效果
        小数据集的时候收敛效果没有批量梯度下降好。
    '''
    # normal gradient descent not stochastic gradient descent
    def update_parameters(self, theta0, theta1, x, y, alpha):
        d_theta0, d_theta1 = self.derivatives(theta0, theta1, x, y)
        theta0 = theta0 - (alpha * d_theta0)
        theta1 = theta1 - (alpha * d_theta1)
        return theta0, theta1

    # stochastic gradient descent
    def sgd_update_parameters(self, theta0, theta1, x, y, alpha):
        d_theta0, d_theta1 = self.sgd_derivatives(theta0, theta1, x, y)
        theta0 = theta0 - (alpha * d_theta0)
        theta1 = theta1 - (alpha * d_theta1)
        return theta0, theta1

    def cost(self, theta0, theta1, x, y):
        cost_value = 0
        for (xi, yi) in zip(x, y):
            cost_value += (self.hypothesis(theta0, theta1, xi) - yi) ** 2

        return 0.5 * cost_value

    @staticmethod
    def plot_line(theta0, theta1, x, y):
        max_x = np.max(x) + 1
        min_x = np.min(x) - 1

        x_plot = np.linspace(min_x, max_x, 10)
        y_plot = theta0 + theta1 * x_plot

        plt.plot(x_plot, y_plot, color='#58b970', label='Regression Line')

        plt.scatter(x, y)
        plt.axis([0, 15, 0, 20])
        plt.show()

    def train(self, x, y):
        self._theta0 = np.random.rand()
        self._theta1 = np.random.rand()

        for i in range(1000):
            # if i % 100 == 0:
            #     self.plot_line(self._theta0, self._theta1, x, y)
            #     print(cost(self._theta0, self._theta1, x, y))
            self._theta0, self._theta1 = self.update_parameters(self._theta0, self._theta1, x, y, self._alpha)
        self.plot_line(self._theta0, self._theta1, x, y)
        # print(self.cost(self._theta0, self._theta1, x, y))

    def sgd_train(self, x, y):
        self._theta0 = np.random.rand()
        self._theta1 = np.random.rand()

        # 按老师的要求
        for i in range(10):
            # self.plot_line(self._theta0, self._theta1, x, y)
            self._theta0, self._theta1 = self.sgd_update_parameters(self._theta0, self._theta1, x, y, self._alpha)
        self.plot_line(self._theta0, self._theta1, x, y)
        # print(self.cost(self._theta0, self._theta1, x, y)

    def predict(self, x):
        return self.hypothesis(self._theta0, self._theta1, x)


if __name__ == '__main__':
    '''
        x 为年份，如果直接用2000，2001 x 数据相对 y 数据太大，且 x 数据之间太粘稠，差异性不大，所以我将它减去1999变为1，2，...
        这样数据不黏稠且相对 y 来说不会太大，以及对于整个模型拟合度也会更高，这样我们的随机梯度下降能在次数少的情况下达到一个非常良好的
        模型拟合效果，同时也不会因为如果用了其他数据标准化的方法进而对预测的数据处理造成麻烦。
        目前来看，相对舍友的直接使用模型还是非常的有效的，整个数据预处理对模型的提升效果。
    '''
    '''
        以下为两种方式的测试，一种是正常的我们对于小数据量使用批量梯度下降迭代了1000次的效果，另一种是用老师要求的10次的随机梯度下降迭代
        简单的测试发现两者差距并不大。但是批量梯度下降的效果肯定会更加好。从损失函数即可分析得出。
    '''
    data_x = np.array([[i-1999] for i in range(2000, 2014)])
    # print(data_x)
    data_y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.853, 7.971, 8.561, 10.000, 11.280, 12.900]
    data_y = np.array(data_y)

    # --------------- normal gradient descent -------------

    # linear_regression_0 = LinearRegression(0.01)
    # linear_regression_0.train(data_x, data_y)
    # print(linear_regression_0.predict(2014-1999))   # result is 12.30953985

    # --------------- stochastic gradient descent ----------

    sgd_linear_regression_0 = LinearRegression(0.01)
    sgd_linear_regression_0.sgd_train(data_x, data_y)
    print(sgd_linear_regression_0.predict(2014-1999))   # result is 13.87409418 or 11.82537556 or .....
