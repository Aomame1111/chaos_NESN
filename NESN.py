# 重みを変えてstep,freeを10回予測
# 重みを保存
# 写真を保存
# 重みと予測結果の関係性を見ることが目的

# 関数のインポート
import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt


# 関数の定義
def make_bias(n_x, connection_rate):
    bias = np.zeros(n_x)
    not_zero_location_number = int(connection_rate * n_x)
    not_zero_locations = np.empty(0)

    for i in range(not_zero_location_number):
        while not_zero_locations.size != i + 1:
            not_zero_location = np.random.randint(0, n_x - 1)
            if not_zero_location not in not_zero_locations:
                not_zero_locations = np.append(not_zero_locations, not_zero_location)

    for j in range(not_zero_location_number):
        bias[int(not_zero_locations[j])] = np.random.uniform(-1, 1)

    bias = bias.reshape(n_x, 1)
    return bias


def make_weight(line, column, range_low, range_high, connection_rate):
    weights = np.zeros(column * line)

    not_zero_location_number = int(connection_rate * line * column)
    not_zero_locations = np.empty(0)

    for i in range(not_zero_location_number):
        while not_zero_locations.size != i + 1:
            not_zero_location = np.random.randint(0, line * column - 1)
            if not_zero_location not in not_zero_locations:
                not_zero_locations = np.append(not_zero_locations, not_zero_location)

    for j in range(not_zero_location_number):
        weights[int(not_zero_locations[j])] = np.random.uniform(range_low, range_high)

    weights = weights.reshape(line, column)
    return weights


def make_weight_reservoir(n_x, connection_rate):
    w_res = make_weight(n_x, n_x, -1, 1, connection_rate)
    eig, _ = la.eig(w_res)
    spectral_radius = max(abs(eig))
    w_res = w_res / spectral_radius
    eig, _ = la.eig(w_res)
    spectral_radius = max(abs(eig))

    while spectral_radius >= 1:
        w_res = make_weight(n_x, n_x, -1, 1, connection_rate)
        eig, _ = la.eig(w_res)
        spectral_radius = max(abs(eig))
        w_res = w_res / spectral_radius
        eig, _ = la.eig(w_res)
        spectral_radius = max(abs(eig))

    print(spectral_radius)
    return w_res


def data_get(filename):
    f = filename
    data = pd.read_csv(f, encoding='cp932',
                       header=0, names=['step', 'x', 'y', 'z'])
    xyz_t = data.to_numpy()
    t_and_xyz = xyz_t.T
    return t_and_xyz


def graph_prepare():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("lorenz")
    return fig, ax


# クラスの定義
class RLS:
    def __init__(self, nx, ny, lam, delta):
        self.lam = lam
        delta = delta
        self.P = 1/delta * np.eye(nx, nx)
        self.w_out = np.zeros((ny, nx))

    def __call__(self, x, output):
        for i in range(10):
            v = output - np.dot(self.w_out, x)
            gain = 1/self.lam * np.dot(self.P, x)
            gain = gain/(1 + 1/self.lam * np.dot(np.dot(x.T, self.P), x))
            self.P = 1/self.lam * (self.P - np.dot(np.dot(gain, x.T), self.P))
            self.w_out = self.w_out + np.dot(v, gain.T)

        return self.w_out


class NESN:
    def __init__(self, n_u, n_x, n_y, filename, training_percent, activation_function=np.tanh):
        self.n_u = n_u
        self.n_x = n_x
        self.n_y = n_y
        self.act_func = activation_function

        self.bias = make_bias(self.n_x, 0.5)
        self.w_in = make_weight(self.n_x, self.n_u, -0.1, 0.1, 0.1)
        self.w_ofb = make_weight(self.n_x, self.n_y, -0.1, 0.1, 0.1)
        self.w_out = np.zeros((self.n_y, self.n_x))
        self.w = make_weight_reservoir(self.n_x, 0.1)

        self.filename = filename
        self.input_data_t_xyz = data_get(self.filename)
        self.input_data_size = self.input_data_t_xyz.shape[1]
        self.time_data, self.input_data = np.split(self.input_data_t_xyz, [1])

        self.training_percent = training_percent
        self.training_data_size = int(self.input_data_size * self.training_percent)
        self.test_data_size = self.input_data_size - self.training_data_size
        self.training_time_data, self.test_time_data = np.hsplit(self.time_data, [self.training_data_size])
        self.training_time_data = np.append(self.training_time_data, self.training_data_size)
        self.training_data, self.test_data = np.hsplit(self.input_data,
                                                       [self.training_data_size])

        self.reservoir_one_step = np.zeros((self.n_x, 1))
        # self.reservoir_one_step = np.ones((self.n_x, 1))
        self.output_one_step = np.dot(self.w_out, self.reservoir_one_step)
        self.reservoir_free_run = np.zeros((self.n_x, 1))
        # self.reservoir_free_run = np.ones((self.n_x, 1))
        self.output_free_run = np.dot(self.w_out, self.reservoir_free_run)

        self.output_test_one_step = self.test_data[:, [0]]
        self.output_test_free_run = self.test_data[:, [0]]

    def train_one_step(self, optimizer, number):
        self.w_out = np.zeros((self.n_y, self.n_x))

        for i in range(self.training_data_size):
            reservoir = self.act_func(np.dot(self.w, self.reservoir_one_step[:, [i]])
                                      + np.dot(self.w_ofb, self.training_data[:, [i]]) + self.bias)
            output = np.dot(self.w_out, reservoir)
            self.reservoir_one_step = np.hstack((self.reservoir_one_step, reservoir))
            self.output_one_step = np.hstack((self.output_one_step, output))
            if i < self.training_data_size-1:
                self.w_out = optimizer(self.reservoir_one_step[:, [i+1]], self.training_data[:, [i+1]])
        np.savez('weight/step_weight' + str(number), bias=self.bias, w_in=self.w_in, w_ofb=self.w_ofb,
                 w_res=self.w, w_out=self.w_out)

    def train_free_run(self, optimizer, number):
        self.w_out = np.zeros((self.n_y, self.n_x))

        for i in range(self.training_data_size):
            reservoir = self.act_func(np.dot(self.w, self.reservoir_free_run[:, [i]])
                                      + np.dot(self.w_ofb, self.output_free_run[:, [i]]) + self.bias)
            output = np.dot(self.w_out, reservoir)
            self.reservoir_free_run = np.hstack((self.reservoir_free_run, reservoir))
            self.output_free_run = np.hstack((self.output_free_run, output))
            if i < self.training_data_size-1:
                self.w_out = optimizer(self.reservoir_free_run[:, [i+1]], self.training_data[:, [i+1]])
        np.savez('weight/free_weight' + str(number), bias=self.bias, w_in=self.w_in, w_ofb=self.w_ofb,
                 w_res=self.w, w_out=self.w_out)

    # 更新済みのw_outを用いて値を出力していく
    def test(self, step_or_free):
        if step_or_free == 'step':
            # one_step
            for j in range(self.test_data_size):
                reservoir_test = self.act_func(np.dot(self.w,
                                                      self.reservoir_one_step[:, [j+self.training_data_size]])
                                               + np.dot(self.w_ofb, self.output_test_one_step[:, [j]]) + self.bias)
                output_test = np.dot(self.w_out, reservoir_test)
                self.reservoir_one_step = np.hstack((self.reservoir_one_step, reservoir_test))
                self.output_test_one_step = np.hstack((self.output_test_one_step, output_test))

        if step_or_free == 'free':
            # free_run
            for j in range(self.test_data_size):
                reservoir_test = self.act_func(np.dot(self.w,
                                                      self.reservoir_free_run[:, [j+self.training_data_size]])
                                               + np.dot(self.w_ofb, self.output_test_free_run[:, [j]]) + self.bias)
                output_test = np.dot(self.w_out, reservoir_test)
                self.reservoir_free_run = np.hstack((self.reservoir_free_run, reservoir_test))
                self.output_test_free_run = np.hstack((self.output_test_free_run, output_test))

    # 2つのグラフを１つの図に描く
    def graph(self, number):
        fig_one_step, ax_one_step = graph_prepare()
        ax_one_step.plot(self.input_data[0, :],
                         self.input_data[1, :], self.input_data[2, :],
                         color="green", lw=1)
        ax_one_step.plot(self.output_one_step[0, :],
                         self.output_one_step[1, :], self.output_one_step[2, :],
                         color="blue", lw=1)
        ax_one_step.plot(self.output_test_one_step[0, :],
                         self.output_test_one_step[1, :], self.output_test_one_step[2, :],
                         color="red", lw=1)
        # plt.show()
        plt.savefig('picture/noisy_xyz_step' + str(number) + '.png')
        plt.close()

        fig_free_run, ax_free_run = graph_prepare()
        ax_free_run.plot(self.input_data[0, :],
                         self.input_data[1, :], self.input_data[2, :],
                         color="green", lw=1)
        ax_free_run.plot(self.output_free_run[0, :],
                         self.output_free_run[1, :], self.output_free_run[2, :],
                         color="blue", lw=1)
        ax_free_run.plot(self.output_test_free_run[0, :],
                         self.output_test_free_run[1, :], self.output_test_free_run[2, :],
                         color="red", lw=1)
        # plt.show()
        plt.savefig('picture/noisy_xyz_free' + str(number) + '.png')
        plt.close()
