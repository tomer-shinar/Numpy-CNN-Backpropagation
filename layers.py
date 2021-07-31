import pickle


class Layer:
    """
    base class for all layers
    """
    def __init__(self):
        import numpy as np_device
        self.device = np_device
        self.on_train = True

    def train(self):
        self.on_train = True

    def eval(self):
        self.on_train = False

    def __call__(self, X):
        if self.on_train:
            self.memory_input = X
        return self.forward(X)

    def to(self, device='cpu'):
        if device == 'cpu':
            import numpy as np_device
            self.device = np_device
        elif device == 'gpu':
            import cupy as cp_device
            self.device = cp_device
        self.memory_input, self.grads = None, None

    def forward(self, X):
        return X

    def backward(self, error):
        assert self.on_train
        return error

    def fit(self, lr, weight_decay):
        assert self.on_train

    def save(self, file_name):
        """
        save the model to pickle file.
        :param file_name: the file name saving to.
        """
        self.to('cpu')
        f = open(file_name, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()


class Dropout(Layer):
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.keep_prob = 1 - drop_rate

    def forward(self, X):
        if self.on_train:
            self.mask = self.device.random.random_sample(X.shape)
            self.mask = self.mask < self.keep_prob
            X = self.device.multiply(X, self.mask)
            X = X / self.keep_prob
        return X

    def backward(self, error):
        assert self.on_train
        return self.device.multiply(error, self.mask) / self.keep_prob

    def to(self, device='cpu'):
        super().to(device)
        self.mask = None

    # empty fit


class Softmax(Layer):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, X):
        exp = self.device.exp(X)
        X = exp / self.device.tile(exp.sum(1), (X.shape[1], 1)).T
        return X

    # empty backward and fit


class Relu(Layer):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, X):
        return self.device.maximum(X, 0)

    def backward(self, error):
        assert self.on_train
        return (self.memory_input > 0).astype(int) * error

    # empty fit


class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, X):
        return self.device.tanh(X)

    def backward(self, error):
        assert self.on_train
        # TODO derive tanh
        return (self.memory_input > 0).astype(int) * error

    # empty fit


def un_pool(device,error, kernel_size):
    b, h, w, c = error.shape
    error = device.stack([error] * kernel_size ** 2)
    error = error.reshape(kernel_size, kernel_size, b, h, w, c)
    error = error.transpose(2, 3, 0, 4, 1, 5)
    error = error.reshape(b, h * kernel_size, w * kernel_size, c)
    return error


class AvgPooling2D(Layer):
    def __init__(self, kernel_size: int):
        super(AvgPooling2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, X):
        # X = (batch_size, width, height, channels)
        assert X.shape[1] % self.kernel_size == 0 and X.shape[2] % self.kernel_size == 0
        X = X.reshape(X.shape[0], X.shape[1] // self.kernel_size, self.kernel_size,
                      X.shape[2] // self.kernel_size, self.kernel_size, X.shape[-1]).mean(axis=(2, 4))
        return X

    def backward(self, error):
        assert self.on_train
        return un_pool(self.device, error, self.kernel_size)

    # empty fit


class MaxPooling2D(Layer):
    def __init__(self, kernel_size: int):
        super(MaxPooling2D, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, X):
        assert X.shape[1] % self.kernel_size == 0 and X.shape[2] % self.kernel_size == 0
        pooled_X = X.reshape(X.shape[0], X.shape[1] // self.kernel_size, self.kernel_size,
                             X.shape[2] // self.kernel_size, self.kernel_size, X.shape[-1]).max(axis=(2, 4))
        self.mask = (X == un_pool(self.device, pooled_X, self.kernel_size))
        return pooled_X

    def backward(self, error):
        assert self.on_train
        error = un_pool(self.device, error, self.kernel_size)
        error = error * self.mask.astype(int)
        return error

    def to(self, device='cpu'):
        super().to(device)
        self.mask = None


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.W = self.xavier_init(out_dim, in_dim)
        self.b = self.device.zeros(out_dim)

    def to(self, device='cpu'):
        super().to(device)
        self.W, self.b = self.device.asarray(self.W), self.device.asarray(self.b)

    def xavier_init(self, m, n):
        limit = (6 / (n * m)) ** 0.5
        weights = self.device.random.uniform(-limit, limit, size=(m, n))
        return weights

    def forward(self, X):
        return self.device.dot(self.W, X.T).T + self.b

    def backward(self, error):
        assert self.on_train
        self.grads = error
        return self.device.dot(self.W.T, error.T).T

    def fit(self, lr, weight_decay=0.05):
        self.W -= lr * (self.device.dot(self.grads.T, self.memory_input) + weight_decay * self.W)
        self.b -= lr * self.grads.sum(0)


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=True):
        super(Conv2d, self).__init__()
        self.W = self.kaiming_init(in_channels, (out_channels, in_channels * kernel_size ** 2))
        self.b = self.device.zeros(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

    def kaiming_init(self, n, shape):
        return self.device.random.normal(size=shape) * (2 / n) ** 0.5

    def to(self, device='cpu'):
        super().to(device)
        self.W, self.b = self.device.asarray(self.W), self.device.asarray(self.b)

    def pad(self, X):
        b, h, w, c = X.shape
        pad = self.kernel_size // 2
        return self.device.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))

    def forward(self, X):
        if self.padding:
            X = self.pad(X)
            self.memory_input = X
        b, h, w, c = X.shape
        output = self.device.zeros((b, h - self.kernel_size + 1, w - self.kernel_size + 1, self.out_channels))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                start = (j + i * self.kernel_size) * self.in_channels
                output += (X[:, i: h + i - self.kernel_size + 1, j: w + j - self.kernel_size + 1, self.device.newaxis,
                           :]
                           @ self.W[:, start:start + self.in_channels].T).reshape(b, h - self.kernel_size + 1,
                                                                                  w - self.kernel_size + 1,
                                                                                  self.out_channels)
        output = b + output
        return output

    def backward(self, error):
        assert self.on_train
        b, h, w, _ = error.shape
        self.grads = error
        error = error @ self.W
        dA = self.device.zeros((b, h + self.kernel_size - 1, w + self.kernel_size - 1, self.in_channels))
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                start = (j + i * self.kernel_size) * self.in_channels
                dA[:, i: h + i, j: w + j, :] += error[:, :, :, start: start + self.in_channels]
        if self.padding:
            pad = self.kernel_size // 2
            dA = dA[:, pad:-pad, pad:-pad, :]
        return dA

    def fit(self, lr, weight_decay=0.05):
        b, h, w, _ = self.grads.shape
        input_frames = [self.memory_input[:, i:h + i, j:w + j, :].reshape(-1, 1, self.in_channels) for i in
                        range(self.kernel_size) for j in range(self.kernel_size)]
        dw = self.grads.reshape(-1, self.out_channels, 1) @ self.device.stack(input_frames)
        dw = dw.sum(1).transpose(0, 1, 2).reshape(self.out_channels, -1)
        self.W -= lr * (dw + weight_decay * self.W)
        self.b -= self.grads.sum(axis=(0, 1, 2))


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(self.shape[0], -1)

    def backward(self, error):
        assert self.on_train
        return error.reshape(*self.shape)


class BatchNorm(Layer):
    def __init__(self, shape):
        super(BatchNorm, self).__init__()
        self.gamma = self.device.ones(shape)
        self.beta = self.device.zeros(shape)

    def to(self, device='cpu'):
        super().to(device)
        self.gamma, self.beta = self.device.asarray(self.gamma), self.device.asarray(self.beta)

    def forward(self, X):
        eps = 1e-5
        sample_mean = X.mean(axis=0)
        sample_var = X.var(axis=0)
        std = self.device.sqrt(sample_var + eps)
        self.memory_std = std
        X = (X - sample_mean) / std
        self.memory_norm = X
        return X * self.gamma + self.beta

    def backward(self, error):
        assert self.on_train
        self.grads = error
        error *= self.gamma
        N = error.shape[0]
        return 1 / N / self.memory_std * (
                    N * error - error.sum(axis=0) - self.memory_norm * (error * self.memory_norm).sum(axis=0))

    def fit(self, lr, weight_decay):
        assert self.on_train
        self.gamma -= (self.grads * self.memory_norm).sum(0) * lr
        self.beta -= self.grads.sum(0) * lr


class ResidualCNN(Layer):
    def __init__(self, kernel, dropout, input_size):
        super(ResidualCNN, self).__init__()
        channels = input_size[-1]
        self.layers = [
            BatchNorm(input_size),
            Relu(),
            Dropout(dropout),
            Conv2d(channels, channels, kernel),
            BatchNorm(input_size),
            Relu(),
            Dropout(dropout),
            Conv2d(channels, channels, kernel),
        ]

    def to(self, device='cpu'):
        self.super().to(device)
        for l in self.layers:
            l.to(device)

    def forward(self, x):
        residual = x
        for l in self.layers:
            x = l(x)
        return x + residual

    def backward(self, error):
        assert self.on_train
        residual_error = error
        for l in self.layers[::-1]:
            error = l.backward(error)
        return error + residual_error

    def fit(self, lr, weight_decay=0.01):
        for l in self.layers:
            l.fit(lr=lr, weight_decay=weight_decay)

    def train(self):
        self.on_train = True
        for l in self.layers:
            l.train()

    def eval(self):
        self.on_train = False
        for l in self.layers:
            l.eval()


class Model(Layer):
    def __init__(self, layers: list):
        super(Model, self).__init__()
        self.layers = layers

    def forward(self, X):
        for l in self.layers:
            X = l(X)
        return X

    def to(self, device='cpu'):
        for l in self.layers:
            l.to(device)

    def backward(self, error):
        for l in self.layers[::-1]:
            error = l.backward(error)
        return error

    def fit(self, lr, weight_decay=0.01):
        for l in self.layers:
            l.fit(lr=lr, weight_decay=weight_decay)

    def train(self):
        self.on_train = True
        for l in self.layers:
            l.train()

    def eval(self):
        self.on_train = False
        for l in self.layers:
            l.eval()
