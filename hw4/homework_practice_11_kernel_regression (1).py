import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(
        self,
        lr=0.01,
        regularization=1.0,
        tolerance=1e-2,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1.0,
        fit_intercept=True
    ):
        """
        :param lr: learning rate
        :param regularization: regularization coefficient
        :param tolerance: stopping criterion for square of euclidean norm of weight difference
        :param max_iter: stopping criterion for iterations
        :param batch_size: size of the batches used in gradient descent steps
        :parame kernel_scale: length scale in RBF kernel formula
        :param fit_intercept: whether to use free coefficient
        """

        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.x_train: np.ndarray | None = None

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.fit_intercept: bool = fit_intercept

        self.loss_history: list[float] = []
        self.kernel = RBF(length_scale=kernel_scale)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculating loss for x and y dataset
        :param x: features array
        :param y: targets array
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        K = self.kernel(x)

        predictions = K @ self.w + self.b
        errors = predictions - y

        mse_part = 0.5 * np.sum(errors ** 2)
        reg_part = 0.5 * self.regularization * (self.w @ K @ self.w)

        return mse_part + reg_part

    def calc_grad(self, x: np.ndarray, y: np.ndarray):
        """
        Calculating gradient for x and y dataset
        :param x: features array
        :param y: targets array
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        K = self.kernel(x)

        predictions = K @ self.w + self.b
        errors = predictions - y

        grad_w = K @ errors + self.regularization * (K @ self.w)

        if self.fit_intercept:
            grad_b = np.sum(errors)
        else:
            grad_b = 0.0

        return grad_w, grad_b

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров с помощью градиентного спуска
        :param x: features array
        :param y: targets array
        :return: self
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_objects = x.shape[0]

        self.x_train = x.copy()
        self.w = np.zeros(n_objects)

        if self.fit_intercept:
            self.b = np.mean(y)
        else:
            self.b = 0.0

        self.loss_history = []

        for _ in range(self.max_iter):
            old_w = self.w.copy()
            old_b = self.b

            grad_w, grad_b = self.calc_grad(x, y)

            self.w -= self.lr * grad_w / n_objects

            if self.fit_intercept:
                self.b -= self.lr * grad_b / n_objects

            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)

            weight_diff = np.sum((self.w - old_w) ** 2) + (self.b - old_b) ** 2

            if weight_diff < self.tolerance:
                break

        return self

    def fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Получение параметров через аналитическое решение
        :param x: features array
        :param y: targets array
        :return: self
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_objects = x.shape[0]

        self.x_train = x.copy()

        K = self.kernel(x)
        I = np.eye(n_objects)

        if self.fit_intercept:
            ones = np.ones((n_objects, 1))

            left = np.zeros((n_objects + 1, n_objects + 1))
            left[:n_objects, :n_objects] = K + self.regularization * I
            left[:n_objects, n_objects:] = ones
            left[n_objects:, :n_objects] = ones.T

            right = np.zeros(n_objects + 1)
            right[:n_objects] = y

            solution = np.linalg.solve(left, right)

            self.w = solution[:n_objects]
            self.b = solution[n_objects]
        else:
            self.w = np.linalg.solve(K + self.regularization * I, y)
            self.b = 0.0

        self.loss_history = [self.calc_loss(x, y)]

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicting targets for x dataset
        :param x: features array
        :return: prediction: np.ndarray
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        K_test = self.kernel(x, self.x_train)

        return K_test @ self.w + self.b
