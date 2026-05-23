from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            initialize_method = lambda size: np.random.normal(size=size) * np.sqrt(2.0 / size[0])
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 必须记录输入，反向传播求导时需要用到

        # 矩阵乘法：X * W + b
        out = np.dot(X, self.W) + self.b
        return out

    def backward(self, grad: np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # 计算对输入X的梯度，这个梯度要return传给上一层
        # dX = grad * W^T
        grad_input = np.dot(grad, self.W.T)

        # 计算对当前层参数W的梯度
        # dW = X^T * grad
        dW = np.dot(self.input.T, grad)

        # 计算对当前层偏置b的梯度
        # db = 对 batch_size 维度求和，保持维度一致 (1, out_dim)
        db = np.sum(grad, axis=0, keepdims=True)

        # 将参数梯度保存起来
        self.grads['W'] = dW
        self.grads['b'] = db

        return grad_input

    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """二维卷积层，使用显式循环实现以保证正确性"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        fan_in = in_channels * kernel_size * kernel_size
        if initialize_method is None:
            initialize_method = lambda size: np.random.normal(size=size) * np.sqrt(2.0 / fan_in)

        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((1, out_channels))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.X_pad = None

        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        X: [batch, in_channels, H, W]
        W: [out_channels, in_channels, k, k]
        b: [1, out_channels]
        out: [batch, out_channels, H_out, W_out]
        """
        self.input = X
        batch, in_ch, H, W = X.shape
        out_ch = self.out_channels
        k = self.kernel_size
        s = self.stride
        p = self.padding

        H_out = (H + 2 * p - k) // s + 1
        W_out = (W + 2 * p - k) // s + 1

        # 对输入做 zero-padding
        X_pad = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        self.X_pad = X_pad

        out = np.zeros((batch, out_ch, H_out, W_out))

        # 对每个输出位置，提取 patch 与卷积核做逐元素乘加
        for oc in range(out_ch):
            for ic in range(in_ch):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, w_start = h * s, w * s
                        patch = X_pad[:, ic, h_start:h_start + k, w_start:w_start + k]
                        out[:, oc, h, w] += np.sum(patch * self.W[oc, ic], axis=(1, 2))

        out += self.b.reshape(1, -1, 1, 1)
        return out

    def backward(self, grads):
        """
        grads: [batch, out_channels, H_out, W_out]
        返回: [batch, in_channels, H, W] — 传给上一层的梯度
        """
        batch, out_ch, H_out, W_out = grads.shape
        in_ch = self.in_channels
        k = self.kernel_size
        s = self.stride
        p = self.padding

        X_pad = self.X_pad

        # db: grads 在 batch, H, W 维度上求和，压缩为 (1, out_channels)
        db = np.sum(grads, axis=(0, 2, 3)).reshape(1, -1)

        # dW: X_pad 与 grads 的互相关，使用跨步切片加速
        dW = np.zeros_like(self.W)
        for oc in range(out_ch):
            for ic in range(in_ch):
                for ki in range(k):
                    for kj in range(k):
                        patch = X_pad[:, ic, ki:ki + H_out * s:s, kj:kj + W_out * s:s]
                        dW[oc, ic, ki, kj] = np.sum(patch * grads[:, oc, :, :])

        # dX: 将每个输出位置的贡献累加回输入（相当于转置卷积）
        _, _, H_pad, W_pad = X_pad.shape
        dX_pad = np.zeros_like(X_pad)
        for oc in range(out_ch):
            for ic in range(in_ch):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start, w_start = h * s, w * s
                        dX_pad[:, ic, h_start:h_start + k, w_start:w_start + k] += \
                            self.W[oc, ic] * grads[:, oc, h, w].reshape(-1, 1, 1)

        self.grads['W'] = dW
        self.grads['b'] = db

        # 去掉 padding 区域，恢复原始尺寸
        if p > 0:
            return dX_pad[:, :, p:-p, p:-p]
        return dX_pad

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class MaxPool2D(Layer):
    """2x2 最大池化，步长 2"""
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.optimizable = False
        self.mask = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        batch, ch, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = np.zeros((batch, ch, H_out, W_out))
        # 记录最大值位置（mask 中 1 表示最大值，0 表示其他）
        self.mask = np.zeros(X.shape, dtype=np.float64)
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * s, w * s
                patch = X[:, :, h_start:h_start + k, w_start:w_start + k]
                max_vals = np.max(patch, axis=(2, 3))
                out[:, :, h, w] = max_vals
                # 标记最大值位置（处理平局情况）
                max_pos = (patch == max_vals[:, :, np.newaxis, np.newaxis])
                self.mask[:, :, h_start:h_start + k, w_start:w_start + k] += max_pos.astype(np.float64)
        return out

    def backward(self, grads):
        # 梯度只传递给前向传播中的最大值位置
        batch, ch, H_out, W_out = grads.shape
        k = self.kernel_size
        s = self.stride
        dx = np.zeros_like(self.mask)
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * s, w * s
                dx[:, :, h_start:h_start + k, w_start:w_start + k] += \
                    self.mask[:, :, h_start:h_start + k, w_start:w_start + k] * \
                    grads[:, :, h, w][:, :, np.newaxis, np.newaxis]
        return dx


class Flatten(Layer):
    """将 [batch, C, H, W] 展平为 [batch, C*H*W]"""
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape  # 保存原始形状，反向传播时恢复
        batch = X.shape[0]
        return X.reshape(batch, -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True  # 默认开启 softmax
        self.probs = None  # 用来保存预测的概率分布
        self.labels = None  # 用来保存真实标签

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels
        batch_size = predicts.shape[0]

        # 计算预测概率
        if self.has_softmax:
            # 调用文件最底部的softmax辅助函数
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        # 提取出当前批次中，真实类别的预测概率
        # 利用NumPy的高级索引，一次性把每个样本正确分类对应的概率提出来
        correct_class_probs = self.probs[np.arange(batch_size), labels]

        # 计算交叉熵损失
        # 加上1e-12是为了防止概率为0时导致log(0)报错
        loss = -np.sum(np.log(correct_class_probs + 1e-12)) / batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.probs.shape[0]

        # 先拷贝一份当前的概率分布矩阵
        self.grads = self.probs.copy()

        # 梯度公式: (预测概率-真实分布)/batch_size
        # 只需要将真实类别所在位置的概率减去1即可
        self.grads[np.arange(batch_size), self.labels] -= 1.0
        self.grads /= batch_size

        # Then send the grads to model for back propagation
        if self.model is not None:
            self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition