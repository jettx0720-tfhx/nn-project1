from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    """带动量的随机梯度下降"""
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu  # 动量系数
        # 为每个可训练参数初始化速度（初始为 0）
        self.velocity = {}
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    self.velocity[(id(layer), key)] = np.zeros_like(layer.params[key])

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    v_key = (id(layer), key)
                    # 动量更新公式: v = mu * v - lr * grad
                    self.velocity[v_key] = self.mu * self.velocity[v_key] - self.init_lr * layer.grads[key]
                    # 先做 weight decay（L2 正则化），再加动量更新项
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] += self.velocity[v_key]