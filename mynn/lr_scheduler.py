from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    """多步长学习率衰减：在预设的迭代次数上触发衰减"""
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = milestones  # 触发衰减的迭代次数列表
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # 当前迭代次数命中里程碑时，学习率乘以 gamma
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(scheduler):
    """指数衰减学习率：每次 step 都乘以 gamma"""
    def __init__(self, optimizer, gamma=0.99) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.init_lr *= self.gamma