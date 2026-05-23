import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an example to train, evaluate, save, load the model. However, some of the function calling may not be correct
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []
        self._last_dev_score = 0
        self._last_dev_loss = 999

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            total_iters = int(X.shape[0] / self.batch_size) + 1

            for iteration in range(total_iters):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)

                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                # 仅在打印日志或每个 epoch 最后一步时评估验证集，避免 CNN 每一步都跑全量验证
                is_last_iter = (iteration == total_iters - 1)
                if (iteration % log_iters == 0) or is_last_iter:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self._last_dev_score = dev_score
                    self._last_dev_loss = dev_loss
                    if iteration % log_iters == 0:
                        print(f"epoch: {epoch}, iteration: {iteration}")
                        print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                        print(f"[Dev] loss: {dev_loss}, score: {dev_score}")
                else:
                    # 非评估步复用上一次 dev 值，保持曲线长度一致
                    dev_score = getattr(self, '_last_dev_score', 0)
                    dev_loss = getattr(self, '_last_dev_loss', 999)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

            # 取本 epoch 最后一次评估的 dev_score 判断是否保存最佳模型
            epoch_dev_score = self._last_dev_score
            if epoch_dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performance has been updated: {best_score:.5f} --> {epoch_dev_score:.5f}")
                best_score = epoch_dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss

    def save_model(self, save_path):
        self.model.save_model(save_path)