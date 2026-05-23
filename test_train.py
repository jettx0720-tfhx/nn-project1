# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import matplotlib
matplotlib.use('Agg')
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

# train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_images_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\train-images-idx3-ubyte.gz'
# train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
train_labels_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# Part A: MLP 基线训练
print("训练 MLP 基线")

linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
optimizer_mlp = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
loss_fn_mlp = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner_mlp = nn.runner.RunnerM(linear_model, optimizer_mlp, nn.metric.accuracy, loss_fn_mlp)
runner_mlp.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                 num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner_mlp, axes)
plt.savefig('training_curves_mlp.png')
print("MLP 训练曲线已保存到 training_curves_mlp.png\n")


# CNN 模型训练
print("训练 CNN 模型")

# CNN 输入需要 [batch, 1, 28, 28] 形状
train_imgs_cnn = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs_cnn = valid_imgs.reshape(-1, 1, 28, 28)

cnn_model = nn.models.Model_CNN()
optimizer_cnn = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
loss_fn_cnn = nn.op.MultiCrossEntropyLoss(model=cnn_model, max_classes=train_labs.max()+1)

runner_cnn = nn.runner.RunnerM(cnn_model, optimizer_cnn, nn.metric.accuracy, loss_fn_cnn)
runner_cnn.train([train_imgs_cnn, train_labs], [valid_imgs_cnn, valid_labs],
                 num_epochs=5, log_iters=100, save_dir=r'./best_models_cnn')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner_cnn, axes)
plt.savefig('training_curves_cnn.png')
print("CNN 训练曲线已保存到 training_curves_cnn.png\n")


# 结果汇总
print("MLP vs CNN 对比")
print(f"MLP 最佳验证准确率: {runner_mlp.best_score:.4f}")
print(f"CNN 最佳验证准确率: {runner_cnn.best_score:.4f}")