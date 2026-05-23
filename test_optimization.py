"""
优化实验脚本：对比 SGD vs Momentum，支持不同学习率设置。
训练完成后保存学习曲线对比图。
"""
import mynn as nn
from draw_tools.plot import plot
import matplotlib
matplotlib.use('Agg')
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle


# 固定随机种子，保证实验可复现
np.random.seed(309)

# 数据加载
train_images_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 加载已有划分索引，保证与基线实验使用相同的训练/验证集划分
with open('idx.pickle', 'rb') as f:
    idx = pickle.load(f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000] / 255.0
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:] / 255.0
train_labs = train_labs[10000:]

num_epochs = 5
num_classes = train_labs.max() + 1

# MLP SGD vs MomentGD 对比
print("实验1: MLP — SGD vs MomentGD")

# MLP + SGD
print("\n[1/4] MLP + SGD")
mlp_sgd = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU')
opt_sgd = nn.optimizer.SGD(init_lr=0.06, model=mlp_sgd)
loss_sgd = nn.op.MultiCrossEntropyLoss(model=mlp_sgd, max_classes=num_classes)
runner_mlp_sgd = nn.runner.RunnerM(mlp_sgd, opt_sgd, nn.metric.accuracy, loss_sgd)
runner_mlp_sgd.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                     num_epochs=num_epochs, log_iters=100, save_dir=r'./best_models_mlp_sgd')

# MLP + MomentGD
print("\n[2/4] MLP + MomentGD")
mlp_mom = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU')
opt_mom = nn.optimizer.MomentGD(init_lr=0.06, model=mlp_mom, mu=0.9)
loss_mom = nn.op.MultiCrossEntropyLoss(model=mlp_mom, max_classes=num_classes)
runner_mlp_mom = nn.runner.RunnerM(mlp_mom, opt_mom, nn.metric.accuracy, loss_mom)
runner_mlp_mom.train([train_imgs, train_labs], [valid_imgs, valid_labs],
                     num_epochs=num_epochs, log_iters=100, save_dir=r'./best_models_mlp_mom')

# CNN SGD vs MomentGD 对比
print("实验2: CNN — SGD vs MomentGD")

train_imgs_cnn = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs_cnn = valid_imgs.reshape(-1, 1, 28, 28)

# --- CNN + SGD ---
print("\n[3/4] CNN + SGD")
cnn_sgd = nn.models.Model_CNN()
opt_cnn_sgd = nn.optimizer.SGD(init_lr=0.06, model=cnn_sgd)
loss_cnn_sgd = nn.op.MultiCrossEntropyLoss(model=cnn_sgd, max_classes=num_classes)
runner_cnn_sgd = nn.runner.RunnerM(cnn_sgd, opt_cnn_sgd, nn.metric.accuracy, loss_cnn_sgd)
runner_cnn_sgd.train([train_imgs_cnn, train_labs], [valid_imgs_cnn, valid_labs],
                     num_epochs=num_epochs, log_iters=100, save_dir=r'./best_models_cnn_sgd')

# --- CNN + MomentGD ---
print("\n[4/4] CNN + MomentGD")
cnn_mom = nn.models.Model_CNN()
opt_cnn_mom = nn.optimizer.MomentGD(init_lr=0.06, model=cnn_mom, mu=0.9)
loss_cnn_mom = nn.op.MultiCrossEntropyLoss(model=cnn_mom, max_classes=num_classes)
runner_cnn_mom = nn.runner.RunnerM(cnn_mom, opt_cnn_mom, nn.metric.accuracy, loss_cnn_mom)
runner_cnn_mom.train([train_imgs_cnn, train_labs], [valid_imgs_cnn, valid_labs],
                     num_epochs=num_epochs, log_iters=100, save_dir=r'./best_models_cnn_mom')

# 绘制四宫格对比图：MLP/CNN × Loss/Accuracy，直观对比 SGD 与 Momentum 收敛曲线
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# MLP 对比
axes[0, 0].plot(runner_mlp_sgd.dev_loss, label='SGD', alpha=0.7)
axes[0, 0].plot(runner_mlp_mom.dev_loss, label='Momentum (mu=0.9)', alpha=0.7, linestyle='--')
axes[0, 0].set_title('MLP — Dev Loss')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()

axes[0, 1].plot(runner_mlp_sgd.dev_scores, label='SGD', alpha=0.7)
axes[0, 1].plot(runner_mlp_mom.dev_scores, label='Momentum (mu=0.9)', alpha=0.7, linestyle='--')
axes[0, 1].set_title('MLP — Dev Accuracy')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()

# CNN 对比
axes[1, 0].plot(runner_cnn_sgd.dev_loss, label='SGD', alpha=0.7)
axes[1, 0].plot(runner_cnn_mom.dev_loss, label='Momentum (mu=0.9)', alpha=0.7, linestyle='--')
axes[1, 0].set_title('CNN — Dev Loss')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()

axes[1, 1].plot(runner_cnn_sgd.dev_scores, label='SGD', alpha=0.7)
axes[1, 1].plot(runner_cnn_mom.dev_scores, label='Momentum (mu=0.9)', alpha=0.7, linestyle='--')
axes[1, 1].set_title('CNN — Dev Accuracy')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('optimization_comparison.png')
print("\n优化实验完成，对比图已保存到 optimization_comparison.png")

# 汇总
print("优化实验结果汇总")
print(f"MLP  SGD     最佳验证准确率: {runner_mlp_sgd.best_score:.4f}")
print(f"MLP  Momentum 最佳验证准确率: {runner_mlp_mom.best_score:.4f}")
print(f"CNN  SGD     最佳验证准确率: {runner_cnn_sgd.best_score:.4f}")
print(f"CNN  Momentum 最佳验证准确率: {runner_cnn_mom.best_score:.4f}")
