"""
错误分析与可视化脚本 (Part C — Direction 5)
功能：
  1. 混淆矩阵
  2. 误分类样本展示
  3. MLP 第一层权重可视化
  4. CNN 卷积核可视化
"""
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os


def load_mnist_test():
    test_images_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    return test_imgs / 255.0, test_labs, rows, cols


def confusion_matrix_plot(preds, labels, save_path='confusion_matrix.png'):
    """绘制并保存混淆矩阵"""
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # 统计每一对 (真实类别, 预测类别)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # 在每个格子内标注数值，深浅色自动适配
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存到 {save_path}")
    return cm


def misclassified_examples(images, true_labels, preds, rows, cols, max_show=20, save_path='misclassified.png'):
    """展示被误分类的样本：真实标签 vs 预测标签"""
    # 找出所有预测错误的样本索引
    errors = true_labels != preds
    error_idx = np.where(errors)[0]
    n = min(len(error_idx), max_show)

    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.reshape(-1)

    for i in range(n):
        idx = error_idx[i]
        axes[i].imshow(images[idx].reshape(rows, cols), cmap='gray')
        axes[i].set_title(f'True:{true_labels[idx]} Pred:{preds[idx]}', fontsize=10)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"误分类样本图已保存到 {save_path} (共 {len(error_idx)} 个错误)")


def mlp_weight_visualization(model, save_path='mlp_weights.png'):
    """MLP 第一层权重可视化：每个神经元的 784 维权重重塑为 28x28 图像"""
    w = model.layers[0].params['W']  # [784, 600]
    n_cols = 20
    n_rows = 15
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))

    for i in range(n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        if i < w.shape[1]:
            # 将第 i 个神经元的权重向量 reshape 为 28x28 并可视化
            axes[r, c].imshow(w[:, i].reshape(28, 28), cmap='RdYlBu', vmin=-0.5, vmax=0.5)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])

    plt.suptitle('MLP First-Layer Weights (28x28 per neuron)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"MLP 权重可视化已保存到 {save_path}")


def cnn_kernel_visualization(model, save_path='cnn_kernels.png'):
    """CNN 卷积核可视化：第一层 8 个核 + 第二层 8 个核（每个取第一个输入通道）"""
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))

    conv1_w = None
    for layer in model.layers:
        if isinstance(layer, nn.op.conv2D):
            if conv1_w is None:
                # 第一层卷积核: [8, 1, 3, 3] -> 取第一个输入通道 -> [8, 3, 3]
                conv1_w = layer.params['W'][:, 0]
                for i in range(8):
                    axes[0, i].imshow(conv1_w[i], cmap='RdYlBu')
                    axes[0, i].set_title(f'k{i}')
                    axes[0, i].set_xticks([])
                    axes[0, i].set_yticks([])
            else:
                # 第二层卷积核: [16, 8, 3, 3] -> 每输出通道取第一个输入通道 -> 展示前 8 个
                conv2_w = layer.params['W'][:, 0, :, :]  # [16, 3, 3]
                for i in range(min(8, conv2_w.shape[0])):
                    axes[1, i].imshow(conv2_w[i], cmap='RdYlBu')
                    axes[1, i].set_title(f'k{i}')
                    axes[1, i].set_xticks([])
                    axes[1, i].set_yticks([])
                break

    axes[0, 0].set_ylabel('Conv1', fontsize=12)
    axes[1, 0].set_ylabel('Conv2 (ch0)', fontsize=12)
    plt.suptitle('CNN Convolution Kernels Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"CNN 卷积核可视化已保存到 {save_path}")


def main():
    test_imgs, test_labs, rows, cols = load_mnist_test()
    os.makedirs('./analysis_output', exist_ok=True)

    # MLP
    print("MLP 错误分析")

    mlp_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\best_models\best_model.pickle'
    if os.path.exists(mlp_path):
        mlp = nn.models.Model_MLP()
        mlp.load_model(mlp_path)
        logits_mlp = mlp(test_imgs)
        preds_mlp = np.argmax(logits_mlp, axis=1)
        acc_mlp = (preds_mlp == test_labs).mean()

        cm_mlp = confusion_matrix_plot(preds_mlp, test_labs, './analysis_output/confusion_matrix_mlp.png')
        misclassified_examples(test_imgs, test_labs, preds_mlp, rows, cols,
                               save_path='./analysis_output/misclassified_mlp.png')
        mlp_weight_visualization(mlp, './analysis_output/mlp_weights.png')

        # 拆解各类别准确率，找出模型最薄弱的类别
        print(f"MLP 总体测试准确率: {acc_mlp:.4f}")
        for c in range(10):
            mask = test_labs == c
            class_acc = (preds_mlp[mask] == test_labs[mask]).mean()
            print(f"  类别 {c}: {class_acc:.4f}  ({mask.sum()} 样本)")
    else:
        print(f"MLP 模型未找到: {mlp_path}")

    # CNN
    print("CNN 错误分析")

    cnn_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\best_models_cnn\best_model.pickle'
    if os.path.exists(cnn_path):
        cnn = nn.models.Model_CNN()
        cnn.load_model(cnn_path)
        test_imgs_cnn = test_imgs.reshape(-1, 1, 28, 28)
        logits_cnn = cnn(test_imgs_cnn)
        preds_cnn = np.argmax(logits_cnn, axis=1)
        acc_cnn = (preds_cnn == test_labs).mean()

        cm_cnn = confusion_matrix_plot(preds_cnn, test_labs, './analysis_output/confusion_matrix_cnn.png')
        misclassified_examples(test_imgs, test_labs, preds_cnn, rows, cols,
                               save_path='./analysis_output/misclassified_cnn.png')
        cnn_kernel_visualization(cnn, './analysis_output/cnn_kernels.png')

        # 拆解各类别准确率，找出模型最薄弱的类别
        print(f"CNN 总体测试准确率: {acc_cnn:.4f}")
        for c in range(10):
            mask = test_labs == c
            class_acc = (preds_cnn[mask] == test_labs[mask]).mean()
            print(f"  类别 {c}: {class_acc:.4f}  ({mask.sum()} 样本)")

        # MLP vs CNN 错误模式对比：分析两个模型的预测差异
        if os.path.exists(mlp_path):
            same_error = (preds_mlp != test_labs) & (preds_cnn != test_labs)      # 两者都错
            mlp_only_right = (preds_mlp == test_labs) & (preds_cnn != test_labs)  # 仅 MLP 正确
            cnn_only_right = (preds_mlp != test_labs) & (preds_cnn == test_labs)  # 仅 CNN 正确
            print(f"\nMLP vs CNN 错误对比:")
            print(f"  两者都错: {same_error.sum()}")
            print(f"  仅MLP正确: {mlp_only_right.sum()}")
            print(f"  仅CNN正确: {cnn_only_right.sum()}")
    else:
        print(f"CNN 模型未找到: {cnn_path}")


if __name__ == '__main__':
    main()
