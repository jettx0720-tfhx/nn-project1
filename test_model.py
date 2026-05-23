import mynn as nn
import numpy as np
from struct import unpack
import gzip
import os

# 测试 MLP 模型
print("MLP 测试集评估:")
mlp = nn.models.Model_MLP()
mlp.load_model(r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\best_models\best_model.pickle')

test_images_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = mlp(test_imgs)
acc_mlp = nn.metric.accuracy(logits, test_labs)
print(f"MLP  测试准确率: {acc_mlp:.4f}")

# 测试 CNN 模型
print("\nCNN 测试集评估:")
cnn_path = r'D:\HuaweiMoveData\Users\86156\Desktop\神经\PJ1\codes\best_models_cnn\best_model.pickle'
if os.path.exists(cnn_path):
    cnn = nn.models.Model_CNN()
    cnn.load_model(cnn_path)
    # CNN 需要 [batch, 1, 28, 28]
    test_imgs_cnn = test_imgs.reshape(-1, 1, 28, 28)
    logits_cnn = cnn(test_imgs_cnn)
    acc_cnn = nn.metric.accuracy(logits_cnn, test_labs)
    print(f"CNN  测试准确率: {acc_cnn:.4f}")

    print(f"\nMLP vs CNN 测试集对比:")
    print(f"  MLP: {acc_mlp:.4f}")
    print(f"  CNN: {acc_cnn:.4f}")
    print(f"  差距: {acc_cnn - acc_mlp:+.4f}")
else:
    print("CNN 模型文件不存在，请先运行 test_train.py 训练 CNN")