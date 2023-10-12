import os
import sys
sys.path.append(os.pardir)
from data.mnist import load_mnist
from deep_convnet import DeepConvNet  
import numpy as np
from common.util import softmax
import random
import matplotlib.pyplot as plt

# 加载模型及权重
model = DeepConvNet() 
model.load_params("params/deep_convnet_params.pkl") 

# 加载数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)

# 随机选择的测试样本和标签
test_size = x_test.shape[0]
batch_size = random.randint(1, 10)

batch_mask = np.random.choice(test_size, batch_size)
imgs = x_test[batch_mask]
labels = t_test[batch_mask]

print('input: 输入的数字是', labels)

# 展示输入图像
def show_imgs(imgs, labels):
    # 创建一个横向的子图
    _, axes = plt.subplots(1, imgs.shape[0], figsize=(12, 3)) 

    for i in range(imgs.shape[0]):
        image = imgs[i].reshape(28, 28)
        ax = axes[i]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"标签: {labels[i]}")

    plt.show()

# show_imgs(imgs, labels)

# 使用模型进行推理
logits = model.predict(imgs)
# 概率分布
probabilities = softmax(logits)

# print(logits)
# print(probabilities)

max_indexs = np.argmax(probabilities, axis=1)
max_values = probabilities[np.arange(len(max_indexs)), max_indexs]
max_values = [f'{value * 100:.2f}%' for value in max_values]

print(f"output: 输入的数字是{max_indexs} 的概率约为 {max_values}")











