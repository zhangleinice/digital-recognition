import os
import sys
sys.path.append(os.pardir)
from data.mnist import load_mnist
from deep_convnet import DeepConvNet  
from PIL import Image
import numpy as np
from common.gradient import softmax

# 加载模型，权重
model = DeepConvNet() 
model.load_params("params/deep_convnet_params.pkl") 

# 加载数据集
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)

# 随机选择一个索引
random_index = np.random.randint(0, len(x_test))

# 获取随机选择的测试样本和标签
img = x_test[random_index]
label = t_test[random_index]

def img_show(img):
    pil = Image.fromarray(np.uint8(img))
    pil.show()

# image = img.reshape(28, 28)
# img_show(image)

print('输入的数字是: ', label)

# 使用模型进行推理
logits = model.predict(img.reshape(1, 1, 28, 28))

# 概率分布
probabilities = softmax(logits)

max_index = np.argmax(probabilities)

# print(logits)

# print(probabilities)

max_value = probabilities[0][max_index]

print(f"输入的数字是 {max_index} 的概率接近: {max_value * 100:.2f}%")










