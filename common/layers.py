# 神经网络层
import sys, os
sys.path.append(os.pardir) 
import numpy as np
from common.util import *

# f(x) = x > 0 ? x : 0;
class ReLU:
    def __init__(self):
        # 布尔数组
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = np.copy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# f(x) = 1/(1+e^x)
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx


# f(x)=x⋅W+b 加权和层，全连接层    
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

        # 输入数据的值和形状
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        # reshape(x.shape[0], -1) 行数不变，列数 = 总元素个数 / 行数
        # 多维数组转成二维数组，主要针对二维以上数组
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


# softmax 和 交叉熵误差
class SoftmaxWithLoss:
    def __init__(self):
        self.y = None  # 神经网络输出
        self.t = None  # 监督数据
        self.loss = None  # 损失

    def forward(self, x, t):
        self.t = t
        # logits ==> 概率
        self.y = softmax(x)
        # 真实值y, 与预测值t的误差（评估整个数据集或一个批次的损失）
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    # 监督数据的表示方式：one-hot encoding:[0, 1, 0]和 class labels: 1
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 监督数据的长度 = 神经网络输出时候，表示是 one-hot encoding
        if self.t.size == self.y.size:  
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


# 卷积层保证了数据的形状；卷积计算（图像 * 滤波层）
class Convolution:
    # W 滤波器(权重)
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        # im2col: 将 x 4维转化为2维
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 将滤波器 W 转化为2维
        col_W = self.W.reshape(FN, -1).T
        # 计算加权和
        out = np.dot(col, col_W) + self.b
        # 还原为4维形状
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx



# max池化 缩小高，长方向运算
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # x ==> 矩阵
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        # Max池化（展开后每行的最大值）
        out = np.max(col, axis=1)
        # 还原输出形状
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

    
# 在训练数据时，随机地将一些神经元的输出设为零
# 这种随机性和多样性有助于减少神经网络对训练数据过拟合
class Dropout:
    # 丢弃50%的神经元
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 随机清零神经元
            return x * self.mask
        else:
            # 测试数据不再随机丢弃，保持与训练时相似的输出分布有助于更好地评估模型的性能。
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask



class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv层的情况下为4维，全连接层的情况下为2维  

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx