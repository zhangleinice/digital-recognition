# 梯度下降法  W = W - η * grads

import numpy as np

# η , grads 不变
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            # params[key] = params[key] - self.lr * grads[key]
            # -= 直接在变量的值上面进行操作，修改了原来变量的值
            params[key] -= self.lr * grads[key]



# grads改变，变成了一个累计梯度
# v = αv - η * grad ; v是动量因子，将存储部分以前的梯度
# W = W + v
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        # 初始化速度
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        # 更新参数
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            # params[key] = params[key] + self.v[key]
            params[key] += self.v[key]


# η变化：根据梯度调整学习率
# h = h + grads * grads ; 累计平方梯度
# W = W - grads * η/√h 
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        # 初始化h
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # 更新params
        for key in grads.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 在 AdaGrad 上加一个衰减系数 ρ (0, 1); 使梯度平方不会一直增大，步长不会减小
# h = ρ*h + (1-ρ) * grads * grads
# W = W - η * grads/√h
class RMSprop:

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# 缝合怪：RMSProp + Momentum
# lr_t = η * √(1 - β₂^t) / (1 - β₁^t)
# m = β₁ * m + (1 - β₁) * grads
# v = β₂ * v + (1 - β₂) * grads^2 
# W = W - lr_t * m / (√v + ε)

class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        # 使得学习率随着迭代步骤的增加而逐渐减小，这有助于保持梯度下降的稳定性和收敛性
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 **
                                 self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)




class Nesterov:

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


