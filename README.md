# digital-recognition
深度学习CNN训练手写数字识别

## 网络结构如下所示
    conv - relu - conv- relu - pool -  
    conv - relu - conv- relu - pool -  
    conv - relu - conv- relu - pool -  
    affine - relu - dropout - affine - dropout - softmax  

## layers
    卷积层conv: 保证数据的形状  
    激活函数:  
        - ReLU  
        - softmax: logits ==> 概率分布  
    损失函数: 使用交叉熵函数cross_entropy_error  
    池化层pool: 使用Max池化。缩小高，长方向的空间运算    
    affine: 加权和 A = X·W + b  
    dropout: 随机删减神经元，减少过拟合  

## 优化
    卷积层基于3 * 3的小型滤波器  
    激活函数为ReLU，使用 He初始值  
    全连接层后面使用Dropout  
    optimizer使用Adam优化  
    


