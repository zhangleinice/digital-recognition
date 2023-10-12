# digital-recognition
CNN数字识别，识别准确率为99.32%  

## Quick Start
    训练：python train.py  
    使用：python use.py  

## 网络结构如下所示
    conv - relu - conv- relu - pool -  
    conv - relu - conv- relu - pool -  
    conv - relu - conv- relu - pool -  
    affine - relu - dropout - affine - dropout - softmax  

    这个神经网络有6个卷积层和2个全连接层，总共8个层  

## 优化
    卷积层基于3 * 3的小型滤波器  
    激活函数为ReLU，使用 He初始值  
    全连接层后面使用Dropout，随机删减神经元，减少过拟合   
    optimizer使用Adam优化  
    


