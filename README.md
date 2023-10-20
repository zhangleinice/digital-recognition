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
    卷积层基于3 * 3的小型滤波器(卷积核)    
    激活函数为ReLU，使用 He初始值  
    全连接层后面使用Dropout，随机删减神经元，减少过拟合   
    optimizer使用Adam优化  

## 特征提取
    低层次特征：底层的卷积层通常用来捕捉低层次特征，如边缘、纹理、颜色等。这些特征是图像中的基本信息，不需要太多的上下文理解。  
    中层次特征：中间层次的卷积层和全连接层可以学习到更复杂的特征，如形状、物体部件等。这些特征需要更多上下文信息来理解  
    高层次特征：更深层次的网络层可以识别抽象的高层次特征，如物体的类别、语义信息等。这些特征需要对整个图像进行全局分析  
    加深网络：通过叠加层提高识别准确率  




    


