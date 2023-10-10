
# 加深网络，精确度达到99.34%
import sys, os
sys.path.append(os.pardir)  
from data.mnist import load_mnist
from deep_convnet import DeepConvNet
from sample_convnet import SimpleConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
# network = DeepConvNet()  

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# 保存参数
network.save_params("params/sample_convnet_params.pkl")
# network.save_params("params/deep_convnet_params.pkl")
print("Saved Network Parameters!")
