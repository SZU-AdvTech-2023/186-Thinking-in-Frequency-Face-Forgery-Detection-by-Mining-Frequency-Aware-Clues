import torch.nn as nn
import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import torch.optim as optim
# torch.cuda.empty_cache()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class resnet(nn.Module):
    def __init__(self):
        super(resnet,self).__init__()
        self.all = models.resnet18(pretrained=False)
        num_ftrs = self.all.fc.out_features
        self.fc = nn.Linear(num_ftrs,1)
    def forward(self,x):
        x=self.all(x)
        x=self.fc(x)
        return x

# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
#             nn.MaxPool2d(kernel_size = 2,stride = 2),
#             nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
#             nn.MaxPool2d(kernel_size = 2,stride = 2),
#             nn.Dropout2d(p = 0.1),
#             nn.AdaptiveMaxPool2d((1,1))
#         )
#         self.dense = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64,32),
#             nn.ReLU(),
#             nn.Linear(32,2),
#             # nn.Sigmoid()
#         )
#     def forward(self,x):
#         x = self.conv(x)
#         y = self.dense(x)
#         return y

# net = Net()
# print(net)
# 单GPU或者CPU
# model = net()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # 判断GPU设备大于1，也就是多个GPU，则使用多个GPU设备的id来加载
# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model,device_ids=[0,1,2])