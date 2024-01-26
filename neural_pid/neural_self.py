'''
Author: zhangL2auto
Date: 2024-01-25 16:09:49
LastEditors: 2auto deutschlei47@126.com
LastEditTime: 2024-01-25 16:24:09
'''
import numpy as np

wi = np.mat([[-0.6394, -0.2696, -0.3756, -0.7023],
                         [-0.8603, -0.2013, -0.5024, -0.2596],
                         [-1.0000, 0.5543, -1.0000, -0.5437],
                         [-0.3625, -0.0724, 0.6463, -0.2859],
                         [0.1425, 0.0279, -0.5406, -0.7660]]
                         )
print(wi.T.shape)
xi = np.mat([10, 9, 1, 5])
print(xi.shape)

class IncrementalPID:
    def __init__(self):
        self.xite_1 = 0.2 # learn rate
        self.alfa = 0.95 
        self.IN = 4
        self.H = 5
        self.Out = 3
        self.wi = np.mat([[-0.6394, -0.2696, -0.3756, -0.7023],
                         [-0.8603, -0.2013, -0.5024, -0.2596],
                         [-1.0000, 0.5543, -1.0000, -0.5437],
                         [-0.3625, -0.0724, 0.6463, -0.2859],
                         [0.1425, 0.0279, -0.5406, -0.7660]])
        
        self.wo = np.mat([[0.7576, 0.2616, 0.5820, -0.1416, -0.1325],
                          [-0.1146, 0.2949, 0.8352, 0.2205, 0.4508],
                          [0.7201, 0.4566, 0.7672, 0.4962, 0.3632]])
        
        self.wi_1 = self.wi
        self.wi_2 = self.wi
        self.wi_3 = self.wi
        self.wo_1 = self.wo
        self.wo_2 = self.wo
        self.wo_3 = self.wo

        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        # self.ts = 40  # 采样周期取值
        self.x = [self.Kp, self.Ki, self.Kd]  # 比例、积分、微分初值

        self.y = 0.0    # 系统输出值
        self.y_1 = 0.0  # 上次系统输出值
        self.y_2 = 0.0  # 上次系统输出值

        self.e = 0.0    # 输出值与输入值的偏差
        self.e_1 = 0.0
        self.e_2 = 0.0
        self.de_1 = 0.0

        self.u = 0.0
        self.u_1 = 0.0
        self.u_2 = 0.0
        self.u_3 = 0.0
        self.u_4 = 0.0
        self.u_5 = 0.0

        self.Oh = np.mat(np.zeros((self.H, 1)))  # 隐含层的输出
        self.I = np.mat(np.zeros((self.H, 1)))   # 隐含层的输入

        self.Oh_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.K_sub = [0.0, 0.0, 0.0]
        self.dK_sub = [0.0, 0.0, 0.0]
        self.delta3_sub = [0.0, 0.0, 0.0]
        self.dO_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.delta2_sub = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.den = -0.8251
        self.num = 0.2099