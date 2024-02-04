'''
Author: zhangL2auto
Date: 2024-01-23 14:47:30
LastEditors: 2auto deutschlei47@126.com
LastEditTime: 2024-02-04 14:12:51
'''
import numpy as np
import matplotlib.pyplot as plt

class IncrementalPID:
    def __init__(self):
        self.lr_1 = 0.20 # 学习率系数
        self.alfa = 0.95 # 惯性保持系数
        self.inputLayer_num = 4
        self.hiddenLayer_num = 5
        self.outputLayer_num = 3
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
        self.wo_1 = self.wo
        self.wo_2 = self.wo

        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        # self.ts = 40  # 采样周期取值
        self.error_value = [0.0, 0.0, 0.0]  # 比例、积分、微分对应误差描述的初值

        self.y_out = 0.0    # 系统输出值
        self.y_1 = 0.0      # 上次系统输出值

        self.e = 0.0        # 输出值与输入值的偏差
        self.e_1 = 0.0
        self.e_2 = 0.0
        self.de_init = 0.0  # 两次误差的差值

        self.u = 0.0
        self.u_1 = 0.0
        self.u_2 = 0.0

        self.hiddenLayer_input = np.mat(np.zeros((self.hiddenLayer_num, 1)))   # 隐含层的输入
        self.hiddenLayer_output = np.mat(np.zeros((self.hiddenLayer_num, 1)))  # 隐含层的输出

        self.hiddenLayer_output_sub = [0.0] * 5
        self.outputLayer_output_sub = [0.0] * 3
        self.outputLayer_d_sub = [0.0] * 3
        self.outputLayer_delta_sub = [0.0] * 3
        self.hiddenLayer_d_sub = [0.0] * 5
        self.hiddenLayer_delta_sub = [0.0] * 5

        # 使用den, num模拟系统的实际输出
        self.den = -0.8251
        self.num = 0.2099
    # 设置PID控制器参数
    def SetStepSignal(self, SetSignal):
        # 系统的实际输出值
        self.y_out = -self.den * self.y_1 + self.num * self.u_2
        # 系统的控制误差值
        self.e = SetSignal - self.y_out
        # 神经网络的输入
        self.r_in = np.mat([SetSignal, self.y_out, self.e, 5])
        # 增量PID各个误差的描述
        self.error_value[0] = self.e - self.e_1  # 比例输出
        self.error_value[1] = self.e  # 积分输出
        self.error_value[2] = self.e - 2 * self.e_1 + self.e_2  # 微分输出
        # 增量PID的误差描述矩阵
        self.error_pid = np.mat([[self.error_value[0]], [self.error_value[1]], [self.error_value[2]]])
        # 神经网络输入到输出，此输出即隐藏层的输入
        self.hiddenLayer_input = np.dot(self.r_in, (self.wi.T))
        
        # 在激活函数作用下隐含层的输出矩阵
        self.hiddenLayer_output_sub = np.tanh(self.hiddenLayer_input)
        self.hiddenLayer_output = self.hiddenLayer_output_sub.reshape(self.hiddenLayer_num, 1)
        # 输出层的输入，即隐含层的输出*权值
        self.outputLayer_input = np.dot(self.wo, self.hiddenLayer_output)  
        
        # 输出层的输出，即PID三个参数
        for i in range(self.outputLayer_num):
            x = self.outputLayer_input.tolist()[i][0]
            self.outputLayer_output_sub[i] = np.exp(x) / (np.exp(x) + np.exp(-x))

        self.Kp = self.outputLayer_output_sub[0]
        self.Ki = self.outputLayer_output_sub[1]
        self.Kd = self.outputLayer_output_sub[2]
        self.Kpid = np.mat([self.Kp, self.Ki, self.Kd])

        self.du = np.dot(self.Kpid, self.error_pid).tolist()[0][0]
        self.u = self.u_1 + self.du
        # 积分抗饱和 输出的限位
        # if self.u >= 10:
        #     self.u = 10
        #
        # if self.u <= -10:
        #     self.u = -10

        # 调整神经网络学习率，进行后续的反向求解
        self.de = self.e - self.e_1
        if self.de > (self.de_init * 1.04):
            self.lr = 0.7 * self.lr_1
        elif self.de < self.de_init:
            self.lr = 1.05 * self.lr_1
        else:
            self.lr = self.lr_1

        #  反向求解，权值在线调整
        
        # sign近似链式法则中的 dy/du, 模型输出变化量/控制增量的变化量（+0.0001避免出现除0）
        self.dyu = np.sin((self.y_out - self.y_1) / (self.u - self.u_1 + 0.0000001))
        # self.dyu = np.sign((self.y_out - self.y_1) / (self.u - 2*self.u_1 + self.u_2 + 0.0000001))
        
        # 输出层激活函数求导
        for i in range(self.outputLayer_num):
            output = self.outputLayer_output_sub[i]
            self.outputLayer_d_sub[i] = 2 / ((np.exp(output) + np.exp(-output)) * (np.exp(output) + np.exp(-output)))
        self.outputLayer_d = np.mat(self.outputLayer_d_sub)
        
        # 链式法则的前四项乘积
        for i in range(self.outputLayer_num):
            self.outputLayer_delta_sub[i] = self.e * self.dyu * self.error_pid.tolist()[i][0] * self.outputLayer_d[0,i]
        self.outputLayer_delta = np.mat(self.outputLayer_delta_sub)

        for l in range(self.outputLayer_num):
            for i in range(self.hiddenLayer_num):
                self.d_wo = (1 - self.alfa) * self.lr * self.outputLayer_delta_sub[l] * self.hiddenLayer_output.tolist()[i][0] + self.alfa * (self.wo_1 - self.wo_2)
                # self.d_wo = self.lr * self.outputLayer_delta_sub[l] * self.hiddenLayer_output.tolist()[i][0] + self.alfa * (self.wo_1 - self.wo_2)

        # self.wo = self.wo_1 + self.d_wo + self.alfa * (self.wo_1 - self.wo_2)
        self.wo = self.wo_1 + self.d_wo
        
        # 中间层激活函数求导
        for i in range(self.hiddenLayer_num):
            hidden = self.hiddenLayer_input.tolist()[0][i]
            self.hiddenLayer_d_sub[i] = 4 / ((np.exp(hidden) + np.exp(-hidden)) * (np.exp(hidden) + np.exp(-hidden)))
        self.hiddenLayer_d = np.mat(self.hiddenLayer_d_sub)
        # 中间层权重学习
        self.hiddeLayer_delta_temp = np.dot(self.outputLayer_delta, self.wo)
        # 中间层激活函数导数 * 中间层权重学习
        for i in range(self.hiddenLayer_num):
             self.hiddenLayer_delta_sub[i] = self.hiddenLayer_d_sub[i] * self.hiddeLayer_delta_temp.tolist()[0][i]
        self.hiddenLayer_delta = np.mat(self.hiddenLayer_delta_sub)

        self.d_wi = (1 - self.alfa) * self.lr * self.hiddenLayer_delta.T * self.r_in + self.alfa * (self.wi_1 - self.wi_2)
        self.wi = self.wi_1 + self.d_wi

        # 参数更新
        self.u_2 = self.u_1
        self.u_1 = self.u

        self.y_1 = self.y_out

        self.wo_2 = self.wo_1
        self.wo_1 = self.wo
        self.wi_2 = self.wi_1
        self.wi_1 = self.wi

        self.e_2 = self.e_1
        self.e_1 = self.e
        
        self.lr_1 = self.lr

        # IncrementValue = self.Kp * (self.Error - self.LastError) + self.Ki * self.Error + self.Kd * (
        #             self.Error - 2 * self.LastError + self.LastLastError)
        # self.PIDOutput += IncrementValue
        # self.LastLastError = self.LastError
        # self.LastError = self.Error
    # 设置一阶惯性环节系统  其中InertiaTime为惯性时间常数
    def SetInertiaTime(self, InertiaTime, SampleTime):
        self.y_out = (InertiaTime * self.y_1 + SampleTime * self.u_2) / (SampleTime + InertiaTime)
        # self.LastSystemOutput = self.SystemOutput

plt.figure("neural pid")  # 创建图表1

def TestPID():
    Process = IncrementalPID()
    ProcessXaxis = [0]
    ProcessYaxis = [0]
    X = [0]
    Y_out = [0]

    for i in range(1, 100):
        Process.SetStepSignal(10)
        # Process.SetInertiaTime(100, 50)
        ProcessXaxis.append(i)
        ProcessYaxis.append(Process.y_out)
        X.append(i)
        Y_out.append(Process.e)
        plt.plot(ProcessXaxis, ProcessYaxis, 'r')
        plt.plot(ProcessXaxis, np.ones(len(ProcessXaxis))*600, 'g')
        plt.text(x = 10, y = 10, s = Process.Kp)
        plt.pause(0.01)
    plt.show()

if __name__ =='__main__':
    TestPID()