import numpy as np
import skfuzzy as fuzzy
import skfuzzy.control as control
import matplotlib.pyplot as plt

# Beispiel
# 输入为服务(service)和质量(quality)两个参数，输出为得到的小费(tip)

# 输入输出模糊集
x_service = np.arange(1, 11, 1, np.float32)
x_quality = np.arange(1, 11, 1, np.float32)
y_tips = np.arange(1, 25, 1, np.float32)
service = control.Antecedent(x_service, 'service')
quality = control.Antecedent(x_quality, 'quality')
tips = control.Consequent(y_tips, 'tips')

# 输入输出隶属度函数

service['L'] = fuzzy.trimf(x_quality, [0, 0, 5])
#定义质量差时的三角隶属度函数横坐标
quality['L'] = fuzzy.trimf(x_quality, [0, 0, 5])  
quality['M'] = fuzzy.trimf(x_quality, [0, 5, 10])
quality['H'] = fuzzy.trimf(x_quality, [5, 10, 10])
#若使用automf自动生成代码为 
##names = ['L', 'M', 'H']
##quality.automf(names=names)  #自动分配
#定义服务差时的三角隶属度函数横坐标
service['L'] = fuzzy.trimf(x_service, [0, 0, 5]) 
service['M'] = fuzzy.trimf(x_service, [0, 5, 10])
service['H'] = fuzzy.trimf(x_service, [5, 10, 10])
#定义小费的三角隶属度函数横坐标
tips['L'] = fuzzy.trimf(y_tips, [0, 0, 13]) 
tips['M'] = fuzzy.trimf(y_tips, [0, 13, 25])
tips['H'] = fuzzy.trimf(y_tips, [13, 25, 25])

#解模糊方法采用质心法
tips.defuzzify_method = 'centroid'

rule1=control.Rule(antecedent=((quality['L'] & service['L'])|(quality['L'] & service['M'])|(quality['M'] & service['L'])),consequent=tips['L'],label='Low')
rule2=control.Rule(antecedent=((quality['M']&service['M'])|(quality['L']&service['H'])|(quality['H']&service['L'])),consequent=tips['M'],label='Medium')
rule3=control.Rule(antecedent=((quality['M']&service['H'])|(quality['H']&service['M'])|(quality['H']&service['H'])),consequent=tips['H'],label='High')


tipping_ctrl = control.ControlSystem([rule1, rule2, rule3])
tipping = control.ControlSystemSimulation(tipping_ctrl)
tipping.input['quality'] = 6.5
tipping.input['service'] = 9.8
tipping.compute()
print (tipping.output['tips'])
tips.view(sim=tipping)
plt.show()