'''
Author: zhangL2auto
Date: 2024-01-22 14:29:06
LastEditors: 2auto deutschlei47@126.com
LastEditTime: 2024-01-23 09:47:13
'''
import numpy as np
import skfuzzy as fuzzy
import skfuzzy.control as control

# 1. 变量论域的范围
x_stain_range = np.arange(1, 11, 1, np.float32)
x_oil_range = np.arange(1, 11, 1, np.float32)
y_powder_range = np.arange(1, 11, 1, np.float32)

# 2. 获得模糊集和隶属度函数(使用三角形函数)
# 创建模糊控制变量
x_stain = control.Antecedent(x_stain_range, 'stain')
x_oil = control.Antecedent(x_oil_range, 'oil')
y_powder = control.Consequent(y_powder_range, 'powder')

# 模糊集和隶属函数
x_stain['N'] = fuzzy.trimf(x_stain_range, [1, 1, 5])
x_stain['M'] = fuzzy.trimf(x_stain_range, [1, 5, 10])
x_stain['P'] = fuzzy.trimf(x_stain_range, [5, 10, 10])
x_oil['N'] = fuzzy.trimf(x_oil_range, [1, 1, 5])
x_oil['M'] = fuzzy.trimf(x_oil_range, [1, 5, 10])
x_oil['P'] = fuzzy.trimf(x_oil_range, [5, 10, 10])
y_powder['N'] = fuzzy.trimf(y_powder_range, [1, 1, 5])
y_powder['M'] = fuzzy.trimf(y_powder_range, [1, 5, 10])
y_powder['P'] = fuzzy.trimf(y_powder_range, [5, 10, 10])

# 设定输出powder的解模糊方法——质心解模糊方式
y_powder.defuzzify_method = 'centroid'

# 3. 建立模糊规则
rule0 = control.Rule(antecedent = ((x_stain['N'] & x_oil['N']) | (x_stain['M'] & x_oil['N'])), consequent = y_powder['N'], label = 'rule N')
rule1 = control.Rule(antecedent = ((x_stain['P'] & x_oil['N']) | (x_stain['N'] & x_oil['M']) | (x_stain['M'] & x_oil['M']) |
                                   (x_stain['P'] & x_oil['M']) | (x_stain['N'] & x_oil['P']) ), consequent = y_powder['M'], label = 'rule M')
rule2 = control.Rule(antecedent = ((x_stain['M'] & x_oil['P']) | (x_stain['P'] & x_oil['P']) ), consequent = y_powder['P'], label = 'rule P')

# 系统运行环境
system = control.ControlSystem(rules = [rule0, rule1, rule2])
sim = control.ControlSystemSimulation(system)

sim.input['stain'] = 143
sim.input['oil'] = 105
sim.compute()
output = sim.output['powder']
print(output)