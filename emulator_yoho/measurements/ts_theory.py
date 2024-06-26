from numpy import polyfit, poly1d
import numpy as np
import math

o_yoho = np.array([0.22 , 0.15,  0.03])
o_mlp = np.array([0.11, 0.003, 0.00015])
# parameter

a = 4246  # alpha
a_tencent = 3432.1369751010816
b = 0.071    # beta
b_tencent = 0.08369626317705427

s = 256 # sigma
t = 500  # theta
m_yoho = 65536
m_mlp = 64 * 3072
R_yoho = 1.4045 # r sum, projuct, 1 ?? 1.40625
R_mlp = 1 + 1/12 + (1/12) * (1/4)
O_yoho = sum(o_yoho)
Omax_yoho = np.max(o_yoho)
O_mlp = sum(o_mlp)
Omax_mlp = np.max(o_mlp)

CDU_yoho = math.sqrt( (b * m_yoho * s * t * a) / (a * R_yoho + s * t * (O_yoho - Omax_yoho)) )
CDU_mlp = math.sqrt( (b * m_mlp * s * t * a) / (a * R_mlp + s * t * (O_mlp - Omax_mlp)) )

print("CDU_yoho:", CDU_yoho)
print("CDU_mlp:", CDU_mlp)