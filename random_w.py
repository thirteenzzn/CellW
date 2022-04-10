# -*- coding = utf-8 -*-
# @Time : 2022/4/10 12:52
# @Author : zzn
# @File : random_w.py
import numpy as np
with open('data/w0.txt','w') as f:
    for i in range(1000):
        f.write(str(np.random.randint(1,10,(5))))
        f.write('\n')