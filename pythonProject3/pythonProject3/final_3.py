import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

index = np.arange(4)
#最长路径
values = [755.80, 753.13, 550.94, 720.43]
SD = [30, 16, 20,14 ]
plt.ylabel('distance')
plt.bar(index, values, yerr = SD, error_kw = {'ecolor' : '0.2', 'capsize' :6}, alpha=0.7, label = 'First')
plt.xticks(index,['GA1', 'GA2', 'RS', 'HC'])
plt.savefig('picture/max_bar.png')
plt.show()
