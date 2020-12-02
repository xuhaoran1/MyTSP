import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

index = np.arange(4)
#最短路径
values = [79.91, 74.58, 482.70, 241.22]
SD = [5, 16, 20,15 ]
plt.ylabel('distance')
plt.bar(index, values, yerr = SD, error_kw = {'ecolor' : '0.2', 'capsize' :6}, alpha=0.7, label = 'First')
plt.xticks(index,['GA1', 'GA2', 'RS', 'HC'])
plt.savefig('picture/min_bar.png')
plt.show()
