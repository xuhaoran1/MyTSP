import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''先画max的图'''
GA_1_max = pd.read_excel('GA_1_max.xlsx')
GA_2_max = pd.read_excel('GA_2_max.xlsx')
hill_cimber_search = pd.read_excel('hill_climber_search.xlsx')
random_search = pd.read_excel('random.xlsx')

plt.style.use('fivethirtyeight')
x_ = [i for i in range(1,10001)]
fig, ax = plt.subplots()
ax.errorbar(x=x_,y=GA_1_max['max_result'],yerr=GA_1_max['max_flag'],label='GA1')
ax.errorbar(x=x_,y=GA_2_max['max_result'],yerr=GA_2_max['max_flag'],label='GA2')
ax.errorbar(x=x_,y=hill_cimber_search['max_result'],yerr=hill_cimber_search['max_flag'],label='HC')
ax.errorbar(x=x_,y=random_search['max_result'],yerr=random_search['max_flag'],label="RS")
ax.legend(loc='lower right')
#设置图注
ax.set_xlabel('times')
ax.set_ylabel('distance')
plt.savefig('picture/'+'max'+'.png')
plt.show()









