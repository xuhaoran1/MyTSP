'''hill_climber,就是爬山法或者说盆地跳跃'''
'''使用的方法简单的来说,就是随机抽取两个城市,而后交换为止,如果效果更好,则保留结果，如果效果不好，则随机抽取结果'''
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process,Queue

#总的存储队列
# q = Queue()

data = pd.read_csv('tsp.txt',names=['x','y'],sep="\t")
#总共1000行数据,也就是1000个点


def getdistance_sum(path,data):
    '''距离是正好算一圈'''
    result = 0
    for i in range(len(path)):
        result+=get_distance(path[i],path[(i+1)%len(path)],data)
    return result


def get_distance(num1,num2,data):
    '''欧式距离'''
    x1,y1 = data.iloc[num1]
    x2,y2 = data.iloc[num2]
    return ((x1-x2)**2+(y1-y2)**2)**(1/2)
#爬山法来计算TSP

def min_exchange(path,min):
    '''最长路径交换位置'''
    len_ = len(path)

    a = random.randint(0,len_-1)
    b = random.randint(0,len_-1)
    while(a==b):
        b = random.randint(0,len_-1)

    path[a], path[b] = path[b], path[a]

    if (getdistance_sum(path, data) < min):
        min = getdistance_sum(path, data)
    else:
        path[a], path[b] = path[b], path[a]

    return min

def max_exchange(path,max):
    '''最短路径交换位置'''
    len_ = len(path)

    a = random.randint(0, len_-1)
    b = random.randint(0, len_-1)
    while (a == b):
        b = random.randint(0, len_-1)

    path[a],path[b] = path[b],path[a]
    if (getdistance_sum(path, data) > max):
        max = getdistance_sum(path, data)
    else:
        path[a],path[b] = path[b],path[a]

    return max



'''把运行10次的结果记录下来不就行了'''
'''那是运行上的问题，代码上只写迭代次数'''

def run():
    '''直接画两个图,随机搜索的情况是直接搜索get这里面出现的最好的一个'''
    # 应该是记录至今为止的最长or最短距离结果
    result_max = []
    result_min = []
    min_cost = 999999
    min_list = []
    max_cost = 0
    max_list = []
    #记录上一次的permutaion
    path  = np.random.permutation(range(len(data)))
    first = getdistance_sum(path, data)
    if(first<min_cost):
        min_cost = first
        min_list = path
    if(first>max_cost):
        max_cost = first
        max_list = path
    for index in range(1,10001):
        if(index%100==0):
            print(index)
        max_cost = max_exchange(max_list,max_cost)
        min_cost = min_exchange(min_list,min_cost)
        result_max.append(max_cost)
        result_min.append(min_cost)

    return(result_max,result_min,max_list,min_list)

def plot_curve_travel_route(path,name):
    for i in range(len(data)):
        plt.annotate("",xy=(data.iat[path[i],0],data.iat[path[i],1]))
    data1 = pd.DataFrame(columns=data.columns)
    X_ = []
    Y_ = []
    for i in path:
        X_.append(data.iat[i,0])
        Y_.append(data.iat[i,1])
    plt.plot(X_,Y_)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Random_Search-"+name)
    plt.savefig('picture/'+ name + ".png")
    plt.show()


#绘制errorbar
#下面是代数,上面的迭代后的路径长度,哦，也算吧,等同于不同的起点的值
def plot_curve_errorbar(result,result_flag,name):
    x_ = [i for i in range(1,10001)]

    plt.errorbar(x=x_,y=result,yerr=result_flag)
    plt.xlabel('times')
    plt.ylabel('distance')
    plt.savefig('picture/'+name+'.png')
    plt.show()



if __name__ == '__main__':

    all = []
    #jobs = []
    #p = multiprocessing.Pool(10)
    for i in range(10):
        #p.apply_async(run(),callback=lambda x:all.append(x))
        # p = Process(target=run())
        # jobs.append(p)
        # p.start()
        result_max,result_min,max_list,min_cost = run()
        all.append((result_max,result_min,max_list,min_cost))

    # for p in jobs:
    #     p.join()
    # all = [q.get() for j in range(10)]
    finish_max =[]
    finish_min =[]
    finish_max_flag = []
    finish_min_flag = []


    #最后10次运行中最长路径or最短路径
    max_list = []
    min_list = []
    max_cost = 0
    min_cost = 99999
    for j in range(10000):
        maxi = 0
        mini = 0
        flag = 0
        if(j%1000==999):
            flag+=1
        if(flag==0):
            for i in range(len(all)):
                maxi += all[i][0][j]
                mini += all[i][1][j]
            maxi=maxi/len(all)
            mini=mini/len(all)
            finish_max.append(maxi)
            finish_min.append(mini)
            finish_max_flag.append(0)
            finish_min_flag.append(0)
        else:
            for i in range(len(all)):
                maxi += all[i][0][j]
                mini += all[i][1][j]
            maxi=maxi/len(all)
            mini=mini/len(all)
            max_bar = 0
            min_bar = 0
            for i in range(len(all)):
                max_bar += (all[i][0][j]-maxi)**2
                min_bar += (all[i][1][j] - mini) ** 2
            max_bar = (max_bar/len(all))**(1/2)/(len(all)**(1/2))
            min_bar = (min_bar / len(all)) ** (1 / 2) / (len(all) ** (1 / 2))

            finish_max.append(maxi)
            finish_min.append(mini)
            finish_max_flag.append(max_bar)
            finish_min_flag.append(min_bar)


    for i in range(len(all)):
        temp_max=getdistance_sum(all[i][2],data)
        temp_min = getdistance_sum(all[i][3],data)
        if(temp_max>max_cost):
            max_cost=temp_max
            max_list = all[i][2]
        if (temp_min < min_cost):
            min_cost = temp_min
            min_list = all[i][3]


    #画图
    plot_curve_travel_route(min_list, "The_shortest_path")
    plot_curve_travel_route(max_list, "The_longest_path")

    plot_curve_errorbar(finish_max, finish_max_flag, "The_longest_path_errorbar")
    plot_curve_errorbar(finish_min, finish_min_flag, "The_shortest_path_errorbar")
