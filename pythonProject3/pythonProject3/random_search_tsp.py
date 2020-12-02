'''随机选一个节点进行画图后计算'''
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def run():
    '''直接画两个图,随机搜索的情况是直接搜索get这里面出现的最好的一个'''
    # 应该是记录至今为止的最长or最短距离结果，所以返回值也要分成不同的
    result_max = []
    result_flag_max = []
    result_min = []
    result_flag_min = []
    min_cost = 999999
    min_list = []
    max_cost = 0
    max_list = []
    '''可以直接维护10个队列长度的mincost'''
    max_ten_times_max = [0 for i in range(10)]
    min_ten_times_min = [999999 for i in range(10)]
    for index in range(1,10001):
        if(index%100==0):
            print(index)
        flag = 0
        if(index%1000==0):
            #记录一下
            flag+=1
        if(flag==1):
            '''设置标准差和根号10'''
            ten_times_max = []
            ten_times_aver_max = 0
            ten_times_min = []
            ten_times_aver_min = 0
            for i in range(10):
                path = np.random.permutation(range(len(data)))
                '''维护总的最长路径和最小路径'''
                temporary = getdistance_sum(path,data)
                if(temporary<min_cost):
                    min_cost = temporary
                    min_list = path
                if(temporary>max_cost):
                    max_cost = temporary
                    max_list = path
                '''维护每次的最长和最短路径'''
                if(temporary>max_ten_times_max[i]):
                    max_ten_times_max[i] = temporary
                if(temporary<min_ten_times_min[i]):
                    min_ten_times_min[i] = temporary

                ten_times_min.append(min_ten_times_min[i])
                ten_times_max.append(max_ten_times_max[i])
                ten_times_aver_min+=min_ten_times_min[i]
                ten_times_aver_max+=max_ten_times_max[i]
            ten_times_aver_max = ten_times_aver_max/10
            ten_times_aver_min = ten_times_aver_min/10

            bar_max = 0
            bar_min = 0
            for i in range(len(ten_times_min)):
                bar_max += (ten_times_max[i]-ten_times_aver_max)**2
                bar_min += (ten_times_min[i]-ten_times_aver_min)**2
            #标准差除以根号10
            bar_max = (bar_max/10)**(1/2)/(10**(1/2))
            bar_min = (bar_min/10)**(1/2)/(10**(1/2))
            result_flag_max.append(bar_max)
            result_flag_min.append(bar_min)
            result_max.append(ten_times_aver_max)
            result_min.append(ten_times_aver_min)
        else:
            ten_times_max = 0
            ten_times_min = 0
            for i in range(10):
                path = np.random.permutation(range(len(data)))
                #维护全局最长最短路径
                temporary = getdistance_sum(path, data)
                if (temporary < min_cost):
                    min_cost = temporary
                    min_list = path
                if (temporary> max_cost):
                    max_cost = temporary
                    max_list = path
                #维护10次最长最短路径

                if (temporary > max_ten_times_max[i]):
                    max_ten_times_max[i] = temporary
                if (temporary < min_ten_times_min[i]):
                    min_ten_times_min[i] = temporary

                ten_times_max+=max_ten_times_max[i]
                ten_times_min+=min_ten_times_min[i]

            result_flag_max.append(0)
            result_flag_min.append(0)
            result_max.append(ten_times_max/10)
            result_min.append(ten_times_min/10)

    max = (max_cost,max_list)
    min = (min_cost,min_list)
    return result_max,result_flag_max,result_min,result_flag_min,max,min


#绘制图片
#序号代表顺序,序号内的节点顺序代表
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




#errorbar变成了标准差除以根号10
#所以,外层的for循环是1000,里面是10次
#比如,运行10次,每次都是1w次迭代,则每次迭代都需要,算的是每次10迭代的平均数和标准差，这个主要是有点针对GA和后面的东西的，前面的影响不是特别大
#比如,运行10次,每次都是1w次迭代,每次都是运行10次,算平均值,没1000次迭代显示错误条
#比如,运行10次,每次迭代肯定都是运行完的,也就是说GA和第二个算法的代码要修改
'''当达到1000代的时候,存储下来结果,GA就直接放到GA函数的里面，然后把值存起来就好了,记录结果和errorbar'''

#中间的就赋值为0就行了嘛,问题基本上就都解决了
if __name__ == '__main__':
    result_max,result_flag_max,result_min,result_flag_min,max,min = run()
    max_cost,max_list = max
    min_cost,min_list = min


    plot_curve_travel_route(min_list,"The_shortest_path")
    plot_curve_travel_route(max_list,"The_longest_path")



    plot_curve_errorbar(result_max,result_flag_max,"The_longest_path_errorbar")
    plot_curve_errorbar(result_min,result_flag_min,"The_shortest_path_errorbar")
