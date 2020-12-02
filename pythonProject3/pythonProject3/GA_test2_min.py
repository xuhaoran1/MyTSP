import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import pandas as pd
import math

matplotlib.rcParams['font.family'] = 'STSong'

# 数据载入
data = pd.read_csv('tsp.txt', names=['x', 'y'], sep="\t")

# 距离矩阵
point_count = len(data)
Distance = np.zeros([point_count, point_count])
for i in range(point_count):
    for j in range(point_count):
        Distance[i][j] = math.sqrt((data.iat[i, 0] - data.iat[j, 0]) ** 2 + (data.iat[i, 1] - data.iat[j, 1]) ** 2)

# 种群数
count = 200
# 进化次数
iteration = 10000
# 设置强者的定义概率，即种群前20%为强者
retain_rate = 0.2
# 变异率
mutation_rate = 0.1
# 设置起点
index = [i for i in range(point_count)]


# 总距离
def get_total_distance(path_new):
    distance = 0
    for i in range(point_count - 1):
        # count为30，意味着回到了开始的点，此时的值应该为0.
        distance += Distance[int(path_new[i])][int(path_new[i + 1])]
    distance += Distance[int(path_new[-1])][int(path_new[0])]
    return distance


# 适应度评估，选择，迭代一次选择一次
def selection(population):
    # 对总距离从小到大进行排序
    graded = [[get_total_distance(x), x] for x in population]
    graded = [x[1] for x in sorted(graded)]
    # 选出适应性强的染色体
    retain_length = int(len(graded) * retain_rate)
    # 适应度强的集合,直接加入选择中
    parents = graded[:retain_length]
    ## 轮盘赌算法选出K个适应性不强的个体，保证种群的多样性
    s = graded[retain_length:]
    # 挑选的不强的个数
    k = count * 0.2
    # 存储适应度
    a = []
    for i in range(0, len(s)):
        a.append(get_total_distance(s[i]))
    sum = np.sum(a)
    b = np.cumsum(a / sum)
    while k > 0:  # 迭代一次选择k条染色体
        t = random.random()
        for h in range(1, len(b)):
            if b[h - 1] < t <= b[h]:
                parents.append(s[h])
                k -= 1
                break
    return parents


# 交叉繁殖
def crossover(parents):
    # 生成子代的个数,以此保证种群稳定
    target_count = count - len(parents)
    # 孩子列表
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)
        female_index = random.randint(0, len(parents) - 1)
        # 在适应度强的中间选择父母染色体
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]

            left = random.randint(0, len(male) - 2)
            right = random.randint(left + 1, len(male) - 1)

            # 交叉片段
            gene1 = male[left:right]
            gene2 = female[left:right]

            # 得到原序列通过改变序列的染色体，并复制出来备用。
            child1_c = male[right:] + male[:right]
            child2_c = female[right:] + female[:right]
            child1 = child1_c.copy()
            child2 = child2_c.copy()

            # 已经改变的序列=>去掉交叉片段后的序列
            for o in gene2:
                child1_c.remove(o)
            for o in gene1:
                child2_c.remove(o)

            # 交换交叉片段
            child1[left:right] = gene2
            child2[left:right] = gene1

            child1[right:] = child1_c[0:len(child1) - right]
            child1[:left] = child1_c[len(child1) - right:]

            child2[right:] = child2_c[0:len(child1) - right]
            child2[:left] = child2_c[len(child1) - right:]

            children.append(child1)
            children.append(child2)

    return children


# 变异
def mutation(children):
    # children现在包括交叉和优质的染色体
    for i in range(len(children)):
        if random.random() < mutation_rate:
            child = children[i]
            # 产生随机数
            u = random.randint(0, len(child) - 4)
            v = random.randint(u + 1, len(child) - 3)
            w = random.randint(v + 1, len(child) - 2)
            child = child[0:u] + child[v:w] + child[u:v] + child[w:]
            children[i] = child
    return children


# 得到最佳纯输出结果
def get_result(population):
    graded = [[get_total_distance(x), x] for x in population]
    graded = sorted(graded)
    return graded[0][0], graded[0][1]


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
    plt.title(name)
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
def plot_dot(result,name):
    x_ = [i for i in range(1, 10001)]
    plt.scatter(x=x_,y=result)
    plt.xlabel('times')
    plt.ylabel('distance')
    plt.savefig('picture/' + name + '.png')
    plt.show()


if __name__ == '__main__':
    # 主函数的问题
    final_min = []
    final_final_min = []
    final_final_min_flag = []
    min_cost = 999999
    min_path = []
    for i in range(10):
        population = []
        for i in range(count):
            # 随机生成个体
            x = index.copy()
            # 随机排序
            random.shuffle(x)
            population.append(x)

        # 主函数：
        register = []
        i = 0
        distance, result_path = get_result(population)
        register.append(distance)
        while i < iteration:
            if (i % 100 == 0):
                print(i)
            # 选择繁殖个体群
            parents = selection(population)
            # 交叉繁殖
            children = crossover(parents)
            # 变异操作
            children = mutation(children)
            # 更新种群
            population = parents + children
            distance, result_path = get_result(population)
            register.append(distance)
            i = i + 1
        final_min.append(register)
        if (distance < min_cost):
            min_cost = distance
            min_path = result_path
    print("运行10次" + "迭代", iteration, "次后，最优值是：", min_cost)
    print("最优路径：", min_path)
    for j in range(10000):
        flag = 0
        if (j % 1000 == 999):
            flag += 1
        if (flag == 0):
            aver = 0
            for i in range(len(final_min)):
                aver += final_min[i][j]
            aver = aver / len(final_min)
            final_final_min.append(aver)
            final_final_min_flag.append(0)
        else:
            aver = 0
            for i in range(len(final_min)):
                aver += final_min[i][j]
            aver = aver / len(final_min)
            min_bar = 0
            for i in range(len(final_min)):
                min_bar += (final_min[i][j] - aver) ** 2
            min_bar = (min_bar / len(final_min)) ** (1 / 2) / (len(final_min) ** (1 / 2))
            final_final_min.append(aver)
            final_final_min_flag.append(min_bar)

        # 画图
    plot_curve_travel_route(min_path, "GA_2_The_shortest_path")
    plot_curve_errorbar(final_final_min, final_final_min_flag, "GA_2_The_shortest_path_errorbar")
    a = pd.DataFrame(final_final_min, columns=['min_result'])
    a['min_flag'] = final_final_min_flag
    a.to_excel("GA_2_min.xlsx")
    plot_dot(final_final_min,"GA_2_min_dot_plot")