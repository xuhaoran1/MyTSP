# MyTSP
使用random research，hill climber，以及2个GA共计四种种方法计算TSP的最长和最短路径，并选出最佳方法,可用于遗传算法解决问题的参考
# 方法解释
总共写了三种大方法:
第一种，随机优化，从原理上讲，就是随机生成1w次的路径规划，总共运行10次，得出每一步的最小优化结果
第二种，是爬山法或者叫盆地跳跃的方法，这种方法是选定一个初始解，在该解的基础上进行优化，代码的表示是通过任取两个路径节点交换，计算是否比原来的效果更好，如果更好则保留，如果没有更好，则不这么优化，这种优化极容易陷入局部最优解中
第三种是遗传算法，遗传算法的代码总共写了两种，一种是在选择方法中使用精英方法和轮盘对赌法，一种是只使用精英遗传的方法。逻辑是这样的，直接使用精英遗传的下一代有可能会也有可能会出现陷入局部最优解的情况，而加入轮盘对赌法则存在一定的概率接受此时较差但未来经过优化后提升幅度较大的解集，这就是精英方法和轮盘对赌法的来源。
