import numpy as np
import pandas as pd
import itertools
from pomegranate import *  # 导入pomegranate库，用于建模贝叶斯网络
from collections import OrderedDict

# 从CSV文件中加载数据并进行预处理
data = np.array(pd.read_csv("asia_data.csv"))  # 读取数据
card = data.max(0)  # 计算每列的最大值，代表每个变量的可能状态数
data_capacity = []  # 初始化，用于存储所有节点的所有可能状态
for i in range(len(card)):
    data_capacity.append(list(np.arange(card[i])))  # 为每个变量生成可能的状态列表
data_capacity = np.array(data_capacity)    
data = data-1  # 数据索引从0开始，因此对所有值减1以适应模型

# 定义函数来计算条件参数（θ）
def theta_copm(Pa,i,j,k):
    Pa_copy = Pa[i].copy()
    Pa_copy.append(i)
        
    j_copy = j.copy()
    j_copy.append(k)
    
    ind = np.where(sum(abs(data[:,Pa_copy] - j_copy).T)==0)
    p_nom = len(ind[0])
    
    ind = np.where(sum(abs(data[:,Pa[i]] - j).T)==0)
    p_denom = len(ind[0])
    if p_denom!=0:
        Theta = p_nom/p_denom
    else:
        Theta = 0
    return Theta

# 计算累积分布函数（CDF）
def cdf_comp(Pa,i,j,k):
    C = 0
    for k1 in range(k+1):
        C += theta_copm(Pa,i,j,k1)
    return C

# 计算一个变量对其它变量的影响分数
def influence_score(Pa,i,n,data_capacity,card):
    parents = Pa[i].copy()
    parents.pop(n)
    parents_comb = list(itertools.product(*data_capacity[parents,:]))
    score = np.zeros(len(parents_comb))
    vote = np.zeros(len(parents_comb))
    for p in range(len(parents_comb)):
        c = np.zeros((card[Pa[i][n]],card[i]-1))
        for s in range(card[Pa[i][n]]):
            for k in range(card[i]-1):
                j = list(parents_comb[p]) + [s]
                c[s,k] = cdf_comp(Pa,i,j,k)
        if sum(sum((c[1:c.shape[0],:]-c[0:c.shape[0]-1,:])>=0))==(c.shape[0]-1)*c.shape[1]:
            score[p] = sum(c[0,:]-c[c.shape[0]-1,:])
            vote[p] = -1
        if sum(sum((c[1:c.shape[0],:]-c[0:c.shape[0]-1,:])<=0))==(c.shape[0]-1)*c.shape[1]:
            score[p] = sum(c[0,:]-c[c.shape[0]-1,:])
            vote[p] = 1
    Score = 0
    if (sum(score>=0)==len(score)) | (sum(score<=0)==len(score)):
        Score = sum(score)
    return Score

# 计算模型中独立参数的数量
def indep_params(Pa,card):    
    params = 1
    for i in range(len(Pa)):
        params*= (card[i]-1)*np.product(card[Pa[i]])
    return params

# 检查图中是否存在循环
def cyclic(g):
    path = set()
    def visit(vertex):
        path.add(vertex)
        for neighbour in g.get(vertex, ()):
            if neighbour in path or visit(neighbour):
                return True
        path.remove(vertex)
        return False
    return any(visit(v) for v in g)

# 计算贝叶斯信息准则（BIC）得分
def BIC_score(data,Pa,card,struct):    
    z = []
    for ii in range(len(Pa)):
        zz = Pa[ii].copy()
        zz.append(ii)
        z.append(zz)
    
    for ii in range(len(z)-1):
        for jj in range(ii+1,len(z)):
            if len(set(z[ii]) - (set(z[ii]) - set(z[jj])))>0:
                z[ii] = list(set(z[ii]+z[jj]))
                z[jj] = list(set(z[ii]+z[jj]))
    ss=100000000
    for ii in range(len(z)):
        if len(z[ii])==len(z):
           ss=0
    model = BayesianNetwork.from_structure(data,struct)
    BIC = model.log_probability(data).sum() - np.log(data.shape[0])*indep_params(Pa,card)/2 - ss
    return BIC

# 初始化种群
def init_pop(ns,data,miu,sigma):
    n = data.shape[1]   
    e = np.round(np.random.normal(miu,sigma,ns))
    e[e>n*(n-1)/2] = n*(n-1)/2
    e[e<n-1] = n-1
    e = e.astype(np.int64)    
    edges = np.arange(n*(n-1)/2)
    all_edges = np.array(list(itertools.product(np.arange(n),np.arange(n))))    
    all_edges = all_edges[all_edges[:,0]<all_edges[:,1]]        
    Pa = []
    for i in range(len(e)):
        Pa.append([])
        for j in range(data.shape[1]):
            Pa[i].append([])    
    for i in range(len(e)):
        s = np.random.randint(0,len(edges),2*len(edges))
        s = list(OrderedDict.fromkeys(s))
        s = s[:e[i]]    
        choose_conj = np.random.randint(0,2,e[i])    
        for j in range(len(s)):        
            conj = (not(choose_conj[j]))
            Pa[i][all_edges[s[j]][choose_conj[j]]].append(all_edges[s[j]][conj+0])
    return Pa

# 变异操作
def mutation(individual):
    n = np.random.randint(0,len(individual))
    binary = np.random.randint(0,1)
    if (binary==0) and (len(individual[n])>0):
        individual[n] = list(np.delete(individual[n],-1))
    if (binary==1) and (len(individual[n])<len(individual)-1):
        m = np.random.randint(0,len(individual))
        if m!=n:
            individual[n].append(m)
            individual[n] = list(set(individual[n]))   
    return individual

# 交叉操作
def crossover(individual1,individual2):
    n = np.random.randint(0,len(individual1))
    ind1 = individual1.copy()
    ind2 = individual2.copy()
    ind1[:n] = individual2[:n]
    ind2[:n] = individual1[:n]
    s1 = 0
    s2 = 0
    for i in range(len(individual1)):
        if len(set(ind1[i])-set([i]))<len(ind1[i]):
            s1 += 1
        if len(set(ind2[i])-set([i]))<len(ind2[i]):
            s2 += 1
    if s1>0:
        ind1 = individual1
    if s2>0:
        ind2 = individual2        
    return ind1,ind2

# 选择操作
def selection(pop,fitness,ns):
    fitness1 = -fitness
    fitness1 = fitness1.max()-fitness1
    fitness1+=50
    rel_fit = fitness1/fitness1.sum()
    rel_fit_cumm = np.cumsum(rel_fit)
    selected_ind = np.zeros((ns))
    for i in range(ns):
        selected_ind[i] = (rel_fit_cumm<np.random.uniform(0, 1)).sum()
        
    selected_ind = selected_ind.astype(int)
    
    selected_pop = [pop[i] for i in selected_ind]    
    return selected_pop,selected_ind     

# 评估种群
def pop_eval(Pa,ns,data,card):
    BICs = []
    ind = []
    for i in range(ns):
        struct = tuple(tuple(x) for x in Pa[i])  
        g = {}
        for j in range(len(struct)):
            g[j] = struct[j]
        if not cyclic(g):    
            BICs.append(BIC_score(data,Pa[i],card,struct))
            ind.append(i)
    BICs = np.array(BICs)
    pop = [Pa[i] for i in ind]
    return BICs,pop

# 主程序开始
# 初始化参数
pr = 0.2  # 变异概率
miu = 10  # 边缘数的平均值
sigma = 3  # 边缘数的标准差
ns = 400  # 种群大小
Pa = init_pop(ns,data,miu,sigma)  # 初始化种群
BICs, pop = pop_eval(Pa, ns, data, card)  # 评估初始种群
ns = len(pop)
if ns % 2 != 0:
    ns -= 1
best_ind = BICs.argmax()
best_individual = pop[best_ind]
best_val = BICs.max()
bests = [best_val]
for iteration in range(10):
    print('iteration=', iteration)
    selected_pop, selected_ind = selection(pop, BICs, ns)  # 选择
    for i in range(ns):
        if np.random.uniform(0, 1) < pr:
            selected_pop[i] = mutation(selected_pop[i])  # 变异
    selected_pop1 = selected_pop.copy()
    for i in range(int(ns / 2)):
        selected_pop1[i * 2], selected_pop1[i * 2 + 1] = crossover(selected_pop1[i * 2], selected_pop1[i * 2 + 1])  # 交叉
    BICs, pop = pop_eval(selected_pop1, ns, data, card)  # 评估种群并实现精英主义
    ind = BICs.argmin()
    pop[ind] = best_individual
    BICs[ind] = best_val
    best_ind = BICs.argmax()
    best_individual = pop[best_ind]    
    best_val = BICs.max()
    bests.append(best_val)

a = np.argsort(BICs)
aa = BICs[a]
bb = [pop[i] for i in a]
p = [[],[0],[],[2],[2],[1,3],[5],[5,4]]
struct = tuple(tuple(x) for x in p) 
print(BIC_score(data, p, card, struct))

# 计算影响分数
Score = 0
for i in range(8):
    if len(bb[156][i]) > 0:
        for n in range(len(bb[156][i])):
            print(influence_score(bb[156], i, 1, data_capacity, card))
