import numpy as np
import random
from pgmpy.models import BayesianNetwork 
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import BDeuScore

def has_cycle_util(model, node, visited, recStack):
    visited[node] = True
    recStack[node] = True
    for neighbour in model.successors(node):
        if not visited[neighbour]:
            if has_cycle_util(model, neighbour, visited, recStack):
                return True
        elif recStack[neighbour]:
            return True
    recStack[node] = False
    return False

def dfs_has_cycle(graph, node, visited, parent):
    visited[node] = True
    for neighbour in graph[node]:
        if not visited[neighbour]:
            if dfs_has_cycle(graph, neighbour, visited, node):
                return True
        elif parent != neighbour:
            return True
    return False

def check_cycle(model):
    graph = {node: [] for node in model.nodes()}
    for edge in model.edges():
        graph[edge[0]].append(edge[1])
    
    visited = {node: False for node in model.nodes()}
    for node in model.nodes():
        if not visited[node]:
            if dfs_has_cycle(graph, node, visited, -1):
                return True
    return False

def mutate_with_v_structure_and_no_cycle(model, variables):
    nodes = list(model.nodes())
    attempts = 0
    success = False

    while attempts < 100 and not success:  # 限制尝试的次数
        node1, node2 = random.sample(nodes, 2)
        # 检查是否已经存在这条边或其逆向边
        if not model.has_edge(node1, node2) and not model.has_edge(node2, node1):
            # 临时添加边以检查是否会形成循环
            model.add_edge(node1, node2)
            if check_cycle(model):
                # 如果添加这条边会形成循环，移除这条边
                model.remove_edge(node1, node2)
            else:
                # 成功添加边而且没有形成循环
                success = True
            attempts += 1

    if not success:
        print("Failed to mutate the model without forming a cycle.")




def elite_learning(population, elite_size, data):
    scores = [BDeuScore(data).score(model) for model in population]
    elite_indices = np.argsort(scores)[-elite_size:]
    elite_models = [population[i] for i in elite_indices]
    
    for i, model in enumerate(population):
        if i not in elite_indices:
            elite_model = random.choice(elite_models)
            for edge in elite_model.edges():
                # 检查是否已经存在这条边或反向边，如果不存在，则尝试添加
                if not model.has_edge(*edge) and not model.has_edge(edge[1], edge[0]):
                    model.add_edge(*edge)
                    # 立即检查是否形成了循环
                    if check_cycle(model):  # 使用之前讨论的循环检测逻辑
                        model.remove_edge(*edge)  # 如果形成了循环，则移除这条边


#生成初始种群
def generate_initial_population(variables, population_size):
    population = []
    for _ in range(population_size):
        model = BayesianNetwork()
        for var in variables:
            model.add_node(var)
        population.append(model)
    return population

def run_SIGA_BN(variables, data, population_size=10, elite_size=2, max_generations=20):
    population = generate_initial_population(variables, population_size)
    for generation in range(max_generations):
        # 精英学习
        elite_learning(population, elite_size, data)
        
        # 变异
        for model in population:
            mutate_with_v_structure_and_no_cycle(model, variables)
        
        # 当前代的最佳适应度分数
        best_score = max(BDeuScore(data).score(model) for model in population)
        print(f"Generation {generation}, Best Score: {best_score}")


# variables = ['A', 'B', 'C', 'D']
# model = BayesianNetwork([('A', 'B'), ('B', 'C'), ('C', 'D')])
# cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.5], [0.5]])
# cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['A'], evidence_card=[2])
# cpd_C = TabularCPD(variable='C', variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['B'], evidence_card=[2])
# cpd_D = TabularCPD(variable='D', variable_card=2, values=[[0.5, 0.5], [0.5, 0.5]], evidence=['C'], evidence_card=[2])
# model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)
sampler = BayesianModelSampling(model)
data = sampler.forward_sample(size=1000)


run_SIGA_BN(variables, data)
