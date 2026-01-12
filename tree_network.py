from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
import random
from collections import deque
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode_db import default_game_mode
import networkx as nx

def create_tree(node_num, entry_node_num, value_node_num, seed=42):
    random.seed(seed)
    
    network = Network(set_random_vulnerabilities=True)  # 实例化网络
    nodes = [Node(str(i)) for i in range(node_num)]  # 创建 N 个节点
    
    # 设置特殊节点，添加节点
    for i in range(entry_node_num):
        nodes[i]._entry_node = True
    for i in range(value_node_num):
        nodes[-1 - i]._high_value_node = True
    for node in nodes:
        network.add_node(node)
    
    # 创建树形网络
    root = nodes[0]
    queue = deque([root])
    used_nodes = {root.name}
    max_branch = node_num // 3  # 最大子节点数量取决于总节点数量
    
    while len(used_nodes) < node_num and queue:
        current_node = queue.popleft()
        
        # 确定当前节点可以有多少个子节点
        remaining_nodes = node_num - len(used_nodes)
        if remaining_nodes <= 0:
            break
        
        # 随机决定子节点数量(1到min(max_branch, remaining_nodes))
        max_possible = min(max_branch, remaining_nodes)
        num_children = random.randint(1, max_possible)
        
        # 添加子节点
        for _ in range(num_children):
            if len(used_nodes) >= node_num:
                break
            
            # 从尚未使用的节点中选择
            available_nodes = [n for n in nodes if n.name not in used_nodes]
            if not available_nodes:
                break
                
            # 顺序选择一个节点作为子节点
            child = available_nodes[0]
            network.add_edge(current_node, child)
            used_nodes.add(child.name)
            queue.append(child)
    
    network.reset()
    
    return network

def calculate_network_efficiency(adj_matrix, weighted=False):
    """
    基于邻接矩阵计算网络的全局效率（连通性指标）
    
    参数:
        adj_matrix (numpy.ndarray): 邻接矩阵（加权或非加权）
        weighted (bool): 是否为加权图，默认False
    
    返回:
        float: 全局网络效率值（范围[0,1]）
    """
    # 将邻接矩阵转换为NetworkX图
    if weighted:
        G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph())
    else:
        G = nx.from_numpy_array(adj_matrix > 0, create_using=nx.Graph())  # 二值化处理
    
    # # 检查网络连通性（若存在孤立节点需特殊处理）
    # if not nx.is_connected(G):
    #     print("警告：网络不连通，效率计算将忽略不可达节点对")
    
    # 计算全局效率
    efficiency = nx.global_efficiency(G)
    return efficiency

def calculate_interfaces_efficiency(interfaces):
    efficiencies = []
    for interface in interfaces:
        base_matrix = nx.to_numpy_array(interface.base_graph)
        current_matrix = interface.adj_matrix
        base_efficiency = calculate_network_efficiency(base_matrix)
        current_efficiency = calculate_network_efficiency(current_matrix)
        subnet_efficiency = current_efficiency / base_efficiency
        efficiencies.append(subnet_efficiency)
    mean_efficiency = sum(efficiencies) / len(efficiencies)
    return mean_efficiency
    
if __name__ == "__main__":
    network1 = create_tree(30, 2, 2)
    network2 = create_tree(30, 2, 2)
    nodes = network1.get_nodes(key_by_name=True)
    
    network1.show(verbose=True)
    # network2.show(verbose=True)
    
    adj1, _ = network1.to_adj_matrix_and_positions()
    adj2, _ = network2.to_adj_matrix_and_positions()
    
    print(adj1)
    print(adj2)
    
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    
    G = nx.from_numpy_array(adj2, create_using=nx.DiGraph())
    pos = nx.spring_layout(G, seed=42, k=0.8)  # k值调整节点间距

    nx.draw_networkx_edges(G, pos, edge_color='black', width=1.5, arrows=True, arrowsize=15)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
    nx.draw_networkx_labels(G, pos, font_color='black')

    plt.axis('off')
    plt.savefig('tree30_2.png')
    
    print(network1.high_value_nodes)
    
    game_mode = default_game_mode()
    network_interface = NetworkInterface(game_mode=game_mode, network=network1)
    nodes = network1.get_nodes(key_by_name=True)
    
    n_nodes = network_interface.get_total_num_nodes()
    obs = network_interface.get_current_observation()
    
    print('entry_nodes:')
    entry_nodes = obs[n_nodes**2 + 5 * n_nodes + 2 : n_nodes**2 + 6 * n_nodes + 2]
    print(entry_nodes)

    print('high value nodes:')
    high_nodes = obs[n_nodes**2 + 6 * n_nodes + 2 : n_nodes**2 + 7 * n_nodes + 2]
    print(high_nodes)
    
    
    network_interface.isolate_node(nodes['1'])
    adj_matrix = network_interface.adj_matrix
    print(calculate_network_efficiency(adj_matrix))
    
    print(calculate_interfaces_efficiency([network_interface]))
    
