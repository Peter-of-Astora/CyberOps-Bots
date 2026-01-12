
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.game_modes.game_mode_db import default_game_mode
from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
import random
import numpy as np
from tree_network import create_tree
from collections import deque
from collections import Counter
import math


def get_attackable_subnet_nodes(interface):
    '''
    Input: A NetworkInterface Instance
    Output: A list of Nodes' name/index
    '''
    # 获取网络信息
    n_nodes = interface.get_total_num_nodes()  # 节点数量
    obs = interface.get_current_observation()  # 状态空间
    connections = obs[: n_nodes**2]
    matrix = connections.reshape((n_nodes, n_nodes))  # 邻接矩阵
    compromised_dict = interface.get_all_node_compromised_states()  # 攻陷字典
    compromised_state = list(compromised_dict.values())  # 攻陷状态
    entry_nodes = obs[n_nodes**2 + 5 * n_nodes + 2 : n_nodes**2 + 6 * n_nodes + 2]  # 入口节点
    entry_nodes_num = np.count_nonzero(entry_nodes == 1)  # 入口节点数量
    
    # 循环每一个节点检查是否是可被攻击的节点
    attackable_nodes_index = []
    for i in range(len(compromised_state)):
        neighbors = matrix[i]  # i节点的邻接向量
        self_compromised_bool = compromised_state[i]  # 节点i自己是否已经被攻陷 
        if entry_nodes[i] == 1:
            # 如果是入口节点，肯定可以攻击，直接往下走(并保证可攻击节点列表不为空)
            attackable_nodes_index.append(int(i))
            continue
        if self_compromised_bool == 1:
            # 如果自己已经被攻陷了，没有必要再攻一次
            continue
        for j in range(len(compromised_state)):
            edge_bool = neighbors[j]  # 判断是否是邻居(邻接矩阵中i=j时为0,不用考虑)
            compromised_bool = compromised_state[j]  # 判断是否被攻陷
            if edge_bool and compromised_bool:  # 如果是邻居且被攻陷，说明i节点可被攻击
                attackable_nodes_index.append(int(i))
            
    # 如果可攻击节点列表中不止有入口节点，说明已经进入网络，没有必要反复攻击入口节点，把入口节点吐掉
    if len(attackable_nodes_index) > entry_nodes_num:
        attackable_nodes_index = attackable_nodes_index[entry_nodes_num: ]
    
    return attackable_nodes_index

def get_all_attackable_nodes(interface_list):
    '''
    Input: A list of NetworkInterface Instance
    Output: A list of list of Nodes' name/index
    '''
    nodes_index = []
    for interface in interface_list:
        nodes_index.append(get_attackable_subnet_nodes(interface))
    return nodes_index

def shortest_path_length(adj_matrix, i, j):
    """
    计算无向无权图中节点i到节点j的最短路径长度
    
    参数:
        adj_matrix: numpy数组表示的邻接矩阵
        i: 起始节点索引
        j: 目标节点索引
        
    返回:
        两节点之间的最短距离（整数）
        如果节点不可达，返回-1
    """
    n = adj_matrix.shape[0]  # 节点数量
    if i == j:
        return 0  # 同一节点距离为0
    
    visited = [False] * n  # 记录已访问节点
    distance = [-1] * n     # 记录各节点到i的距离
    queue = deque()
    
    queue.append(i)
    visited[i] = True
    distance[i] = 0
    
    while queue:
        current = queue.popleft()
        
        # 遍历当前节点的所有邻居
        for neighbor in range(n):
            if adj_matrix[current][neighbor] == 1 and not visited[neighbor]:
                if neighbor == j:  # 找到目标节点
                    return distance[current] + 1
                
                visited[neighbor] = True
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    
    return -1  # 节点不可达

def calculate_penetrate_depth(interface):
    '''
    计算某子网中距离关键节点最近的被攻陷节点
    
    interface: NetworkInterface Class of the subnet
    
    return: node_index, distance
    '''
    # 获取网络信息
    n_nodes = interface.get_total_num_nodes()  # 节点数量
    obs = interface.get_current_observation()  # 状态空间
    connections = obs[: n_nodes**2]
    matrix = connections.reshape((n_nodes, n_nodes))  # 邻接矩阵
    compromised_dict = interface.get_all_node_compromised_states()  # 攻陷字典
    compromised_state = list(compromised_dict.values())  # 攻陷状态
    compromised_node_indexes = [i for i, x in enumerate(compromised_state) if x == 1]  # 攻陷节点索引
    value_node_state = obs[n_nodes**2 + 6 * n_nodes + 2 : n_nodes**2 + 7 * n_nodes + 2]  # 关键节点
    value_node_indexes = [i for i, x in enumerate(value_node_state) if x == 1]  # 关键节点索引
    
    # 保存与关键节点的最小距离
    min_distance = -1
    node_index = 0
    
    # 循环遍历攻陷节点，找到与关键节点距离最短的
    for compromised_node in compromised_node_indexes:
        # 对于每一个关键节点计算距离
        for value_node in value_node_indexes:
            dis = shortest_path_length(matrix, compromised_node, value_node)
            # 如果路径不存在，则跳过
            if dis < 0:
                continue
            # 如果路径更短，打擂台
            if min_distance < 0 or dis < min_distance:
                min_distance = dis
                node_index = compromised_node
    
    return node_index, min_distance

def dijkstra_shortest_path(adjacency_matrix, start_node, end_node):
    """
    使用Dijkstra算法计算两个节点之间的最短路径
    
    参数:
        adjacency_matrix: 二维数组形式的邻接矩阵，0表示无连接
        start_node: 起始节点的索引
        end_node: 目标节点的索引
        
    返回:
        包含路径节点索引的列表，如果路径不存在则返回空列表
    """
    n = len(adjacency_matrix)  # 节点数量
    INF = float('inf')
    
    # 初始化距离数组和访问标记
    distances = [INF] * n
    distances[start_node] = 0
    visited = [False] * n
    
    # 前驱节点数组，用于重建路径
    previous_nodes = [-1] * n
    
    for _ in range(n):
        # 找到当前未访问节点中距离最小的节点
        current_node = -1
        min_distance = INF
        for node in range(n):
            if not visited[node] and distances[node] < min_distance:
                min_distance = distances[node]
                current_node = node
        
        # 如果没有找到可到达的节点，说明剩余节点不可达
        if current_node == -1:
            break
        
        # 标记当前节点为已访问
        visited[current_node] = True
        
        # 如果已经到达目标节点，可以提前终止
        if current_node == end_node:
            break
        
        # 更新邻居节点的距离
        for neighbor in range(n):
            if (adjacency_matrix[current_node][neighbor] != 0 and  # 有连接
                not visited[neighbor]):  # 未访问
                new_distance = distances[current_node] + adjacency_matrix[current_node][neighbor]
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
    
    # 重建路径
    path = []
    current_node = end_node
    
    # 如果无法到达目标节点，返回空列表
    if distances[end_node] == INF:
        return path
    
    # 从终点回溯到起点
    while current_node != -1:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    
    # 反转路径，使其从起点到终点
    path.reverse()
    
    return path

def calculate_entropy(attack_sequence):
    """
    计算单个子网攻击序列的熵值
    
    参数:
        attack_sequence: list, 单个子网的攻击节点历史列表，如 ['0', '4', '0', '0', '1', ...]
        
    返回:
        float: 该子网攻击序列的熵值
    """
    # 计算总攻击次数
    total_attacks = len(attack_sequence)
    if total_attacks == 0:
        return 0.0
    
    # 统计每个节点出现的频率
    node_counts = Counter(attack_sequence)
    
    # 计算熵值
    entropy = 0.0
    for count in node_counts.values():
        probability = count / total_attacks
        entropy -= probability * math.log2(probability)
    
    return entropy

def calculate_attack_concentration(attack_history):
    """
    计算所有子网的攻击链路集中度(基于熵)
    
    参数:
        attack_history: list of lists, 攻击历史列表，每个子网是一个子列表
        
    返回:
        list: 每个子网对应的攻击链路集中度(1-归一化熵值)
    """
    concentration_scores = []
    
    for subnet_sequence in attack_history:
        # 计算原始熵值
        entropy = calculate_entropy(subnet_sequence)
        
        # 计算最大可能熵值(当所有节点均匀分布时)
        unique_nodes = len(set(subnet_sequence))
        if unique_nodes <= 1:
            max_entropy = 0.0
        else:
            max_entropy = math.log2(unique_nodes)
        
        # 计算集中度得分(1-归一化熵值)
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
            concentration = 1 - normalized_entropy
        else:
            concentration = 1.0  # 只有一种攻击节点，完全集中
        
        concentration_scores.append(concentration)
    
    return concentration_scores

def provoke_long_memory(interface):
    '''
    计算响应记忆中的关键路径
    
    return:
        enter_path: 已保存的攻击链路
        predict_path: 攻击者可能对关键节点进行的攻击链路
    '''
    # 获取网络信息
    n_nodes = interface.get_total_num_nodes()  # 节点数量
    obs = interface.get_current_observation()  # 状态空间
    connections = obs[: n_nodes**2]
    matrix = connections.reshape((n_nodes, n_nodes))  # 邻接矩阵
    compromised_dict = interface.get_all_node_compromised_states()  # 攻陷字典
    compromised_state = list(compromised_dict.values())  # 攻陷状态
    compromised_node_indexes = [i for i, x in enumerate(compromised_state) if x == 1]  # 攻陷节点索引
    value_node_state = obs[n_nodes**2 + 6 * n_nodes + 2 : n_nodes**2 + 7 * n_nodes + 2]  # 关键节点
    value_node_indexes = [i for i, x in enumerate(value_node_state) if x == 1]  # 关键节点索引
    # 计算最近的节点与距离
    node_index, dis = calculate_penetrate_depth(interface)
    # 计算入口到这个节点的路径
    enter_path = dijkstra_shortest_path(matrix, 0, node_index)
    # 计算这个节点到关键节点的路径
    all_paths = [dijkstra_shortest_path(matrix, node_index, value_node) for value_node in value_node_indexes]
    valid_paths = [path for path in all_paths if path]  # 只保留非空路径
    predict_path = min(valid_paths, key=len)
    
    return enter_path, predict_path
    
if __name__ == "__main__":
    game_mode = default_game_mode()
    network = create_tree(30, 2, 2)
    nodes = network.get_nodes(key_by_name=True)
    interface = NetworkInterface(game_mode=game_mode, network=network)
    
    network.show(verbose=True)

    interface.attack_node(nodes['0'])
    interface.attack_node(nodes['2'])
    interface.attack_node(nodes['8'])
    # interface.isolate_node(nodes['4'])
    
    provoke_long_memory(interface)
    
    # print(get_attackable_subnet_nodes(interface))
    
    # n_nodes = interface.get_total_num_nodes()
    # obs = interface.get_current_observation()
    # connections = obs[: n_nodes**2]
    # matrix = connections.reshape((n_nodes, n_nodes))
    
    # distance = shortest_path_length(matrix, 0, 9)
    # print(distance)
    
    # print(calculate_penetrate_depth(interface))
    
    # path = dijkstra_shortest_path(matrix, 5, 9)
    # print(path)
    
    # attack_history = [
    #     ['0', '4', '0', '0', '1', '0', '4', '6', '3', '16', '15', '6', '10', '11', '9', '11', '10', '14', '6', '13', '14', '12', '2', '17', '10', '8', '13', '13'],
    #     ['0', '0', '0', '0', '0', '0', '0', '0', '2']
    # ]
    
    # concentrations = calculate_attack_concentration(attack_history)
    
    # for i, score in enumerate(concentrations):
    #     print(f"子网 {i} 的攻击链路集中度: {score:.4f}")