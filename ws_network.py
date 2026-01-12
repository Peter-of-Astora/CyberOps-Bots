from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
import random

def create_ws(node_num, entry_node_num, value_node_num, K=6, p=0.1):
    network = Network(set_random_entry_nodes=True, 
                      num_of_random_entry_nodes=entry_node_num, 
                      set_random_high_value_nodes=True, 
                      num_of_random_high_value_nodes=value_node_num, 
                      set_random_vulnerabilities=True)  # 实例化网络
    nodes = [Node(str(i)) for i in range(node_num)]  # 创建 N 个节点
    
    # 初始化规则环状网络，每个节点连接 K 个最近邻
    for node in nodes:
        network.add_node(node)
    
    # 连接每个节点的 K/2 个左邻居和 K/2 个右邻居
    potential_edges = set()  # 记录所有可能的边（避免重复）
    for i in range(node_num):
        for j in range(1, K // 2 + 1):
            left = (i - j) % node_num
            right = (i + j) % node_num
            potential_edges.add((i, left))  # 将边以元组形式保存
            potential_edges.add((i, right))

    # 按概率p选择是否重连
    for u, v in potential_edges:
        if random.random() < p:  # 重连：跳过原始边，随机加新边
            candidates = [x for x in range(node_num) if x != u and x != v]
            if candidates:  # 检查候选列表是否为空
                new_v = random.choice(candidates)
                network.add_edge(nodes[u], nodes[new_v])
            else:
                network.add_edge(nodes[u], nodes[v])
        else:  # 保留原始边
            network.add_edge(nodes[u], nodes[v])
    # 设置特殊节点与漏洞值
    network.reset()
    
    return network

if __name__ == "__main__":
    network = create_ws(30, 2, 2)
    network.show(verbose=True)
    adj, _ = network.to_adj_matrix_and_positions()
    print(adj)
    
    from yawning_titan.envs.generic.core.network_interface import NetworkInterface
    from yawning_titan.game_modes.game_mode_db import default_game_mode
    
    game_mode = default_game_mode()
    network_interface = NetworkInterface(game_mode=game_mode, network=network)
    
    n_nodes = network_interface.get_total_num_nodes()
    obs = network_interface.get_current_observation()

    connections = obs[: n_nodes**2]
    matrix = connections.reshape((n_nodes, n_nodes))
    print(matrix)
    print(n_nodes)
    
    print(network.get_nodes(key_by_name=True))
    print(network.get_nodes(key_by_name=True)['0'], type(network.get_nodes(key_by_name=True)['0']))
    

