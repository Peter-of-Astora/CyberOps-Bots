import numpy as np
import attack

def obs2matrix(interfaces):
    '''
    将状态空间转换成自然语言描述的网络描述所需的指标\n
    interfaces: A list of NetworkInterface instances\n
    '''
    depths = []
    iso_num = []
    compromised_num = []
    for interface in interfaces:
        deep_node_index, depth = attack.calculate_penetrate_depth(interface)
        depths.append(depth)
        n_nodes = interface.get_total_num_nodes()
        obs = interface.get_current_observation()
        isolated_state = obs[n_nodes**2: n_nodes**2 + n_nodes]
        compromised_state = obs[n_nodes**2 + n_nodes: n_nodes**2 + 2 * n_nodes]
        isolated_node_num = np.count_nonzero(isolated_state == 1)
        compromised_node_num = np.count_nonzero(compromised_state == 1)
    return

if __name__ == "__main__":
    pass
