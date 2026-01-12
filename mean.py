import numpy as np
import itertools

def calculate_average_cumulative_reward(rewards):
    '''
    输入reward列表,返回平均累积reward列表
    '''
    cumulative_rewards = list(itertools.accumulate(rewards))  # 高效计算累积和
    running_averages = [cum_sum / (i + 1) for i, cum_sum in enumerate(cumulative_rewards)]
    return running_averages

def concatenate_data(data1, data2, data3):
    """
    将三组(step, value)数据按顺序拼接，并确保step值连续递增。

    参数:
        data1 (tuple): 第一组数据，格式为(steps_list, values_list)
        data2 (tuple): 第二组数据，格式为(steps_list, values_list)
        data3 (tuple): 第三组数据，格式为(steps_list, values_list)

    返回:
        tuple: 拼接后的新数据，格式为(concatenated_steps, concatenated_values)
    """
    steps1, values1 = data1
    steps2, values2 = data2
    steps3, values3 = data3
    
    # 计算第一组数据的最大step值
    max_step1 = steps1[-1]
    
    # 计算第二组数据的最大step值
    max_step2 = steps2[-1]
    
    # 调整第二组数据的step：加上第一组的最大值
    adjusted_steps2 = [step + max_step1 for step in steps2]
    
    # 调整第三组数据的step：加上前两组的最大值之和
    adjusted_steps3 = [step + max_step1 + max_step2 for step in steps3]
    
    # 使用简单的列表拼接合并所有step和value
    print(steps1)
    concatenated_steps = list(itertools.chain(steps1, adjusted_steps2, adjusted_steps3))
    concatenated_values = list(itertools.chain(values1, values2, values3))
    print(concatenated_steps[-1])
    
    return concatenated_steps, concatenated_values

if __name__ == "__main__":
    print(calculate_average_cumulative_reward([1, 2, 3]))

    print(concatenate_data(([1, 2, 5], [1, 2, 5]), ([1, 2, 5], [1, 2, 5]), ([1, 2, 5], [1, 2, 5])))
