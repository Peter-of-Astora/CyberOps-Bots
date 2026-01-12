import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from typing import Optional
import os
import re

def read_tensorboard_log(event_path, tag):
    '''
    Read event log with scalar tag and return the steps and values.
    '''
    # 指定你下载的 TensorBoard 事件文件路径
    event_file_path = event_path

    # 初始化 EventAccumulator 并加载数据
    ea = event_accumulator.EventAccumulator(event_file_path)
    ea.Reload()  # 加载事件文件中的所有数据

    # 获取事件文件中记录的所有标签（tags）
    tags = ea.Tags()

    scalar_tag = tag
    scalar_data = ea.Scalars(scalar_tag)

    steps = np.array([event.step for event in scalar_data])  # X轴列表
    values = np.array([event.value for event in scalar_data])  # Y轴列表

    return steps, values

def find_log_directory(
    base_log_dir: str,
    algorithm: str,
    subnet_num: int,
    attacker_num: int,
    attack_mode: str
) -> Optional[str]:
    """
    在给定的日志文件夹中查找符合要求的日志目录。

    参数:
        base_log_dir (str): 日志根目录路径
        algorithm (str): 算法名，如 'IPPO'
        subnet_num (int): 子网数量
        attacker_num (int): 攻击者数量
        attack_mode (str): 攻击模式，如 'recon'

    返回:
        Optional[str]: 匹配到的日志目录完整路径，未找到返回 None
    """

    # 构造正则表达式
    pattern = re.compile(
        rf"^{re.escape(algorithm)}_{subnet_num}x\d+with.*{attacker_num}e.*{attacker_num}v{attacker_num}att_{re.escape(attack_mode)}_\d{{8}}-\d{{4}}$"
    )

    # 遍历日志目录
    for dirname in os.listdir(base_log_dir):
        if pattern.match(dirname):
            full_path = os.path.join(base_log_dir, dirname)
            if os.path.isdir(full_path):
                return full_path

    return None


if __name__ == "__main__":
    dir = find_log_directory('/root/work/project/log/env_log', algorithm='MAPPO', subnet_num=1, attacker_num=2, attack_mode='recon')
    print(dir)

    steps, values = read_tensorboard_log(dir, 'Reward-Steps')

    # 打印数组形状以验证
    print(f"步骤数数组形状: {steps.shape}")
    print(f"值数组形状: {values.shape}")
    print(f"前10个步骤: {steps[:10]}")
    print(f"前10个值: {values[:10]}")
    print(type(steps), type(values))
