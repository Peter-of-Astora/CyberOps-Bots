import sys
import os
sys.path.append("/root/work/project")  

from ma_net_env import NetMAEnv

# 读取配置
import argparse
from xuance.common import get_configs # type: ignore
configs_dict = get_configs(file_dir="/root/work/project/configs/mappo_net.yaml")
configs = argparse.Namespace(**configs_dict)

# 注册环境
from xuance.environment import REGISTRY_MULTI_AGENT_ENV # type: ignore
REGISTRY_MULTI_AGENT_ENV[configs.env_name] = NetMAEnv

# 训练
from xuance.environment import make_envs # type: ignore
from xuance.torch.agents import MAPPO_Agents # type: ignore
envs = make_envs(configs)  # Make parallel environments.
# 把下面几行注释掉即可只测试
Agent = MAPPO_Agents(config=configs, envs=envs)  # Create a IPPO agent from XuanCe.
Agent.train(configs.running_steps)  # Train the model for numerous steps.
Agent.save_model("MAPPO.pth")  # Save the model to model_dir.
Agent.finish()  # Finish the training.
print('train finish')

def env_fn():
    envs = make_envs(configs)
    return envs
    
# agent_test = MAPPO_Agents(config=configs, envs=envs)
# agent_test.load_model("/root/work/project/models/MAPPO")  # 如果不指定后一个model参数，自动会载入最新的
# agent_test.test(env_fn, 0)