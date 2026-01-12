import numpy as np
from xuance.environment import RawMultiAgentEnv # type: ignore
from net_env import NetEnv


class NetMAEnv(RawMultiAgentEnv):
    def __init__(self, env_config):
        super(NetMAEnv, self).__init__()
        # 获取配置
        self.env_id = env_config.env_id
        self.subnet_num = env_config.subnet_num
        self.subnet_node_num = env_config.subnet_node_num
        self.entry_node_num = env_config.entry_node_num
        self.value_node_num = env_config.value_node_num
        self.attack_mode = env_config.attack_mode
        self.reward_mode = env_config.reward_mode
        self.attack_intensity = env_config.attack_intensity
        self.algorithm = env_config.algorithm
        # 实例化NetEnv类
        self.env = NetEnv(self.subnet_num, self.subnet_node_num, 
                        self.entry_node_num, self.value_node_num, attack_mode=self.attack_mode, attack_intensity=self.attack_intensity, 
                        algorithm=self.algorithm, 
                        reward_mode=self.reward_mode)
        self.env.reset()
        # 获取单个智能体的空间定义
        self.single_observation_space = self.env.observation_space
        self.single_action_space = self.env.action_space
        # 实现必要属性（注意名字不能写错了）
        # 智能体
        self.num_agents = env_config.agent_num
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]  # 智能体名字的列表，用于作字典的键
        # 空间定义
        self.state_space = self.single_observation_space  # 全局的状态空间，没有必要设置成字典
        self.observation_space = {agent: self.single_observation_space for agent in self.agents}  # 观察空间字典
        self.action_space = {agent: self.single_action_space for agent in self.agents}  # 动作空间字典（设置成连续的了）
        # 步数
        self.max_episode_steps = 1000
        self._current_step = 0
        
    def get_env_info(self):
        return {'state_space': self.state_space,
                'observation_space': self.observation_space,
                'action_space': self.action_space,
                'agents': self.agents,
                'num_agents': self.num_agents,
                'max_episode_steps': self.max_episode_steps}
    
    # 需要动作屏蔽时，要实现这个函数，否则return None即可
    def avail_actions(self):
        return None
    
    # 当前存活的agent（如果游戏中agent可能只有部分活跃，需要对这部分修改）
    def agent_mask(self):
        return {agent: True for agent in self.agents}
    
    # 返回当前的状态空间
    def state(self):
        """Returns the global state of the environment."""
        return self.env.observe()

    def reset(self):
        obs, _ = self.env.reset()
        observation = {agent: obs for agent in self.agents}
        info = {}
        self._current_step = 0
        return observation, info
    
    def step(self, action_dict):
        # 执行动作
        i = 1
        for action in action_dict.values():
            if i == len(action_dict.values()):  # 最后一个智能体执行动作时，调用step(避免多次攻击)，然后跳出循环
                obs, reward, terminated, truncated, info = self.env.step(action)
                break
            self.env.defend(action)
            i += 1
        # 得到观察字典(以及其他字典)
        observation = {agent: obs for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = False
        self._current_step += 1
        
        return observation, rewards, terminations, truncations, info

    def render(self, *args, **kwargs):
        return
    
    def close(self):
        self.env.close()
        return


if __name__ == "__main__":
    # 读取配置
    import argparse
    from xuance.common import get_configs # type: ignore
    configs_dict = get_configs(file_dir="/root/work/project/configs/ippo_net.yaml")
    configs = argparse.Namespace(**configs_dict)
    
    # 注册环境
    from xuance.environment import REGISTRY_MULTI_AGENT_ENV # type: ignore
    REGISTRY_MULTI_AGENT_ENV[configs.env_name] = NetMAEnv
    
    # 训练
    from xuance.environment import make_envs # type: ignore
    from xuance.torch.agents import IPPO_Agents # type: ignore

    envs = make_envs(configs)  # Make parallel environments.
    Agent = IPPO_Agents(config=configs, envs=envs)  # Create a IPPO agent from XuanCe.
    Agent.train(configs.running_steps)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.
    print('train finish')
    
    
    