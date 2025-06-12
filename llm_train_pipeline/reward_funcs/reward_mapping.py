from reward_funcs import basic
import os
import inspect

# 初始化奖励函数映射字典
# 定义奖励函数映射
reward_funcs_mapping = {
    'format':[basic.strict_format_reward_func, basic.CoT_format_reward_fuc],
    'deep thinking': [basic.language_consistency_reward, basic.cot_reward_func, basic.infer_reward]
}


def get_reward_function(reward_funcs_types):
    return reward_funcs_mapping.get(reward_funcs_types)

