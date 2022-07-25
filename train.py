import json
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs import TrainingEnvV2
from feature_extractors import CustomCombinedExtractor
from callbacks import TensorboardCallback
from config import make_config
from loading_utils import load_train_df
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    config = make_config('config.json')

    agent = config['agent']
    curr_time = int(time.time())
    agent_id = f'{agent}-{curr_time}'
    modelsdir = f'models/{agent_id}'
    logdir = config['log_dir']
    tensorboard_log = config['tensorboard_log']
    tb_log_name = f'{agent_id}'

    df = load_train_df(config['data_dir'], intervals=config['intervals'], coins=config['coins'],
                       fiat=config['fiat'], index_col='date', end_date=config['train_end'])

    env = TrainingEnvV2(df=df, **config['env_kwargs'])
    env = Monitor(env, filename=f'{logdir}/{agent_id}', info_keywords=tuple(config['monitor']))

    tensorboard_callback = TensorboardCallback(verbose=0, monitor_kws=config['tb_monitor'])

    # models = os.listdir(modelsdir)
    # if len(models) > 0:
    #     models.sort(key=lambda item: (len(item), item))
    #     model = PPO.load(f'{modelsdir}/{models[-1]}', env=env)
    #     curr_model_id = int(models[-1].split('.')[0])
    # else:
    # policy_kwargs = dict(
    #     # features_extractor_class=CustomCombinedExtractor,
    #     # features_extractor_kwargs=dict(ts_extractor_arch=extractor_net_arch,
    #     #                                ts_extractor_arch_mlp=extractor_net_arch_mlp,
    #     #                                activation_f=extractor_activation_f),
    #     net_arch=net_arch, activation_fn=activation_f
    # )

    curr_model_id = config['checkpoint_timesteps']
    model = PPO(config['policy'], env, verbose=0, tensorboard_log=tensorboard_log,
                policy_kwargs=config['policy_kwargs'], **config['agent_kwargs'])

    # agent = DRLAgent(env=lambda cfg: env, price_array=None, tech_array=None, turbulence_array=None)
    # PPO_PARAMS = {
    #     "n_steps": 2048,
    #     "ent_coef": 0.005,
    #     "learning_rate": 0.0001,
    #     "batch_size": 128,
    # }
    # model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

    n_loops = config['total_timesteps'] // config['checkpoint_timesteps']
    for ep in range(1, n_loops + 1):
        print(f'Episode {ep}/{n_loops}')
        curr_model_id += config['checkpoint_timesteps']
        model.learn(total_timesteps=config['checkpoint_timesteps'], tb_log_name=tb_log_name, reset_num_timesteps=False,
                    callback=tensorboard_callback)
        model.save(f'{modelsdir}/{curr_model_id}')
        # print('here')
        # model_ppo = agent.train_model(model=model_ppo,
        #                               tb_log_name=tb_log_name,
        #                               total_timesteps=80000, cwd=tensorboard_log)
