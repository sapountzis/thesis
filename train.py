import json
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import ReLU, Tanh
from envs import TrainingEnv
from feature_extractors import CustomCombinedExtractor


activation_f_map = {'relu': ReLU, 'tanh': Tanh}
agent_map = {'PPO': PPO}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, monitor_kws=()):
        super(TensorboardCallback, self).__init__(verbose)
        self.monitor_kws = monitor_kws

    def _on_step(self) -> bool:
        for monitor_kw in self.monitor_kws:
            self.logger.record(monitor_kw, self.training_env.get_attr(monitor_kw)[0])

        return True


if __name__ == '__main__':

    with open("config.json", "r") as configfile:
        config = json.load(configfile)

        try:
            data_dir = config['data_dir']
            intervals = config['intervals']
            past_steps = config['past_steps']
            loss_tolerance = config['loss_tolerance']
            loss_penalty_coef = config['loss_penalty_coef']
            fee = config['fee']
            fiat = config['fiat']
            capital = config['capital']
            net_arch = config['net_arch']
            if config['activation_f'] == 'relu':
                learning_rate = 0.001
                policy_kwargs = dict(
                    log_std_init=-2,
                    ortho_init=False,
                    activation_fn=activation_f_map[config['activation_f']],
                    net_arch=[dict(pi=[256, 256, 64], vf=[256, 256, 64])]
                )
            else:
                learning_rate = 0.001
                policy_kwargs = dict()
            activation_f = activation_f_map[config['activation_f']]
            extractor_net_arch = config['extractor_net_arch']
            extractor_net_arch_mlp = config['extractor_net_arch_mlp']
            extractor_activation_f = activation_f_map[config['extractor_activation_f']]
            agent = config['agent']
            curr_time = int(time.time())
            agent_id = f'{agent}-{curr_time}'
            modelsdir = f'models/{agent_id}'
            tb_log_name = f'{agent_id}'
            agent = agent_map[agent]
            TIMESTEPS = config['timesteps_per_checkpoint']
            total_timesteps = config['total_timesteps']
            n_loops = total_timesteps // TIMESTEPS
            logdir = 'logs'
            tensorboard_log = 'tb_logs'
            device = config['device']
            policy = config['policy']

            tb_monitor_kws = tuple(config['tb_monitor'])
            monitor_kws = tuple(config['monitor'])

            os.makedirs(logdir, exist_ok=True)
            os.makedirs(tensorboard_log, exist_ok=True)
            os.makedirs(modelsdir, exist_ok=True)
        except Exception as e:
            print(e.with_traceback())
            print('Invalid config.json file')
            exit()
        print("Read successful")

    env = TrainingEnv(capital=capital, data_dir=data_dir, intervals=intervals, past_steps=past_steps, fee=fee,
                      loss_tolerance=loss_tolerance, loss_penalty_coef=loss_penalty_coef)
    env = Monitor(env, filename=f'{logdir}/{agent_id}', info_keywords=monitor_kws)

    with open(f'{logdir}/{agent_id}.config.json', 'w') as f:
        json.dump(config, f)

    tensorboard_callback = TensorboardCallback(verbose=0, monitor_kws=tb_monitor_kws)

    models = os.listdir(modelsdir)
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

    curr_model_id = 0
    model = PPO(policy, env, verbose=0, device=device, tensorboard_log=tensorboard_log, learning_rate=learning_rate,
                n_steps=256, batch_size=64, policy_kwargs=policy_kwargs)

    for ep in range(1, n_loops+1):
        print(f'Episode {ep}/{n_loops}')
        curr_model_id += TIMESTEPS
        model.learn(total_timesteps=TIMESTEPS, tb_log_name=tb_log_name, reset_num_timesteps=False,
                    callback=tensorboard_callback)
        model.save(f'{modelsdir}/{curr_model_id}')

