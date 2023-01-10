import os
from sb3_contrib import RecurrentPPO
import json
from config import make_config, make_dirs
from envs import TradingEnv
from loading_utils import load_train_df

if __name__ == '__main__':

    config = make_config('config.json')

    episode_timesteps = config['env_kwargs']['episode_length']

    df = load_train_df(config['data_dir'], intervals=config['intervals'], coins=config['coins'],
                       fiat=config['fiat'], index_col='date',
                       end_date=config['train_end'], start_date=config['train_start'])

    # btc_hold = df['15min']['btcusdt'].iloc[:episode_timesteps]['open']
    # btc_hold = btc_hold.iloc[-1] / btc_hold.iloc[0]
    # print(f'BTC hold return: {btc_hold}')
    # plt.show()

    learning_rate = 1e-4
    clip_range = 0.1

    ep_start = 1
    models = sorted(os.listdir('models'))
    if config['continue'] and len(models) > 0 and False:
        prev_agent_id = models[-1]
        if '_cont' in prev_agent_id:
            agent_id = prev_agent_id.split('_cont')[0]
            cont = int(prev_agent_id.split('_cont')[1])
            agent_id += f'_cont{cont + 1}'
        else:
            agent_id = prev_agent_id + '_cont1'
        checkpoints = sorted([int(chkpnt.removesuffix('.zip')) for chkpnt in os.listdir(f'models/{prev_agent_id}')])
        curr_model_id = checkpoints[-1]
        print(f'Loading model {agent_id}, checkpoint {curr_model_id}')
        eps = os.listdir(f'model_logs/{prev_agent_id}')
        eps = sorted([int(ep.split('_')[1]) for ep in eps if ep.endswith('trades.csv')])
        ep_start = eps[-1] + 1
        env = TradingEnv(df, capital=config['env_kwargs']['capital'], ep_len=config['env_kwargs']['episode_length'],
                         fee=config['env_kwargs']['fee'], env_id=agent_id)
        env.episode = ep_start - 1
        model = RecurrentPPO.load(f'models/{prev_agent_id}/{curr_model_id}.zip', env=env)
    else:
        print('creating new model')
        make_dirs(config)
        agent_id = config['agent_id']
        tensorboard_log = config['tensorboard_log']
        curr_model_id = config['checkpoint_timesteps']
        env = TradingEnv(df, capital=config['env_kwargs']['capital'], ep_len=config['env_kwargs']['episode_length'],
                         fee=config['env_kwargs']['fee'], env_id=agent_id, log_freq=config['log_freq'])
        # model = PPO('MultiInputPolicy', env, verbose=0, tensorboard_log=tensorboard_log)
        model = RecurrentPPO('MultiInputLstmPolicy', env, verbose=1, tensorboard_log=tensorboard_log,
                             **config['agent_kwargs'], policy_kwargs=config['policy_kwargs'])
        # model = RecurrentPPO('MultiInputLstmPolicy', env, verbose=1, device='cpu',
        #                      tensorboard_log=tensorboard_log,
        #                      learning_rate=config['agent_kwargs']['learning_rate'],
        #                      batch_size=config['agent_kwargs']['batch_size'],
        #                      n_steps=config['agent_kwargs']['n_steps'],
        #                      clip_range=config['agent_kwargs']['clip_range'],
        #                      n_epochs=config['agent_kwargs']['n_epochs'],
        #                      policy_kwargs={'net_arch': config['policy_kwargs']['net_arch'],
        #                                     'n_lstm_layers': config['policy_kwargs']['n_lstm_layers'],
        #                                     'lstm_hidden_size': config['policy_kwargs']['lstm_hidden_size']})

    tb_log_name = f'{agent_id}'
    modelsdir = f'models/{agent_id}'

    # dump config to json
    with open(f'logs/{config["agent_id"]}.config.json', 'w') as f:
        json.dump(config, f)

    n_loops = config['total_timesteps'] // config['checkpoint_timesteps']
    for ep in range(1, n_loops + 1):
        print(f'Episode {ep}/{n_loops}')

        curr_model_id += config['checkpoint_timesteps']
        model.learn(total_timesteps=config['checkpoint_timesteps'], reset_num_timesteps=False, tb_log_name=tb_log_name)

        model.save(f'{modelsdir}/{curr_model_id}')
