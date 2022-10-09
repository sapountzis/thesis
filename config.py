import json
from torch.nn import ReLU, Tanh
import torch
from feature_extractors import AdaptiveNormalizationExtractor, DainLstmExtractor, CNNFeaturesExtractor
import time
import os


def make_config(config_path):
    activation_f_map = {'relu': ReLU, 'tanh': Tanh}
    features_extractor_class_map = {'adaptive_normalization': AdaptiveNormalizationExtractor,
                                    'dain_lstm': DainLstmExtractor,
                                    'cnn': CNNFeaturesExtractor}

    with open(config_path, "r") as configfile:
        config = json.load(configfile)
        print("Config parsed successfully")

    agent = config['agent']
    curr_time = int(time.time())
    agent_id = f'{agent}-{curr_time}'
    modelsdir = f'models/{agent_id}'
    logdir = config['log_dir']
    tensorboard_log = config['tensorboard_log']

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(modelsdir, exist_ok=True)

    with open(f'{logdir}/{agent_id}.config.json', 'w') as f:
        json.dump(config, f)

    if config['policy_kwargs'].get('activation_fn') is not None:
        config['policy_kwargs']['activation_fn'] = activation_f_map[config['policy_kwargs']['activation_fn']]

    if config['policy_kwargs'].get('features_extractor_class') is not None:
        torch.device(config['agent_kwargs']['device'])
        config['policy_kwargs']['features_extractor_class'] = features_extractor_class_map[config['policy_kwargs']['features_extractor_class']]

    if config['policy_kwargs'].get('features_extractor_kwargs') is not None:
        if config['policy_kwargs']['features_extractor_kwargs'].get('activation_fn') is not None:
            config['policy_kwargs']['features_extractor_kwargs']['activation_fn'] = activation_f_map[config['policy_kwargs']['features_extractor_kwargs']['activation_fn']]

    return config
