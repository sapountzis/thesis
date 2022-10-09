from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from custom_layers import DAIN_Layer, DAIN_LSTM_Layer
import torch as th
from typing import List
import gym
import math


def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class extract_tensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor[:, -1, :]


class DainLstmExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, lstm_units=16):
        super(DainLstmExtractor, self).__init__(observation_space, features_dim=1)

        input_dim = 0
        for key, subspace in observation_space.spaces.items():
            if key != 'portfolio':
                input_dim = subspace.shape[-1]
                break

        self.price_net = [DAIN_LSTM_Layer(input_dim=input_dim), nn.LSTM(input_dim, lstm_units), extract_tensor(),
                          nn.Flatten()]
        self.price_net = nn.Sequential(*self.price_net)
        self.volatility_net = [DAIN_LSTM_Layer(input_dim=input_dim), nn.LSTM(input_dim, lstm_units), extract_tensor(),
                               nn.Flatten()]
        self.volatility_net = nn.Sequential(*self.volatility_net)
        self.volume_net = [DAIN_LSTM_Layer(input_dim=input_dim), nn.LSTM(input_dim, lstm_units), extract_tensor(),
                           nn.Flatten()]
        self.volume_net = nn.Sequential(*self.volume_net)

        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if 'price' in key:
                extractors[key] = self.price_net
                total_concat_size += lstm_units
            if 'volatility' in key:
                extractors[key] = self.volatility_net
                total_concat_size += lstm_units
            if 'volume' in key:
                extractors[key] = self.volume_net
                total_concat_size += lstm_units
            if key == 'portfolio':
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:

        encoded_tensor_list = [extractor(observations[key]) for key, extractor in self.extractors.items()]
        return th.cat(encoded_tensor_list, dim=1)


class AdaptiveNormalizationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(AdaptiveNormalizationExtractor, self).__init__(observation_space, features_dim=1)

        input_dim = 0
        for key, subspace in observation_space.spaces.items():
            if key != 'portfolio':
                input_dim = subspace.shape[-1]
                break

        self.price_dain = nn.Sequential(*[DAIN_Layer(input_dim=input_dim), nn.Flatten()])
        self.volatility_dain = nn.Sequential(*[DAIN_Layer(input_dim=input_dim), nn.Flatten()])
        self.volume_dain = nn.Sequential(*[DAIN_Layer(input_dim=input_dim), nn.Flatten()])

        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if 'price' in key:
                total_concat_size += subspace.shape[1] * subspace.shape[2]
            if 'volatility' in key:
                total_concat_size += subspace.shape[1] * subspace.shape[2]
            if 'volume' in key:
                total_concat_size += subspace.shape[1] * subspace.shape[2]
            if key == 'portfolio':
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:

        encoded_tensor_list = [extractor(observations[key]) for key, extractor in self.extractors.items()]
        return th.cat(encoded_tensor_list, dim=1)


class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, channels=2, kernel_size=5, activation_fn=nn.ReLU,
                 dense_layers=None, dense_units=256):

        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim=1)

        non_ts_features = ['portfolio_alloc', 'ath_ratio', 'atl_ratio']
        ts_features = {'pricesPctChg', 'volatility', 'openclose', 'volumes'}

        price_obs_shape = (0, 0)
        for key, space in observation_space.spaces.items():
            if key not in non_ts_features:
                price_obs_shape = space.shape
                break

        # cnn extractors
        out_len = 0
        self.ts_extractors = {}
        for ts_feature in ts_features:
            self.ts_extractors[ts_feature], out_len = self.make_cnn_extractor(kernel_size, channels, activation_fn,
                                                                              price_obs_shape[1], price_obs_shape[2])
        # out_len = price_obs_shape[1]
        # ts_extractor = []
        # in_channels = 1
        # while out_len > kernel_size:
        #     ts_extractor.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(kernel_size, 1)))
        #     ts_extractor.append(activation_fn())
        #     in_channels = channels
        #     out_len = calculate_output_length(out_len, kernel_size)
        #     if price_obs_shape[1] % out_len == 0:
        #         ts_extractor.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        #         out_len = calculate_output_length(out_len, kernel_size=2, stride=2)
        # ts_extractor.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(out_len, 1)))
        # ts_extractor.append(activation_fn())
        # ts_extractor.append(nn.Flatten())
        # out_len = price_obs_shape[2] * channels
        #
        # self.ts_extractor = nn.Sequential(*ts_extractor)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key not in non_ts_features:
                for extractor_key in self.ts_extractors.keys():
                    if extractor_key in key:
                        extractors[key] = self.ts_extractors[extractor_key]
                        total_concat_size += out_len
            else:
                self.n_coins = subspace.shape[0]
                extractors[key] = nn.Flatten()
                total_concat_size += self.n_coins

        self.dense_layers = dense_layers
        self.dense_units = dense_units
        if self.dense_layers is not None:
            concat_extractor = []
            for i in range(dense_layers):
                concat_extractor.append(nn.Linear(total_concat_size, dense_units)
                                        if i == 0 else nn.Linear(dense_units, dense_units))
                concat_extractor.append(activation_fn())
            concat_extractor = nn.Sequential(*concat_extractor)
            extractors.update({'concat_extractor': concat_extractor})

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size if self.dense_layers is None else self.dense_units

    def make_cnn_extractor(self, kernel_size, channels, activation_fn, timesteps, coins):
        out_len = timesteps
        ts_extractor = []
        in_channels = 1
        while out_len > kernel_size:
            ts_extractor.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(kernel_size, 1)))
            ts_extractor.append(activation_fn())
            in_channels = channels
            out_len = calculate_output_length(out_len, kernel_size)
            if timesteps % out_len == 0:
                ts_extractor.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                out_len = calculate_output_length(out_len, kernel_size=2, stride=2)
        ts_extractor.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(out_len, 1)))
        ts_extractor.append(activation_fn())
        ts_extractor.append(nn.Flatten())
        out_len = coins * channels

        return nn.Sequential(*ts_extractor), out_len

    def forward(self, observations: dict) -> th.Tensor:
        encoded_tensor_list = th.cat([self.extractors[key](obs) for key, obs in observations.items()], dim=1)
        # Concatenate all the tensors
        # encoded_tensor = th.cat(encoded_tensor_list, dim=1)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        if self.dense_layers is not None:
            return self.extractors['concat_extractor'](encoded_tensor_list)

        return encoded_tensor_list


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, ts_extractor_arch,
                 ts_extractor_arch_mlp, activation_f=nn.ReLU):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        price_obs_shape = (0, 0)
        for key, space in observation_space.spaces.items():
            if key != 'portfolio':
                price_obs_shape = space.shape
                break

        # cnn extractor
        out_len = price_obs_shape[1]
        ts_extractor = []
        for idx, channels in enumerate(ts_extractor_arch['channels']):
            # kernel_size = 7 - 2 * (idx // 2)
            ts_extractor.append(nn.Conv2d(in_channels=ts_extractor_arch['channels'][idx - 1] if idx > 0 else 1,
                                          out_channels=channels,
                                          kernel_size=ts_extractor_arch['kernel_sizes'][idx]))
            ts_extractor.append(activation_f())
            out_len = calculate_output_length(out_len, ts_extractor_arch['kernel_sizes'][idx][0])
            if idx % 3 == 2:
                ts_extractor.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                out_len = calculate_output_length(out_len, kernel_size=2, stride=2)
        ts_extractor.append(nn.Flatten())
        out_len = out_len * ts_extractor_arch['channels'][-1]

        # for out_dim in ts_extractor_arch_mlp:
        #     ts_extractor.append(nn.Linear(in_dim, out_dim))
        #     ts_extractor.append(activation_f())
        #     in_dim = out_dim
        # out_len = in_dim
        # ts_extractor = [nn.LSTM(input_size=price_obs_shape[1], hidden_size=ts_extractor_arch[-1], batch_first=True)]
        # out_len = ts_extractor_arch[-1]

        self.ts_extractor = nn.Sequential(*ts_extractor)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key != "portfolio":
                extractors[key] = self.ts_extractor
                total_concat_size += out_len
            else:
                self.n_coins = subspace.shape[0]
                extractors[key] = nn.Flatten()
                total_concat_size += self.n_coins

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:

        # self.extractors contain nn.Modules that do all the processing.
        # for key, extractor in self.extractors.items():
        #     print(observations[key].shape)
        #     tmp = extractor(observations[key])
        #     print(tmp[0].shape)
        encoded_tensor_list = [extractor(observations[key]) for key, extractor in self.extractors.items()]

        # Concatenate all the tensors
        # encoded_tensor = th.cat(encoded_tensor_list, dim=1)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
