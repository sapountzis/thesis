from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from customLayers import DAIN_Layer
import torch as th
from typing import List
import gym
import math

th.backends.cuda.matmul.allow_tf32 = True


def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


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
                extractors[key] = self.price_dain
                total_concat_size += subspace.shape[1] * subspace.shape[2]
            if 'volatility' in key:
                extractors[key] = self.volatility_dain
                total_concat_size += subspace.shape[1] * subspace.shape[2]
            if 'volume' in key:
                extractors[key] = self.volume_dain
                total_concat_size += subspace.shape[1] * subspace.shape[2]
            if key == 'portfolio':
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:

        encoded_tensor_list = [extractor(observations[key]) for key, extractor in self.extractors.items()]
        return th.cat(encoded_tensor_list, dim=1)



class CNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, coins, activation_f=nn.ReLU):

        super(CNNFeaturesExtractor, self).__init__(observation_space, features_dim=1)

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
