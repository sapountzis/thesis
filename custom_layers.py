import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAIN_LSTM_Layer(nn.Module):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_LSTM_Layer, self).__init__()

        self.first_run = True
        self.input_dim = input_dim
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        if self.first_run:
            with torch.no_grad():
                self.scaling_layer.weight.data = torch.eye(self.input_dim)
                self.mean_layer.weight.data = torch.eye(self.input_dim)
            self.first_run = False

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.reshape(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 1)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), 1, adaptive_avg.size(1))
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 1)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.reshape(adaptive_std.size(0), 1, adaptive_std.size(1))
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, 1)
            gate = torch.sigmoid(self.gating_layer(avg))
            gate = gate.reshape(gate.size(0), 1, gate.size(1))
            x = x * gate

        else:
            assert False

        return x


# Deep Adaptive Input Normalization
class DAIN_Layer(nn.Module):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()

        self.first_run = True
        self.mode = mode
        self.input_dim = input_dim
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr
        # self.mask = torch.eye(input_dim)

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        # self.mean_layer = prune.custom_from_mask(self.mean_layer, name='weight', mask=self.mask)

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        # self.scaling_layer = prune.custom_from_mask(self.scaling_layer, name='weight', mask=self.mask)

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        if self.first_run:
            with torch.no_grad():
                self.scaling_layer.weight.data = torch.eye(self.input_dim)
                self.mean_layer.weight.data = torch.eye(self.input_dim)
            self.first_run = False

        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Nothing to normalize
        if self.mode is None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1, avg.size(2))
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            x = torch.log(x)
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1, adaptive_avg.size(2))
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1, adaptive_avg.size(2))
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1, adaptive_std.size(2))
            x = (x / adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1, adaptive_avg.size(2))
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1, adaptive_std.size(2))
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, 2)
            gate = torch.sigmoid(self.gating_layer(avg))
            gate = gate.reshape(gate.size(0), gate.size(1), 1, gate.size(2))
            x = x * gate

        else:
            assert False

        return x

