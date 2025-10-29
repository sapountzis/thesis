# Cryptocurrency Portfolio Optimization with Recurrent Proximal Policy Optimization

A reinforcement learning framework for optimizing cryptocurrency portfolios using Recurrent Proximal Policy Optimization (RPPO). This project implements a deep RL agent that learns to allocate capital across multiple cryptocurrencies using historical market data.

## Overview

This thesis project explores the application of recurrent reinforcement learning to cryptocurrency portfolio optimization. The agent learns optimal trading strategies by interacting with a simulated trading environment, using LSTM networks to capture temporal dependencies in market data.

## Key Features

- **Multi-Asset Portfolio Management**: Simultaneously trades across 9 cryptocurrencies (BTC, ETH, ADA, LTC, VET, XRP, AVAX, DOT, SOL)
- **Recurrent Architecture**: Uses LSTM layers to maintain memory of market dynamics
- **Multi-Timeframe Features**: Processes market data from multiple time intervals
- **Custom Feature Extractors**: Implements DAIN (Deep Adaptive Input Normalization) and CNN-based feature extraction
- **Realistic Trading Simulation**: Includes transaction fees, portfolio constraints, and realistic market conditions
- **Comprehensive Evaluation**: Compares performance against buy-and-hold baselines using Sharpe and Sortino ratios

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

- `torch` - PyTorch for neural network operations
- `stable-baselines3` - RL algorithms implementation
- `sb3_contrib` - Additional RL algorithms including RecurrentPPO
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` & `seaborn` - Visualization
- `pyarrow` - Feather file format support
- `tensorboard` - Training monitoring

## Data Setup

### Downloading Binance Data

The project uses Binance historical data in Feather format. To download data:

1. Update `binance_dump.py` with your desired date range and trading pairs
2. Run the script to download data:

```bash
python binance_dump.py
```

Data will be saved in `binance_data/all/` directory.

### Data Format

Expected data structure:
- Files: `{COIN}{FIAT}.feather` (e.g., `btcusdt.feather`)
- Columns: `date`, `open`, `high`, `low`, `close`, `volume`

## Configuration

Edit `config.json` to customize training parameters:

```json
{
  "agent": "RecurrentPPO",
  "checkpoint_timesteps": 100000,
  "total_timesteps": 10000000,
  "coins": ["BTC", "ETH", "ADA", "LTC", "VET", "XRP", "AVAX", "DOT", "SOL"],
  "intervals": ["4H"],
  "train_start": "1-1-2021",
  "train_end": "1-8-2022",
  "env_kwargs": {
    "capital": 1000,
    "episode_length": 1024,
    "fee": 0.00022
  },
  "agent_kwargs": {
    "learning_rate": 0.0003,
    "batch_size": 512,
    "n_epochs": 7,
    "clip_range": 0.1
  }
}
```

## Usage

### Training

Train a new model:

```bash
python train.py
```

The training script will:
- Load and preprocess cryptocurrency data
- Create a new model with unique ID
- Train the agent in episodes
- Save checkpoints periodically
- Log metrics to TensorBoard

To continue training from a checkpoint, set `"continue": true` in `config.json`.

### Evaluation

Test a trained model:

```bash
python test_model.py
```

This will:
- Load a trained model
- Evaluate on test data
- Generate performance visualizations
- Compare against buy-and-hold baseline

### Visualization

Analyze training results:

```bash
python visualize.py
```

Generates visualizations for:
- Portfolio allocation correlation
- Price correlation
- Trade rate over episodes
- Sharpe and Sortino ratios
- Model vs buy-and-hold performance

## Project Structure

```
.
├── train.py                  # Main training script
├── test_model.py            # Model evaluation
├── visualize.py             # Performance visualization
├── config.py                # Configuration utilities
├── config.json              # Training configuration
├── envs.py                  # Trading environment (Gym)
├── feature_extractors.py    # Custom neural network architectures
├── custom_layers.py         # DAIN normalization layers
├── loading_utils.py         # Data loading utilities
├── preprocessing_utils.py   # Feature engineering
├── callbacks.py             # TensorBoard callbacks
├── binance_dump.py          # Data download script
├── create_dataset.py        # Dataset creation utilities
└── requirements.txt         # Python dependencies
```

## Environment Details

### Observation Space

The agent observes:
- **Multi-timeframe features**: Price percentage changes, volume, volatility for each interval
- **Portfolio allocation**: Current distribution of capital across assets
- **Reward**: Previous step's reward signal

### Action Space

Continuous actions:
- **Portfolio allocation** (softmax): Distribution of capital across cryptocurrencies + USDT
- **Trade threshold**: Binary decision on whether to execute trades (threshold >= 0.5)

### Reward Function

Reward is calculated as the percentage change in portfolio value normalized by initial capital:

```
reward = (portfolio_value_t - portfolio_value_{t-1}) / initial_capital
```

This encourages the agent to maximize returns while penalizing losses.

## Model Architecture

### Feature Extractors

1. **AdaptiveNormalizationExtractor**: Uses DAIN layers for adaptive normalization
2. **DainLstmExtractor**: Combines DAIN normalization with LSTM processing
3. **CNNFeaturesExtractor**: Convolutional neural network for feature extraction

### Policy Network

- **Base**: MultiInputLstmPolicy from stable-baselines3
- **Architecture**: Configurable MLP layers (default: 3 layers of 64 units)
- **LSTM**: 2 layers with 64 hidden units
- **Activation**: ReLU or Tanh (configurable)

## Monitoring

Training metrics are logged to TensorBoard:

```bash
tensorboard --logdir tb_logs
```

Monitored metrics include:
- Episode returns
- Trade rate
- Portfolio value changes
- Portfolio allocations
- Loss functions

## Results

Model checkpoints and logs are saved in:
- `models/{agent_id}/` - Trained model checkpoints
- `model_logs/{agent_id}/` - Episode logs and trade records
- `logs/{agent_id}.config.json` - Configuration snapshot
- `tb_logs/` - TensorBoard logs

## Citation

If you use this code in your research, please cite:

```
Cryptocurrency Portfolio Optimization with Recurrent Proximal Policy Optimization
[Your Name]
[Your Institution]
[Year]
```

## License

[Specify your license here]

## Acknowledgments

- Built on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- Uses DAIN (Deep Adaptive Input Normalization) for feature normalization
- Data from Binance cryptocurrency exchange

