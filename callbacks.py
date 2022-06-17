from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, monitor_kws=()):
        super(TensorboardCallback, self).__init__(verbose)
        self.monitor_kws = monitor_kws

    def _on_step(self) -> bool:
        for monitor_kw in self.monitor_kws:
            if monitor_kw == 'holdings_pc':
                holdings = self.training_env.get_attr(monitor_kw)[0]
                for k, v in holdings.items():
                    self.logger.record(f'holdings_pc/{k}', v)
            else:
                self.logger.record(monitor_kw, self.training_env.get_attr(monitor_kw)[0])

        return True
