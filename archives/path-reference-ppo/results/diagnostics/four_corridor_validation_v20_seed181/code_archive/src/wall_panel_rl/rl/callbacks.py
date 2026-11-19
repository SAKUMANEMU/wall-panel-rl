from stable_baselines3.common.callbacks import BaseCallback


class RewardTermsCallback(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            for key, value in info.items():
                if key.startswith("reward_"):
                    self.logger.record_mean("components/" + key, value)
        return True
