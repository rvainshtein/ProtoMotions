from pathlib import Path

import lightning as L

from phys_anim.agents.ppo import PPO
from phys_anim.envs.humanoid.common import BaseHumanoid


class SaveBestModelCallback:
    training_loop: PPO
    env: BaseHumanoid

    def __init__(self):
        """
        Callback to save the model when rewards exceed the previous best.
        """
        self.best_reward = float('-inf')

    def after_train(self, training_loop: PPO):
        """
        Check and save the model at the end of each epoch if the reward is higher.

        Args:
            training_loop (PPO): The trainer instance.
        """
        # Assuming `self.experience_buffer.total_rewards` is a tensor
        current_reward = training_loop.experience_buffer.total_rewards.mean().item()

        if current_reward > self.best_reward:
            self.best_reward = current_reward
            print(f"New best reward: {self.best_reward}. Saving model...")

            # Save the model to the specified directory
            ckpt_name = "best_model.ckpt"
            training_loop.save(name=ckpt_name)  #  the path is None, so it will save to the default directory
            print(f"Model saved with reward: {self.best_reward}")