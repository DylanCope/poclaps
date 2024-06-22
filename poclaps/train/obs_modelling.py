from poclaps.train.training_cb import TrainerCallback
from poclaps.train.losses import categorical_cross_entropy

from typing import Iterable

import optax
import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class TrainState:
    params: struct.PyTreeNode
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState


class LoggerCallback(TrainerCallback):

    def __init__(self, log_freq: int = 1):
        self.log_freq = log_freq

    def on_iteration_end(self, iteration, training_state, metrics):
        if iteration % self.log_freq == 0:
            print('Iteration:', iteration, '- Loss:', metrics['loss'].item())


class ObsModelTrainer:

    def __init__(self,
                 obs_model,
                 pretrained_policy,
                 callback=None):
        self.obs_model = obs_model
        self.pretrained_policy = pretrained_policy
        self.callback = callback or TrainerCallback()

    def compute_loss(self, variables, model_inps, actions, seq_lens):
        carry = self.obs_model.initialize_carry(model_inps.shape)
        _, obs_preds = self.obs_model.apply(variables, carry, model_inps, seq_lens)
        pred_action_dist, _ = self.pretrained_policy(obs_preds)
        loss = categorical_cross_entropy(pred_action_dist.logits, actions)
        return (loss.sum(axis=-1) / seq_lens).mean()
    
    def train_step(self, train_state: TrainState, data: Iterable):
        losses = []
        for inputs in data:
            loss, grads = jax.value_and_grad(self.compute_loss)(train_state.params, *inputs)
            updates, opt_state = train_state.optimizer.update(grads, train_state.opt_state)
            next_params = optax.apply_updates(train_state.params, updates)
            train_state = train_state.replace(
                opt_state=opt_state,
                params=next_params
            )
            losses.append(loss)
        
        metrics = {'loss': jnp.mean(jnp.stack(losses))}
        return train_state, metrics

    def train(train_state: TrainState,
              n_steps: int,
              data: Iterable,
              loss_fn: callable,
              start_step=0,
              callback: TrainerCallback = None) -> TrainState:
        callback = callback or TrainerCallback()
        for i in range(n_steps):
            callback.on_iteration_end(start_step+i, train_state, metrics)

        return train_state

