from .training_cb import TrainerCallback

from pathlib import Path
from typing import Any, List
import json

import numpy as np
import orbax.checkpoint as ocp
from flax import struct, core


def get_best_ckpt_dir(ckpts_dir: Path,
                      comparison_method: str = 'max',
                      metric_name: str = 'mean_total_reward'):

    comparator = max if comparison_method == 'max' else min
    default_metric = -np.inf if comparison_method == 'max' else np.inf

    def get_metric(ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        with open(ckpt_dir / 'metrics.json') as metrics_file:
            metrics = json.load(metrics_file)
        return metrics.get(metric_name, default_metric)

    return comparator(ckpts_dir.iterdir(), key=get_metric)


def get_best_ckpt_step(ckpts_dir: Path | str,
                       comparison_method: str = 'max',
                       metric_name: str = 'mean_total_reward') -> int:
    ckpts_dir = Path(ckpts_dir)
    best_ckpt_dir = get_best_ckpt_dir(ckpts_dir, comparison_method, metric_name)
    return int(best_ckpt_dir.name)


def load_best_ckpt(ckpts_dir: Path | str,
                   abstract_pytree: struct.PyTreeNode,
                   comparison_method: str = 'max',
                   metric_name: str = 'mean_total_reward'):
    ckpts_dir = Path(ckpts_dir)
    best_ckpt_dir = get_best_ckpt_dir(ckpts_dir, comparison_method, metric_name)
    best_ckpt_step = int(best_ckpt_dir.name)
    return load_ckpt(ckpts_dir, best_ckpt_step, abstract_pytree)


def load_ckpt(ckpts_dir: Path | str,
              step: int,
              abstract_pytree: struct.PyTreeNode):
    ckpts_dir = Path(ckpts_dir)
    checkpoint_manager = ocp.CheckpointManager(ckpts_dir.absolute())
    return checkpoint_manager.restore(
        step, args=ocp.args.StandardRestore(abstract_pytree)
    )


class CheckpointerCallback(TrainerCallback):

    def __init__(self,
                 ckpts_dir : str | Path,
                 log_metrics: List[str] = None,
                 only_save_last: bool = False,
                 **ckpt_kwargs):
        self.log_metrics = set(log_metrics or []) | {
            'mean_total_reward', 'total_env_steps',
            'training_iteration', 'mean_episode_length'
        }
        self.ckpts_dir = ckpts_dir
        self.only_save_last = only_save_last
        self.checkpoint_manager = ocp.CheckpointManager(
            ckpts_dir,
            options=ocp.CheckpointManagerOptions(**ckpt_kwargs)
        )

    def on_iteration_end(self,
                         iteration: int,
                         training_state: struct.PyTreeNode,
                         metric: core.FrozenDict[str, Any]):
        if self.only_save_last:
            return

        if isinstance(iteration, np.ndarray):
            iteration = int(iteration.item())

        created_ckpt = self.checkpoint_manager.save(
            iteration,
            args=ocp.args.StandardSave(training_state),
            metrics=metric
        )
        self.checkpoint_manager.wait_until_finished()

        if created_ckpt:
            ckpt_dir = self.ckpts_dir / f'{iteration}'
            with open(ckpt_dir / 'metrics.json', 'w') as f:
                json.dump({
                    k: v.tolist() for k, v in metric.items()
                    if k in self.log_metrics
                }, f)

    def on_train_end(self, training_state: struct.PyTreeNode):
        *_, iteration = training_state
    
        if not isinstance(iteration, int):
            iteration = int(iteration.item())

        self.checkpoint_manager.save(
            iteration,
            args=ocp.args.StandardSave(training_state),
            force=True
        )
        self.checkpoint_manager.wait_until_finished()
