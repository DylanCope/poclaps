from poclaps.train.ppo import make_train as create_mappo_train_fn
from poclaps.train.training_cb import ChainedCallback
from poclaps.train.wandb_cb import WandbCallback
from poclaps.train.metrics_cb import MetricsLogger
from poclaps.train.ckpt_cb import CheckpointerCallback


def create_ppo_trainer(config: dict) -> callable:

    output_dir = config['OUTPUT_DIR']
    cb = ChainedCallback(
        WandbCallback(tags=[config['ALGORITHM'], "RNN", config["ENV_NAME"]]),
        CheckpointerCallback(
            output_dir / 'checkpoints',
            max_to_keep=config.get('KEEP_CHECKPOINTS', 1),
            save_interval_steps=config.get('CHECKPOINT_INTERVAL', 200),
            only_save_last=config.get('ONLY_SAVE_LAST', True),
        ),
        MetricsLogger(output_dir)
    )

    init_state, train_fn = create_mappo_train_fn(config, cb)

    def run_training():
        cb.on_train_begin(config)
        out = train_fn(init_state)
        cb.on_train_end(out['runner_state'])

    return run_training
