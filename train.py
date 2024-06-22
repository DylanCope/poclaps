from poclaps.train import create_ppo_trainer
from poclaps.utils.hydra_utils import get_current_hydra_output_dir

import json
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None,
            config_path="config",
            config_name="simple_gridworld_ppo")
def main(config):
    config = OmegaConf.to_container(config)
    print('Config:\n', json.dumps(config, indent=4))

    config['output_dir'] = get_current_hydra_output_dir()

    algorithm = config.get('algorithm', 'PPO')
    if algorithm == 'PPO':
        train = create_ppo_trainer(config)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

    train()


if __name__=="__main__":
    main()