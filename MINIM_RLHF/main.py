import yaml
import torch
from pathlib import Path
from typing import Dict, Tuple, List
from src.synthetic.synthetic_system import SyntheticSystem
from src.selector.receive_selector import ReceiveSelector
from src.rl.policy import RLPolicy
from src.raters.rating_system import RatingSystem
import logging
import wandb
from datetime import datetime
from src.utils.training_utils import (
    validate_data_file, setup_wandb, setup_optimizers_and_schedulers,
    train_epoch, update_schedulers, save_checkpoints_if_needed,
    load_checkpoints
)

def setup_logging(config: Dict):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

def train(config: Dict):
    validate_data_file(config)
    run_id = setup_wandb(config)
    
    selector = ReceiveSelector(config)
    policy = RLPolicy(config)
    synthetic_system = SyntheticSystem(config)
    rating_system = RatingSystem(config)
    
    setup_optimizers_and_schedulers(selector, policy, config)
    start_epoch = load_checkpoints(selector, policy, synthetic_system, config)
    
    stage = config["training"]["stage"]
    logging.info(f"Training in stage {stage}")
    
    for epoch in range(start_epoch, config["training"]["max_epochs"]):
        try:
            train_epoch(
                epoch=epoch,
                stage=stage,
                selector=selector,
                policy=policy,
                synthetic_system=synthetic_system,
                rating_system=rating_system,
                config=config
            )
            
            update_schedulers(selector, policy, synthetic_system, config)
            save_checkpoints_if_needed(epoch, selector, policy, synthetic_system, config)
            
        except Exception as e:
            logging.error(f"Error in epoch {epoch}: {str(e)}")
            continue

def main():
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    setup_logging(config)
    train(config)

if __name__ == "__main__":
    main()