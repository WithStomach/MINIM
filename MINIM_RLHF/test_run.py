from src.utils.device_utils import get_device
import logging
import torch
from pathlib import Path
import yaml
from src.synthetic.synthetic_system import SyntheticSystem
from src.selector.receive_selector import ReceiveSelector
from src.rl.policy import RLPolicy

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config() -> dict:
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def test_generation():
    config = load_config()
    setup_logging()
    device = get_device(config)
    
    logging.info(f"Using device: {device}")
    
    synthetic_system = SyntheticSystem(config)
    selector = ReceiveSelector(config)
    policy = RLPolicy(config)
    
    test_prompt = "OCT: healthy retinal layers"
    logging.info(f"Generating image for prompt: {test_prompt}")
    
    try:
        images, attempts = synthetic_system.generate_images([test_prompt], selector, policy)
        
        if images:
            logging.info(f"Successfully generated image in {attempts[0]} attempts")
            
            output_path = Path("outputs/test_generation.png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            images[0].save(output_path)
            
            selector_score = selector.evaluate_image(images[0])
            policy_score = policy.evaluate_image(images[0], test_prompt)
            
            logging.info(f"Generated image saved to {output_path}")
            logging.info(f"Selector score: {selector_score:.2f}")
            logging.info(f"Policy score: {policy_score:.2f}")
            
    except Exception as e:
        logging.error(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_generation()