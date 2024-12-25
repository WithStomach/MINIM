import torch
import argparse
import yaml
from pathlib import Path
from src.synthetic.synthetic_system import SyntheticSystem
from src.selector.receive_selector import ReceiveSelector
from src.rl.policy import RLPolicy
import logging
from PIL import Image
import os


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_output_dir(output_dir: str, modality: str) -> str:
    full_path = os.path.join(output_dir, modality)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def generate_images(
    prompt: str,
    num_images: int,
    config_path: str,
    policy_checkpoint: str,
    output_dir: str,
    modality: str = "OCT",
):
    setup_logging()
    config = load_config(config_path)
    output_dir = setup_output_dir(output_dir, modality)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    synthetic_system = SyntheticSystem(config)
    selector = ReceiveSelector(config)
    policy = RLPolicy(config)

    if policy_checkpoint and os.path.exists(policy_checkpoint):
        checkpoint = torch.load(policy_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Loaded policy checkpoint from {policy_checkpoint}")

    policy.eval()

    logging.info(f"Generating {num_images} images for prompt: {prompt}")
    images, attempts = synthetic_system.generate_images(
        [prompt] * num_images, selector, policy
    )

    for i, (image, attempt) in enumerate(zip(images, attempts)):
        selector_score = selector.evaluate_image(image)
        policy_score = policy.evaluate_image(image, prompt)

        output_path = os.path.join(
            output_dir,
            f"{modality}_{i+1}_selector_{selector_score:.2f}_policy_{policy_score:.2f}.png",
        )
        image.save(output_path)
        logging.info(f"Saved image {i+1}/{num_images} (attempts: {attempt})")
        logging.info(
            f"Scores - Selector: {selector_score:.2f}, Policy: {policy_score:.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--policy_checkpoint", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--modality",
        type=str,
        default="OCT",
        choices=["OCT", "Chest CT", "Chest X-Ray", "Brain MRI", "Breast MRI", "Fundus"],
    )

    args = parser.parse_args()

    generate_images(
        prompt=args.prompt,
        num_images=args.num_images,
        config_path=args.config,
        policy_checkpoint=args.policy_checkpoint,
        output_dir=args.output_dir,
        modality=args.modality,
    )


if __name__ == "__main__":
    main()
