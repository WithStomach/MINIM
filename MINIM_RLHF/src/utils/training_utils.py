import torch
import logging
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from torch.optim import lr_scheduler

def validate_data_file(config: Dict):
    """Check if the ratings file exists and log the number of entries."""
    data_file = Path(config["data"]["ratings_file"])
    if not data_file.exists():
        raise FileNotFoundError(f"Ratings file not found: {data_file}")
    
    with open(data_file) as f:
        num_lines = sum(1 for line in f) - 1
    logging.info(f"Found {num_lines} ratings in {data_file}")

def setup_wandb(config: Dict) -> str:
    """Initialize Weights & Biases for experiment tracking."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project="medical-rlhf", config=config, name=run_id)
    return run_id

def setup_optimizers_and_schedulers(selector, policy, config: Dict):
    """Set up optimizers and learning rate schedulers for models."""
    selector.optimizer = torch.optim.Adam(
        selector.model.parameters(),
        lr=config["selector"]["training"]["learning_rate"],
        weight_decay=config["selector"]["training"]["optimizer"]["weight_decay"]
    )
    selector.scheduler = lr_scheduler.CosineAnnealingLR(
        selector.optimizer,
        T_max=config["training"]["max_epochs"],
        eta_min=1e-6
    )

def train_epoch(epoch: int, stage: int, selector, policy, synthetic_system, rating_system, config: Dict):
    """Train models for one epoch."""
    logging.info(f"Starting epoch {epoch}")
    
    image_paths, ratings, prompts = rating_system.get_batch_ratings(
        config["data"]["ratings_file"],
        config["training"]["batch_size"]
    )
    
    if not image_paths:
        raise ValueError("No valid ratings found")
    
    ratings = torch.tensor(ratings, dtype=torch.float32).to(policy.device)
    images = torch.stack([
        policy.preprocess_image(path) for path in image_paths
    ]).to(policy.device)
    
    metrics = {
        "epoch": epoch,
        "total_samples": len(image_paths),
    }
    
    if stage == 1:
        policy_loss = policy.update(images, prompts, ratings, stage=1)
        metrics["policy_loss"] = policy_loss
        
        selector_loss = selector.train_step(image_paths, ratings, stage=1)
        metrics["selector_loss"] = selector_loss.item()

        if config["synthetic_system"]["enable_training"]:
            high_quality_indices = [i for i, r in enumerate(ratings.cpu().tolist()) if r > 2]
            if high_quality_indices:
                high_quality_images = images[high_quality_indices]
                high_quality_prompts = [prompts[i] for i in high_quality_indices]
                synthetic_loss = synthetic_system.update(high_quality_images, high_quality_prompts)
                metrics["synthetic_loss"] = synthetic_loss
        else:
            metrics["synthetic_loss"] = None
    
    else:
        selected_paths, selected_ratings, selected_prompts = selector.filter_samples(
            image_paths, 
            ratings.cpu().tolist(),
            prompts,
            threshold=config["training"]["stage2"]["selector_threshold"]
        )
        
        if selected_paths:
            selected_images = torch.stack([
                policy.preprocess_image(path) for path in selected_paths
            ]).to(policy.device)
            
            selected_ratings = torch.tensor(selected_ratings, dtype=torch.float32).to(policy.device)
            
            policy_loss = policy.update(selected_images, selected_prompts, selected_ratings, stage=2)
            metrics["policy_loss"] = policy_loss
            
            selector_loss = selector.train_step(selected_paths, selected_ratings, stage=2)
            metrics["selector_loss"] = selector_loss.item()
            
            if config["synthetic_system"]["enable_training"]:
                high_quality_indices = [i for i, r in enumerate(selected_ratings.cpu().tolist()) if r > 2]
                if high_quality_indices:
                    high_quality_images = selected_images[high_quality_indices]
                    high_quality_prompts = [selected_prompts[i] for i in high_quality_indices]
                    synthetic_loss = synthetic_system.update(high_quality_images, high_quality_prompts)
                    metrics["synthetic_loss"] = synthetic_loss
        else:
            logging.warning("No samples passed selector threshold")
            metrics["policy_loss"] = None
            metrics["selector_loss"] = None
    
    passed_images = count_passed_images(images, selector, config)
    pass_rate = passed_images / len(images)
    metrics.update({
        "passed_samples": passed_images,
        "pass_rate": pass_rate,
        "policy_lr": policy.optimizer.param_groups[0]["lr"],
        "selector_lr": selector.optimizer.param_groups[0]["lr"]
    })
    
    wandb.log({k: v for k, v in metrics.items() if v is not None})
    log_training_progress(metrics)

def count_passed_images(images: torch.Tensor, selector, config: Dict) -> int:
    """Count the number of images passing the selector threshold."""
    with torch.no_grad():
        passed = 0
        for img in images:
            score = selector.model(img.unsqueeze(0))
            if score.item() >= config["training"]["stage2"]["selector_threshold"]:
                passed += 1
    return passed

def log_training_progress(metrics: Dict):
    """Log the training progress."""
    log_msg = [f"Epoch {metrics['epoch']}:"]
    
    if "policy_loss" in metrics and metrics["policy_loss"] is not None:
        log_msg.append(f"Policy Loss = {metrics['policy_loss']:.4f}")
    else:
        log_msg.append("Policy Loss = No Update")
        
    if "selector_loss" in metrics:
        log_msg.append(f"Selector Loss = {metrics['selector_loss']:.4f}")
    
    log_msg.append(
        f"Pass Rate = {metrics['pass_rate']:.2%} "
        f"({metrics['passed_samples']}/{metrics['total_samples']})"
    )
    
    if "synthetic_loss" in metrics:
        log_msg.append(f"Synthetic Loss = {metrics['synthetic_loss']:.4f}")
    
    logging.info(", ".join(log_msg))

def update_schedulers(selector, policy, synthetic_system, config: Dict):
    """Update learning rate schedulers."""
    policy.scheduler.step()
    selector.scheduler.step()
    if config["synthetic_system"]["enable_training"] and synthetic_system.scheduler:
        synthetic_system.scheduler.step()

def save_checkpoints_if_needed(epoch: int, selector, policy, synthetic_system, config: Dict):
    """Save model checkpoints if needed."""
    save_frequency = config["training"]["checkpoints"]["save_frequency"]
    save_dir = Path(config["training"]["checkpoints"]["save_dir"])
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    should_save = (epoch + 1) % save_frequency == 0
    is_final_epoch = epoch + 1 == config["training"]["max_epochs"]
    
    if should_save or is_final_epoch:
        logging.info(f"Saving checkpoints at epoch {epoch + 1}")
        save_checkpoint(policy, "policy", epoch + 1, policy.optimizer, policy.scheduler, save_dir, current_time)
        save_checkpoint(selector.model, "selector", epoch + 1, selector.optimizer, selector.scheduler, save_dir, current_time)
        if config["synthetic_system"]["enable_training"]:
            save_checkpoint(
                synthetic_system.model, "synthetic", epoch + 1,
                synthetic_system.optimizer, synthetic_system.scheduler,
                save_dir, current_time
            )

def load_checkpoints(selector, policy, synthetic_system, config: Dict):
    """Load the latest model checkpoints if configured to do so."""
    if not config["training"].get("load_latest_checkpoint", True):
        logging.info("Skipping loading of latest checkpoints as per configuration.")
        return 0  # Start from epoch 0 if not loading checkpoints

    save_dir = Path(config["training"]["checkpoints"]["save_dir"])
    
    latest_run_id, policy_checkpoint = get_latest_checkpoint(save_dir, "policy")
    _, selector_checkpoint = get_latest_checkpoint(save_dir, "selector")
    _, synthetic_checkpoint = get_latest_checkpoint(save_dir, "synthetic")
    
    start_epoch = 0
    if latest_run_id:
        logging.info(f"Found previous run: {latest_run_id}")
        policy_epoch = load_checkpoint(policy, policy.optimizer, policy.scheduler, policy_checkpoint)
        selector_epoch = load_checkpoint(selector.model, selector.optimizer, selector.scheduler, selector_checkpoint)
        if synthetic_checkpoint and config["synthetic_system"]["enable_training"]:
            synthetic_epoch = load_checkpoint(synthetic_system.model, synthetic_system.optimizer, synthetic_system.scheduler, synthetic_checkpoint)
        start_epoch = max(policy_epoch, selector_epoch) if policy_epoch and selector_epoch else 0
    return start_epoch

def get_latest_checkpoint(base_dir: Path, model_name: str) -> Tuple[str, Path]:
    """Get the latest checkpoint for a given model."""
    if not base_dir.exists():
        return None, None
        
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    
    for run_dir in run_dirs:
        epoch_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
            key=lambda x: int(x.name.split("_")[1]),
            reverse=True
        )
        
        for epoch_dir in epoch_dirs:
            checkpoint_path = epoch_dir / f"{model_name}_checkpoint.pt"
            if checkpoint_path.exists():
                return run_dir.name, checkpoint_path
                
    return None, None

def load_checkpoint(model: torch.nn.Module, optimizer, scheduler, checkpoint_path: Path) -> int:
    """Load a model checkpoint."""
    if not checkpoint_path:
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    
    logging.info(f"Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch

def save_checkpoint(model: torch.nn.Module, name: str, epoch: int, optimizer, scheduler, save_dir: Path, run_id: str):
    """Save a model checkpoint."""
    run_dir = save_dir / run_id
    epoch_dir = run_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    
    save_path = epoch_dir / f"{name}_checkpoint.pt"
    torch.save(checkpoint, save_path)
    logging.info(f"Saved {name} checkpoint to {save_path}")

def get_device(config: Dict) -> torch.device:
    """Determine the device to use for computation."""
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move the model to the specified device."""
    model = model.to(device)
    if device.type == "cuda" and model.config["device"]["precision"] == "float16":
        model = model.half()
    return model