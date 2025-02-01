import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image
import logging
from tqdm import tqdm
import json
from datetime import datetime
from CDAR import CDAR
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=4, patch_size=192, augment=True):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.image_files = [f for f in os.listdir(
            hr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.image_files[idx])
        lr_path = os.path.join(
            self.lr_dir, self.image_files[idx].replace('.png', 'x4.png'))

        hr_img = Image.open(hr_path).convert('RGB')
        lr_img = Image.open(lr_path).convert('RGB')

        if self.augment:
            lr_patch_size = self.patch_size // self.scale
            lr_w = random.randint(0, lr_img.width - lr_patch_size)
            lr_h = random.randint(0, lr_img.height - lr_patch_size)
            hr_w = lr_w * self.scale
            hr_h = lr_h * self.scale

            lr_img = lr_img.crop(
                (lr_w, lr_h, lr_w + lr_patch_size, lr_h + lr_patch_size))
            hr_img = hr_img.crop(
                (hr_w, hr_h, hr_w + self.patch_size, hr_h + self.patch_size))

            if random.random() < 0.5:
                lr_img = TF.hflip(lr_img)
                hr_img = TF.hflip(hr_img)
            if random.random() < 0.5:
                lr_img = TF.vflip(lr_img)
                hr_img = TF.vflip(hr_img)

        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)

        return lr_tensor, hr_tensor


class MetricLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'val_psnr': [],
            'learning_rate': []
        }

    def update(self, metrics_dict):
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)

    def save_metrics(self):
        metrics_path = os.path.join(self.log_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)


def setup_logging(save_dir):
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'training.log'))
    file_handler.setFormatter(log_format)

    logger = logging.getLogger('CSRT')
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def save_checkpoint(model, optimizer, epoch, best_psnr, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_psnr': best_psnr
    }, save_path)

def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    # Added small epsilon to prevent log(0)
    return -10 * torch.log10(mse + 1e-8)


def train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
    model.train()
    model = model.to(device)
    criterion = criterion.to(device)
    total_loss = 0
    pbar = tqdm(train_loader, desc='Training')

    for lr_imgs, hr_imgs in pbar:
        # Move data to GPU asynchronously
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        hr_imgs = hr_imgs.to(device, non_blocking=True)

        optimizer.zero_grad()
        sr_imgs = model(lr_imgs)
        loss = criterion(sr_imgs, hr_imgs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Clear cache periodically
        if torch.xpu.is_available():
            if torch.xpu.memory_allocated() > 0.8 * torch.xpu.get_device_properties(device).total_memory:
                torch.xpu.empty_cache()
        elif torch.cuda.is_available():
            if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(device).total_memory:
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    logger.info(f'Training - Average Loss: {avg_loss:.4f}')
    return avg_loss


def validate(model, val_loader, criterion, device, logger):
    model.eval()
    model = model.to(device)
    total_psnr = 0
    total_loss = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(val_loader, desc='Validating'):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            psnr = calculate_psnr(sr_imgs, hr_imgs)

            total_psnr += psnr.item()
            total_loss += loss.item()

            if torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    logger.info(
        f'Validation - Average Loss: {avg_loss:.4f}, Average PSNR: {avg_psnr:.2f}')
    return avg_loss, avg_psnr


def main():
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 500,
        'scale': 4,
        'patch_size': 192,
        'channels': 96,
        'num_cascade_units': 8,
        'lr_init': 1e-4,
        'lr_final': 1e-5,
        'betas': [0.9, 0.99],
        'adam_eps': 1e-7,
        'weight_decay': 1e-8,
        'div2k_hr_train': './datasets/DIV2K/DIV2K_train_HR',
        'div2k_lr_train': './datasets/DIV2K/DIV2K_train_LR_bicubic/X4',
        'div2k_hr_valid': './datasets/DIV2K/DIV2K_valid_HR',
        'div2k_lr_valid': './datasets/DIV2K/DIV2K_valid_LR_bicubic/X4'
    }

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('./experiments', f'CSRT_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Setup logging and metrics tracking
    logger = setup_logging(save_dir)
    metric_logger = MetricLogger(save_dir)

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.xpu.is_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    # Create datasets and dataloaders
    train_dataset = DIV2KDataset(
        config['div2k_hr_train'],
        config['div2k_lr_train'],
        scale=config['scale'],
        patch_size=config['patch_size']
    )

    val_dataset = DIV2KDataset(
        config['div2k_hr_valid'],
        config['div2k_lr_valid'],
        scale=config['scale'],
        patch_size=config['patch_size'],
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')

    # Initialize model, criterion, optimizer
    model = CDAR(
        channels=config['channels'],
        num_cascade_units=config['num_cascade_units'],
        scale=config['scale']
    ).to(device)

    criterions = {
        'L1': torch.nn.L1Loss().to(device),
        'L2': torch.nn.MSELoss().to(device)
    }
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['lr_init'],
                            betas=config['betas'],
                            eps=config['adam_eps'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['lr_final']
    )

    # Training loop
    best_psnr = 0
    for epoch in range(config['num_epochs']):
        if epoch < epoch // 2:
            criterion = criterions['L1']
        else:
            criterion = criterions['L2']
        logger.info(f"Starting epoch {epoch+1}/{config['num_epochs']}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, logger)

        # Validate
        val_loss, val_psnr = validate(
            model, val_loader, criterion, device, logger)

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_psnr)

        # Log metrics
        metric_logger.update({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'learning_rate': current_lr
        })
        metric_logger.save_metrics()

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(
                model, optimizer, epoch, best_psnr,
                os.path.join(save_dir, 'best_model.pth')
            )
            logger.info(f'New best model saved with PSNR: {best_psnr:.2f}')

        # Save latest model
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, best_psnr,
                os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            )
            logger.info(f'Checkpoint saved for epoch {epoch+1}')


if __name__ == '__main__':
    main()
