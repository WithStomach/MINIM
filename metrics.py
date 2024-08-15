import torch
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
import numpy as np

from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from PIL import Image
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real_images",
        type=str,
        default=None,
        required=True,
        help="A csv file contains all real images' path.",
    )
    parser.add_argument(
        "--sync_images",
        type=str,
        default=None,
        required=True,
        help="A csv file contains all fake images' path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    args = parser.parse_args()
    return args


def convert_to_rgb(image):
    # 如果图像是灰度图像，将其转换为RGB图像
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = TF.resize(image, (299, 299))  # Inception模型输入大小
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image


def get_inception_activations(images, batch_size=32, dims=2048, device="cuda:0"):
    model = inception_v3(pretrained=True, transform_input=False).eval()
    dataloader = DataLoader(ImageDataset(images), batch_size=batch_size, shuffle=False)

    activations = np.zeros((len(images), dims))

    start_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            pred = model(batch)[0] if type(model(batch)) == tuple else model(batch)
            pred = F.adaptive_avg_pool2d(pred, (1, 1))
            activations[start_idx : start_idx + pred.size(0)] = (
                pred.cpu().numpy().reshape(pred.size(0), -1)
            )
            start_idx += pred.size(0)

    return activations


def calculate_FID(real_images, fake_images, device="cuda:0"):
    act1 = get_inception_activations(real_images, device=device)
    act2 = get_inception_activations(fake_images, device=device)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_IS(images, splits=10, device="cuda:0"):
    preds = get_inception_activations(images, dims=1000, device=device)

    split_scores = []
    for k in range(splits):
        part = preds[
            k * (preds.shape[0] // splits) : (k + 1) * (preds.shape[0] // splits), :
        ]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(np.exp(np.sum(pyx * np.log(pyx / py))))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_SSIM(real_images, fake_images, device="cuda:0"):
    real_dataset = ImageDataset(real_images)
    fake_dataset = ImageDataset(fake_images)

    real_loader = DataLoader(real_dataset, batch_size=1, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=1, shuffle=False)

    ssim_values = []

    for real, fake in zip(real_loader, fake_loader):
        real = real.squeeze().numpy()
        fake = fake.squeeze().numpy()

        if real.shape != fake.shape:
            raise ValueError("Input images must have the same dimensions.")

        ssim_value = ssim(real, fake, multichannel=True)
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)


def main():
    args = parse_args()
    device = args.device
    real_images_data = pd.read_csv(args.real_images)
    fake_images_data = pd.read_csv(args.sync_images)

    real_images = []  # 真实图片列表
    fake_images = []  # 生成图片列表

    for path in real_images_data["path"]:
        img = Image.open(path)
        img = convert_to_rgb(img)
        real_images.append(img)

    for path in fake_images_data["path"]:
        img = Image.open(path)
        img = convert_to_rgb(img)
        fake_images.append(img)

    fid = calculate_FID(real_images, fake_images, device=device)
    print(f"FID: {fid}")

    ssim = calculate_SSIM(real_images, fake_images, device=device)
    print(f"SSIM: {ssim}")

    is_mean, is_std = calculate_IS(fake_images, device=device)
    print(f"Inception Score: {is_mean} ± {is_std}")


if __name__ == "__main__":
    main()
