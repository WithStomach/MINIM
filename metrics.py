import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os

from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim
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
        "--generated_images",
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


def load_inception_model():
    inception = models.inception_v3(pretrained=True)
    inception.eval()
    return inception


def read_images(folder_path, transform=None):
    image_files = [
        f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    images = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path).convert("RGB")
        if transform == None:
            input_tensor = transforms.ToTensor(image)
        else:
            input_tensor = transform(image)
        images.append(input_tensor)
    images = torch.stack(images)
    return images


def calculate_FID(inception, real_images, generated_images):
    real_features = inception(real_images)
    generated_features = inception(generated_images)

    mu_real = torch.mean(real_features, dim=0)
    mu_generated = torch.mean(generated_features, dim=0)

    cov_real = np.cov(real_features.detach().numpy().T)
    cov_generated = np.cov(generated_features.detach().numpy().T)

    diff = mu_real - mu_generated
    diff = diff.detach().numpy()
    sqrt_cov = sqrtm(cov_real.dot(cov_generated))
    fid = np.real(diff.dot(diff) + np.trace(cov_real + cov_generated - 2 * sqrt_cov))
    return np.sqrt(fid)


def calculate_IS(inception, images):
    pred = inception(images)
    pred = F.softmax(pred).cpu().numpy()

    py = np.mean(pred, axis=0)
    scores = []
    for i in range(pred.shape[0]):
        pyx = pred[i, :]
        scores.append(np.log(pyx / py))
    scores = np.exp(np.mean(scores))

    return np.mean(scores)


def calculate_SSIM(real_images, generated_images):
    ssim_values = []

    for real, generated in zip(real_images, generated_images):
        real = real.cpu().numpy()
        generated = generated.cpu().numpy()

        if real.shape != generated.shape:
            raise ValueError("Input images must have the same dimensions.")

        ssim_value = ssim(real, generated, data_range=1.0, channel_axis=0)
        ssim_values.append(ssim_value)

    return np.mean(ssim_values)


def main():
    args = parse_args()
    device = args.device
    real_images = read_images(args.real_images)
    generated_images = read_images(args.generated_images)
    inception_model = load_inception_model()

    fid = calculate_FID(inception_model, real_images, generated_images)
    print(f"FID: {fid}")

    ssim = calculate_SSIM(real_images, generated_images)
    print(f"SSIM: {ssim}")

    inception_score = calculate_IS(inception_model, generated_images)
    print(f"Inception Score: {inception_score}")


if __name__ == "__main__":
    main()
