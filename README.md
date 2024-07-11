# Our Name

## Introduction

This project addresses the challenge of scarce high-quality medical imaging datasets by exploring a domain adaptation strategy for generative models in the medical field. Given the complexity of medical images, the intricacies of medical language text, and the private nature of medical data, traditional vision-language models are not readily applicable. Our study introduces a latent diffusion-based generative model designed to produce synthetic multi-modal images, guided by report prompts. Rigorous objective and subjective assessments have been conducted to ensure the quality and potential utility of these synthetic images in ophthalmic diagnostics.

## Clinical Applications

- **Diagnostic Assistance**: The generated images, once vetted by the filtering model, are used to assist in diagnostic processes, providing additional insights to medical professionals.
- **Data Augmentation**: High-quality synthetic images also serve as valuable data augmentation for training other diagnostic models, improving their generalizability and performance.

![clinical](./pic/clinical.png)

## Framework

- **Diffusion Model**: We started with a pretrained diffusion model [stable-diffusion-v1-4](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md) and fine-tune it on our dataset containing multi-modal medical images.
- **Reinforcement Learning**: After training , our model generates a variety of medical images across different modalities.These images are then presented to medical professionals who assess the congruence between the generated images and their corresponding prompts. Utilizing the scoring data from the physicians, we train a filtering model that learns to distinguish between high-quality and low-quality image generations.

![framework](./pic/framework.png)

## Installation

Our framework is mainly based on [diffusers](https://github.com/huggingface/diffusers.git),
Follow these steps to install and run the project:

1. Clone the repository and navigate to the project directory

    ```bash
    git clone {repo_name}
    cd {repo_name}
    ```

2. Install diffusers

    ```bash
    pip install git+https://github.com/huggingface/diffusers.git
    pip install -U -r requirements.txt
    ```

3. Initialize an Accelerate environment

    ```bash
    accelerate config
    ```

For more detailed installation instructions, refer to <https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image>

## Data Format

The input data should be a single csv file containing two columns: `path` and `Text`, where `path` represents the path to an image and `Text` is the description to it.

```csv
path, Text
image/1.jpg, breast MRI with tumor
image/2.jpg, chest CT without tumor
```

## Usage

### Train

To train the model, follow these steps:

1. Ensure you have the correctly formatted data and a pretrained diffusion model.

2. Navigate to the following directory

    ```bash
    cd examples/text_to_image
    ```

3. Edit the first few lines of `train.sh`

    ```bash
    export MODEL_NAME="path/to/pretrained_model"
    export DATASET_NAME="path/to/data.csv"
    ```

4. Run the training script

    ```bash
    bash train.sh
    ```

This will execute the `train.sh` script, which contains all the necessary commands to start the training process. And the checkpoints will be saved in `./checkpoint` by default.

### Deployment

For the sake of using the model to generate images according to given prompt, you just need to run:

```bash
python generate.py --model_path=path/to/pretrained_model --checkpoint=path/to/checkpoint/unet --prompt=prompt --output_dir=output/
```
