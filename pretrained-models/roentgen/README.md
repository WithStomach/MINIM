---
widget:
  - text: "Large right-sided pleural effusion. Cardiomegaly."
language:
  - en
tags:
  - stable-diffusion
  - text-to-image
license: creativeml-openrail-m
inference: true
---

Important: The generated images are for research and educational purposes only and cannot replace real chest x-rays for medical diagnosis.

By using RoentGen you confirm that you are credentialed and allowed to use [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), and will only share access to the model with people that are also credentialed for MIMIC-CXR.

Relevant data use agreement: https://physionet.org/content/mimic-cxr/view-dua/2.0.0/

Preprint: https://arxiv.org/abs/2211.12737

Project Website: https://stanfordmimi.github.io/RoentGen/

## Use

You can use the model by loading it with the huggingface diffusers library:

```python
from diffusers import StableDiffusionPipeline

model_path = "/path/to/roentgen"
device='gpu'  # or mps, cpu...
pipe = StableDiffusionPipeline.from_pretrained(model_path).to(torch.float32).to(device)

prompt = "big right-sided pleural effusion"

pipe([prompt], num_inference_steps=75, height=512, width=512, guidance_scale=4)
```

Sometimes the safety checker module included in the original Stable Diffusion pipeline will lead to black images. If this is happens frequently, it and can be disabled:

```python
pipe.safety_checker = lambda imgs, _: (imgs, False)
```

## Citation

```bibtex
@misc{chambon2022roentgen,
  doi = {10.48550/ARXIV.2211.12737},
  url = {https://arxiv.org/abs/2211.12737},
  author = {Chambon, Pierre and Bluethgen, Christian and Delbrouck, Jean-Benoit and Van der Sluijs, Rogier and Polacin, Malgorzata and Chaves, Juan Manuel Zambrano and Abraham, Tanishq Mathew and Purohit, Shivanshu and Langlotz, Curtis P. and Chaudhari, Akshay},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RoentGen: Vision-Language Foundation Model for Chest X-ray Generation},
  publisher = {arXiv},
  year = {2022},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
