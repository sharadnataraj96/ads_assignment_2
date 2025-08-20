# UBeauty: Automated Brand Grammar Extraction and LoRA Training Pipeline

This project provides a pipeline for extracting brand grammar from images, training a LoRA (Low-Rank Adaptation) model using Stable Diffusion XL, and generating image variants using the [Fooocus API](https://github.com/mrhan1993/Fooocus-API). The workflow is designed for luxury beauty product imagery, but can be adapted for other domains.

## 1. Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training and inference)
- [Git](https://git-scm.com/)
- [Fooocus API](https://github.com/lllyasviel/Fooocus) (for image generation)
- [Hugging Face account](https://huggingface.co/) (for dataset/model uploads)

## 2. Install Fooocus API

Clone and set up the Fooocus API repository:

```shell
git clone https://github.com/mrhan1993/Fooocus-API.git
cd Fooocus-API
```

Create the conda env
```shell
conda env create -f environment.yaml
conda activate fooocus-api
```

Start the API server

```shell
python main.py
```

## 3. In the stealth folder

Run the file

```shell
python main.py
```

The resuls will be stored in the outputs directory


The code does the following

- Ingests a given dataset and generates captions using brand grammer as reference vocabulary and LLava Image to text model
- Trains a LoRA(style LoRA) to learn the brand grammer and reports metrics to weights and biases(WandB)
- Creates masks for a given input image : text pair
- Uses the input image, LoRA and the mask to inpaint the scene around the product.

Expected outputs

- captions.json contains the brand grammer for each image
- lora : contains the trained lora
- output_images : contains the output images






