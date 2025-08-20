import os
import json
import subprocess
import torch
import numpy as np

from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image

import time
import io
import base64
from typing import List
import sys
import importlib.util
from pathlib import Path
import shutil
import requests

import matplotlib.pyplot as plt
import cv2




from datasets import Dataset
import datasets


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"  #change this to adapt dynamically
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMG_SERVER = ""
MASK_DIR = os.path.join(OUTPUT_DIR,"masks")

LORA_TRAINING_ARGS = {
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "dataset_name": "sharad-nataraj/ubeauty",
        "caption_column": "caption",
        "validation_prompt": "An advertisement image of a luxury beauty product",
        "output_dir": f"{OUTPUT_DIR}/lora_models",
        "learning_rate": 1e-4,
        "num_train_epochs": 100,
        "report_to": "wandb",
    }


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
else: 
    raise FileExistsError(f"Output directory {OUTPUT_DIR} already exists")

def extract_brand_grammer(image_dir,output_json,model_id = "llava-hf/llava-1.5-7b-hf"):

    """
    Extracts the brand grammer using an image captioning model.
    stores the results in a json file.
    Args:
        image_dir: path to the image directory
        output_json: path to the output json file
        model_id: model id to use for image captioning
    Returns:
        captions: dictionary of image filenames and their captions
    """

    print("Extracting brand grammer...")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    
    # if not os.path.exists(output_json):
    #     raise FileNotFoundError(f"Output JSON file {output_json} does not exist")
    

    if not os.path.exists("prompts/llava_prompt_v2.txt"):
        raise FileNotFoundError("VLM_prompt.txt file not found in prompts directory")
    

    captions = {}

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    processor = LlavaProcessor.from_pretrained(model_id)

    print("Model loaded successfully")

    system_prompt = open("prompts/llava_prompt.txt").read()

    for filename in os.listdir(image_dir):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image = Image.open(os.path.join(image_dir,filename))

        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": system_prompt}
            ]
        },
    ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=200)
        decoded_output = processor.decode(output[0][inputs["input_ids"].shape[-1]:])

        captions[filename] = decoded_output
        text_file = os.path.join(image_dir,filename.replace(".png",".txt"))
        with open(text_file,"w") as f:
            f.write(decoded_output)

        print(f"Extracted brand grammer for {filename}")

        del inputs, output, decoded_output

    with open(os.path.join(OUTPUT_DIR,output_json), "w") as f:
        json.dump(captions, f,indent=4)

    print("Brand grammer extracted successfully")

    upload_dataset_to_hf(image_dir)


    return captions

def upload_dataset_to_hf(dataset_dir):

    """
    Uploads a dataset to the Hugging Face Hub.
    """

    print("Uploading dataset to Hugging Face Hub...")

    images,captions = [],[]
    for file in os.listdir(dataset_dir):
        if file.endswith((".png",".jpg",".jpeg")):
            base = file.replace(".png","").replace(".jpg","").replace(".jpeg","")
            images.append((os.path.join(dataset_dir,file)))
            with open(os.path.join(dataset_dir,base + ".txt")) as f:
                captions.append(f.read().strip())

    dataset = Dataset.from_dict({"image":images,"caption":captions})
    dataset = dataset.cast_column("image", datasets.Image()) 

    dataset.push_to_hub("sharad-nataraj/ubeauty",private=False)

    print("Dataset uploaded to Hugging Face Hub successfully")

def train_lora(dataset_dir=None,captions_path=None,output_dir=None,target_resolution=(1024,1024)):

    """
    Trains a LoRA model using the dataset and captions.
    """

    print("Training LoRA model...")

    script_path = Path("diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py")


    args = [
        f"--{k}={v}" for k, v in LORA_TRAINING_ARGS.items()
    ]

    command = [
            "accelerate",
            "launch",
            str(script_path),
        ] + args

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with error: {e}")

    if os.path.exists(f"{OUTPUT_DIR}/lora_models"):
        print("LoRA model trained successfully")
    else:
        raise FileNotFoundError("LoRA model not found")
    
    # Move the trained LoRA model to Fooocus-API/repositories/Fooocus/models/loras
    src_dir = os.path.join(OUTPUT_DIR, "lora_models")
    dest_dir = os.path.join("Fooocus-API", "repositories", "Fooocus", "models", "loras")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    # Move all files and subdirectories from src_dir to dest_dir
    
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)
        if os.path.isdir(src_path):
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

def generate_image_mask(image_path,text_prompt:str):

    """
    Generates a mask for the image.
    """

    print("Generating image mask...")

    image = Image.open(image_path)

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(text=[text_prompt],images=[image],padding=True,return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    preds = outputs.logits.unsqueeze(1)

    if not os.path.exists(MASK_DIR):
        os.makedirs(MASK_DIR)

    
    prob_map = torch.sigmoid(preds).squeeze()
    binary_mask_np = (prob_map > 0.5).cpu().numpy()
    final_mask = (binary_mask_np*255).astype(np.uint8)
    inverted_mask = 255 - final_mask

    mask_path = os.path.basename(image_path)

    plt.imsave(os.path.join(MASK_DIR,f"{mask_path}"),inverted_mask)
    _,encoded_mask = cv2.imencode('.jpeg',inverted_mask)
    base64_mask = base64.b64encode(encoded_mask.tobytes()).decode("utf-8")

    return base64_mask





def generate_variants(prompts_list:List[str],input_image_path:str,mask_text:str):

    """
    Generates variants of the prompts.
    """

    input_image = Image.open(input_image_path)
    base64_image = base64.b64encode(input_image.tobytes()).decode("utf-8")
    base64_mask = generate_image_mask(input_image,mask_text)

    for i,prompt in enumerate(prompts_list):

        lora_params = {
            "enabled" : True,
            "model_name":"u_beauty_style.safetensors",
            "weight":1.0
        }
        
        params ={
            "input_image":base64_image,
            "input_mask":base64_mask,
            "prompt":"Open desk drawer scene,laptop at the edge of the image frame,lipstick and keys on the desk,neutral office lighting,soft,even grade lighting,fron-view,luxury",
            "negative_prompt":"",
            "inpaint_additional_prompt" : "an image of a luxurious beauty product on an office desk",
            "require_base64 " : True,
            "async_process": False,
            "loras" :[lora_params]
        }

        response = requests.post(
            url=f"{IMG_SERVER}/v2/generation/image-inpaint-outpaint",
            data=json.dumps(params),
            headers={"Content-Type":"application/json"}
            )
        output_img_dir = os.path.join(OUTPUT_DIR,"output_images")
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)
        if response.status_code == 200:
            image_response = requests.get(response.json()[0]['url'])
            generated_image = Image.open(io.BytesIO(image_response.content))
            generated_image.save(f"{output_img_dir}/{i}.png")
        else:
            print(f"Failed to generate image for {prompt}")
            continue
        
       
        
       






if __name__ == "__main__":
    extract_brand_grammer(
        image_dir="dataset/train",
        output_json=f"{OUTPUT_DIR}/brand_grammer.json"
    )

    train_lora()

    prompts_list = [
        "Open desk drawer scene,laptop at the edge of the image frame,lipstick and keys on the desk,neutral office lighting,soft,even grade lighting,fron-view,luxury",
        "Bathroom counter with kids toothbrush cup partially visible in the image frame, hair tie near the object,warm lighting, approachable easy grade lighting from the right side,confortable mood"
    ]

    generate_variants(
        prompts_list=prompts_list,
        input_image_path="",
        mask_text="an image of a luxurious beauty product"
    )



    




