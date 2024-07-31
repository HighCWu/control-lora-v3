import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)



import cv2
import torch
import numpy as np

from diffusers.utils import load_image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from PIL import Image

from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline


model_id = "SG161222/Realistic_Vision_V4.0_noVAE" # "ckpt/anything-v3.0" # "runwayml/stable-diffusion-v1-5" # 
vae_model_path = "stabilityai/sd-vae-ft-mse" # "ckpt/anything-v3.0" # "runwayml/stable-diffusion-v1-5" # 
vae_subfolder = None # "vae" # "vae" # 

vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder=vae_subfolder, torch_dtype=torch.float16)

unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
unet = unet.add_extra_conditions("sd-control-lora-v3-pose")

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-pose-half-rank128-conv_in-rank128")
pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-pose-half-rank128-conv_in-rank128")
pipe = pipe.to(device="cuda", dtype=torch.float16)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


images = [(path, load_image(path)) for path in ["./imgs/pose1.jpg", "./imgs/pose2.jpg", "./imgs/pose3.jpg", "./imgs/pose4.jpg"]]
prompts = [
    "medieval rich kingpin sitting in a tavern with his thugs, drinking and cheering, few subjecs, elegant, close frontal shot, digital painting, concept art, smooth, sharp focus, illustration, from d & d by ruan jia and mandy jurgens and artgerm and william - adolphe bouguerea",
    "sensual good looking pale young indian doctors wearing jeans in celebrating after passing an exam, portrait, elegant, intricate, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha",
    "a psychedelic detailed gorgeous acid trip painting of an extremely sexy elegant and attractive female character, wearing a tight-fitting tan detective jacket, detective had on her head, beautiful [[long red hair]] in loose curls, slender woman, very curvy, noir, smoking a fancy long french cigarette, in the rain in the early evening, cinematic, dramatic lighting, full body view, cool pose, artwork by Artgerm, Rutkowski, Dale Keown and Van Sciver, featured on artstation, cgsociety, behance hd",
    "wow! fanart, ork from warhammer 4 0 k with, d & d, high fantasy, detailed, digital art, artstation, smooth, sharp focus, art by artgerm, greg rutkowski, alphonse mucha"
]

os.makedirs("./out/grids", exist_ok=True)
for (path, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]

    output = pipe(
        [prompt]*3,
        [image]*3,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 3,
        num_inference_steps=20,
        generator=generator,
    )

    grid = image_grid([image] + output.images, 2, 2)
    grid.save("./out/grids/%s.png" % os.path.basename(path).split(".")[0])


images = [(path, load_image(path)) for path in ["./imgs/pose5.jpg", "./imgs/pose6.jpg", "./imgs/pose7.jpg", "./imgs/pose8.jpg"]]
prompts = [
    "a tiny worlds by greg rutkowski, sung choi, mitchell mohrhauser, maciej kuciara, johnson ting, maxim verehin, peter konig, bloodborne, 8 k photorealistic, cinematic lighting, hd, high details, dramatic, dark atmosphere, trending on artstation",
    "office secretary anime, d & d, fantasy, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and magali villeneuve",
    "isometric chubby 3 d game cannon, with detailed, clean, cartoon, octane render, unreal engine, artgerm, artstation",
    "fullbody!! dynamic movement pose, beautiful ethnic woman with flowing hair, big natural horns on her head, gold jewellery, dnd, face, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha"
]

os.makedirs("./out/grids", exist_ok=True)
for (path, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]

    output = pipe(
        [prompt]*3,
        [image]*3,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 3,
        num_inference_steps=20,
        generator=generator,
    )

    grid = image_grid([image] + output.images, 2, 2)
    grid.save("./out/grids/%s.png" % os.path.basename(path).split(".")[0])
