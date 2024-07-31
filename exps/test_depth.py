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
unet = unet.add_extra_conditions("sd-control-lora-v3-depth")

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-depth-half-rank8-conv_in-rank128")
pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-depth-half-rank8-conv_in-rank128")
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


images = [(path, load_image(path)) for path in ["./imgs/depth1.jpg", "./imgs/depth2.jpg", "./imgs/depth3.jpg", "./imgs/depth4.jpg"]]
prompts = [
    "a gorgeous woman with long light-blonde hair wearing a low cut tanktop, standing in the rain on top of a mountain, highly detailed, artstation, concept art, sharp focus, illustration, art by artgerm and alphonse mucha",
    "a colored colorized portrait of andy warhol, hyperrealistic, extremely realistic, highly realistic, hd quality, 4 k resolution, 8 k resolution, detailed, very detailed, highly detailed, extremely detailed, intricate details, real, very real, oil painting, digital painting, painting, trending on deviantart, trending on artstation, in the style of david levine, in the style of greg rutkowski",
    "lofi bus from school portrait, Pixar style, by Tristan Eaton Stanley Artgerm and Tom Bagshaw.",
    "( cyberpunk 2 0 7 7, bladerunner 2 0 4 9 ), a complex thick bifurcated robotic cnc surgical arm cybernetic symbiosis hybrid mri 3 d printer machine making a bio chemical lab, art by artgerm and greg rutkowski and alphonse mucha, biomechanical, lens orbs, global illumination, octane render, architectural, f 3 2,"
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
