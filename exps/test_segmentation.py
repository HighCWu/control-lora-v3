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
unet = unet.add_extra_conditions("sd-control-lora-v3-segmentation")

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-segmentation-half_skip_attn-rank128-conv_in-rank128")
pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-segmentation-half_skip_attn-rank128-conv_in-rank128")
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


images = [(path, load_image(path)) for path in ["./imgs/segmentation1.jpg", "./imgs/segmentation2.jpg", "./imgs/segmentation3.jpg", "./imgs/segmentation4.jpg"]]
prompts = [
    "realistic painting of a tardigrade kaiju, with 6 legs in a desert storm, by james gurney, slime, big globule eye, godzilla, vintage, concept art, oil painting, tonalism, crispy",
    "portrait of a beautiful cute strong brave realistic! female gnome engineer, textured undercut black hair, d & d, micro detail, intricate, elegant, highly detailed, centered, rule of thirds, artstation, sharp focus, illustration, artgerm, tomasz alen kopera, donato giancola, wlop",
    "beautiful digital painting of a stylish asian female forest with high detail, 8 k, stunning detail, works by artgerm, greg rutkowski and alphonse mucha, unreal engine 5, 4 k uhd",
    "duotone noir scifi concept dynamic illustration of 3 d mesh of robotic cats inside box floating zero gravity glowing 3 d mesh portals futuristic, glowing eyes, octane render, surreal atmosphere, volumetric lighting. accidental renaissance. by sachin teng and sergey kolesov and ruan jia and heng z. graffiti art, scifi, fantasy, hyper detailed. trending on artstation"
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


images = [(path, load_image(path)) for path in ["./imgs/segmentation5.jpg", "./imgs/segmentation6.jpg", "./imgs/segmentation7.jpg", "./imgs/segmentation8.jpg"]]
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
