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
unet = unet.add_extra_conditions([
    "pose",
    "segmentation",
    "tile"
])

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights([
#     "out/sd-control-lora-v3-pose-half-rank128-conv_in-rank128",
#     "out/sd-control-lora-v3-segmentation-half_skip_attn-rank128-conv_in-rank128",
#     "out/sd-control-lora-v3-tile-half_skip_attn-rank16-conv_in-rank64",
# ])
pipe.load_lora_weights(
    "HighCWu/control-lora-v3", subfolder=[
        "sd-control-lora-v3-pose-half-rank128-conv_in-rank128",
        "sd-control-lora-v3-segmentation-half_skip_attn-rank128-conv_in-rank128",
        "sd-control-lora-v3-tile-half_skip_attn-rank16-conv_in-rank64",
    ]
)
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


images = [
    (path1, path2, [load_image(path1), load_image(path2), Image.new('RGB', load_image(path2).size)]) for path1, path2 in zip(
        ["./imgs/pose5.jpg", "./imgs/pose6.jpg", "./imgs/pose7.jpg", "./imgs/pose8.jpg"],
        ["./imgs/segmentation5.jpg", "./imgs/segmentation6.jpg", "./imgs/segmentation7.jpg", "./imgs/segmentation8.jpg"]
    )
]
prompts = [
    "a tiny worlds by greg rutkowski, sung choi, mitchell mohrhauser, maciej kuciara, johnson ting, maxim verehin, peter konig, bloodborne, 8 k photorealistic, cinematic lighting, hd, high details, dramatic, dark atmosphere, trending on artstation",
    "office secretary anime, d & d, fantasy, portrait, highly detailed, headshot, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and magali villeneuve",
    "isometric chubby 3 d game cannon, with detailed, clean, cartoon, octane render, unreal engine, artgerm, artstation",
    "fullbody!! dynamic movement pose, beautiful ethnic woman with flowing hair, big natural horns on her head, gold jewellery, dnd, face, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha"
]

os.makedirs("./out/grids", exist_ok=True)

for (path1, path2, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]

    output = pipe(
        [prompt]*4,
        [[img.copy() for img in image] for _ in range(4)],
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
        extra_condition_scale=[1,0,0]
    )

    grid = image_grid(image[:1] + [Image.new('RGB', image[0].size)] + output.images, 3, 2)
    path1 = os.path.basename(path1).split(".")[0]
    path = path1 + "_2"
    grid.save("./out/grids/%s.png" % path)

for (path1, path2, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]

    output = pipe(
        [prompt]*4,
        [image]*4,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
        extra_condition_scale=[0,1,0]
    )

    grid = image_grid([Image.new('RGB', image[1].size)] + image[1:2] + output.images, 3, 2)
    path2 = os.path.basename(path2).split(".")[0]
    path = path2 + "_2"
    grid.save("./out/grids/%s.png" % path)

tile_images = []
for (path1, path2, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]

    output = pipe(
        [prompt]*4,
        [image]*4,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
        extra_condition_scale=[1,1,0]
    )

    tile_images.append([img.resize([img.size[0]*2, img.size[1]*2], Image.Resampling.NEAREST) for img in output.images])
    grid = image_grid(image[:2] + output.images, 3, 2)
    path1 = os.path.basename(path1).split(".")[0]
    path2 = os.path.basename(path2).split(".")[0]
    path = path1 + "_" + path2
    grid.save("./out/grids/%s.png" % path)

for (path1, path2, image), tile_image, prompt in zip(images, tile_images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]

    image_lr = [img.resize([img.size[0]*2, img.size[1]*2], Image.Resampling.NEAREST) for img in image[:2]]
    image = [[Image.new('RGB', img.size), Image.new('RGB', img.size), img] for img in tile_image]

    output = pipe(
        [prompt]*4,
        image,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
        num_inference_steps=20,
        generator=generator,
        extra_condition_scale=[0,0,1]
    )

    grid = image_grid(image_lr + output.images, 3, 2)
    path1 = os.path.basename(path1).split(".")[0]
    path2 = os.path.basename(path2).split(".")[0]
    path = path1 + "_" + path2 + "_" + "tile"
    grid.save("./out/grids/%s.png" % path)
