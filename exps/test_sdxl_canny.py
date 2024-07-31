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
from pipeline_sdxl import StableDiffusionXLControlLoraV3Pipeline


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
vae_model_path = "madebyollin/sdxl-vae-fp16-fix" # "stabilityai/stable-diffusion-xl-base-1.0"
vae_subfolder = None # "vae"

vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder=vae_subfolder, torch_dtype=torch.float16)

unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
unet = unet.add_extra_conditions("sd-control-lora-v3-canny")

pipe: StableDiffusionXLControlLoraV3Pipeline = StableDiffusionXLControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, torch_dtype=torch.float16
)
# load attention processors
pipe.load_lora_weights("out/sdxl-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
# pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sdxl-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
# pipe.load_lora_weights("HighCWu/sdxl-control-lora-v3-canny")
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


images = [(path, load_image(path)) for path in ["./imgs/canny1.jpg", "./imgs/canny2.jpg", "./imgs/canny3.jpg", "./imgs/canny4.jpg"]]
prompts = [
    "portrait of a beautiful winged goddess with horns, long wavy black hair, long black dress with silver jewels by tom bagshaw", 
    "an emo portrait painting. short dark brown messy pixie haircut, large black eyes, antichrist eyes, slightly rounded face, pointed chin, thin lips, small nose, black tank top, black leather jacket, black knee - length skirt, black choker, gold earring, by peter mohrbacher, by rebecca guay, by ron spencer",
    "a photograph of a futuristic street scene, brutalist style, straight edges, finely detailed oil painting, impasto brush strokes, soft light, 8 k, dramatic composition, dramatic lighting, sharp focus, octane render, masterpiece, by adrian ghenie and jenny saville and zhang jingna", 
    "portrait of a dancing eagle woman, beautiful blonde haired lakota sioux goddess, intricate, highly detailed art by james jean, ray tracing, digital painting, artstation, concept art, smooth, sharp focus, illustration, artgerm and greg rutkowski and alphonse mucha, vladimir kush, giger, roger dean, 8 k"
]

os.makedirs("./out/grids", exist_ok=True)
for (path, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]

    output = pipe(
        [prompt]*3,
        image=[image]*3,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 3,
        num_inference_steps=20,
        generator=generator
    )

    grid = image_grid([image] + output.images, 2, 2)
    grid.save("./out/grids/sdxl_%s.png" % os.path.basename(path).split(".")[0])



# vermeer canny


image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
# image.show()


image_ori = image
image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
# canny_image.show()
image = canny_image

prompt = ["best quality, extremely detailed, " + t for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
generator = [torch.Generator(device="cpu").manual_seed(i) for i in range(len(prompt))]

output = pipe(
    prompt,
    image=image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    num_inference_steps=20,
    generator=generator,
)

grid = image_grid([image, image_ori] + output.images, 3, 2)
grid.save("./out/grids/sdxl_canny_vermeer.png")
