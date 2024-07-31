<center>
<h1>ControlLoRA Version 3: LoRA Is All You Need to Control the Spatial Information of Stable Diffusion
</h1>
<a href="https://huggingface.co/HighCWu/control-lora-v3">[Models]</a>
<a href="https://huggingface.co/spaces/HighCWu/control-lora-v3">[Spaces]</a>
</center>

## Introduction

I introduce a faster and lighter way to add image conditions to control the spatial information of the generated images of stable diffusion. To do this, LoRA is all you need.
The idea is very simple, increase the input channel size of the first convolution of stable diffusion `unet` by 1 times and fill the new weight part with zero values. Then configure and train a LoRA like training a normal linear&conv2d LoRA of `text-to-image`, the only difference is that you need to provide the noise latent and  the conditional image latent together as the inputs to `unet`. After training, we can even merge the LoRA weights completely into `unet`.


![Canny Result](./imgs/canny_vermeer.png)


***Note:*** It should be noted that control lora v3 uses the latent encoded by `vae` as conditional input, so it is not very sensitive to certain types of images and needs to be processed in advance to encode images that can retain details.


## Pretrained Models

[control-lora-v3 sd canny](https://huggingface.co/HighCWu/sd-control-lora-v3-canny)

[control-lora-v3 sdxl canny](https://huggingface.co/HighCWu/sdxl-control-lora-v3-canny)

[control-lora-v3 pretrained models collection](https://huggingface.co/HighCWu/control-lora-v3)

Try `exps/test_<TASK>.py` to test different type conditions.

Try `exps/test_multi_load.py` to load multi lora and switch between them.

Try `exps/test_multi_inputs.py` to use multi lora at the same time.

I used a high learning rate and a short number of steps for training, and the dataset was also generated, so the generation results may not be very good. It is recommended that researchers use real data, lower learning and longer training steps to train to achieve better generation results.


## Prepare Environment

```sh
git clone https://github.com/HighCWu/control-lora-v3
cd control-lora-v3
pip install -r requirements.txt
```

## Prepare Dataset

Make a dataset like [HighCWu/diffusiondb_2m_first_5k_canny](https://huggingface.co/datasets/HighCWu/diffusiondb_2m_first_5k_canny) or custom a dataset script like [`exps/sd1_5_tile_pair_data.py`](exps/sd1_5_tile_pair_data.py).

## Training

It's very easy to train control-lora-v3 stable diffusion or stable diffusion xl with resolution 512 images on a gpu with 16GB vram.

I put my detail training code in `exps/train_*.py`. You can refer to my code to configure the hyperparameters used for training, but the training data is local to my computer, so you need to modify `dataset_name` before you can use it normally.

For Stable Diffusion, use:

```sh
accelerate launch train.py \
 --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
 --output_dir=<YOUR_SAVE_DIR> \
 --tracker_project_name=<WANDB_PROJECT_NAME> \
 --dataset_name=<HF_DATASET> \
 --proportion_empty_prompts=0.1 \
 --conditioning_image_column=guide \
 --image_column=image --caption_column=text \
 --rank=16 \
 --lora_adapter_name=<YOUR_TASK_LORA_NAME> \
 --init_lora_weights=gaussian \
 --loraplus_lr_ratio=1.0 \
 --half_or_full_lora=half_skip_attn \
 --extra_lora_rank_modules conv_in \
 --extra_lora_ranks 64 \
 --resolution=512 \
 --learning_rate=0.0001 \
 --seed=42 \
 --validation_image <PATH_1> <PATH_2> <PATH_N> \
 --validation_prompt <PROMPT_1> <PROMPT_2> <PROMPT_N> \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --mixed_precision=fp16 \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --max_train_steps=50000 \
 --resume_from_checkpoint=latest \
 --report_to wandb \
 --push_to_hub
```

For Stable Diffusion XL, use: 

(***Note***: It will encounter the NaN problem when validating the generation results during the training process, but it does not affect the final training results. To be repaired.)

```sh
accelerate launch train_sdxl.py \
 --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
 --output_dir=<YOUR_SAVE_DIR> \
 --tracker_project_name=<WANDB_PROJECT_NAME> \
 --dataset_name=<HF_DATASET> \
 --proportion_empty_prompts=0.1 \
 --conditioning_image_column=guide \
 --image_column=image --caption_column=text \
 --rank=16 \
 --lora_adapter_name=<YOUR_TASK_LORA_NAME> \
 --init_lora_weights=gaussian \
 --loraplus_lr_ratio=1.0 \
 --half_or_full_lora=half_skip_attn \
 --extra_lora_rank_modules conv_in \
 --extra_lora_ranks 64 \
 --resolution=512 \
 --learning_rate=0.0001 \
 --seed=42 \
 --validation_image <PATH_1> <PATH_2> <PATH_N> \
 --validation_prompt <PROMPT_1> <PROMPT_2> <PROMPT_N> \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --mixed_precision=bf16 \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --max_train_steps=50000 \
 --resume_from_checkpoint=latest \
 --report_to wandb \
 --push_to_hub
```


You can init lora weights with the powerful [PiSSA](https://github.com/GraphPKU/PiSSA) by:
`
--init_lora_weights=pissa
`
or faster init with pissa_niter_n
`
--init_lora_weights=pissa_niter_2
`


You can custom the dataset training script by replace 
`
--dataset_name=<HF_DATASET>
`
with a script that includes a custom `torch.utils.data.Dataset` class:
`
--dataset_script_path=<YOUR_SCRIPT_PATH>
`.
You can refer to [`exps/sd1_5_tile_pair_data.py`](exps/sd1_5_tile_pair_data.py).


## Inference

For stable diffusion, use:

```python
# !pip install opencv-python transformers accelerate
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline
import numpy as np
import torch

import cv2
from PIL import Image

# download an image
image = load_image(
    "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
)
image = np.array(image)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# load stable diffusion v1-5 and control-lora-v3 
unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=torch.float16
)
unet = unet.add_extra_conditions(["canny"])
pipe = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
# pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
pipe.load_lora_weights("HighCWu/sd-control-lora-v3-canny")

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
    "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
).images[0]
image.show()
```

For stable diffusion xl, use:

```python
# !pip install opencv-python transformers accelerate
from diffusers import AutoencoderKL
from diffusers.utils import load_image
from model import UNet2DConditionModelEx
from pipeline_sdxl import StableDiffusionXLControlLoraV3Pipeline
import numpy as np
import torch

import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

# download an image
image = load_image(
    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

# initialize the models and pipeline
unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet", torch_dtype=torch.float16
)
unet = unet.add_extra_conditions(["canny"])
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlLoraV3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", unet=unet, vae=vae, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sdxl-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
# pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sdxl-control-lora-v3-canny-half_skip_attn-rank16-conv_in-rank64")
pipe.load_lora_weights("HighCWu/sdxl-control-lora-v3-canny")
pipe.enable_model_cpu_offload()

# get canny image
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# generate image
image = pipe(
    prompt, image=canny_image
).images[0]
image.show()
```

## TODO

[ ] Fix train_sdxl validate pipeline `NaN`.

[ ] Release more sdxl control-lora-v3 pretrained models.

[ ] Train multi lora at the same time to support inference multi lora. 


## Citation

    @software{wu2024controllorav3,
        author = {Wu Hecong},
        month = {7},
        title = {{ControlLoRA Version 3: LoRA Is All You Need to Control the Spatial Information of Stable Diffusion}},
        url = {https://github.com/HighCWu/control-lora-3},
        version = {1.0.0},
        year = {2024}
    }
