import os
import sys
import copy
import torch
import random
import numpy as np

from PIL import Image, ImageFilter  
from datasets import load_dataset
from torchvision import transforms
from accelerate.logging import get_logger


logger = get_logger(__name__)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        args = copy.deepcopy(args)
        args.dataset_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/sd-generated", "sd1_5_pair_data"))

        self.args = args
        self.tokenizer = tokenizer
        
        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
        
        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
        else:
            if args.train_data_dir is not None:
                dataset = load_dataset(
                    args.train_data_dir,
                    cache_dir=args.cache_dir,
                )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        if args.image_column is None:
            self.image_column = column_names[0]
            logger.info(f"image column defaulting to {self.image_column}")
        else:
            self.image_column = args.image_column
            if self.image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if args.caption_column is None:
            self.caption_column = column_names[1]
            logger.info(f"caption column defaulting to {self.caption_column}")
        else:
            self.caption_column = args.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if args.conditioning_image_column is None:
            self.conditioning_image_column = column_names[2]
            logger.info(f"conditioning image column defaulting to {self.conditioning_image_column}")
        else:
            self.conditioning_image_column = args.conditioning_image_column


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"]
        # train_dataset = train_dataset.cast_column("image", ImageFeature())
        # train_dataset = train_dataset.cast_column("guide", ImageFeature())
        train_dataset = train_dataset.with_transform(self.preprocess_train)

        self.dataset = train_dataset

    def tokenize_captions(self, examples, is_train=True):
        captions = []
        for caption in examples[self.caption_column]:
            if random.random() < self.args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{self.caption_column}` should contain either strings or lists of strings."
                )
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def preprocess_train(self, examples):
        image_0 = examples[self.image_column][0]
        if isinstance(image_0, str):
            if sys.platform != 'win32' and any([image_0.startswith(chr(i) + ':') for i in range(65, 123)]): # "A: ~ z:"
                examples[self.image_column] = [
                    f'/mnt/{image[0].lower()}/{image[2:]}' 
                    for image in examples[self.image_column]
                ]
            examples[self.image_column] = [
                Image.open(image.replace("\\", "/")) 
                for image in examples[self.image_column]
            ]
        images = [(image).convert("RGB") for image in examples[self.image_column]]
        images = [self.image_transforms(image) for image in images]

        conditioning_images = [self.preprocess_conditions(image) for image in examples[self.image_column]]
        conditioning_images = [self.conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["input_ids"] = self.tokenize_captions(examples)

        return examples
    
    def preprocess_conditions(self, image: Image.Image):
        args = self.args
        image = image.convert("RGB")
        random_code = np.random.randint(0, 5)
        if random_code == 0:
            res = 2 ** np.random.randint(3, int(np.log2(args.resolution)))
            image = image.resize((res, res), resample=Image.Resampling.NEAREST)
        elif random_code == 1:
            res = 2 ** np.random.randint(3, int(np.log2(args.resolution)))
            image = image.resize((res, res), resample=Image.Resampling.BILINEAR)
        elif random_code == 2:
            image = image.filter(ImageFilter.GaussianBlur(radius = np.random.randint(5, 11)))
        elif random_code == 3:
            image = np.asarray(image).astype(np.float32) / 127.5 - 1
            noise = np.random.uniform(0.1, 0.5) * np.random.randn(*image.shape)
            image = image + noise
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif random_code == 4:
            image = image.resize((args.resolution//2, args.resolution//2), resample=Image.Resampling.NEAREST)

        return image
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
