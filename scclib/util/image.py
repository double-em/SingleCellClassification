from PIL import Image
from fastai.data.transforms import get_image_files

import pandas as pd
import numpy as np

import os
import io
import base64
from tqdm import tqdm
from PIL import Image

from fastai.vision.all import *
import torch.nn.functional as nnf
from torch import Tensor


def select_images(folder_path: str, size_image: int) -> list:
    return [img_path for img_path in get_image_files(folder_path) if is_image_size(img_path, size_image)]


def is_image_size(path: str, size: int) -> bool:
    w, h = Image.open(path).size
    return w >= size and h >= size


def read_img(interp_size: int, path: str, device: str = None, interpolation_mode="bicubic") -> Tensor:
    img = Image.open(path)

    size = (interp_size, interp_size)

    img = pill_to_tensor(img)

    if device is not None:
        img = img.to(torch.device(device))
    
    # Make another dimension so we can use interpolation
    img = img.unsqueeze(0)
    
    # Make the tensor float32 so we can use interpolation
    img = img.float()

    if interpolation_mode is not None:
        img = nnf.interpolate(img, size=size, mode=interpolation_mode, align_corners=False)
    
    # Convert the tensor back to UINT8 for a valid RGB (255) image
    img = img.type(torch.uint8)

    # Remove the added dimension
    img = img[0,:,:,:]

    return img


def pill_to_tensor(image: Image.Image)->TensorImage:
    arr = np.asarray(image)

    if arr.ndim==2 : arr = np.expand_dims(arr,2)

    # Transpose width, height to height,width
    arr = np.transpose(arr, (1,0,2))

    # Move channels to the first position
    arr = np.transpose(arr, (2, 1, 0))

    return torch.from_numpy(arr)


def create_samples(img_size: int, df: pd.DataFrame, img_source: str, img_destination: str, csv_path: str, device: str = None, img_format="jpg"):
    all_cells = []
    num_files = len(df)

    for idx in tqdm(range(num_files)):
        image_id = df.iloc[idx].ID
        labels = df.iloc[idx].Label

        fname = Path(f'{img_destination}/{image_id}.{img_format}')

        all_cells.append({
            'image_id': image_id,
            'image_labels': labels
        })

        if fname.is_file():
            continue

        # We only read RGB and not RGBY
        # Because in previous models trained on the HPA set
        # the Yellow channel did not make a difference in accuracy
        red = read_img(f'{img_source}/{image_id}_red', img_size, device=device)
        green = read_img(f'{img_source}/{image_id}_green', img_size, device=device)
        blue = read_img(f'{img_source}/{image_id}_blue', img_size, device=device)

        stacked_image = torch.stack([blue, green, red], axis=1)
        stacked_image = stacked_image[0,:,:,:]

        stacked_image = torch.transpose(stacked_image, dim0=2, dim1=0)

        # Array needs to be on the CPU to read the image via. 'Image.fromarray()'
        arr = stacked_image.cpu().numpy()
        im = Image.fromarray(arr, "RGB")
        
        # Format needs to be 'JPEG' to save as jpg.
        if img_format.lower() == "jpg":
            img_format = "JPEG"

        im.save(fname, format=img_format)

    cell_df = pd.DataFrame(all_cells)
    cell_df.to_csv(csv_path, index=False)