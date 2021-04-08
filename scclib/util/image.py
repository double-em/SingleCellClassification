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

class HPAImage:

    def __init__(self, id, ext, path, size: int = None, device):
        self.id = id
        self.ext = ext
        self.path = path
        self.size = size
        self.device = device

    def rgb(self, size: int = None):
        if size is None:
            size = self.size

        assemble_rgb_image(self.r(), self)

    def r(self):
        return self.get_channel("red")

    def g(self):
        return self.get_channel("green")

    def b(self):
        return self.get_channel("blue")

    def get_channel(self, channel: str):
        path = f'{self.path}/{self.id}_{channel}.{self.ext}'
        return read_img(path=path, interp_size=interp_size, device=device)

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


def pill_to_tensor(image: Image.Image) -> TensorImage:
    arr = np.asarray(image)

    if arr.ndim==2 : arr = np.expand_dims(arr,2)

    # Transpose width, height to height,width
    arr = np.transpose(arr, (1,0,2))

    # Move channels to the first position
    arr = np.transpose(arr, (2, 1, 0))

    return torch.from_numpy(arr)

def get_rgb_pieces_tensors(path: str, img_format: str, interp_size: int, device: str = None):
    # We only read RGB and not RGBY
    # Because in previous models trained on the HPA set
    # the Yellow channel did not make a difference in accuracy
    red = read_img(path=f'{path}_red.{img_format}', interp_size=interp_size, device=device)
    green = read_img(path=f'{path}_green.{img_format}', interp_size=interp_size, device=device)
    blue = read_img(path=f'{path}_blue.{img_format}', interp_size=interp_size, device=device)

    return red, green, blue

def image_from_tensor(tensor: Tensor) -> Image.Image:
    # Array needs to be on the CPU to read the image via. 'Image.fromarray()'
    arr = tensor.cpu().numpy()
    return Image.fromarray(arr, "RGB")

def assemble_rgb_image(r: Tensor, g: Tensor, b: Tensor, device: str = None) -> Image.Image:
    stacked_image = torch.stack([b, g, r], axis=1)
    stacked_image = stacked_image[0, :, :, :]

    stacked_image = torch.transpose(stacked_image, dim0=2, dim1=0)

    return image_from_tensor(stacked_image)


def create_samples(img_size: int, df: pd.DataFrame, img_source: str, img_destination: str, csv_path: str, device: str = None, img_format="png"):
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
        
        # Format needs to be 'JPEG' to save as jpg.
        if img_format.lower() == "jpg":
            img_format = "JPEG"

        r, g, b = get_rgb_pieces_tensors(f'{img_source}/{image_id}', img_format, img_size, device)

        im = assemble_rgb_image(image_id, img_size, img_source, img_format, device)
        im.save(fname, format=img_format.upper())

    cell_df = pd.DataFrame(all_cells)
    cell_df.to_csv(csv_path, index=False)