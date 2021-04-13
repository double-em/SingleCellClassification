import matplotlib.pyplot as plt
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

    def __init__(self, id: str, ext: str, path: Path, device: str, size: int = None):
        self.id = id
        self.ext = ext
        self.path = path
        self.size = size
        self.device = device

        self.fname = path/f'{id}.{ext}'

        self.r_ten = None
        self.g_ten = None
        self.b_ten = None

    def rgb_image(self, size: int = None) -> Image.Image:
        if size is None:
            size = self.size

        return assemble_rgb_image(self.r_tensor(), self.g_tensor(), self.b_tensor())

    def r_image(self):
        return image_from_tensor(self.r_tensor(), mode="P")

    def r_tensor(self):
        if self.r_ten is None:
            self.r_ten = self.get_channel("red")
            return self.r_ten
        return self.r_ten

    def g_image(self):
        return image_from_tensor(self.g_tensor(), mode="P")

    def g_tensor(self):
        if self.g_ten is None:
            self.g_ten = self.get_channel("green")
            return self.g_ten
        return self.g_ten

    def b_image(self):
        return image_from_tensor(self.b_tensor(), mode="P")

    def b_tensor(self):
        if self.b_ten is None:
            self.b_ten = self.get_channel("blue")
            return self.b_ten
        return self.b_ten

    def get_channel(self, channel: str):
        path = f'{self.path}/{self.id}_{channel}.{self.ext}'
        return read_img(path=path, interp_size=self.size, device=self.device)

    def plot_all(self, rows=2, cols=2, color="#0A0"):
        titles = ["red", "green", "blue", "rgb"]
        images = [self.r_image(), self.g_image(), self.b_image(), self.rgb_image()]
        figure, ax = plt.subplots(nrows=rows, ncols=cols)
        for idx, image in enumerate(images):
            axes = np.ravel(ax)[idx]
            axes.imshow(image)
            axes.set_axis_off()
            axes.set_title(titles[idx], color=color)
        plt.tight_layout()
        plt.show()


def show_images(images: list, rows=1, cols=1):
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for idx, image in enumerate(images):
        np.ravel(ax)[idx].imshow(image)
        np.ravel(ax)[idx].set_axis_off()
        np.ravel(ax)[idx].set_title(image_id)
    plt.tight_layout()
    plt.show()


def select_images(folder_path: str, size_image: int) -> list:
    return [img_path for img_path in get_image_files(folder_path) if is_image_size(img_path, size_image)]


def is_image_size(path: str, size: int) -> bool:
    w, h = Image.open(path).size
    return w >= size and h >= size


def read_img(interp_size: int, path: str, device: str = None, interpolation_mode="bilinear") -> Tensor:
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


def pill_to_tensor(image: Image.Image) -> Tensor:
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

def image_from_tensor(tensor: Tensor, mode: str = None) -> Image.Image:
    tensor = torch.transpose(tensor, dim0=2, dim1=0)

    if mode is not "RGB":
        tensor = tensor[:,:,0]

    # Array needs to be on the CPU to read the image via. 'Image.fromarray()'
    arr = tensor.cpu().numpy()
    return Image.fromarray(arr, mode)

def assemble_rgb_image(r: Tensor, g: Tensor, b: Tensor, device: str = None) -> Image.Image:
    stacked_image = torch.stack([r, g, b], axis=1)
    stacked_image = stacked_image[0, :, :, :]

    return image_from_tensor(stacked_image, mode="RGB")


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

        image = HPAImage(id=image_id,
                         ext=img_format,
                         path=img_source,
                         device=device,
                         size=512)

        im = image.rgb_image()

        im.save(fname, format=img_format.upper())

    cell_df = pd.DataFrame(all_cells)
    cell_df.to_csv(csv_path, index=False)