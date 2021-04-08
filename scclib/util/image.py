from PIL import Image
from fastai.data.transforms import get_image_files


def select_images(folder_path: str, image_size: int) -> list:
    return [img_path for img_path in get_image_files(folder_path) if is_image_size(img_path, image_size)]


def is_image_size(path: str, size: int) -> bool:
    w, h = Image.open(path).size
    return w >= size and h >= size
