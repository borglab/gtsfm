"""MVSDataset io utils
    reference: https://github.com/FangjinhuaWang/PatchmatchNet

"""
import re
import sys
from typing import Tuple

import numpy as np


def read_pfm(filename: str) -> Tuple[np.ndarray, float]:
    """Read data from a .pfm file

    Args:
        filename: string of input .pfm file path

    Returns:
        data: data read from .pfm file in the of shape (H, w, C)
        scale: float scale parameter loaded from .pfm file
    """
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename: str, image: np.ndarray, scale: float = 1) -> None:
    """Save data to a .pfm file

    Args:
        filename: string of output .pfm file path
        image: data in the shape of (H, W, C) to save in the .pfm file
        scale: float scale parameter to save in the .pfm file
    """
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n".encode("utf-8") if color else "Pf\n".encode("utf-8"))
    file.write("{} {}\n".format(image.shape[1], image.shape[0]).encode("utf-8"))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write(("%f\n" % scale).encode("utf-8"))

    image.tofile(file)
    file.close()
