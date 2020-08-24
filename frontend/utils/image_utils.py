import numpy as np


def rgb_to_gray_matlab(image_array: np.ndarray) -> np.ndarray:
    if len(image_array.shape) == 2:
        pass
    elif image_array.shape[2] == 3 or image_array.shape[2] == 4:
        image_array = (np.average(image_array[:, :, :3].astype(np.float32), weights=[
            0.2989, 0.5870, 0.1140], axis=2)).astype(np.uint8)
    else:
        raise ValueError('Input image dimensions are wrong')

    return image_array
