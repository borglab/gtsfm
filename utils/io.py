"""
Functions to provide I/O APIs for all the modules

Authors: Ayush Baid
"""

from PIL import Image


def load_image(img_path: str, scale_factor: float = None) -> np.array:
    """
    Load the image from disk

    Args:
        img_path (str): the path of image to load
        scale_factor (float, optional): the multiplicative scaling factor for height and width. Defaults to None.

    Returns:
        np.array: loaded image
    """
    if scale_factor is None:
        return np.asarray(Image.open(img_path))

    highres_img = Image.open(img_path)

    width, height = highres_img.size

    small_width = int(width*scale_factor)
    small_height = int(height*scale_factor)

    return np.asarray(highres_img.resize((small_width, small_height)))


def save_image(image: np.array, img_path: str):
    """
    Saves the image to disk

    Args:
        image (np.array): image
        img_path (str): the path on disk to save the image to
    """
    im = Image.fromarray(image)
    im.save(img_path)


'''
Code from DFE:

def get_folder_path(data_type, tag):
    folder_name = os.path.join('..', 'results', tag, data_type)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    return folder_name


def save_verified_features(features, image_idxes, tag, verifier_tag):
    folder_name = get_folder_path(
        os.path.join(verifier_tag, 'features'),
        tag
    )

    file_name = os.path.join(folder_name, '%d_%d.npy' % image_idxes)

    np.save(file_name, features)


def load_verified_features(image_idxes, tag, verifier_tag):
    folder_name = get_folder_path(
        os.path.join(verifier_tag, 'features'),
        tag
    )

    file_name = os.path.join(folder_name, '%d_%d.npy' % image_idxes)

    temp = np.load(file_name).astype(np.float32)

    return temp[:, :2], temp[:, 2:]


def check_verified_features(image_idxes, tag, verifier_tag):
    folder_name = get_folder_path(
        os.path.join(verifier_tag, 'features'),
        tag
    )

    file_name = os.path.join(folder_name, '%d_%d.npy' % image_idxes)

    return os.path.exists(file_name)


def save_fundamental_matrix(fundamental_matrix, image_idxes, tag, verifier_tag):
    folder_name = get_folder_path(
        os.path.join(verifier_tag, 'fundamental'),
        tag
    )

    file_name = os.path.join(folder_name, '%d_%d.npy' % image_idxes)

    np.save(file_name, fundamental_matrix)


def load_fundamental_matrix(image_idxes, tag, verifier_tag):
    folder_name = get_folder_path(
        os.path.join(verifier_tag, 'fundamental'),
        tag
    )

    file_name = os.path.join(folder_name, '%d_%d.npy' % image_idxes)

    return np.load(file_name).astype(np.float32)


def check_fundamental_matrix(image_idxes, tag, verifier_tag):
    folder_name = get_folder_path(
        os.path.join(verifier_tag, 'fundamental'),
        tag
    )

    file_name = os.path.join(folder_name, '%d_%d.npy' % image_idxes)

    return os.path.exists(file_name)


def save_features(features, image_idx, tag):
    folder_name = get_folder_path('features', tag)

    file_name = os.path.join(folder_name, '%d.npy' % image_idx)

    np.save(file_name, features)


def load_features(image_idx, tag):
    folder_name = get_folder_path('features', tag)

    file_name = os.path.join(folder_name, '%d.npy' % image_idx)

    return np.load(file_name).astype(np.float32)


def check_features(image_idx, tag):
    folder_name = get_folder_path('features', tag)

    file_name = os.path.join(folder_name, '%d.npy' % image_idx)

    return os.path.exists(file_name)


def save_descriptors(descriptors, image_idx, tag):
    folder_name = get_folder_path('descriptors', tag)

    file_name = os.path.join(folder_name, '%d.npy' % image_idx)

    np.save(file_name, descriptors)


def load_descriptors(image_idx, tag):
    folder_name = get_folder_path('descriptors', tag)

    file_name = os.path.join(folder_name, '%d.npy' % image_idx)

    return np.load(file_name).astype(np.float32)


def check_descriptors(image_idx, tag):
    folder_name = get_folder_path('descriptors', tag)

    file_name = os.path.join(folder_name, '%d.npy' % image_idx)

    return os.path.exists(file_name)


def save_matches(match_data, image_idxes, tag):
    folder_name = get_folder_path('matches', tag)
    file_name = os.path.join(folder_name,
                             '%d_%d.npy' % image_idxes
                             )

    np.save(file_name, match_data)


def load_matches(image_idxes, tag):
    folder_name = get_folder_path('matches', tag)

    file_name = os.path.join(folder_name,
                             '%d_%d.npy' % image_idxes
                             )

    return np.load(file_name).astype(np.int32)


def check_matches(image_idxes, tag):
    folder_name = get_folder_path('matches', tag)
    file_name = os.path.join(folder_name,
                             '%d_%d.npy' % image_idxes
                             )

    return os.path.exists(file_name)


def load_match_pts(image_idxes, tag):
    match_indices = load_matches(image_idxes, tag)

    pts1 = load_features(image_idxes[0], tag)
    pts2 = load_features(image_idxes[1], tag)

    return pts1[match_indices[:, 0]][:, :2], pts2[match_indices[:, 1]][:, :2]

'''
