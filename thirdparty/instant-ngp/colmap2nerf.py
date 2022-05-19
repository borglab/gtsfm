"""Convert COLMAP-style SfM results to NeRF input format

Ref: https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py

This is a truncated version edited by Ren Liu. 
"""

import argparse
import json
import math
import os
from typing import Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse arguments from user input
    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, \
            and optionally run colmap in the first place"
    )

    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument(
        "--text",
        default="colmap_text",
        help="input path to the colmap text files (set automatically if run_colmap is used)",
    )
    parser.add_argument(
        "--aabb_scale",
        default=16,
        choices=["1", "2", "4", "8", "16"],
        help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16",
    )
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
    parser.add_argument("--out", default=".", help="output path")
    return parser.parse_args()


def variance_of_laplacian(image: cv2.Mat) -> float:
    """compute the variance of laplacian of the input image

    Args:
        image: input image, in shape of (H, W)
    Returns:
        the variance of laplacian
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath: str) -> float:
    """compute the sharpness of the input image

    Args:
        imagePath: input image path
    Returns:
        the sharpness of laplacian
    """
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion rotation vector to 3x3 rotation matrix

    Args:
        qvec: a quaternion rotation vector, in shape of (4,)

    Returns:
        the corresponding 3x3 rotation matrix
    """
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """calculate the 3x3 rotation matrix from vector a to vector b

    Args:
        a: orginal vector, in shape of (3,)
        b: target vector, in shape of (3,)

    Returns:
        result rotation matrix, in shape of (3, 3)
    """
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def closest_point_2_lines(oa: np.ndarray, da: np.ndarray, ob: np.ndarray, db: np.ndarray) -> Tuple[np.ndarray, float]:
    """returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel

    Args:
        oa: origin of the first ray, in shape of (3, )
        da: direction of the first ray, in shape of (3, )
        ob: origin of the second ray, in shape of (3, )
        db: direction of the second ray, in shape of (3, )

    Returns:
        1. the point closest to both rays of form o+t*d, in shape of (3, )
        2. a weight factor that goes to 0 if the lines are parallel
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


if __name__ == "__main__":
    args = parse_args()

    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)
    IMAGE_FOLDER = args.images
    TEXT_FOLDER = args.text
    OUT_PATH = args.out
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(
                    " "
                )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                # name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                b = sharpness(name)
                print(name, "sharpness=", b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                c2w[0:3, 2] *= -1  # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3], :]  # swap y and z
                c2w[2, :] *= -1  # flip whole world upside down

                up += c2w[0:3, 1]

                frame = {"file_path": name, "sharpness": b, "transform_matrix": c2w}
                out["frames"].append(frame)
    nframes = len(out["frames"])
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"])  # rotate up to be the z axis

    # find a central point they are all looking at
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in out["frames"]:
        mf = f["transform_matrix"][0:3, :]
        for g in out["frames"]:
            mg = g["transform_matrix"][0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.01:
                totp += p * w
                totw += w
    totp /= totw
    print(totp)  # the cameras are looking at totp
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] -= totp

    avglen = 0.0
    for f in out["frames"]:
        avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
    avglen /= nframes
    print("avg camera distance from origin", avglen)
    for f in out["frames"]:
        f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes, "frames")
    print(f"writing {OUT_PATH}/transforms.json")
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    with open(os.path.join(OUT_PATH, "transforms.json"), "w") as outfile:
        json.dump(out, outfile, indent=2)
