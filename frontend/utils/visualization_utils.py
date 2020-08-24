'''
Functions for visualization/plotting
'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from frontend.utils.feature_utils import keypoints_of_array
from utils.io import save_image_array


def visualize_keypoints(image, descriptors):
    kpts = keypoints_of_array(descriptors)

    image_with_kpts = cv.drawKeypoints(image, kpts, None)

    plt.figure()
    plt.imshow(image_with_kpts)

    plt.show()


def hstack_images(imgA, imgB):
    """
    Stacks 2 images side-by-side
    :param imgA:
    :param imgB:
    :return:

    NOTE: copied from Frank's assignment
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]

    newImg = np.ones((Height, Width, 3), dtype=imgA.dtype)*255
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

    return newImg


def vstack_images(imgA, imgB):
    """
    Stacks 2 images top-bottom
    :param imgA:
    :param imgB:
    :return:

    NOTE: copied from Frank's assignment
    """
    Height = imgA.shape[0] + imgB.shape[0]
    Width = max(imgA.shape[1], imgB.shape[1])

    newImg = np.ones((Height, Width, 3), dtype=imgA.dtype)*255
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[imgA.shape[0]:, :imgB.shape[1], :] = imgB

    return newImg


def visualize_matches(image1, image2, features1, features2, inlier_mask=None, file_name=None, match_width=False):
    '''

    NOTE: copied from Frank's assignment and modified
    '''
    scale_im1 = 1.0
    scale_im2 = 1.0
    if match_width:
        max_width = int(max(image1.shape[1], image2.shape[1]))

        if image1.shape[1] != max_width:
            scale_im1 = float(max_width)/image1.shape[1]
            image1 = cv.resize(
                image1,
                (max_width, int(image1.shape[0]*scale_im1)),
                interpolation=cv.INTER_CUBIC
            )
        elif image2.shape[1] != max_width:
            scale_im2 = float(max_width)/image2.shape[1]

            image2 = cv.resize(
                image2,
                (max_width, int(image2.shape[0]*scale_im2)),
                interpolation=cv.INTER_CUBIC
            )

    stacked_image = vstack_images(image1, image2)

    shift_y = image1.shape[0]

    if inlier_mask is None:
        inlier_mask = np.full((features1.shape[0], ), True)

    for feature_idx in range(features1.shape[0]):
        stacked_image = cv.circle(
            stacked_image, (int(features1[feature_idx, 0]*scale_im1), int(features1[feature_idx, 1]*scale_im1)), 3, (0, 0, 255), -1)
        stacked_image = cv.circle(
            stacked_image, (int(features2[feature_idx, 0]*scale_im2), int(features2[feature_idx, 1]*scale_im2) + shift_y), 3, (0, 0, 255), -1)

        if inlier_mask[feature_idx]:
            line_color = (0, 255, 0)
        else:
            line_color = (255, 0, 0)

        stacked_image = cv.line(
            stacked_image, (int(features1[feature_idx, 0]*scale_im1),
                            int(features1[feature_idx, 1]*scale_im1)),
            (int(features2[feature_idx, 0]*scale_im2), int(features2[feature_idx, 1]*scale_im2)+shift_y), line_color, 1, cv.LINE_AA)

    highres_image = cv.resize(
        stacked_image, (stacked_image.shape[1]*2, stacked_image.shape[0]*2), interpolation=cv.INTER_CUBIC)

    if file_name is not None:
        save_image_array(highres_image, file_name)
    else:
        plt.figure()
        plt.imshow(highres_image)
        plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1.shape
    for idx, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        color = colors[idx % 10]
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 2, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 2, color, -1)
    return img1, img2


def drawEpipolarLines(img1, img2, pts1, pts2, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img5)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show()
