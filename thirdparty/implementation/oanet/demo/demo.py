#!/usr/bin/env python3
import os
import cv2
import numpy as np
from learnedmatcher import LearnedMatcher
from extract_sift import ExtractSIFT

def draw_match(img1_path, img2_path, corr1, corr2):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    corr1 = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]

    assert len(corr1) == len(corr2)

    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]

    display = cv2.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              flags=4
                              )
    return display


def main():
    """The main function."""
    model_path = os.path.join('../model', 'gl3d/sift-4000/model_best.pth')
    img1_name, img2_name = 'test_img1.jpg', 'test_img2.jpg'

    detector = ExtractSIFT(8000)
    lm = LearnedMatcher(model_path, inlier_threshold=1, use_ratio=0, use_mutual=0)
    kpt1, desc1 = detector.run(img1_name)
    kpt2, desc2 = detector.run(img2_name)
    _, corr1, corr2 = lm.infer([kpt1, kpt2], [desc1, desc2])
    display = draw_match(img1_name, img2_name, corr1, corr2)
    cv2.imwrite(img1_name+"-"+img2_name+".png", display)

    cv2.imshow("display", display)
    cv2.waitKey()


if __name__ == "__main__":
    main()
