import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt
import createMask
import argparse
from seamlessCloningPoisson import seamlessCloningPoisson
import cv2


def update_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--foreground", type=str, required=True)
    parser.add_argument("--background", type=str, required=True)
    parser.add_argument("--mask", type=str, default="")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--x", type=int, default=200)
    parser.add_argument("--y", type=int, default=200)
    return parser


def get_cfg():
    parser = argparse.ArgumentParser()
    parser = update_parser(parser)
    config = parser.parse_args()
    return config


if __name__ == "__main__":
    cfg = get_cfg()
    if cfg.mask:
        mask = cv2.imread(cfg.mask, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 5, 1, 0)
    else:
        mask = createMask.main(cfg.foreground)
    targetImg = np.array(Image.open(cfg.background.encode()).convert('RGB'))
    sourceImg = np.array(Image.open(cfg.foreground.encode()).convert('RGB'))
    resultImg = seamlessCloningPoisson(sourceImg, targetImg, mask, cfg.x, cfg.y)
    if cfg.out:
        plt.imsave(cfg.out)
    plt.imshow(resultImg)
    plt.show()
