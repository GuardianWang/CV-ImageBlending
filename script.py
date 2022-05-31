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
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--foreground_scale", type=float, default=1.)
    return parser


def obj_crop_region(mask):
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)

    return x, y, w, h


def crop_foreground(foreground, mask):
    x, y, w, h = obj_crop_region(mask)
    foreground = foreground[y: y + h, x: x + w]
    mask = mask[y: y + h, x: x + w]
    return foreground, mask


def get_cfg():
    parser = argparse.ArgumentParser()
    parser = update_parser(parser)
    config = parser.parse_args()
    return config


def main():
    cfg = get_cfg()
    if cfg.mask:
        mask = cv2.imread(cfg.mask, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask > 5, 1, 0)
    else:
        mask = createMask.main(cfg.foreground)

    sourceImg = np.array(Image.open(cfg.foreground.encode()).convert('RGB'))
    mask = cv2.resize(mask, None, fx=cfg.foreground_scale, fy=cfg.foreground_scale, interpolation=cv2.INTER_NEAREST)
    sourceImg = cv2.resize(sourceImg, None, fx=cfg.foreground_scale, fy=cfg.foreground_scale, interpolation=cv2.INTER_CUBIC)
    sourceImg, mask = crop_foreground(sourceImg, mask)

    targetImg = np.array(Image.open(cfg.background.encode()).convert('RGB'))
    resultImg = seamlessCloningPoisson(sourceImg, targetImg, mask, cfg.x, cfg.y)
    if cfg.out:
        plt.imsave(cfg.out)
    plt.imshow(resultImg)
    plt.show()


if __name__ == "__main__":
    main()
