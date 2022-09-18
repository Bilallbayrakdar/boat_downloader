import cv2 as cv
import os
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", type=str, default="./datasets", help="Path to the input dataset.")
    parser.add_argument("--path_out", type=str, default="./datasets/merged", help="Path to the output dataset.")
    parser.add_argument("--n", type=int, default=2, help="Number of augmentations.")
    parser.add_argument("--m", type=int, default=9, help="Magnitude of augmentations.")
    parser.add_argument("--copy_test", default='store_true', default=False, help="Copy test images.")
    args = parser.parse_args()

    in_path = args.path_in
    out_path = args.path_out
    N = args.n
    M = args.m

    image_path = f"{in_path}/imgs/training"
    images = [f"{image_path}/{img}" for img in os.listdir(image_path) if img.endswith('.jpg')]

    if not os.path.exists(out_path): os.mkdir(out_path)
    transformation = transforms.Compose([transforms.RandAugment(N, M),transforms.ToTensor()])

    for img in tqdm(images):
        image = transformation(Image.open(img)).numpy()*255
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        cv.imwrite(f"{out_path}/{img.split('/')[-1]}", cv.cvtColor(image, cv.COLOR_BGR2RGB))
