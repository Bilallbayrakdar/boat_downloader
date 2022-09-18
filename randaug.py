import cv2 as cv
import os
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", type=str, default="./datasets/boats", help="Path to the input dataset.")
    parser.add_argument("--path_out", type=str, default="./datasets/boats_randaugment", help="Path to the output dataset.")
    parser.add_argument("--n", type=int, default=2, help="Number of augmentations.")
    parser.add_argument("--m", type=int, default=9, help="Magnitude of augmentations.")
    parser.add_argument("--copy_others", default='store_false', help="Copy test images.")
    args = parser.parse_args()

    in_path = args.path_in
    out_path = args.path_out
    N = args.n
    M = args.m
    copy_others = args.copy_others

    image_path = f"{in_path}/imgs/training"
    images = [f"{image_path}/{img}" for img in os.listdir(image_path) if img.endswith('.jpg')]

    if not os.path.exists(out_path): os.mkdir(out_path)
    if not os.path.exists(f"{out_path}/imgs"): os.mkdir(f"{out_path}/imgs")
    if not os.path.exists(f"{out_path}/imgs/training"): os.mkdir(f"{out_path}/imgs/training")
    if not os.path.exists(f"{out_path}/annotations"): os.mkdir(f"{out_path}/annotations")
    if not os.path.exists(f"{out_path}/annotations/training"): os.mkdir(f"{out_path}/annotations/training")

    transformation = transforms.Compose([transforms.RandAugment(N, M),transforms.ToTensor()])

    print("Augmenting images...")
    for img in tqdm(images):
        image = transformation(Image.open(img)).numpy()*255
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
        cv.imwrite(f"{out_path}/imgs/training/{img.split('/')[-1]}", cv.cvtColor(image, cv.COLOR_BGR2RGB))

    print("Copying annotations...")
    ann_path = f"{in_path}/annotations/training"
    annotations = [f"{ann_path}/{ann}" for ann in os.listdir(ann_path) if ann.endswith('.txt')]
    for ann in tqdm(annotations): copyfile(ann, f"{out_path}/annotations/training/{ann.split('/')[-1]}")

    if copy_others:
        print("Copying test images...")
        if not os.path.exists(f"{out_path}/imgs/test"): os.mkdir(f"{out_path}/imgs/test")
        if not os.path.exists(f"{out_path}/annotations/test"): os.mkdir(f"{out_path}/annotations/test")

        image_path = f"{in_path}/imgs/test"

        images = [f"{image_path}/{img}" for img in os.listdir(image_path) if img.endswith('.jpg')]
        for img in tqdm(images): copyfile(img, f"{out_path}/imgs/test/{img.split('/')[-1]}")

        print("Copying test annotations...")
        ann_path = f"{in_path}/annotations/test"
        annotations = [f"{ann_path}/{ann}" for ann in os.listdir(ann_path) if ann.endswith('.txt')]
        for ann in tqdm(annotations): copyfile(ann, f"{out_path}/annotations/test/{ann.split('/')[-1]}")

        
