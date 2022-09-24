import os 
from glob import glob
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", type=str, default="./datasets", help="Path to the input dataset.")
    parser.add_argument("--path_out", type=str, default="./datasets/merged", help="Path to the output dataset.")
    args = parser.parse_args()

    # path = "./datasets"
    # new_path = "./datasets/merged"

    path = args.path_in
    new_path = args.path_out

    datasets = [f"{path}/{folder}" for folder in os.listdir(path) if not folder.startswith('.')]

    train_images = []
    train_annotations = []
    test_images = []
    test_annotations = []

    for dataset in datasets:
        train_images += glob(f"{dataset}/imgs/training/*.jpg")
        train_annotations += glob(f"{dataset}/annotations/training/*.txt")
        test_images += glob(f"{dataset}/imgs/test/*.jpg")
        test_annotations += glob(f"{dataset}/annotations/test/*.txt")   

    new_imgs_path = f"{new_path}/imgs"
    new_annotations_path = f"{new_path}/annotations"

    train_images = sorted(train_images)
    test_images = sorted(test_images)

    train_annotations = sorted(train_annotations)
    test_annotations = sorted(test_annotations)

    if not os.path.exists(new_path): os.mkdir(new_path)
    if not os.path.exists(new_imgs_path): os.mkdir(new_imgs_path)
    if not os.path.exists(new_annotations_path): os.mkdir(new_annotations_path)
    if not os.path.exists(f"{new_imgs_path}/training"): os.mkdir(f"{new_imgs_path}/training")
    if not os.path.exists(f"{new_imgs_path}/test"): os.mkdir(f"{new_imgs_path}/test")
    if not os.path.exists(f"{new_annotations_path}/training"): os.mkdir(f"{new_annotations_path}/training")
    if not os.path.exists(f"{new_annotations_path}/test"): os.mkdir(f"{new_annotations_path}/test")


    for i in range(len(train_images)):
        img = train_images[i]
        ann = train_annotations[i]
        shutil.copy(img, f"{new_imgs_path}/training/img_{i+1}.jpg")
        shutil.copy(ann, f"{new_annotations_path}/training/gt_img_{i+1}.txt")

    for i in range(len(test_images)):
        img = test_images[i]
        ann = test_annotations[i]
        shutil.copy(img, f"{new_imgs_path}/test/img_{i+1}.jpg")
        shutil.copy(ann, f"{new_annotations_path}/test/gt_img_{i+1}.txt")


