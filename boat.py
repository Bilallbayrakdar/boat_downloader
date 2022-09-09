import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np
from json import load as json_load
from typing import List, Tuple, Dict, Any, Union, Optional
from math import ceil
from pprint import pprint
import shutil
from numpy.random import default_rng
from argparse import ArgumentParser


def error_log(r_type, path, image, bb1, bb2, gt_text, pred_text):
    print(f"ERROR: {r_type}-> {path}")
    plt.title("image:")
    plt.imshow(image)
    plt.show()
    try:
        plt.title(f"gt_crop:{gt_text}")
        plt.imshow(image[bb1[1]:bb1[3], bb1[0]:bb1[2]])
        plt.show()
    except:
        print(bb1)
    try:
        plt.title(f"pred_crop:{pred_text}")
        plt.imshow(image[bb2[1]:bb2[3], bb2[0]:bb2[2]])
        plt.show()
    except:
        print(bb2)


def convert_rectangle(bbox):
    x1 = bbox[:,0].min()
    y1 = bbox[:,1].min()
    x2 = bbox[:,0].max()
    y2 = bbox[:,1].max()

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return [x1, y1, x2, y2, x1, y2, x2, y1]


def asciify(string): return string.replace('ğ','g').replace('ı','i').replace('ö','o').replace('ü','u').replace('ş','s').replace('ç','c')

def build_data(path:str, batch_size:int=8, verbose:bool=False):
    """
    Builds the data for the model.
    
    Parameters
    path: str 
        Path to the folders of images.
    save_path: str  
        Path to save the crops in TP,FP,FN folders.
    weights: List[str]
        List of weights to use. First element is the path of the detector weights, 
        second element is the path of the recognizer weights.
    threshold: float
        Threshold for the IoU.
    batch_size: int
        Batch size for the model.
    verbose: bool
        If True, prints the progress.
    """

    labellings = {f"{path}/{folder}":{} for folder in os.listdir(f"{path}/") if not (folder.startswith('.') or folder.endswith(".zip") or folder.endswith("out"))}

    gt = {}

    for labelling in labellings:
        print(f"labelling:{labelling}")
        labellings[labelling]["image_names"] = [image for image in os.listdir(labelling) if image.endswith('.jpg')]
        labellings[labelling]["labels"] = [json_load(open(f"{labelling}/{label}")) for label in os.listdir(labelling) if label.endswith('.json') and not label.endswith('json.json')][0]
        
    print("Iterating over folders...")
    for labelling in labellings:
        empty_images = []
        print(f"{labelling}")
        gt_labels_bboxes = {}
        for i in range(0,ceil(len(labellings[labelling]["image_names"])/batch_size)):
            b_idx = i*batch_size
            e_idx = min((i+1)*batch_size, len(labellings[labelling]["image_names"]))

            for name in labellings[labelling]["image_names"][b_idx:e_idx]:
                k = [e for e in list(labellings[labelling]['labels']['_via_img_metadata'].keys()) if name in e][0]
                temp = labellings[labelling]['labels']['_via_img_metadata'][k]
                empty = [x for x in temp["regions"] if x['shape_attributes'].get('all_points_x',[]) != []]
                    
                if len(empty) == len(temp['regions']):
                    temp = {x['region_attributes'].get('boat_name', x['region_attributes'].get('other_text', '')):np.stack((x['shape_attributes']['all_points_x'],x['shape_attributes']['all_points_y']), axis=-1) for x in temp["regions"]}
                    # temp = {asciify(k.lower()):convert_rectangle(v) for k,v in temp.items() if k != ''}
                    temp = {asciify(k.lower()):v.reshape(-1).tolist() for k,v in temp.items() if k != ''}
                    gt_labels_bboxes[name] = temp
                        
                else:
                    print(f"Empty image: {name}")
                    gt_labels_bboxes[name] = None
        
            labellings[labelling]['gt_labels_bboxes'] = gt_labels_bboxes

        for label in labellings[labelling]['gt_labels_bboxes']: gt[label] = labellings[labelling]['gt_labels_bboxes'][label]

    return gt

def convert2gt(x):
    if x == {}: return ""

    v = list(x.items())
    labels = []
    values = []
    gt = ""

    for k,v in v:
        v = str(v).replace("[","").replace("]","")
        gt += f"{v}, {k}\n"

    return gt


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--download", type=int, default=0, help="Download the dataset.")
    parser.add_argument("--path_in", type=str, default="./images", help="Path to the input dataset.")
    parser.add_argument("--path_out", type=str, default="./data", help="Path to the output dataset.")

    args = parser.parse_args()

    if args.download ==1:
        os.system("sh boat_data.sh")

    src = args.path_in
    dst = args.path_out
    
    ## Parse the boat data

    labellings = build_data(src)
    folders = [entity for entity in os.listdir(src)]

    images = [f"{src}/{folder}/{name}" for folder in folders for name in os.listdir(f"{src}/{folder}") if name.endswith(".jpg")]

    rng = default_rng()
    train = rng.choice(len(images), size=int(len(images)*.8), replace=False)
    test = np.array(list(set(range(len(images))).difference(set(train))))


    ## Convert the data to the format to ICDAR and save it

    if not os.path.exists(dst): os.mkdir(dst)
    if not os.path.exists(f"{dst}/train"): os.mkdir(f"{dst}/train")
    if not os.path.exists(f"{dst}/train/img"): os.mkdir(f"{dst}/train/img")
    if not os.path.exists(f"{dst}/train/gt"): os.mkdir(f"{dst}/train/gt")
    train_txt = ""

    for i in train:
        image = images[i]
        print(image)
        shutil.copyfile(f"{image}", f"{dst}/train/img/{i}.jpg")
        with open(f"{dst}/train/gt/{i}.txt", "w+") as f: 
            f.write(convert2gt(labellings[image.split("/")[-1]]))

        train_txt += f"{dst}/train/img/{i}.jpg\t{dst}/train/gt/{i}.txt\n"

    with open(f"{dst}/train.txt", "w+") as f: f.write(train_txt)

    if not os.path.exists(dst): os.mkdir(dst)
    if not os.path.exists(f"{dst}/test"): os.mkdir(f"{dst}/test")
    if not os.path.exists(f"{dst}/test/img"): os.mkdir(f"{dst}/test/img")
    if not os.path.exists(f"{dst}/test/gt"): os.mkdir(f"{dst}/test/gt")

    test_txt = ""

    for i in test:
        image = images[i]
        print(image)
        shutil.copyfile(f"{image}", f"{dst}/test/img/{i}.jpg")
        with open(f"{dst}/test/gt/{i}.txt", "w+") as f: 
            f.write(convert2gt(labellings[image.split("/")[-1]]))

        test_txt += f"{dst}/test/img/{i}.jpg\t{dst}/test/gt/{i}.txt\n"

    with open(f"{dst}/test.txt", "w+") as f: f.write(train_txt)




