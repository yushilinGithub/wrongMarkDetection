
"""
Dyhead Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys
import itertools
import logging
import time
import cv2
import json
# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

from typing import Any, Dict, List, Set
import time
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_setup
import numpy as np
from dyhead import add_dyhead_config
from extra import add_extra_config
from pathlib import Path

def image(imageID,width,height,file_name):
    image = {}
    image["height"] = height
    image["width"] = width
    image["id"] = imageID
    image["file_name"] = file_name
    image['license']= 0
    image['date_captured'] = ''
    return image

def category(row):
    category = {}
    category["supercategory"] = 'None'
    category["id"] = row.categoryid
    category["name"] = row[2]
    return category

def annotation(bbox,boxID,file_name):
    annotation = {}
    location = bbox["location"]
    area = (location["right"] -location["left"])*(location["bottom"] - location["top"])
    annotation["segmentation"] = [[location["left"],location["top"],location["right"],location["top"],location["right"],location["bottom"],location["left"],location["bottom"]]]
    annotation["iscrowd"] = 0
    annotation["area"] = area
    annotation["image_id"] = file_name

    annotation["bbox"] = [location["left"], location["top"], location["right"]-location["left"],location["bottom"] - location["top"]]

    annotation["category_id"] = bbox["type"]
    annotation["id"] = boxID
    return annotation

def convert_to_coco(result):
    images = []
    categories = []
    annotations = []

    category = {}
    category["supercategory"] = 'none'
    category["id"] = 0
    category["name"] = 'wrong'
    categories.append(category)
    boxID = 1
    for imageID,(imageName,pred_info) in enumerate(result.items()):
        for bbox in pred_info["bbox"]:  #{'type':0,'location': {'left': 630, 'top': 1508, 'right': 764, 'bottom': 1569}, 'score': 0.5009988}
            annotations.append(annotation(bbox,boxID,imageID+1))
            boxID = boxID+1
        images.append(image(imageID+1,width=pred_info["width"],height=pred_info["height"],file_name=imageName))
    
    data_coco = {}
    data_coco["info"]={"year": 2022,
                         "version": "1.0", 
                        "description": "VIA project exported to COCO format using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)", 
                        "contributor": "",
                        "url": "http://www.robots.ox.ac.uk/~vgg/software/via/",
                         "date_created": "Tue Mar 01 2022 11:12:33 GMT+0800 (China Standard Time)"}
    data_coco["licenses"] = [{'id': 0, 'name': 'Unknown License', 'url': ''}]
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations
    return data_coco


class default_argument():
    config = "configs/dyhead_swint_atss_fpn_2x_ms_short.yaml"
    num_gpus = 1


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_dyhead_config(cfg)
    add_extra_config(cfg)
    cfg.merge_from_file(args.config)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class OkayQuesCut():
    def __init__(self):
        args = default_argument()
        cfg = setup(args)
        self.predictor = DefaultPredictor(cfg)
        if hasattr(torch.cuda,"set_per_process_memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(0.3, 0) 
    def __call__(self, image: np.ndarray):
        boxes = []
        outputs = self.predictor(image)
        instances = outputs["instances"]
        confident_detections = instances[instances.scores > 0.35]

        for box in confident_detections.pred_boxes.__iter__():
            # boxes.append([int(i) for i in box.cpu().tolist()])
            locs = [int(i) for i in box.cpu().tolist()]
            if locs and len(locs) == 4:
                boxes.append({"type":0,
                            "location":{
                                        'left': locs[0],
                                        'top': locs[1],
                                        'right': locs[2],
                                        'bottom': locs[3]
                                      }            
                            })
        return boxes


okay_cut = OkayQuesCut()


def main():
    
    dir = "/home/public/yushilin/wrongSynbol/data/finish_status0_2/"
    output = "/home/public/yushilin/wrongSynbol/data/finish_status0_2_modelSelect_all_short_for_train"
    if not os.path.isdir(output):
        os.mkdir(output)
    results = {}
    print(dir)
    times = []
    for imageName in tqdm.tqdm(os.listdir(dir)[5000:20000]):
        
        if  imageName.endswith("json"):
            continue 
        imagePath = os.path.join(dir,imageName)
        image = cv2.imread(imagePath)
        if image is not None:
            time1 = time.time()
            outputs = okay_cut(image)
            if len(outputs)>=1:
                time2 =time.time()
                times.append(time2-time1)
                results[imageName] = {}
                results[imageName]["bbox"] = outputs
                results[imageName]["width"] = image.shape[1]
                results[imageName]["height"] = image.shape[0]
                cv2.imwrite(os.path.join(output,imageName),image)
        else:
            print(imagePath)

    print("mean time on {} images is {}".format(len(times),np.mean(times)))
    coco = convert_to_coco(results)
    with open(os.path.join(output,"coco.json"),"w",encoding="utf-8") as f:
        json.dump(coco,f)
if __name__ == "__main__":
    main()
