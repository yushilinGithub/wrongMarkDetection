import json
import argparse
from pathlib import Path
import cv2
import uuid
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
def main(args):
    inputPath = Path(args.input)
    sub_dirs = [dir for dir in inputPath.iterdir() if dir.is_dir()]
    outputPath = Path(args.output)
    if not outputPath.is_dir():
        outputPath.mkdir()
    results = {}
    for sub_dir in sub_dirs:
        inputAnnotation = sub_dir/"label.json"
        if not inputAnnotation.is_file():
            raise "{} doesn't exist".format(inputAnnotation)
        with open(inputAnnotation,"r",encoding="utf-8") as f:
            annotations = json.load(f)
        for imageID,value in annotations.items():
            filename = value["filename"]
            image = cv2.imread(str(sub_dir/filename))
            if image is None:
                continue
            if args.renameImage:
                filename = str(uuid.uuid4())+".jpg"
            cv2.imwrite(str(outputPath/filename),image)

            results[filename] = {}
            results[filename]["bbox"] = []
            results[filename]["width"] = image.shape[1]
            results[filename]["height"] = image.shape[0]

            regionlist = value["regions"]
            for region in regionlist:
                results[filename]["bbox"].append({"location":{"left":region["shape_attributes"]["x"],
                                                            "top":region["shape_attributes"]["y"],
                                                            "right":region["shape_attributes"]["x"]+region["shape_attributes"]["width"],
                                                            "bottom":region["shape_attributes"]["y"]+region["shape_attributes"]["height"]},
                                                "type":0
                                                    })
    
    coco = convert_to_coco(results)
    with open(outputPath/"coco.json","w") as f:
        json.dump(coco,f,indent=4)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str,default="D:\\workspace\\wrongSymbolDetection\\data")
    parser.add_argument("--process",type=str,default="to_coco")
    parser.add_argument("--output",type=str,default ="D:\\workspace\\wrongSymbolDetection\\coco_data")
    parser.add_argument("--renameImage",type=str,default=True)
    args = parser.parse_args()
    main(args)