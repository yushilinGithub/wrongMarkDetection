from __future__ import annotations
import shutil
import numpy as np
import json
import argparse
from pathlib import Path 
import shutil
from PIL import Image
#合并分批标注的数据,并将包含真阳性和假阳性的图片保存到目标目录下,coco格式的标注保存到目标目录下,命名为coco.json
def main(args):
    
    input_path = Path(args.input_data)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise Exception("{} doesn't exist")
    if not output_path.is_dir():
        output_path.mkdir()

    batchPaths = [child for child in input_path.iterdir() if child.is_dir()]
    imageID = []
    annID = []
    result_coco = {}
    max_image_id = 0
    max_annotation_id = 0
    for batchPath in batchPaths:
        print("batchPath",batchPath)
        machinePredict = batchPath/"machinePredict.json"
        humanLabel = batchPath/"humanLabel.json"
        if (not machinePredict.exists()) or (not humanLabel.exists()):

            with open(output_path/"coco.json","w",encoding="utf-8") as f:
                json.dump(result_coco,f)
            
            print("result label saved to {}".format(output_path/"coco.json"))
            print("{} images with usable wrong sign".format(len(imageID)))

            raise Exception("predicted label or human annotations in {} doesn't exist".format(batchPath))
        with open(machinePredict,"r",encoding="utf-8") as f:
            machinePredict_anns = json.load(f)
        with open(humanLabel,"r",encoding="utf-8") as f:
            humanLabel_anns = json.load(f)
            
        #只有一个类别，不同的标注文件可能存在区别
        if humanLabel_anns["categories"][0]["id"] != 0:
            for ann in humanLabel_anns["annotations"]:
                ann["category_id"] = 0
        #存在正样本的预测的图片的image_id
        machinePredict_imgID =np.unique([instance["image_id"] 
                                         for instance in machinePredict_anns["annotations"]])
       # 存在正样本的标注的图片的image_id
        humanLabel_imgID = np.unique([instance["image_id"] 
                                      for instance in humanLabel_anns["annotations"]])

        #依次叠加max_annotation_id和max_image_id 以免重复
        max_annotation_id += np.max([instance["id"] 
                                     for instance in humanLabel_anns["annotations"]])
        max_image_id += np.max([image["id"] 
                                 for image in humanLabel_anns["images"]])

            
        humanLabel_filenames = [image["file_name"] for image in humanLabel_anns["images"]]
        machinePredict_filenames_existsPositve = [image["file_name"] 
                                                  for image in machinePredict_anns["images"] 
                                                    if image["id"] in machinePredict_imgID 
                                                            and image["file_name"] in humanLabel_filenames]
        humanLabel_filenames_existsPositive = [image["file_name"] 
                                               for image in humanLabel_anns["images"]
                                                if image["id"] in humanLabel_imgID]
        #预测和标注同时存在正样本的
        filters = np.unique(machinePredict_filenames_existsPositve+humanLabel_filenames_existsPositive)
        new_images=[]
        for image in humanLabel_anns["images"]:
            if not args.filter_out:
                if args.copyImages and not (output_path/image["file_name"]).exists():
                    if (batchPath/image["file_name"]).exists():
                        shutil.copy(batchPath/image["file_name"],output_path/image["file_name"])
                    else:
                        print("{} not exist".format(batchPath/image["file_name"]))
                
            elif image["file_name"] in filters:
                IMG = Image.open(str(batchPath/image["file_name"]))
                image["height"] = IMG.size[1]
                image["width"] = IMG.size[0]
                new_images.append(image)
                if args.copyImages and not (output_path/image["file_name"]).exists():
                    shutil.copy(batchPath/image["file_name"],output_path/image["file_name"])
        if args.filter_out:
            humanLabel_anns["images"] = new_images

        if len(result_coco)==0:
            result_coco = humanLabel_anns
        else:
            for images in humanLabel_anns["images"]:
                images["id"] += int(max_image_id)
            for ann in humanLabel_anns["annotations"]:
                ann["id"] += int(max_annotation_id)
                ann["image_id"]+= int(max_image_id)

            if np.intersect1d(imageID,[image["id"] for image in humanLabel_anns["images"]]).shape[0]==0:
                result_coco["images"].extend(humanLabel_anns["images"])
            if np.intersect1d(annID,[ann["id"] for ann in humanLabel_anns["annotations"]]).shape[0]==0:
                result_coco["annotations"].extend(humanLabel_anns["annotations"])
        imageID.extend(humanLabel_imgID)
        annID.extend(np.unique([ann["id"] for ann in humanLabel_anns["annotations"]]))
    result_coco["categories"] = [{'supercategory': 'name', 'id': 0, 'name': 'wrong'}]
    #对图片的id 以及instance de id重新设置index
    if args.reindex_result:
        image_id_reindex = {image["id"]:index+1 for index,image in enumerate(result_coco["images"])}
        ann_id_reindex = {ann["id"]:index+1 for index,ann in enumerate(result_coco["annotations"])}
        for i in range(len(result_coco["images"])):
            result_coco["images"][i]["id"] = image_id_reindex[result_coco["images"][i]["id"]]
        for i in range(len(result_coco["annotations"])):
            result_coco["annotations"][i]["image_id"] = image_id_reindex[result_coco["annotations"][i]["image_id"]]
            result_coco["annotations"][i]["id"] = ann_id_reindex[result_coco["annotations"][i]["id"]]
            
    with open(output_path/"coco.json","w",encoding="utf-8") as f:
        json.dump(result_coco,f)
    
    print("result label saved to {}".format(output_path/"coco.json"))
    print("{} images in annotations".format(len(result_coco["images"])))
    print("{} images with usable wrong sign".format(len(imageID)))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",
                            type=str,
                            required=True,
                            help = "分批标注的数据,命名为batch_0、batch_1..., 文件夹下面有machinePredict.json,为机器预测的,humanLabel.json为人为标注的")
    parser.add_argument("--output_path",
                            type=str,
                            required=True,
                            help = "整理好的数据放在这个目录下,其中图片为人为标注存在正样本,以及包含机器预测的假阳性案例的图片")
    parser.add_argument("--copyImages",
                            type=str,
                            default=True,
                            help = "是否将图片拷贝到目标目录下面")
    parser.add_argument("--correctImageShape",
                        type=str,
                        default=True)
    parser.add_argument("--filter_out",
                        type=bool,
                        required=False,
                        help="删除掉在预测以及人工标注的图片中都没有正样本的图片")
    parser.add_argument("--reindex_result",
                        required=True,
                        action="store_true",
                        help = "重新更新index")
    args = parser.parse_args()
    main(args)