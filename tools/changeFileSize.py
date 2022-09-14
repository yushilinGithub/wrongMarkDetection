import json
import os
import cv2
def main():
    path = "/home/public/yushilin/wrongSynbol/data/ready_to_train"
    with open(os.path.join(path,"coco.json"),"r",encoding="utf-8") as f:
        data = json.load(f)
    for imageInfo in data["images"]:
        imageData = cv2.imread(os.path.join(path,imageInfo["file_name"]))
        width = imageData.shape[1]
        height = imageData.shape[0]
        if imageInfo["height"]!=height or imageInfo["width"]!=width:
            imageInfo["height"] = height
            imageInfo["width"] = width
            print(imageInfo["file_name"])
    
    with open(os.path.join(path,"coco2.json"),"w",encoding="utf-8") as f:
        json.dump(data,f)

if __name__ == "__main__":
    main()