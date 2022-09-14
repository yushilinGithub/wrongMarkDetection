import json
import os
import shutil
def main():
    path = "/home/public/yushilin/wrongSynbol/data/finish_status0_2_modelSelect_all_short_for_train/"
    backup = "/home/public/yushilin/wrongSynbol/data/backup"
    with open(os.path.join(path,"coco.json"),"r") as f:
        data = json.load(f)
    images = [image["file_name"] for image in data["images"]]
    for name in os.listdir(path):
        if name.endswith("json"):
            continue
        if name not in images:
            print(name)
            shutil.move(os.path.join(path,name),backup)
if __name__ == "__main__":
    main()