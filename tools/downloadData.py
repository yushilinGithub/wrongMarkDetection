import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import os
import requests

#从后台拉取数据
def save_image_from_url(url, output_folder):
    output_path = output_folder / url.split("/")[-1]
    if not output_path.exists():
        image = requests.get(url)
        with open(output_path, "wb") as f:
            f.write(image.content)

def load(csv_data, output_folder):    
    with concurrent.futures.ThreadPoolExecutor( max_workers=5) as executor:
        future_to_url = {
            executor.submit(save_image_from_url, url, output_folder): url
            for url in csv_data["img_url"].values
        }
        for future in concurrent.futures.as_completed(
            future_to_url
        ):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    "%r generated an exception: %s" % (url, exc)
                )


def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise "{} not exist".format(input_path)
    if not output_path.exists():
        output_path.mkdir()
    csv_data = pd.read_csv(input_path)
    print("data size {} ".format(csv_data.shape))
    if args.remove_downloaded:
        print("remove downloaded is set to true")
        downloaded = Path(args.already_download)
        if not downloaded.exists():
            raise "already_download csv file not exist ,{} ".format(args.already_download)
        already_downloaded = pd.read_csv(downloaded)
        common = csv_data.merge(already_downloaded,how="inner" ,on="img_url")
        if common.shape[0] == 0:
            print("no common image url exist")
        else:
            print("common shape ",common.shape)
            csv_data = csv_data[~csv_data.img_url.isin(common.img_url)]
            print("removed",csv_data.shape)
    load(csv_data,output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=str,default="/home/public/yushilin/wrongSynbol/data/Yearning_Data_entity_question_ocr-pre_status0_2.csv")
    parser.add_argument("--already_download",type=str,default="/home/public/yushilin/wrongSynbol/data/Yearning_Data.csv")
    parser.add_argument("--remove_downloaded",type=str,default=True)
    parser.add_argument('--output',type=str,default="/home/public/yushilin/wrongSynbol/data/finish_status0_2/")
    args = parser.parse_args()
    main(args)