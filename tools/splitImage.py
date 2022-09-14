from ast import parse
import imp
import shutil
import argparse
from pathlib import Path
def main(args):
    inputPath = Path(args.inputPath)
    outputPath = Path(args.outputPath)
    if not outputPath.exists():
        outputPath.mkdir()
    i=0
    for filePath in inputPath.iterdir():
        if str(filePath).endswith(".jpg") or str(filePath).endswith(".png") or str(filePath).endswith(".jpeg"):
            batch_name = "batch_{}".format(i//2000)
            dstPath = outputPath/batch_name
            if not dstPath.exists():
                dstPath.mkdir()
            else:
                shutil.copy(filePath,dstPath)
                i=i+1
        else:
            print(filePath)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputPath",type=str,default="/home/public/yushilin/wrongSynbol/data/finish_status0_1")
    parser.add_argument("--outputPath",type=str,default="/home/public/yushilin/wrongSynbol/data/finish_status0_1_splited")
    args = parser.parse_args()
    main(args)