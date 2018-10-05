from PIL import Image
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InceptionV3 output processor')
    parser.add_argument('file_name', type=str, help='file name')
    args = parser.parse_args()

    pred = np.fromfile(args.file_name, dtype=np.float32)

    classes = []
    with open('labels.txt') as f:
        classes = f.readlines()

    topk = np.argsort(pred)[::-1]
    for idx in topk[:5]:
        print('({}) {}'.format(pred[idx], classes[idx][9:].replace('\n', '')))
