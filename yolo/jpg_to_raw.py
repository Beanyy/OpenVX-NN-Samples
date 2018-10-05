from PIL import Image
import numpy as np
import argparse
from skimage import io, transform

def PreprocessImage(path, show_img=True):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # resize
    resized_img = transform.resize(img, (416, 416))

    # convert to numpy.ndarray
    sample = np.asarray(resized_img, dtype=np.float32)
    sample = np.transpose(sample, (2, 0, 1))

    return np.reshape(sample, (1, 3, 416, 416))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVX Yolo input pre-processor')
    parser.add_argument('img_name', type=str, help='jpg image')
    parser.add_argument('processed_img', type=str, help='raw input to net')
    args = parser.parse_args()

    pix = PreprocessImage(args.img_name)
    pix.tofile(args.processed_img)
