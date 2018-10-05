from PIL import Image
import numpy as np
import argparse
from skimage import io, transform

def PreprocessImage(path, show_img=True):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 299, 299
    resized_img = transform.resize(crop_img, (299, 299))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img, dtype=np.float32) * 256
    # swap axes to make image from (299, 299, 3) to (3, 299, 299)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - 128.
    normed_img /= 128.

    return np.reshape(normed_img, (1, 3, 299, 299))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InceptionV3 input pre-processor')
    parser.add_argument('img_name', type=str, help='jpg image')
    parser.add_argument('processed_img', type=str, help='raw input to net')
    args = parser.parse_args()

    pix = PreprocessImage(args.img_name)
    pix.tofile(args.processed_img)