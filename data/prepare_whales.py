"""Split the WHALES dataset into train/dev/test and resize images to 64x64.

Original images have different size.
Resizing to (64, 64) reduces the dataset size , and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we have a lot of unique images here so we have to make a larger dataset by applying rotation, increasing, flip.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import os
import sys

sys.path.extend(['..'])

import PIL
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.utils import get_args
from utils.config import process_config


def agment_and_save(filename, output_dir, size):
    """Resize the image contained in `filename` and save it to the `output_dir` and augment it"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    # augment images based on quantity in accordant class
    lists = pd.read_csv('./data_set/train.csv')
    transform = pd.read_csv('./data_set/results.csv')
    ids = lists[lists['Image'] == os.path.basename(filename)]['Id'].values[0]
    random.seed()
    # print('!!!!!!!!!!!!!!!! {}'.format(transform[transform['Id'] == ids]['checked'].values[0]))

    # I will set true to images already cheked, not to repeate process again for new image
    # Also value should be positive. Means number of images less than 100
    if (transform[transform['Id'] == ids]['size_add'].values[0] >= 0) & (transform[transform['Id'] == ids]['checked'].values[0] != True):
        transform[transform['Id'] == ids]['checked'] = True
        for i in range(transform[transform['Id'] == ids]['size_add'].values[0]):
            deg = random.randint(0, 180)
            temp_image = image.rotate(deg)
            # randomly flip image 
            if random.choice([1, 0]):
                temp_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            # random crop of part of image
            if random.choice([1, 0]):
                temp_image.crop((0, 10, size, size-10))
            temp_image.save(os.path.join(output_dir, str(i) + '_' + filename.split('\\')[-1]))

def main(image_size):
    # Define the data directories
    train_data_dir = os.path.join('data_set', 'train')
    test_data_dir = os.path.join('data_set', 'test')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # fliped, rotated = processImage(filenames)
    # images = filenames + fliped + rotated

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    # Define output dir
    output_dir = 'READY_WHALES'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print('Warning: output dir {} already exists"'.format(output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(output_dir, '{}_whales'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print('Warning: dir {} already exists'.format(output_dir_split))

        print('Processing {} data, saving preprocessed data to {}'.format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            agment_and_save(filename, output_dir_split, image_size)

    print('Done building dataset')

# def processImage(filenames):
#     flipped = []
#     rotated = []
#     for i in filenames:
#         print("Image {}".format(i))
#         # flipped.append(np.flipud(Image.open(i)))
#         rotated.append(Image.open(i).rotate(40))
#     print("This is the end")
#     return flipped, rotated


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config.image_size)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
        print('Using default 64x64 size')
        main(224)
