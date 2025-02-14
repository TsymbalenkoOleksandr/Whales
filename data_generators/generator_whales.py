"""Input pipeline for the SIGNS dataset.

The filenames have format "{label}_IMG_{id}.jpg".
For instance: "data_dir/2_IMG_4584.jpg".
"""

import os
import sys

sys.path.extend(['..'])

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

from utils.utils import get_args
from utils.config import process_config


class WhalesTfLoader:
    def __init__(self, config):
        self.config = config

        data_dir = os.path.join('..', 'data', 'READY_WHALES')
        label_dir = os.path.join('..', 'data', 'data_set', 'train.csv')
        train_dir = os.path.join(data_dir, 'train_whales')
        eval_dir = os.path.join(data_dir, 'dev_whales')
        test_dir = os.path.join(data_dir, 'test_whales')

        # Get the file names from the train and dev sets
        self.train_filename_path = np.array([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')])
        self.eval_filename_path = np.array([os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith('.jpg')])
        self.test_filename_path = np.array([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')])

        self.train_filenames = np.array([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
        self.eval_filenames = np.array([f for f in os.listdir(eval_dir) if f.endswith('.jpg')])
        self.test_filenames = np.array([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

        # To read labels of whales in respect to id of image from csv file
        self.labeles = pd.read_csv(label_dir)

        # Define datasets sizes
        self.train_size = self.train_filename_path.shape[0]
        self.eval_size = self.eval_filename_path.shape[0]
        self.test_size = self.test_filename_path.shape[0]

        # Define number of iterations per epoch
        self.num_iterations_train = (self.train_size + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_eval = (self.eval_size + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test = (self.test_size + self.config.batch_size - 1) // self.config.batch_size

        self.encoder = preprocessing.LabelEncoder()

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self._build_dataset_api()


    @staticmethod
    def _parse_function(filename, label, size):
        """Obtain the image from the filename (for both training and validation).

        The following operations are applied:
            - Decode the image from jpeg format
            - Convert to float and to range [0, 1]
        """
        image_string = tf.read_file(filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)

        resized_image = tf.image.resize_images(image, [size, size])

        return resized_image, label


    @staticmethod
    def _train_preprocess(image, label, use_random_flip, mode='train'):
        """Image preprocessing for training.

        Apply the following operations:
            - Horizontally flip the image with probability 1/2
            - Apply random brightness and saturation
        """
        if mode == 'train':
            if use_random_flip:
                image = tf.image.random_flip_left_right(image)

            image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

            # Make sure the image is still in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label


    def _build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.string, [None, ])
            self.labels_placeholder = tf.placeholder(tf.int64, [None, ])
            self.mode_placeholder = tf.placeholder(tf.string, shape=())

            # Create a Dataset serving batches of images and labels
            # We don't repeat for multiple epochs because we always train and evaluate for one epoch
            parse_fn = lambda f, l: self._parse_function(f, l, self.config.image_size)
            train_fn = lambda f, l: self._train_preprocess(f, l, self.config.use_random_flip, self.mode_placeholder)

            self.dataset = (tf.data.Dataset.from_tensor_slices(
                    (self.features_placeholder, self.labels_placeholder)
                )
                .map(parse_fn, num_parallel_calls=self.config.num_parallel_calls)
                .map(train_fn, num_parallel_calls=self.config.num_parallel_calls)
                .batch(self.config.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

            # Create reinitializable iterator from dataset
            self.iterator = self.dataset.make_initializable_iterator()

            self.iterator_init_op = self.iterator.initializer

            self.next_batch = self.iterator.get_next()

    def initialize(self, sess, mode='train'):
        if mode == 'train':
            idx = np.array(range(self.train_size))
            np.random.shuffle(idx)

            self.train_filename_path = self.train_filename_path[idx]

            self.encoder.fit(self.labeles['Id'].tolist())
            self.labeles['Id_encoded'] = self.encoder.transform(self.labeles['Id'].tolist())
            print(self.labeles['Id_encoded'].shape)

            mas1 = []
            mas2 = []
            # mas3 = []

            for i in self.train_filenames:
                mas1.append(self.labeles['Id_encoded'][self.labeles['Image'] == i].tolist()[0])

            for i in self.eval_filenames:
                mas2.append(self.labeles['Id_encoded'][self.labeles['Image'] == i].tolist()[0])

            # print(self.test_filenames)
            # for i in self.test_filenames:
            #     mas3.append(self.labeles['Id_encoded'][self.labeles['Image'] == i].tolist()[0])

            self.train_labels = mas1
            self.eval_labels = mas2
            # self.test_labels = mas3

            # print('!!!! {}'.format(min(self.train_labels)))

            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.train_filename_path,
                                                   self.labels_placeholder: self.train_labels,
                                                   self.mode_placeholder: mode})
        elif mode == 'eval':
            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.eval_filename_path,
                                                   self.labels_placeholder: self.eval_labels,
                                                   self.mode_placeholder: mode})
        else:
            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.eval_filename_path,
                                                   self.labels_placeholder: self.eval_labels,
                                                   self.mode_placeholder: mode})


    def get_inputs(self):
        return self.next_batch


def main(config):
    """
    Function to test from console
    :param config:
    :return:
    """
    tf.reset_default_graph()

    sess = tf.Session()


    data_loader = WhalesTfLoader(config)

    images, labels = data_loader.get_inputs()

    print('Train')
    data_loader.initialize(sess, mode='train')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)

    print('Eval')
    data_loader.initialize(sess, mode='eval')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)

    print('Test')
    data_loader.initialize(sess, mode='test')

    out_im, out_l = sess.run([images, labels])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
