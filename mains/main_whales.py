import sys
import logging

sys.path.append('..')

import tensorflow as tf

from data_generators.generator_whales import WhalesTfLoader
from models.model_whales import SignsModel
from trainers.trainer_whales import WhalesTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import DefinedSummarizer
from utils.utils import get_args


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])

    # Create tensorflow session and limit gpu usage
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth=True
    sess = tf.Session(config=conf)

    # create your data generator
    data_loader = WhalesTfLoader(config)

    # create instance of the model you want
    model = SignsModel(data_loader, config)


    # create tensorboard logger
    logger = DefinedSummarizer(sess, summary_dir=config.summary_dir,
                               scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
                                            'eval/loss_per_epoch','eval/acc_per_epoch'])

    # create trainer and path all previous components to it
    trainer = WhalesTrainer(sess, model, config, logger, data_loader)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
