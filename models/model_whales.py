from base.base_model import BaseModel
import tensorflow as tf


class SignsModel(BaseModel):
    def __init__(self, data_loader, config):
        super(SignsModel, self).__init__(config)
        # Get the data_generators to make the joint of the inputs in the graph
        self.data_loader = data_loader

        # define some important variables
        self.x = None
        self.y = None
        self.is_training = None
        self.out_argmax = None

        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        """

        :return:
        """

        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y = self.data_loader.get_inputs()

            assert self.x.get_shape().as_list() == [None, self.config.image_size, self.config.image_size, 3]

            self.is_training = tf.placeholder(tf.bool, name='Training_flag')


        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """
        out = self.x
        filters = [self.config.num_filters, self.config.num_filters * 2, self.config.num_filters * 4, self.config.num_filters * 8]

        with tf.variable_scope('network'):
            for i, c in enumerate(filters):
                with tf.variable_scope('block_{}'.format(i + 1)):
                    out = tf.layers.conv2d(out, c, 3, padding='same')
                    if self.config.use_batch_norm:
                        out = tf.layers.batch_normalization(out, momentum=self.config.bn_momentum, training=self.is_training)
                    out = tf.nn.relu(out)
                    out = tf.layers.max_pooling2d(out, 2, 2)

            assert out.get_shape().as_list() == [None, 4, 4, self.config.num_filters * 8]

            out = tf.reshape(out, [-1, 4 * 4 * self.config.num_filters * 8])

            with tf.variable_scope('fc_1'):
                out = tf.layers.dense(out, self.config.num_filters * 8)
                if self.config.use_batch_norm:
                    out = tf.layers.batch_normalization(out, momentum=self.config.bn_momentum, training=self.is_training)
                out = tf.nn.relu(out)

            with tf.variable_scope('out'):
                self.out = tf.layers.dense(out, self.config.num_classes, name='out')

                tf.add_to_collection('out', self.out)

        """
        Some operators for the training process
        """
        with tf.variable_scope('predictions'):
            self.predictions = tf.argmax(self.out, 1, name='predictions')
            tf.add_to_collection('predictions', self.predictions)

        with tf.variable_scope('metrics'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.out)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(self.y, self.predictions), tf.float32))

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

            if self.config.use_batch_norm:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            else:
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)
        tf.add_to_collection('train', self.acc)



    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
