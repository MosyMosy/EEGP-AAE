import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotUniform

from dataset import Dataset

class Encoder(tf.Module):
    def __init__(self, latent_space_size, num_filters, kernel_size, strides, batch_norm, gen_obj_code, name='encoder'):
        super(Encoder, self).__init__(name=name)
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm
        self._gen_obj_code = gen_obj_code

    @property
    def latent_space_size(self):
        return self._latent_space_size

    @tf.function
    def encoder_layers(self, x):
        layers = []
        layers.append(x)

        for filters, stride in zip(self._num_filters, self._strides):
            padding = 'same'
            x = layers.Conv2D(
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                activation=tf.nn.relu,
                name='ecd_{:d}'.format(_id)
            )(x)
            if self._batch_normalization:
                x = layers.BatchNormalization()(x, training=self._is_training)
            layers.append(x)
        return layers

    @tf.function
    def encoder_out(self, x):
        x = self.encoder_layers(x)[-1]
        encoder_out = tf.layers.flatten(x)
        return encoder_out

    @tf.function
    def z(self, x):
        x = self.encoder_out(x)
        z = layers.Dense(
            units=self._latent_space_size,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            name=None
        )(x)
        return z

    def __call__(self, x, is_training=False):
        self._is_training = is_training
        return self.z(x)

class Decoder(tf.Module):
    def __init__(self, reconstruction_shape, num_filters, kernel_size, strides,
                 auxiliary_mask, batch_norm, name='decoder'):
        super(Decoder, self).__init__(name=name)
        self._reconstruction_shape = reconstruction_shape
        self._auxiliary_mask = auxiliary_mask
        if self._auxiliary_mask:
            self._xmask = None
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._batch_normalization = batch_norm

    @tf.function
    def __call__(self, latent_code, is_training=False):
        z = latent_code
        h, w, c = self._reconstruction_shape[0:3]
        print(h, w, c)
        layer_dimensions = [[h // np.prod(self._strides[i:]), w // np.prod(self._strides[i:])] for i in range(0, len(self._strides))]
        print(layer_dimensions)

        if c != 1:
            with tf.name_scope('decoder_bgr'):
                x = layers.Dense(
                    units=layer_dimensions[0][0] * layer_dimensions[0][1] * self._num_filters[0],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                )(z)
                if self._batch_normalization:
                    x = layers.BatchNormalization()(x, training=self._is_training)
                x = tf.reshape(x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0]])

                for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
                    x = tf.image.resize(x, size=layer_size, method='nearest')
                    x = layers.Conv2D(
                        filters=filters,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        activation=tf.nn.relu
                    )(x)
                    if self._batch_normalization:
                        x = layers.BatchNormalization()(x, training=self._is_training)

                x = tf.image.resize(x, size=[h, w], method='nearest')

                if self._auxiliary_mask:
                    self._xmask = layers.Conv2D(
                        filters=1,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        activation=tf.nn.sigmoid
                    )(x)

                x_bgr = layers.Conv2D(
                    filters=3,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    activation=tf.nn.sigmoid
                )(x)
        else:
            x_bgr = None

        if c != 3:
            with tf.name_scope('decoder_edge'):
                x = layers.Dense(
                    units=layer_dimensions[0][0] * layer_dimensions[0][1] * self._num_filters[0],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                )(z)
                if self._batch_normalization:
                    x = layers.BatchNormalization()(x, training=self._is_training)
                x = tf.reshape(x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0]])

                _id = len(layer_dimensions[1:])
                for filters, layer_size in zip(self._num_filters[1:], layer_dimensions[1:]):
                    x = tf.image.resize(x, size=layer_size, method='nearest')
                    x = layers.Conv2D(
                        filters=filters,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        activation=tf.nn.relu
                    )(x)
                    _id -= 1
                    if self._batch_normalization:
                        x = layers.BatchNormalization()(x, training=self._is_training)

                x = tf.image.resize(x, size=[h, w], method='nearest')

                if self._auxiliary_mask:
                    self._xmask = layers.Conv2D(
                        filters=1,
                        kernel_size=self._kernel_size,
                        padding='same',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(),
                        activation=tf.nn.sigmoid
                    )(x)

                activation_final = None
                x_edge = layers.Conv2D(
                    filters=1,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    activation=activation_final
                )(x)
                x_edge_vis = tf.nn.sigmoid(x_edge)
        else:
            x_edge = None
            x_edge_vis = None

        return {'x_bgr': x_bgr,
                'x_edge': x_edge,
                'x_edge_vis': x_edge_vis}

def build_encoder(args, gen_obj_code=False):
    LATENT_SPACE_SIZE = args.getint('Network', 'LATENT_SPACE_SIZE')
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_ENCODER = args.getint('Network', 'KERNEL_SIZE_ENCODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')
    encoder = Encoder(
        latent_space_size=LATENT_SPACE_SIZE,
        num_filters=NUM_FILTER,
        kernel_size=KERNEL_SIZE_ENCODER,
        strides=STRIDES,
        batch_norm=BATCH_NORM,
        gen_obj_code=gen_obj_code
    )
    return encoder

def build_decoder(args):
    NUM_FILTER = eval(args.get('Network', 'NUM_FILTER'))
    KERNEL_SIZE_DECODER = args.getint('Network', 'KERNEL_SIZE_DECODER')
    STRIDES = eval(args.get('Network', 'STRIDES'))
    AUXILIARY_MASK = args.getboolean('Network', 'AUXILIARY_MASK')
    BATCH_NORM = args.getboolean('Network', 'BATCH_NORMALIZATION')

    R_H = args.getint('Dataset', 'H')
    R_W = args.getint('Dataset', 'W')
    R_CO = args.getint('Dataset', 'CO')

    decoder = Decoder(
        reconstruction_shape=[R_H, R_W, R_CO],
        num_filters=list(reversed(NUM_FILTER)),
        kernel_size=KERNEL_SIZE_DECODER,
        strides=list(reversed(STRIDES)),
        auxiliary_mask=AUXILIARY_MASK,
        batch_norm=BATCH_NORM,
    )

    return decoder

def build_dataset(dataset_path, fg_path_format, codebook_path_format, list_objs, args):
    dataset_args = {k: v for k, v in
                    args.items('Dataset') +
                    args.items('Augmentation')}
    dataset = Dataset(dataset_path, fg_path_format, codebook_path_format, list_objs, **dataset_args)
    return dataset

class VectorQuantizerEMA(tf.Module):
    def __init__(self, embedding_dim, num_embeddings, epsilon=1e-5, name='vq_center'):
        super(VectorQuantizerEMA, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._epsilon = epsilon

        initializer = tf.keras.initializers.RandomNormal()
        self._w = tf.Variable(initial_value=initializer((embedding_dim, num_embeddings)), trainable=True, name='embedding')
        self._ema_cluster_size = tf.Variable(initial_value=tf.zeros(num_embeddings), trainable=False, name='ema_cluster_size')
        self._ema_w = tf.Variable(initial_value=self._w.numpy(), trainable=False, name='ema_dw')

    @tf.function
    def quantize(self, encoding_indices):
        w = tf.transpose(self._w, [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

    @tf.function
    def __call__(self, inputs, decay=0.99, temperature=0.07, encoding_1nn_indices=None, encodings=None, mask_roi=None, is_training=False):
        w = self._w

        if not (mask_roi is None):
            masked_w = w[:, mask_roi[0]:mask_roi[1]]
        else:
            masked_w = w

        input_shape = tf.shape(inputs)

        with tf.control_dependencies([tf.debugging.assert_equal(input_shape[-1], self._embedding_dim, message="Input dimension does not match embedding dimension")]):
            flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

        if encoding_1nn_indices is None:
            distances = -tf.matmul(tf.nn.l2_normalize(flat_inputs, axis=1), tf.nn.l2_normalize(masked_w, axis=0))
            encoding_1nn_indices = tf.argmax(-distances, 1)
            if not (mask_roi is None):
                encoding_1nn_indices += mask_roi[0]

        if encodings is None:
            encodings = tf.squeeze(tf.one_hot(encoding_1nn_indices, self._num_embeddings))
            print('VQEMA: Encodings 1NN shape', encoding_1nn_indices.shape, 'Encodings is ONE-HOT with shape', encodings.shape)

        encoding_1nn_indices = tf.reshape(encoding_1nn_indices, tf.shape(inputs)[:-1])

        quantized_1nn = self.quantize(encoding_1nn_indices)

        normalized_inputs = tf.nn.l2_normalize(flat_inputs, axis=1)
        normalized_w = tf.nn.l2_normalize(w, axis=0)
        e_multiply = tf.matmul(normalized_inputs, tf.stop_gradient(normalized_w)) / temperature

        flat_encodings = tf.reshape(encodings, [-1, self._num_embeddings])
        print('Flatten Encodings with Shape', flat_encodings.shape, '// Prediction shape', e_multiply.shape)
        e_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(flat_encodings), logits=e_multiply))

        if is_training:
            updated_ema_cluster_size = tfa.metrics.MovingAverage(
                'ema_cluster_size', decay=decay, dtype=tf.float32)(tf.reduce_sum(encodings, 0))
            print('Dw shape', normalized_inputs.shape, 'Encoding shape', encodings.shape)
            dw = tf.matmul(normalized_inputs, encodings, transpose_a=True)
            updated_ema_w = tfa.metrics.MovingAverage(
                'ema_dw', decay=decay, dtype=tf.float32)(dw)
            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)

            normalised_updated_ema_w = (updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
            with tf.control_dependencies([e_loss]):
                update_w = tf.assign(self._w, normalised_updated_ema_w)
                with tf.control_dependencies([update_w]):
                    loss = e_loss
        else:
            loss = e_loss
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return {'quantize_1nn': quantized_1nn,
                'loss': loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_1nn_indices}
