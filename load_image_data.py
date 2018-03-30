from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def record_parser(value, preprocessor=None, max_classes=-1):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/channels':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    #  height, width = parsed['image/height'], parsed['image/width']
    #  image = tf.image.decode_jpeg(
        #  tf.reshape(parsed['image/encoded'], shape=[]), 3)
    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if preprocessor is not None:
        image = preprocessor(image)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32)

    text = parsed['image/class/text']

    return image, label, text


def read_shards(shardnames, preprocessor, batch_size, is_training=False,
                buffer_size=1500, num_epochs=1):
    """Input function which provides batches for train or eval.

    The data_dir, batch_size and num_epochs parameters can be overwritten if
    they are captured by the ingredient.

    Parameters
    ----------
    shardnames: list(str)
        List of the files to load. If want to shuffle, do before passing
    preprocessor: callable
        Function to apply to each image as they're being loaded. Should accept a
        3d tensor and return a 3d tensor.
    batch_size: int
        How big the batch size should be
    """
    dataset = tf.data.Dataset.from_tensor_slices(shardnames)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value, preprocessor),
                          num_parallel_calls=5)
    dataset = dataset.prefetch(batch_size)
    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels, texts = iterator.get_next()

    return images, labels, texts
