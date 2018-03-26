import argparse
import os
import glob

import tensorflow as tf

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500
_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

parser = argparse.ArgumentParser()
parser.add_argument('--training', action='store_true',
                    help='Use train shards.')
parser.add_argument('--data_dir', type=str, default='data/',
                    help='The path to the shards directory.')
parser.add_argument('--batch_size', type=int, default=4,
                    help='The batch size.')
parser.add_argument('--num_epochs', type=int, default=-1,
                    help='The number of epochs to fetch data for.')


def filenames(is_training, data_dir):
    """Return filenames for dataset."""
    if is_training:
        return glob.glob(os.path.join(data_dir, 'train*'))
    else:
        return glob.glob(os.path.join(data_dir, 'val*'))


def record_parser(value, is_training):
    """Parse an ImageNet record from `value`."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/channels':
            tf.FixedLenFeature((), tf.int64, default_value=-1),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/text':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/class/synset':
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

    image = tf.image.decode_jpeg(
        tf.reshape(parsed['image/encoded'], shape=[]),
        _NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.image.resize_images(image, [128, 128])
    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32)

    return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input function which provides batches for train or eval."""
    dataset = tf.data.Dataset.from_tensor_slices(
        filenames(is_training, data_dir))

    if is_training:
        dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value, is_training),
                          num_parallel_calls=5)
    dataset = dataset.prefetch(batch_size)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    FLAGS, unparsed = parser.parse_known_args()

    # Create the input pipeline
    print('Starting...')
    images, labels = input_fn(FLAGS.training, FLAGS.data_dir, FLAGS.batch_size,
                              FLAGS.num_epochs)

    # Don't want to grab gpus for this demo
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = sess.run(images)
        print('After resizing (to enable batching), the returned result ' +
              'has shape {}'.format(x.shape))
