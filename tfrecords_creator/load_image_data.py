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
        'image/class/synset':
            tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/object/number':
            tf.FixedLenFeature([], tf.int64, default_value=0),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)

    image = tf.image.decode_jpeg(parsed['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    height, width = parsed['image/height'], parsed['image/width']
    if preprocessor is not None:
        image = preprocessor(image)

    label = tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[]),
        dtype=tf.int32)
    text = parsed['image/class/text']
    synset = parsed['image/class/synset']

    # Load the bbox data
    num_bboxes = tf.cast(parsed['image/object/number'], tf.int32)
    xmin = tf.expand_dims(parsed['image/object/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(parsed['image/object/bbox/ymin'].values, 0)
    xmax = tf.expand_dims(parsed['image/object/bbox/xmax'].values, 0)
    ymax = tf.expand_dims(parsed['image/object/bbox/ymax'].values, 0)
    bbox_coords = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bbox_coords = tf.transpose(bbox_coords, [1, 0])

    bbox_labels = tf.sparse_tensor_to_dense(parsed['image/object/bbox/label'])

    return (image, height, width, label, text,
            synset, num_bboxes, bbox_coords, bbox_labels)


class Examples(object):
    """
    Returns a batch of data from a tfrecords file.

    For variable length items like the number of bboxes in an image, need to pad
    out the data to make it all fit in a batch. To do this, we specify the max
    number of likely bboxes to expect. If this is too small, a runtime error
    will be thrown later on.
    """
    def __init__(self, dataset, batch_size, max_bboxes=None):
        self.max_bboxes = max_bboxes
        self.batch_size = batch_size
        img_shape = [None, None, 3]
        height_shape = []
        width_shape = []
        label_shape = []
        text_shape = []
        synset_shape = []
        num_bbox_shape = []
        bbox_shape = [max_bboxes, 4]
        bbox_label_shape = [max_bboxes]

        self.padded_shapes = (img_shape, height_shape, width_shape, label_shape,
                              text_shape, synset_shape, num_bbox_shape,
                              bbox_shape, bbox_label_shape)

        dataset = dataset.padded_batch(self.batch_size, self.padded_shapes)
        iterator = dataset.make_one_shot_iterator()

        item = iterator.get_next()

        self._images = item[0]
        self._heights = item[1]
        self._widths = item[2]
        self._labels = item[3]
        self._texts = item[4]
        self._synsets = item[5]
        self._num_bboxes = item[6]
        self._bbox_coords = item[7]
        self._bbox_labels = item[8]

    @property
    def images(self):
        return self._images

    @property
    def heights(self):
        return self._heights

    @property
    def widths(self):
        return self._widths

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def texts(self):
        return self._texts

    @property
    def synsets(self):
        return self._synsets

    @property
    def num_bboxes(self):
        return self._num_bboxes

    @property
    def bbox_coords(self):
        return self._bbox_coords

    @property
    def bbox_labels(self):
        return self._bbox_labels


def read_shards(shardnames, preprocessor, batch_size, is_training=False,
                buffer_size=1500, num_epochs=1, max_bboxes=None):
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
    max_bboxes: int
        Need to pad variable length sequence for bounding boxes. This is the
        maximum number we'll expect in any one image.
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
    examples = Examples(dataset, batch_size, max_bboxes)

    return examples
