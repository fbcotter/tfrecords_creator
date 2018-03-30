# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

    data_dir/label_0/image0.jpeg
    data_dir/label_0/image1.jpg
    ...
    data_dir/label_1/weird-image.jpeg
    data_dir/label_1/my-image.jpeg
    ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

    train_directory/train-00000-of-01024
    train_directory/train-00001-of-01024
    ...
    train_directory/train-01023-of-01024

and

    validation_directory/validation-00000-of-00128
    validation_directory/validation-00001-of-00128
    ...
    validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

    image/encoded: string containing JPEG encoded image in RGB colorspace
    image/height: integer, image height in pixels
    image/width: integer, image width in pixels
    image/colorspace: string, specifying the colorspace, always 'RGB'
    image/channels: integer, specifying the number of channels, always 3
    image/format: string, specifying the format, always 'JPEG'
    image/filename: string containing the basename of the image file
        e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
    image/class/label: integer specifying the index in a classification layer.
        The label ranges from [0, num_labels] where 0 is unused and left as
        the background class.
    image/class/text: string specifying the human-readable version of the label
        e.g. 'dog'

If your data set involves bounding boxes, please look at build_imagenet_data.py.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import sys
import threading
import random

import six
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    if isinstance(value, six.string_types):
        value = six.binary_type(value, encoding='utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, synset, human, bbox,
                        height, width):
    """Build an Example proto for an example.

    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        synset: string, unique WordNet ID specifying the label, e.g.,
            'n02323233'
        human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
        bbox: list of bounding boxes; each box is a list of integers
            specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
            to the same label as the image label.
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/synset': _bytes_feature(synset),
        'image/class/text': _bytes_feature(human),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature([label] * len(xmin)),
        'image/format': _bytes_feature(image_format),
        'image/filename': _bytes_feature(os.path.basename(filename)),
        'image/encoded': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data. Should be able to
        # handle PNG and CMYK data now
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _process_image(filename, coder):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)
    height = image.shape[0]
    width = image.shape[1]

    return image_data, height, width


def _process_image_files_thread(coder, thread_index, ranges, name, filenames,
                                synsets, humans, labels, bboxes, num_shards,
                                output_directory):
    """Processes and saves list of images as a TFRecord shard. This is the thread
    function that handlels 1 or more shards.

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within [0,
            len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
            analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        synsets: list of strings; each string is a unique WordNet ID.
        humans: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        bboxes: list of bounding boxes for each image. Note that each entry in
            this list might contain from 0+ entries corresponding to the number
            of bounding box annotations for the image. Can be None if not using
            bboxes.
        num_shards: integer number of shards for this data set.
        output_directory: str;
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g.
        # 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]
            synset = synsets[i]
            human = humans[i]
            bbox = bboxes[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image_buffer, label,
                                          synset, human, bbox,
                                          height, width)
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: ' % (datetime.now(), thread_index) +
                      'Processed %d of %d images in thread batch.' %
                      (counter, num_files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def create_tfrecords(name, filenames, labels, humans, num_shards,
                     synsets=None, bboxes=None, output_dir='/tmp',
                     num_threads=2):
    """Process and save list of images as TFRecord of Example protos.

    Args:
        name: string, unique identifier specifying the data set. E.g. train or
            val
        filenames: list of strings; each string is a path to an image file
        synsets: list of strings; each string is a unique WordNet ID. If you're
            not using ImageNet, can just be a list of the class strings for each
            image.
        labels: list of integer; each integer identifies the ground truth
        humans: list of strings; each string is a human-readable label
        num_shards: integer number of shards for this data set.
        bboxes: list of bounding boxes for each image. Note that each entry in
            this list might contain from 0+ entries corresponding to the number
            of bounding box annotations for the image. Can be None or an empty
            list to not use bounding boxes
        output_dir: str; The path to store the output sharded data
        num_threads: int; How many threads to spin up to load the images.
    """
    if bboxes is None:
        bboxes = [[] for _ in range(len(labels))]
    if synsets is None:
        synsets = ['' for _ in range(len(labels))]

    assert len(filenames) == len(synsets)
    assert len(filenames) == len(humans)
    assert len(filenames) == len(labels)
    assert len(filenames) == len(bboxes)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image
    # codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                synsets, humans, labels, bboxes, num_shards, output_dir)
        t = threading.Thread(target=_process_image_files_thread, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def find_image_files(data_dir, label_order=None):
    """Build a list of all images files and labels in the data set.

    Args:
        data_dir: string, path to the root directory of images.
            Assumes that the image data set resides in JPEG files located in
            the following directory structure.

                data_dir/dog/another-image.JPEG
                data_dir/dog/my-image.jpg

            where 'dog' is the label associated with these images.
        label_order: list or str or None; indicates the order for which the
            labels should be enumerated. If None, will use alphabetical order on
            the data_dir. If a list, will use that, if a str, will interpret as
            a filename.

    Returns:
        filenames: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    if label_order is None:
        label_order = os.listdir(data_dir)
    elif isinstance(label_order, str):
        label_order = [l.strip() for l in tf.gfile.FastGFile(
            label_order, 'r').readlines()]
    elif isinstance(label_order, tuple) or isinstance(label_order, list):
        pass
    else:
        raise ValueError("Unkown parameter type label_order")

    # Make sure each label is a directory
    for label in label_order:
        if not os.path.isdir(os.path.join(data_dir, label)):
            print('{} not found as a directory in the data_dir'.format(label) +
                  '. Dropping')
            label_order.remove(label)

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for text in label_order:
        jpeg_file_path = '%s/%s/*' % (data_dir, text)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(labels)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(label_order), data_dir))

    return filenames, texts, labels
