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
IMG_TYPES = ['*.jpg', '*.jpeg', '*.png',
             '*.JPG', '*.JPEG', '*.PNG']


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


def _convert_to_example(filename, image_buffer, img_label, synset, human,
                        bbox_coords, bbox_labels, height, width):
    """Build an Example proto for an example.

    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        synset: string, unique WordNet ID specifying the label, e.g.,
            'n02323233'
        human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
        bbox_coords: list of bounding boxes; each box is a list of floats
            specifying [xmin, ymin, xmax, ymax].
        bbox_labels: list of bounding box labels.
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    # Split the bounding boxes into 4 separate lists
    for b in bbox_coords:
        assert len(b) == 4
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]

    assert len(xmin) == len(bbox_labels)

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(colorspace),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(img_label),
        'image/class/synset': _bytes_feature(synset),
        'image/class/text': _bytes_feature(human),
        'image/object/number': _int64_feature(len(xmin)),
        'image/object/bbox/xmin': _float_feature(xmin),
        'image/object/bbox/xmax': _float_feature(xmax),
        'image/object/bbox/ymin': _float_feature(ymin),
        'image/object/bbox/ymax': _float_feature(ymax),
        'image/object/bbox/label': _int64_feature(bbox_labels),
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


def _parse_bbox_info(bboxes, filename, enumeration, height, width):
    file = os.path.basename(filename)
    entry = None
    bbox_labels = []
    bbox_coords = []
    if filename in bboxes.keys():
        entry = bboxes[filename]
    if file in bboxes.keys():
        entry = bboxes[file]

    if entry is not None:
        # entry['labels'] could be a list of folder names, or a list of
        # integers. If it is the former, need to map these to the correct
        # integer labels.
        bbox_labels = [enumeration[x] if isinstance(x, str) else x for x in
                       entry['labels']]

        # enrty['bboxes'] could be a list of floats or integers. If a list of
        # integers, need to map these to the range [0,1]
        bbox_coords = []
        for box in entry['bboxes']:
            if isinstance(box[0], float):
                bbox_coords.append(box)
            else:
                xmin = box[0] / width
                ymin = box[1] / height
                xmax = box[2] / width
                ymax = box[3] / height
                # Sometimes xmax < xmin. Check for this. Also ensure that the
                # bounds does not go outside the image
                min_x = min(xmin, xmax)
                max_x = max(xmin, xmax)
                min_x = min(max(min_x, 0.0), 1.0)
                max_x = min(max(max_x, 0.0), 1.0)

                min_y = min(ymin, ymax)
                max_y = max(ymin, ymax)
                min_y = min(max(min_y, 0.0), 1.0)
                max_y = min(max(max_y, 0.0), 1.0)

                bbox_coords.append([min_x, min_y, max_x, max_y])

    return bbox_coords, bbox_labels


def _process_image_files_thread(coder, thread_index, ranges, name, filenames,
                                texts, text_mappings, labels, bboxes,
                                num_shards, output_directory, enumeration=None):
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
        texts: list of strings; each string is human readable, e.g. 'dog'
        text_mappings: Dictionary to map folder names to some other text (e.g.
            to map wordnet IDs to their descriptions.
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

    # Print out the maximum number of bboxes seen
    max_bboxes = 0
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
            if texts[i] in text_mappings.keys():
                synset = texts[i]
                human = text_mappings[synset]
            else:
                synset = texts[i]
                human = texts[i]

            try:
                image_buffer, height, width = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            # Get the bounding box info from the dict passed.
            bbox_coords, bbox_labels = _parse_bbox_info(
                bboxes, filename, enumeration, height, width)
            if len(bbox_coords) > max_bboxes:
                max_bboxes = len(bbox_coords)

            example = _convert_to_example(filename, image_buffer, label,
                                          synset, human, bbox_coords,
                                          bbox_labels, height, width)
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
    print('%s [thread %d]: Maximum bbox length for this thread was %d' %
          (datetime.now(), thread_index, max_bboxes))
    sys.stdout.flush()


def create_tfrecords(name, filenames, texts, labels, text_mappings=None,
                     num_shards=2, num_threads=2, bboxes=None,
                     enumeration=None, output_dir='/tmp'):
    """Process and save list of images as TFRecord of Example protos.

    Args:
        name: string, unique identifier specifying the data set. E.g. train or
            val
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is a human-readable label
        labels: list of integer; each integer identifies the ground truth
        text_mappings: Dictionary to map folder names to some other text (e.g.
            to map wordnet IDs to their descriptions.
        num_shards: integer number of shards for this data set.
        num_threads: int; How many threads to spin up to load the images.
        bboxes: dictionary of bounding boxes for the images. The keys of this
            dictionary indicate the image names (either their full path or just
            their filename if this is unique). Each item in this dictionary is
            again a dictionary of two keys, 'labels' - a list of labels for each
            bbox, and 'bboxes' - a list of coordinates for each bbox. Note that
            each entry in this list might contain from 0+ entries corresponding
            to the number of bounding box annotations for the image. Can be None
            to not use bboxes.
        enumeration: dict mapping strings to integers
        output_dir: str; The path to store the output sharded data
    """
    if text_mappings is None:
        text_mappings = {}
    if bboxes is None:
        bboxes = {}

    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (num_threads, ranges))
    sys.stdout.flush()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image
    # codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, text_mappings, labels, bboxes, num_shards, output_dir,
                enumeration)
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
            the data_dir. If a list, will use that as the order. Can also be a
            dict, with key, val pairs giving the mapping from string to integer.

    Returns:
        filenames: list of strings; each string is a path to an image file.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth.
        enumeration: mapping from folder names to label integers
    """
    print('Determining list of input files and labels from %s.' % data_dir)

    special_order = False
    if label_order is None:
        label_order = os.listdir(data_dir)
    elif isinstance(label_order, str):
        label_order = [l.strip() for l in tf.gfile.FastGFile(
            label_order, 'r').readlines()]
    elif isinstance(label_order, tuple) or isinstance(label_order, list):
        pass
    elif isinstance(label_order, dict):
        enumeration = label_order
        special_order = True
        import operator
        # Sort the dictionary by values
        label_order = sorted(enumeration.items(), key=operator.itemgetter(1))
        label_order = [a[0] for a in label_order]
    else:
        raise ValueError("Unkown parameter type label_order")

    # Make sure each label is a directory - drop those that aren't
    # Below is an order(n) method of doing this (see the discussion here
    # https://stackoverflow.com/a/8313120/6437741)
    i = 0
    keep = [os.path.isdir(os.path.join(data_dir, l)) for l in label_order]
    for k, l in zip(keep, label_order):
        if k:
            label_order[i] = l
            i += 1
        else:
            print('{} not found as a directory in the data_dir'.format(l) +
                  '. Dropping')
    del label_order[i:]

    labels = []
    filenames = []
    texts = []

    # Leave label index 0 empty as a background class.
    if not special_order:
        enumeration = {'n/a': 0}

    # Construct the list of JPEG files and labels.
    for idx, text in enumerate(label_order):
        matching_files = []
        for files in IMG_TYPES:
            jpeg_file_path = '%s/%s/%s' % (data_dir, text, files)
            matching_files.extend(tf.gfile.Glob(jpeg_file_path))
            # Add files that are in an images subfolder of the dataset
            try:
                jpeg_file_path2 = '%s/%s/images/%s' % (data_dir, text, files)
                matching_files.extend(tf.gfile.Glob(jpeg_file_path2))
            except tf.errors.NotFoundError:
                pass

        if special_order:
            label_index = enumeration[text]
        else:
            label_index = idx + 1

        enumeration[text] = label_index
        labels.extend([label_index] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        if not idx % 100:
            print('Finished finding files in %d of %d classes.' % (
                idx, len(labels)))

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

    # Check the input enumeration matches the output enumeration
    if special_order:
        pass

    return filenames, texts, labels, enumeration
