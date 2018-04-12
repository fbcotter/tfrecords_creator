import sys
import os
import pytest
import tensorflow as tf
import csv

TEST_BASE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(TEST_BASE, '..'))
from tfrecords_creator import create_tfrecords, find_image_files, read_shards


def setup():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    dest_dir = os.path.join(TEST_BASE, 'example_ds2_tfrecords')
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)


def clear_dir(dest_dir):
    #  Remove the old files in the dest_dir
    for file in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file)
        if os.path.isfile(file_path) and 'train' in file:
            os.unlink(file_path)


def test_find(capsys):
    data_dir = os.path.join(TEST_BASE, 'example_ds1')
    filenames, texts, labels, _ = find_image_files(data_dir)
    out, err = capsys.readouterr()
    warnstr = "redherring.txt not found as a directory in the " + \
        "data_dir. Dropping\n"
    assert warnstr in out
    print(filenames)
    print(texts)
    print(labels)


@pytest.mark.parametrize("num_shards, num_threads",
                         [(1,1),(2,2),(3,3),(4,2)])
def test_createtrain(num_shards, num_threads):
    data_dir = os.path.join(TEST_BASE, 'example_ds1')
    filenames, texts, labels, _ = find_image_files(data_dir)
    dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')

    clear_dir(dest_dir)
    create_tfrecords('train', filenames, texts, labels,
                     num_shards=num_shards, output_dir=dest_dir,
                     num_threads=num_threads)

    assert len(os.listdir(dest_dir)) == num_shards


def test_createloadtrain():
    data_dir = os.path.join(TEST_BASE, 'example_ds1')
    filenames, texts, labels, _ = find_image_files(data_dir)
    dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')

    # Create the records if they weren't already there.
    clear_dir(dest_dir)
    create_tfrecords('train', filenames, texts, labels,
                     num_shards=4, output_dir=dest_dir,
                     num_threads=2)

    assert len(os.listdir(dest_dir)) == 4

    files = [os.path.join(dest_dir, x) for x in os.listdir(dest_dir)]
    preprocessor = lambda x: tf.image.resize_images(x, [224,224])
    examples = read_shards(files, preprocessor, 6)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x,y,t = sess.run((examples.images, examples.labels, examples.texts))

@pytest.mark.parametrize("ds, preprocessor, result",
                         [('ds1', None, tf.errors.InvalidArgumentError),
                          ('ds1', lambda x: tf.image.resize_images(x, [224,224]), (224,224)),
                          ('ds2', None, (64,64)),
                          ('ds2', lambda x: tf.image.resize_images(x, [32,32]), (32, 32))])
def test_preprocessor(ds, preprocessor, result):
    if ds == 'ds1':
        data_dir = os.path.join(TEST_BASE, 'example_ds1')
        filenames, texts, labels, _ = find_image_files(data_dir)
        dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')
    else:
        data_dir = os.path.join(TEST_BASE, 'example_ds2')
        filenames, texts, labels, _ = find_image_files(data_dir)
        dest_dir = os.path.join(TEST_BASE, 'example_ds2_tfrecords')

    # Create the records if they weren't already there.
    clear_dir(dest_dir)
    create_tfrecords('train', filenames, texts, labels,
                     num_shards=2, output_dir=dest_dir,
                     num_threads=2)

    # Make sure the files are there
    assert len(os.listdir(dest_dir)) == 2

    tfrecords_files = [os.path.join(dest_dir, x) for x in os.listdir(dest_dir)]
    examples = read_shards(tfrecords_files, preprocessor, 6)
    if isinstance(result, tuple):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x = sess.run((examples.images))
            assert x.shape[1:3] == result
    else:
        # Check that it runs. Will pad the images naturally
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x = sess.run((examples.images))


@pytest.mark.parametrize("num_shards, num_threads",
                         [(2,2)])
def test_bboxes(num_shards, num_threads):
    data_dir = os.path.join(TEST_BASE, 'example_ds2')
    dest_dir = os.path.join(TEST_BASE, 'example_ds2_tfrecords')
    boxes_file = os.path.join(TEST_BASE, 'example_ds2', 'boxes.txt')

    #  Remove the old files in the dest_dir
    for file in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file)
        if os.path.isfile(file_path) and 'train' in file:
            os.unlink(file_path)

    # Open the csvfile of bounding box annotations
    with open(boxes_file) as csvfile:
        spam = csv.reader(csvfile, delimiter='\t')
        bboxes = {}
        for row in spam:
            if row[0] not in bboxes.keys():
                bboxes[row[0]] = {'labels': [], 'bboxes': []}
            bboxes[row[0]]['labels'].append(row[0].split('_')[0])
            bboxes[row[0]]['bboxes'].append(
                [int(row[1]), int(row[2]), int(row[3]), int(row[4])])

    # Get the file names and mappings
    filenames, texts, labels, enumerations = find_image_files(data_dir)
    # Create the records if they weren't already there.
    create_tfrecords('train', filenames, texts, labels,
                     num_shards=2, num_threads=2,
                     output_dir=dest_dir, bboxes=bboxes,
                     enumeration=enumerations)


def test_bboxes_load():
    data_dir = os.path.join(TEST_BASE, 'example_ds2')
    dest_dir = os.path.join(TEST_BASE, 'example_ds2_tfrecords')
    boxes_file = os.path.join(TEST_BASE, 'example_ds2', 'boxes.txt')

    #  Remove the old files in the dest_dir
    for file in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file)
        if os.path.isfile(file_path) and 'train' in file:
            os.unlink(file_path)

    with open(boxes_file) as csvfile:
        spam = csv.reader(csvfile, delimiter='\t')
        bboxes = {}
        for row in spam:
            if row[0] not in bboxes.keys():
                bboxes[row[0]] = {'labels': [], 'bboxes': []}
            bboxes[row[0]]['labels'].append(row[0].split('_')[0])
            bboxes[row[0]]['bboxes'].append(
                [int(row[1]), int(row[2]), int(row[3]), int(row[4])])

    filenames, texts, labels, enumerations = find_image_files(data_dir)

    # Create the records if they weren't already there.
    create_tfrecords('train', filenames, texts, labels,
                     num_shards=2, output_dir=dest_dir,
                     num_threads=2, enumeration=enumerations)
    tfrecords_files = [os.path.join(dest_dir, x) for x in os.listdir(dest_dir)]
    examples = read_shards(tfrecords_files, preprocessor=None, batch_size=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run((examples.images, examples.texts, examples.synsets, examples.num_bboxes, examples.bbox_coords, examples.bbox_labels))

