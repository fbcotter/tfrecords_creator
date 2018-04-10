import sys
import os
import pytest
import tensorflow as tf

TEST_BASE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(TEST_BASE, '..'))
from build_image_data import create_tfrecords, find_image_files
from load_image_data import read_shards


def setup():
    dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)


def test_find(capsys):
    data_dir = os.path.join(TEST_BASE, 'example_ds1')
    filenames, texts, labels = find_image_files(data_dir)
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
    filenames, texts, labels = find_image_files(data_dir)
    dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')

    # Remove the old files in the dest_dir
    for file in os.listdir(dest_dir):
        file_path = os.path.join(dest_dir, file)
        if os.path.isfile(file_path) and 'train' in file:
            os.unlink(file_path)

    create_tfrecords('train', filenames, labels, texts,
                     num_shards=num_shards, output_dir=dest_dir,
                     num_threads=num_threads)

    assert len(os.listdir(dest_dir)) == num_shards


def test_createloadtrain():
    data_dir = os.path.join(TEST_BASE, 'example_ds1')
    filenames, texts, labels = find_image_files(data_dir)
    dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')

    # Create the records if they weren't already there.
    create_tfrecords('train', filenames, labels, texts,
                     num_shards=4, output_dir=dest_dir,
                     num_threads=2)

    assert len(os.listdir(dest_dir)) == 4

    files = [os.path.join(dest_dir, x) for x in os.listdir(dest_dir)]
    preprocessor = lambda x: tf.image.resize_images(x, [224,224])
    images, labels, texts = read_shards(files, preprocessor, 6)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x,y,t = sess.run((images, labels, texts))


@pytest.mark.parametrize("num_shards, num_threads",
                         [(2,2)])
def test_bboxes(num_shards, num_threads):
    pass
    #  data_dir = os.path.join(TEST_BASE, 'example_ds1')
    #  filenames, texts, labels = find_image_files(data_dir)
    #  dest_dir = os.path.join(TEST_BASE, 'example_ds1_tfrecords')

    #  # Remove the old files in the dest_dir
    #  for file in os.listdir(dest_dir):
        #  file_path = os.path.join(dest_dir, file)
        #  if os.path.isfile(file_path) and 'train' in file:
            #  os.unlink(file_path)

    #  create_tfrecords('train', filenames, labels, texts,
                     #  num_shards=num_shards, output_dir=dest_dir,
                     #  num_threads=num_threads)

    #  assert len(os.listdir(dest_dir)) == num_shards
