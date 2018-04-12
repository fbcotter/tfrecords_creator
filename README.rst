Image TFRecords Creator
=======================

This repo is a standalone repository to create sharded Tensorflow Records from
a dataset on disk. It is based off the inception data prep in
http://github.com/tensorflow/models. They have some nice scripts in there but
they use bazel and are quite restrictive in general. This repo aims at being
a commandline tool to accept a location of any dataset and shard it.

Usage
-----
Usage is split into two stages: conversion and loading. Given the right work
done in the conversion stage, the loading stage should be relatively
straightforward.

Conversion
``````````
This stage converts your dataset of images to shards of TFRecords data. It
assumes you have folders of images organized as follows::
    
    ├── train
    │   ├── label1
    │   │   ├── train_image_0001.jpg
    │   │   ├── train_image_0002.jpg
    │   │   ├── train_image_0003.jpg
    │   │   └── train_image_0004.jpg
    │   ├── label2
    │   │   ├── train_image_0001.jpg
    │   │   ├── train_image_0002.jpg
    │   │   ├── train_image_0003.jpg
    │   │   └── train_image_0004.jpg
    │   ├── label3
    │   │   └── images
    │   │       ├── train_image_0001.jpg
    │   │       ├── train_image_0002.jpg
    │   │       ├── train_image_0003.jpg
    │   │       └── train_image_0004.jpg
    │   ├── ...
    │ 
    └── val
        ├── label1
        │   ├── image_0001.jpg
        │   ├── image_0002.jpg
        ├── label2
        │   ├── image_0001.jpg
        │   ├── image_0002.jpg
        ├── ...
         
We see the hierarchy is as follows:

1. Two or three folders for train/val/test in your dataset folder
2. Inside each sub-dataset folder, multiple folders representing the classes.
   The names of the folders can either be the label names themselves (e.g.
   'dog', 'cat', 'car', ...) or can be WordNet identifiers (as is the case in
   ImageNet, e.g. 'n01737021', 'n02091831', ...).
3. Inside each folder there is a collection of images belonging to the same
   class. Each image must have an extension like ".jpg", ".jpeg" or ".png".
   These should either be directly below the class folder, or in a sub-folder
   called 'images' (as is the case for the `tiny-imagenet`__ dataset).

__ https://tiny-imagenet.herokuapp.com/

TFRecord Format
~~~~~~~~~~~~~~~
Taking a look at the function `_convert_to_example` in `build_image_data.py` we
can see the format of the saved images:

.. code:: python

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

Things like the image height, width, and channels can be determined
automatically from the image file. However there are some fields which need
a little guidance from the user. In particular:

1. image/class/text - str - This is the human readable label. In the default
   case, this will come from the sub-directory names containing the images.
2. image/class/label - int - The enumeration of the texts to class integers. In
   the default case this is done alphabetically, with the label 0 being reserved
   for the unknown class. I.e. counting starts from 1. If you want to change
   this order, see the docstring for `find_image_files` (the label_order field).
3. image/class/synset - str - The synset id for the class. Can be blank strings
   if not using wordnet, or can be 'n001440764', ...
4. image/object/bbox - Bounding boxes for the images. Used for object
   recognition. Needs to be saved in another file and fed as a list to the
   `create_tfrecords` function.

For more information on these fields, see the Advanced Running section.

Running
~~~~~~~
Running is only a matter of calling the `find_image_files` function to return
the data for all the images, then calling the `create_tfrecords` function to
turn this data into TFRecords files.

Let us call the path to the dataset DATADIR. Say we want to put the dataset in
OUTDIR. The simplest thing to do would be to put it all in a single shard. Then
the code to run would be:

.. code:: python
    
    from build_image_data import find_image_files, create_tfrecords
    filenames, texts, labels, enumeration = find_image_files(DATADIR)
    create_tfrecords('train', filenames, texts, labels, output_dir=OUTDIR)


The 'train' string as the first argument is the prefix on the output shards. Of
course this could be whatever you choose it to be. So our output directory would
now have one file in it like so::

    └── OUTDIR
        └── train-00001-of-00001

This example isn't alltogether too useful for later on loading, as we would like to
shard our data so we can load it in parallel. I.e. we would like to store the
dataset in multiple large files rather than one enormous file. To speed up the
writing of the dataset, we can use multiple threads to write these files as
well. For simplicity, one thread can only write an integer number of shards. Now
we can expand the above example by trying:

.. code:: python
    
    from build_image_data import find_image_files, create_tfrecords
    filenames, texts, labels, enumeration = find_image_files(DATADIR)
    create_tfrecords('train', filenames, labels, texts, output_dir=OUTDIR,
    num_shards=4, num_threads=2)

Here, 2 threads are spun up to read-and-write image files, and each one will
write 2 shards. Our output directory will now look like::

    └── OUTDIR
        ├── train-00001-of-00004
        ├── train-00002-of-00004
        ├── train-00003-of-00004
        └── train-00004-of-00004

Advanced Running
~~~~~~~~~~~~~~~~
In the Format section, we also talked about changing the label order, using
bboxes and potentially WordNet for our dataset. Here are some examples of how to
do these things.

1. Changing the label order.
   Say if our train folder has three labels called 'cat', 'dog', and 'emu'. The
   default enumeration would be to set 'cat' to label 1, 'dog' to 2 and 'emu' to
   3. If we want to change this, we can manually create the order in a list and
   pass it to the `find_image_files` function. I.e.

   .. code:: python
        
       from build_image_data import find_image_files, create_tfrecords
       label_order = ['emu', 'cat', 'dog']
       filenames, texts, labels, enumeration = find_image_files(DATADIR, label_order)
       print(enumeration)
       # Prints: {'n/a': 0, 'emu': 1, 'cat': 2, 'dog': 3}
       create_tfrecords('train', filenames, texts, labels, output_dir=OUTDIR,
           num_shards=4, num_threads=2)

   Here we've specified the order as a list. The enumeration return value then
   gives the mapping from folder name to label. We can specify this directly
   ourselves by providing a dictionary instead.

   .. code:: python
        
       from build_image_data import find_image_files, create_tfrecords
       label_order = {'emu': 2, 'cat': 3, 'dog': 1}
       filenames, texts, labels, enumeration = find_image_files(DATADIR, label_order)
       print(enumeration)
       # Prints: {'emu': 2, 'cat': 3, 'dog': 1}

2. Using WordNet
   If you are using WordNet folder names, or for any other reason you want to
   map your directory names to other text labels, then you can do this with
   a simple dictionoary

   .. code:: python
        
       from build_image_data import find_image_files, create_tfrecords
       filenames, texts, labels, enumeration = find_image_files(DATADIR)
       text_mappings = {'emu': 'Big Australian Bird', 'cat': 'Feline', 
           'dog': 'woofer'}
       create_tfrecords('train', filenames, texts, labels,
           text_mappings=text_mappings, output_dir=OUTDIR)

   Now when you load the data, the 'text' field will be 'Big Australian Bird',
   and the 'synset' field will be 'emu'. 

3. Bounding Boxes
   Bounding boxes can be saved in a large number of different formats. For
   simplicity, we leave the parsing of the bounding box raw data up to the user.
   If the data is to be saved alongside the image, it must be given to the
   `create_tfrecords` function in a specific format, e.g.:

   .. code:: python

       bboxes = {'img1_name': {
                    'labels': [obj1_label, obj2_label, ojb3_label],
                    'bboxes': [[xmin1, ymin1, xmax1, ymax1],
                               [xmin2, ymin2, xmax2, ymax2],
                               [xmin3, ymin3, xmax3, ymax3]]
                   },
                 'img2_name': {
                     ...
                 }
       }

    There is some flexibility on the exact format of the fields:
    
    - The first level dictionary keys ('img1_name', 'img2_name') can be either
      unique names or the full path to the image file.
    - The labels can either be integers or strings. If strings, will assume they
      match the folder names, and the label_order field will be used to map
      these to the correct integers.
    - The bounding box lists can either be floats in the range of 0 to 1, or
      integers representing the pixel values. Pixel values will by default be
      converted to floats in the range 0 to 1.

Tests
-----

Loading from TFRecords
----------------------

   .. code:: python
        
       from build_image_data import read_shards
       tfrecords_files = [os.path.join(DEST_DIR, x) for x in
           os.listdir(DEST_DIR)]
       preprocessor = lambda x: tf.image.resize_images(x, [224,224])
       examples = read_shards(tfrecords_files, preprocessor, batch_size=64)


