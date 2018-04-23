# coding: utf-8
import csv
from tfrecords_creator import find_image_files, create_tfrecords
import glob
import os
import json

TI_DIR = '/scratch/share/Tiny_ImageNet'
OUT_DIR = '/scratch/share/Tiny_ImageNet_tfrecords'

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load the wordnet mappings
with open(os.path.join(TI_DIR, 'words.txt')) as csvfile:
    spam = csv.reader(csvfile, delimiter='\t')
    mappings = {}
    for line in spam:
        mappings[line[0]] = line[1]

# Prepare a global train bbox file. Currently, each class has their own bbox
# file with an example row looking like (tab delimited):
#
#       n07873807_0.JPEG	0	6	63	61
#
# We need to find all 200 of these files, read them in and spit out the data in
# one big file
#
#       n07873807_0.JPEG	n07873807	0	6	63	61
bbox_files = glob.glob(os.path.join(TI_DIR, 'train/n*/*.txt'))
train_outfile = os.path.join(TI_DIR, 'train', 'bboxes.txt')
print('Found {} bbox files'.format(len(bbox_files)))
outfile = open(train_outfile, 'w')
for f in bbox_files:
    data = open(f, 'r').read()
    outfile.write(data)
outfile.close()


# Load the train bbox file and put this into a dictionary. The dictionary format
# will be:
#   bboxes[image_file] = {'labels': [bbox1_label, bbox2_label],
#                         'bboxes': [[x1start, y1start, x1end, y1end],
#                                    [x2start, y2start, x2end, y2end]]}
#
# Of course, all the tiny imagenet files only have a single bbox in them, but
# the interface is meant to accommodate more. The image_file keys can be the
# full relative path from the train directory, or can be just the filenames
# themselves. The latter assumes that the filenames are unique across the
# dataset.
bboxes = {}
csvfile = open(os.path.join(TI_DIR, 'train', 'bboxes.txt'), 'r')
spam = csv.reader(csvfile, delimiter='\t')
for line in spam:
    if line[0] not in bboxes.keys():
        bboxes[line[0]] = {'labels': [], 'bboxes': []}
    bboxes[line[0]]['labels'].append(line[0].split('_')[0])
    bboxes[line[0]]['bboxes'].append([
        int(line[1]), int(line[2]), int(line[3]), int(line[4])])


# Find the full file paths for the images in the train dataset. This method also
# chooses default enumerations for these labels which we should store to
# be able to map integers to words later.
filenames, texts, labels, enumerations = find_image_files(
    os.path.join(TI_DIR, 'train'))
fp = open(os.path.join(OUT_DIR, 'enumerations.json'), 'w')
json.dump(enumerations, fp, indent=2)
fp.close()

# Use the gathered information to load the images and save as TFRecords
out_dir = os.path.join(OUT_DIR, 'train')
create_tfrecords('train', filenames, texts, labels, text_mappings=mappings,
                 output_dir=out_dir, num_shards=20, num_threads=5,
                 bboxes=bboxes, enumeration=enumerations)

# Load the val bbox file. These have a slightly different format to the train
# bboxes files. I.e.
#       val_0.JPEG	n03444034	0	32	44	62
bboxes = {}
csvfile = open(os.path.join(TI_DIR, 'val', 'val_annotations.txt'), 'r')
spam = csv.reader(csvfile, delimiter='\t')
for line in spam:
    if line[0] not in bboxes.keys():
        bboxes[line[0]] = {'labels': [], 'bboxes': []}
    bboxes[line[0]]['labels'].append(line[1])
    bboxes[line[0]]['bboxes'].append([
        int(line[2]), int(line[3]), int(line[4]), int(line[5])])

# Find the list of all the files in the val dataset first. Unlike the train
# dataset, these aren't nicely organized into folders, so the texts, labels and
# enumerations outputs will be useless. Still, we need the full paths.

try:
    filenames, _, _, _ = find_image_files(
        os.path.join(TI_DIR, 'val'))
# Don't catch normal exceptions
except (KeyboardInterrupt, SystemExit):
    raise
except Exception as e:
    import ipdb, sys
    print(e)
    ipdb.post_mortem(sys.exc_info()[2])

# Compile the texts, labels for all the files using their filenames and synsets
texts = [bboxes[os.path.basename(file)]['labels'][0] for file in filenames]
labels = [enumerations[text] for text in texts]
out_dir = os.path.join(OUT_DIR, 'val')
create_tfrecords('val', filenames, texts, labels, text_mappings=mappings,
                 output_dir=out_dir, num_shards=20, num_threads=5,
                 bboxes=bboxes, enumeration=enumerations)
