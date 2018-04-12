# coding: utf-8
import csv
from tfrecords_creator import find_image_files, create_tfrecords
import glob
import os

TI_DIR = 'Tiny_ImageNet'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# Load the wordnet mappings
with open(os.path.join(TI_DIR, 'words.txt')) as csvfile:
    spam = csv.reader(csvfile, delimiter='\t')
    mappings = {}
    for line in spam:
        mappings[line[0]] = line[1]

# Prepare the train bbox file
bbox_files = glob.glob(os.path.join(TI_DIR, 'train/n*/*.txt'))
train_outfile = os.path.join(TI_DIR, 'train', 'bboxes.txt')

print('Found {} bbox files'.format(len(bbox_files)))
outfile = open(train_outfile, 'w')
for f in bbox_files:
    data = open(f, 'r').read()
    outfile.write(data)
outfile.close()

# Prepare the val bbox file
bbox_files = glob.glob(os.path.join(TI_DIR, 'val/n*/*.txt'))
print('Found {} bbox files'.format(len(bbox_files)))
val_outfile = os.path.join(TI_DIR, 'val', 'bboxes.txt')

outfile = open(val_outfile, 'w')
for f in bbox_files:
    data = open(f, 'r').read()
    outfile.write(data)
outfile.close()

# Load the train bbox file
bboxes = {}
csvfile = open('Tiny_ImageNet/train/bboxes.txt', 'r')
spam = csv.reader(csvfile, delimiter='\t')
for line in spam:
    if line[0] not in bboxes.keys():
        bboxes[line[0]] = {'labels': [], 'bboxes': []}
    bboxes[line[0]]['labels'].append(line[0].split('_')[0])
    bboxes[line[0]]['bboxes'].append([
        int(line[1]), int(line[2]), int(line[3]), int(line[4])])

# Convert the train dataset
filenames, texts, labels, enumerations = find_image_files(os.path.join(TI_DIR, 'train'))
out_dir = os.path.join(TI_DIR, '..', 'Tiny_ImageNet_tfrecords', 'train')
create_tfrecords('train', filenames, texts, labels,
                 output_dir=out_dir, num_shards=20, num_threads=5,
                 bboxes=bboxes, enumeration=enumerations)

# Load the val bbox file
bboxes = {}
csvfile = open('Tiny_ImageNet/val/bboxes.txt', 'r')
spam = csv.reader(csvfile, delimiter='\t')
for line in spam:
    if line[0] not in bboxes.keys():
        bboxes[line[0]] = {'labels': [], 'bboxes': []}
    bboxes[line[0]]['labels'].append(line[0].split('_')[0])
    bboxes[line[0]]['bboxes'].append([
        int(line[1]), int(line[2]), int(line[3]), int(line[4])])

# Convert the train dataset
filenames, texts, labels, enumerations = find_image_files(os.path.join(TI_DIR, 'val'))
out_dir = os.path.join(TI_DIR, '..', 'Tiny_ImageNet_tfrecords', 'val')
create_tfrecords('val', filenames, texts, labels,
                 output_dir=out_dir, num_shards=20, num_threads=5,
                 bboxes=bboxes, enumeration=enumerations)
