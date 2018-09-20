import tensorflow as tf
import numpy as np
from PIL import Image
import os

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '/Users/apple/Documents/Smokedetector/TFRecorder.txt', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(examplestr):
  # TODO(user): Populate the following variables from your example.
  example = examplestr.split()
  height = int(example[len(example)-2])  # Image height
  width = int(example[len(example)-3])  # Image width
  filename = example[0]  # Filename of the image. Empty if image is not from file
  num_of_features = int(example[1])
  image_raw = np.array(Image.open(filename))  # Encoded image bytes
  image_bytes = image_raw.tostring()
  image_format = b'jpeg'  # b'jpeg' or b'png'

  idx = 2
  xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
  ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
  if num_of_features == 1:
      classes_text = [b'gasdetector']  # List of string class name of bounding box (1 per box)
      classes = [1]  # List of integer class id of bounding box (1 per box)
  elif num_of_features == 2:
      classes_text = [b'helmet', b'goggle']
      classes = [2, 3]
  elif num_of_features == 4:
      classes_text = [b'gasdetector', b'gasdetector', b'helmet', b'goggle']
      classes = [1, 1, 2, 3]

  while idx < len(example)-1:
      xmins.append(float(example[idx])/width)
      ymins.append(float(example[idx+1])/height)
      xmaxs.append(float(example[idx+2])/width)
      ymaxs.append(float(example[idx+3])/height)
      idx = idx + 6

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode()),
      'image/source_id': dataset_util.bytes_feature(filename.encode()),
      'image/encoded': dataset_util.bytes_feature(image_bytes),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  boxfile = open('/Users/apple/Documents/Smokedetector/output.txt')
  examples = boxfile.readlines()
  boxfile.close()

  for examplestr in examples:
      tf_example = create_tf_example(examplestr)
      writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
    tf.app.run()