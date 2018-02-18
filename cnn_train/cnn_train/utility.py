import os.path
import scipy
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes

class DataSet(object):

  def __init__(self,
               images,
               labels):

      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      self._images = images
      self._labels = labels
      self._epochs_completed = 0
      self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples


#def load_images(image_dir, image_width, image_height):
def load_images(dir_list, label_list, image_width, image_height):

    t_images = []
    t_labels = []
    for idir in range(len(dir_list)):
        image_dir = dir_list[idir]
        class_id = label_list[idir]

        if not gfile.Exists(image_dir):
            tf.logging.error("Image directory '" + image_dir + "' not found")
            return None
    
        file_glob = os.path.join(image_dir, '*')
        file_list = gfile.Glob(file_glob)

        if len(file_list) < 1:
            tf.logging.error("No images to process")
            return None

        tf.logging.info("Loading %d images from %s...", len(file_list), image_dir)

        for index in range(len(file_list)):
    #        decoded_image = tf.image.decode_jpeg(file_name, channels=1)
    #        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
            file_name = file_list[index]
            image = scipy.ndimage.imread(file_name, flatten = True)
        
            if image.shape[0] != image_width or image.shape[1] != image_height:
                tf.logging.error("All images should be same size")
                return None
        
            image = image.reshape(-1)
    #        np.concatenate((images, image), axis=0)
            t_images.append(image)
            t_labels.append(class_id)

        tf.logging.info("[...done]")

    images = np.array(t_images).astype(np.float32)
    labels = np.array(t_labels).astype(np.int32)

    return DataSet(images, labels)

