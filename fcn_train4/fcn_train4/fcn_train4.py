from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os.path
import sys
import cv2
import models

tf.logging.set_verbosity(tf.logging.INFO)

###############################################################################
# Data processing
###############################################################################
def get_filenames(dir_set):
    """Return filenames for dataset."""
    file_list = []
    class_list = []
    total = 0
 
    for i in range(len(dir_set['dirs'])):
        image_dir = dir_set['dirs'][i]
        class_id = dir_set['labels'][i]

        if not gfile.Exists(image_dir):
            tf.logging.error("Image directory '" + image_dir + "' not found")
            return None
            
        file_glob = os.path.join(image_dir, '*.jpg')
        files = gfile.Glob(file_glob)

        if len(files) < 1:
            tf.logging.error("No images to process")
            return None

        file_list += files
        class_list += ([class_id] * len(files))
    
    filenames = {'files': file_list, "labels": class_list}
        
    return filenames


def preprocess_image(image, is_training):
    # Convert the image pixels to [0,1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Convert from [depth, height, width] to [height, width, depth]
    image.set_shape([None, None, 3])
    # Substract mean image
    image = mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

    return image

def mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def parse_record(raw_record, is_training, num_classes=0, width=0, height=0, num_channels=0):
  """Parse image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  file = raw_record['files']
  label = raw_record['labels']

  label = tf.one_hot(label, num_classes)

  raw_image = tf.read_file(file)
  image = tf.image.decode_image(raw_image)

  if is_training:
      # Explicitly reshape image tensor to have dimensions properly recoreded in the graph
      image = tf.reshape(image, [height, width, num_channels])

  image = preprocess_image(image, is_training)

  image = {'image': image, 'file': file}
 
  return image, label

def process_record_dataset(dataset, is_training, batch_size,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1,
                           num_classes=0, width=0, height=0, num_channels=0):
  """Given a Dataset with raw records, parse each record into images and labels,
  and return an iterator over the records.
  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)

  ## If we are training over multiple epochs before evaluating, repeat the
  ## dataset for the appropriate number of epochs.
  #dataset = dataset.repeat(num_epochs)

  # Parse the raw records into images and labels
  dataset = dataset.map(lambda value: parse_record_fn(value, is_training, num_classes, width, height, num_channels),
                        num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path.
  dataset = dataset.prefetch(1)

  return dataset


def input_fn(is_training, train_set, eval_set, batch_size, num_epochs=1,
             num_parallel_calls=1,
             num_classes=0, width=0, height=0, num_channels=0):
  """Input function which provides batches for train or eval.
  Args:
    is_training: A boolean denoting whether the input is for training.
    train_set, eval_set: Dictionary with list of the directories and labels.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    A dataset that can be used for iteration.
  """
  if is_training:
      filenames = get_filenames(train_set)
  else:
      filenames = get_filenames(eval_set)

  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  # Shuffle the input files
  dataset = dataset.shuffle(buffer_size=len(filenames['files']))

  return process_record_dataset(dataset, is_training, batch_size,
                                parse_record, num_epochs, num_parallel_calls,
                                num_classes, width, height, num_channels)


def serve_input_fn(test_set, batch_size, num_epochs=1,
             num_parallel_calls=1):
  """Input function which provides batches for train or eval.
  Args:
    is_training: A boolean denoting whether the input is for training.
    train_set, eval_set: Dictionary with list of the directories and labels.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(test_set)

  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  return process_record_dataset(dataset, False, batch_size,
                                parse_record, num_epochs, num_parallel_calls)


######################################################################3
_R_MEAN = 113.34 / 255
_G_MEAN = 107.30 / 255
_B_MEAN = 100.98 / 255

model = 'alexnet_rs'
hparams = dict(model_fn=getattr(models, model),
              num_classes=5,
              class_weights=[1/17,4/17,4/17,4/17,4/17],
              #class_weights=[1.0,1.0,1.0,1.0,1.0],
              bg_train_dir='c:\\train\\cnn\\train\\background\\227',
              c1_train_dir='c:\\train\\cnn\\train\\pepsi\\can_classic_0.33\\227',
              c2_train_dir='c:\\train\\cnn\\train\\cocacola\\pet_classic_0.5\\227',
              c3_train_dir='c:\\train\\cnn\\train\\cocacola\\pet_zero_0.5\\227',
              c4_train_dir='c:\\train\\cnn\\train\\cocacola\\can_classic_0.33\\227',
              bg_eval_dir='c:\\train\\cnn\\eval\\background\\227',
              c1_eval_dir='c:\\train\\cnn\\eval\\pepsi\\can_classic_0.33\\227',
              c2_eval_dir='c:\\train\\cnn\\eval\\cocacola\\pet_classic_0.5\\227',
              c3_eval_dir='c:\\train\\cnn\\eval\\cocacola\\pet_zero_0.5\\227',
              c4_eval_dir='c:\\train\\cnn\\eval\\cocacola\\can_classic_0.33\\227',
              #bg_eval_dir='c:\\train\\cnn\\test\\227\\background',
              #c1_eval_dir='c:\\train\\cnn\\test\\227\\pepsi\\can_classic_0.33',
              #c2_eval_dir='c:\\train\\cnn\\test\\227\\cocacola\\pet_classic_0.5',
              #c3_eval_dir='c:\\train\\cnn\\test\\227\\cocacola\\pet_zero_0.5',
              #c4_eval_dir='c:\\train\\cnn\\test\\227\\cocacola\\can_classic_0.33',
              test_dir='C:\\train\\cnn\\test\\227',
              sample_width=227,
              sample_height=227,
              num_channels=3,
              model_dir="./models/sku_model",
              learning_rate=0.001,
              #learning_rate=0.0001,
              batch_size=50,
              train_steps=10000,
              num_epochs=0, # num-set / batch_size = num_iterations to complete one epoch 
              save_dir=".\\save",
              weight_decay=0.0001,
              save_checkpoints_steps=1000)

######################################################################3
def main(argv):
    # Parse command line
    do_train = False
    do_serve = False
    do_save = False
    for arg in argv[1:]:
        if arg == "-train":
            do_train = True
        elif arg == "-serve":
            do_serve = True
        elif arg == "-save":
            do_save = True

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create the Estimator
    run_config = tf.estimator.RunConfig().replace(model_dir='./'+model,
                                                  save_checkpoints_steps=hparams['save_checkpoints_steps'],
                                                  keep_checkpoint_max=3)
    
    obj_classifier = tf.estimator.Estimator(model_fn=hparams['model_fn'],
                                            config=run_config,
                                            params=hparams)

    if do_train:        
        # Load training and eval data
        train_set = {'dirs': [hparams['bg_train_dir'],
                              hparams['c1_train_dir'],
                              hparams['c2_train_dir'],
                              hparams['c3_train_dir'],
                              hparams['c4_train_dir']],
                     'labels': [0,1,2,3,4]}
        eval_set = {'dirs': [hparams['bg_eval_dir'],
                              hparams['c1_eval_dir'],
                              hparams['c2_eval_dir'],
                              hparams['c3_eval_dir'],
                              hparams['c4_eval_dir']],
                     'labels': [0,1,2,3,4]}

        # Set up logging for predictions
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_accuracy': 'train_accuracy'
        }

        ## Dirty debugging trick. Uncomment if needed. Add corresponding tf.identities to the model
        #tensors_to_log = {
        #    'image': 'image',
        #    'files': 'files',
        #    'labels': 'labels'
        #}
        
        #def formatter_fn(log):
        #    ff = open("./images.txt", "a")
        #    image = log['image']
        #    fn = open("./files.txt", "a")
        #    for i in range(0, len(log['files'])):
        #        file = log['files'][i].decode()
        #        label_one_hot = log['labels'][i]
        #        label = label_one_hot[0] * 0 + label_one_hot[1] * 1 + label_one_hot[2] * 2 + label_one_hot[3] * 3 + label_one_hot[4] * 4
        #        fn.write("{0};{1:.0f}\n".format(file, label))
        #    f.close()
        #    return "logging done"

        #logging_hook = tf.train.LoggingTensorHook(
        #    tensors=tensors_to_log, every_n_iter=1, formatter=formatter_fn)

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

        # Train the model
        def train_input_fn():
            return input_fn(is_training=True,
                            train_set=train_set,
                            eval_set=[],
                            batch_size=hparams['batch_size'],
                            num_classes=hparams['num_classes'],
                            width=hparams['sample_width'],
                            height=hparams['sample_height'],
                            num_channels=hparams['num_channels'])

        def eval_input_fn():
            return input_fn(is_training=False,
                            train_set=[],
                            eval_set=eval_set,
                            batch_size=hparams['batch_size'],
                            num_classes=hparams['num_classes'])

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=hparams['train_steps'],
                                            hooks=[logging_hook])
        
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

        tf.estimator.train_and_evaluate(obj_classifier, train_spec, eval_spec)

 
    elif do_serve:
        test_set = {'dirs': [hparams['test_dir']], 'labels': [0]}
        def test_input_fn():
            return serve_input_fn(test_set=test_set,
                                  batch_size=hparams['batch_size'])

        test_results = obj_classifier.predict(input_fn=test_input_fn)

        #for file in test_files:
        #    result = next(test_results)
        #    print("%s: " % (file), end='')
        #    print(result)
        for result in test_results:
            print(result)


    elif do_save:
        feature_spec = {"image": tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3]), "file": tf.placeholder(dtype=tf.string, shape=[1])}
        input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        obj_classifier.export_savedmodel(hparams['save_dir'], input_receiver_fn)


if __name__ == "__main__":
  tf.app.run()




