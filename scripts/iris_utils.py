from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
import tensorflow_transform as tft # Step 4

from tfx.components.trainer.executor import TrainerFnArgs

_FLOAT_FEATURE_KEYS = ['petal-length', 'petal-width', 'sepal-length', 'sepal-width']

_LABEL_KEY = 'class'

# START Step 4 -----------------------------------------------------------

def preprocessing_fn(inputs):

  outputs = inputs.copy()   

  for key in _FLOAT_FEATURE_KEYS:
    outputs[key] = tft.scale_to_z_score(inputs[key])

  initializer = tf.lookup.KeyValueTensorInitializer(
      keys=['Iris-virginica', 'Iris-versicolor', 'Iris-setosa'],
      values=tf.cast(tf.range(3), tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  table = tf.lookup.StaticHashTable(initializer, default_value=-1)

  outputs[_LABEL_KEY] = table.lookup(outputs[_LABEL_KEY])

  return outputs

# END Step 4 -------------------------------------------------------------

# START Step 5 -----------------------------------------------------------
def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT"""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples,
                                          feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn

def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_LABEL_KEY)

  return dataset

def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying data.

  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).

  Returns:
    A keras Model.
  """
  real_valued_columns = [
      tf.feature_column.numeric_column(key, shape=())
      for key in _FLOAT_FEATURE_KEYS
  ]

  model = _wide_and_deep_classifier(
      wide_columns=real_valued_columns,
      deep_columns=real_valued_columns,
      dnn_hidden_units=hidden_units or [100, 70, 50, 25])
  return model


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units):
  """Build a simple keras wide and deep model.

  Args:
    wide_columns: Feature columns wrapped in indicator_column for wide
      (linear) part of the model.
    deep_columns: Feature columns for deep part of the model.
    dnn_hidden_units: [int], the layer sizes of the hidden DNN.

  Returns:
    A Wide and Deep Keras model
  """
  input_layers = {
      colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
      for colname in _FLOAT_FEATURE_KEYS
  }

  deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)

  for numnodes in dnn_hidden_units:
    deep = tf.keras.layers.Dense(numnodes)(deep)
  wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

  output = tf.keras.layers.Dense(
      1, activation='sigmoid')(
          tf.keras.layers.concatenate([deep, wide]))

  model = tf.keras.Model(input_layers, output)
  
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(lr=0.001),
      metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])

  model.summary(print_fn=absl.logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  first_dnn_layer_size = 100 # número de neurônios na primeira camada
  num_dnn_layers = 4 # número de camadas
  dnn_decay_factor = 0.7 # diminuição de neurônios por camada

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output) # pega o output do transformer

  train_dataset = _input_fn(fn_args.train_files, tf_transform_output, 40) # carrega o dataset
  eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output, 40)

  # If no GPUs are found, CPU is used.
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model(
        # Construct layers sizes with exponetial decay
        hidden_units=[
            max(2, int(first_dnn_layer_size * dnn_decay_factor**i))
            for i in range(num_dnn_layers)
        ])

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf',
             signatures=signatures)
# END Step 5 -------------------------------------------------------------
