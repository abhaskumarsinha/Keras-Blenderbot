import tensorflow as tf
import math
import keras
import numpy as np

class SharedEmbedding(keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.embedding = keras.layers.Embedding(8008, 1280)

  def call(self, inputs):
    return self.embedding(inputs)

class PositionalEmbedding(keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.embedding = keras.layers.Embedding(128, 1280)

  def call(self, inputs, length = 0):
    bsz, seq_len = inputs[:2]

    ones = keras.ops.ones((bsz, seq_len))
    seq = keras.ops.reshape(keras.ops.arange(length, length + seq_len), (1, seq_len))

    y = self.embedding(keras.ops.einsum('bi, xi -> bi', ones, seq))

    return y
