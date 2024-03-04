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

    y = self.embedding(keras.ops.einsum('j, bij -> bij', keras.ops.arange(length, length + seq_len), keras.ops.ones((bsz, seq_len, seq_len))))

    return y[0]
