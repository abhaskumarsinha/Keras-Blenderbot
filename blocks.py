import tensorflow as tf
import keras
import math
import numpy as np

class Decoder(keras.layers.Layer):
  def __init__(self, embedding_layer = None, embed_positions = None, layers = None):
    super().__init__()

    self.dropout = keras.layers.Dropout(0.1)
    self.layerdrop = 0 #ToDo
    self.max_target_positions = 128
    self.embed_scale = math.sqrt(1280)

    self.embed_tokens = embedding_layer
    self.embed_positions = embed_positions

    self.layers = layers

    self.layer_norm = keras.layers.LayerNormalization(-1)

  def call(self, inputs, encoder_hidden_states=None, past_key_values=None):

    input_shape = keras.ops.shape(inputs)
    x = keras.ops.reshape(inputs, (-1, input_shape[-1]))

    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    inputs_embeds = self.embed_tokens(x) * self.embed_scale #TODO
    positions = self.embed_positions(inputs = input_shape, length = past_key_values_length) #TODO
    x = x + positions

    x = self.dropout(x)

    next_decoder_cache = ()

    for idx, decoder_layer in enumerate(self.layers):

      past_key_value = past_key_values[idx] if past_key_values is not None else None

      x, cache = decoder_layer(x, encoder_hidden_states = encoder_hidden_states, past_key_value = past_key_value)

      next_decoder_cache = (cache, )

    x = self.layer_norm(x)

    return x, next_decoder_cache



class Encoder(keras.layers.Layer):
  def __init__(embed_tokens, embed_positions, layers):
    super().__init__()

    self.dropout = keras.layers.Dropout(-1)
    self.layerdrop = 0 #ToDo
    self.embed_dim = 1280
    self.max_source_position = 128
    self.embed_scale = math.sqrt(self.embed_dim)
    self.embed_tokens = embed_tokens
    self.embed_positions = embed_positions
    self.layers = layers
    self.layer_norm = keras.layers.LayerNormalization(-1)


  def call(self, inputs):

    input_shape = keras.ops.shape(inputs)
    input_ids = keras.ops.reshape(inputs, (-1, input_shape[-1]))

    inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embed_pos = self.embed_positions(inputs = input_shape)

    x = input_embeds + embed_pos
    x = self.dropout(x)

    for idx, encoder_layer in enumerate(self.layers):
      x = encoder_layer(x)

    x = self.layer_norm(x)
    return x
