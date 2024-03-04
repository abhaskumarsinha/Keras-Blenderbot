import tensorflow as tf
import numpy as np
import keras
import math

class CachedAttention(keras.layers.Layer):
  def __init__(self):
    super().__init__()
    self.embed_dim = 1280
    self.num_heads = 32
    self.dropout = keras.layers.Dropout(0)

    try:
      self.head_dim = self.embed_dim // self.num_heads
    except:
      raise Exception("`embed_dim` should be divisible by `num_heads`, provided values are: " + str(self.embed_dim) + " and " + str(self.num_heads) + " respectively.")

    self.scaling = self.head_dim**-0.5

    self.k_proj = keras.layers.Dense(self.embed_dim, activation='linear')
    self.v_proj = keras.layers.Dense(self.embed_dim, activation='linear')
    self.q_proj = keras.layers.Dense(self.embed_dim, activation='linear')

    self.out_proj = keras.layers.Dense(self.embed_dim, activation='linear')

  def _shape(self, x, seq_len, bsz):
    y = keras.ops.reshape(x, newshape=(bsz, seq_len, self.num_heads, self.head_dim))
    return keras.ops.transpose(y, (0, 2, 1, 3))

  def call(self, inputs, past_key_value = None, attention_mask = None, key_value_states = None, is_cross_attention = False):

    bsz, tgt_len, _ = keras.ops.shape(inputs)
    query_states = self.q_proj(inputs) * self.scaling

    if (is_cross_attention and past_key_value is not None and past_key_value[0].shape[2] == key_value_states.shape[1]):
      # reuse k,v, cross_attentions
      key_states = past_key_value[0]
      value_states = past_key_value[1]
    elif is_cross_attention:
      # cross_attentions
      key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
      value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    elif past_key_value is not None:
      # reuse k, v, self_attention
      key_states = self._shape(self.k_proj(inputs), -1, bsz) #(10, 32, 1, 40)
      value_states = self._shape(self.v_proj(inputs), -1, bsz) #(10, 32, 1, 40)
      key_states = keras.ops.concatenate([past_key_value[0], key_states], 2)
      value_states = keras.ops.concatenate([past_key_value[1], value_states], 2)
    else:
      # self_attention
      key_states = self._shape(self.k_proj(inputs), -1, bsz)
      value_states = self._shape(self.v_proj(inputs), -1, bsz)


    past_key_value = (key_states, value_states)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim) #(320, -1, 40)
    query_states = keras.ops.reshape(self._shape(query_states, tgt_len, bsz), (proj_shape)) #(320, 1, 40)
    key_states = keras.ops.reshape(key_states, (proj_shape)) #(320, 2, 40)
    value_states = keras.ops.reshape(value_states, (proj_shape)) #(320, 2, 40)

    _, src_len, _ = keras.ops.shape(key_states)

    attn_weights = keras.ops.einsum("bij, bkj -> bik", query_states, key_states)

    attn_weights = keras.ops.softmax(attn_weights, axis=-1)
    attn_probs = self.dropout(attn_weights)

    attn_output = keras.ops.einsum("bij, bjk -> bik", attn_probs, value_states)
    attn_output = keras.ops.reshape(attn_output, (bsz, self.num_heads, tgt_len, self.head_dim))
    attn_output = keras.ops.transpose(attn_output, (0, 2, 1, 3))
    attn_output = keras.ops.reshape(attn_output, (bsz, tgt_len, self.embed_dim))


    attn_output = self.out_proj(attn_output)

    return attn_output, past_key_value


class DecoderLayer(keras.layers.Layer):

  def __init__(self):
    super().__init__()
    self.embed_dim = 1280
    self.self_attn = CachedAttention()
    self.dropout = keras.layers.Dropout(0.1)
    self.dropout2 = keras.layers.Dropout(0.1)
    self.dropout3 = keras.layers.Dropout(0.1)
    self.dropout4 = keras.layers.Dropout(0.1)
    self.activation_fn = keras.ops.gelu
    self.self_attn_layer_norm = keras.layers.LayerNormalization(-1)
    self.encoder_attn = CachedAttention()
    self.encoder_attn_layer_norm = keras.layers.LayerNormalization(-1)
    self.fc1 = keras.layers.Dense(5120)
    self.fc2 = keras.layers.Dense(self.embed_dim)
    self.final_layer_norm = keras.layers.LayerNormalization(-1)


  def call(self, inputs, encoder_hidden_states = None, past_key_value = None, use_cache = True):
    x = inputs

    residual = x
    x = self.self_attn_layer_norm(x)

    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    x, present_key_value = self.self_attn(x, past_key_value = self_attn_past_key_value)
    x = self.dropout(x)
    x = x + residual

    if encoder_hidden_states is not None:
      residual = x
      x = self.encoder_attn_layer_norm(x)

      cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

      x, cross_attn_present_key_value = self.encoder_attn(x, key_value_states = encoder_hidden_states, past_key_value = cross_attn_past_key_value, is_cross_attention = True)

      x = self.dropout2(x)

      x = residual + x

      present_key_value = present_key_value + cross_attn_present_key_value

    residual = x
    x = self.final_layer_norm(x)
    x = self.fc1(x)
    x = self.activation_fn(x, approximate = False)
    x = self.dropout3(x)
    x = self.fc2(x)
    x = self.dropout4(x)
    x = residual + x

    outputs = (x, )
    outputs += (present_key_value, )

    return outputs


class DecoderLayer(keras.layers.Layer):

  def __init__(self):
    super().__init__()
    self.embed_dim = 1280
    self.self_attn = CachedAttention()
    self.dropout = keras.layers.Dropout(0.1)
    self.dropout2 = keras.layers.Dropout(0.1)
    self.dropout3 = keras.layers.Dropout(0.1)
    self.dropout4 = keras.layers.Dropout(0.1)
    self.activation_fn = keras.ops.gelu
    self.self_attn_layer_norm = keras.layers.LayerNormalization(-1)
    self.encoder_attn = CachedAttention()
    self.encoder_attn_layer_norm = keras.layers.LayerNormalization(-1)
    self.fc1 = keras.layers.Dense(5120)
    self.fc2 = keras.layers.Dense(self.embed_dim)
    self.final_layer_norm = keras.layers.LayerNormalization(-1)


  def call(self, inputs, encoder_hidden_states = None, past_key_value = None, use_cache = True):
    x = inputs

    residual = x
    x = self.self_attn_layer_norm(x)

    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    x, present_key_value = self.self_attn(x, past_key_value = self_attn_past_key_value)
    x = self.dropout(x)
    x = x + residual

    if encoder_hidden_states is not None:
      residual = x
      x = self.encoder_attn_layer_norm(x)

      cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

      x, cross_attn_present_key_value = self.encoder_attn(x, key_value_states = encoder_hidden_states, past_key_value = cross_attn_past_key_value, is_cross_attention = True)

      x = self.dropout2(x)

      x = residual + x

      present_key_value = present_key_value + cross_attn_present_key_value

    residual = x
    x = self.final_layer_norm(x)
    x = self.fc1(x)
    x = self.activation_fn(x, approximate = False)
    x = self.dropout3(x)
    x = self.fc2(x)
    x = self.dropout4(x)
    x = residual + x

    outputs = (x, )
    outputs += (present_key_value, )

    return outputs



class EncoderLayer(keras.layers.Layer):
  def __init__(self):
    super().__init__()

    self.self_attn = CachedAttention()
    self.embed_dim = 1280
    self.self_attn_layer_norm = keras.layers.LayerNormalization(-1)
    self.dropout = keras.layers.Dropout(0.1)
    self.dropout2 = keras.layers.Dropout(0.1)
    self.dropout3 = keras.layers.Dropout(0.1)
    self.activation_fn = keras.ops.gelu
    self.fc1 = keras.layers.Dense(5120)
    self.fc2 = keras.layers.Dense(self.embed_dim)
    self.final_layer_norm = keras.layers.LayerNormalization(-1)


  def call(self, inputs):
    x = inputs

    residual = inputs

    x = self.self_attn_layer_norm(x)
    x, _ = self.self_attn(x)
    x = self.dropout(x)
    x = residual + x

    residual = x
    x = self.final_layer_norm(x)
    x = self.fc1(x)
    x = self.activation_fn(x, approximate = False)
    x = self.dropout2(x)
    x = self.fc2(x)
    x = self.dropout3(x)

    x = residual + x

    return x
