import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math as m
import gc
#from tensorflow.python import keras
import numpy as np
import math
import sys
import time
import utils

class DynamicPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, max_seq=1024, **kwargs):
        super().__init__(**kwargs)
        embed_sinusoid_list = np.array([[
            [
                m.sin(
                    pos * m.exp(-m.log(10000) * i/embedding_dim) *
                    m.exp(m.log(10000)/embedding_dim * (i % 2)) + 0.5 * m.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = tf.constant(embed_sinusoid_list, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        return tf.add(inputs, self.positional_embedding[:,:inputs.shape[1],:])

class BaselineAttention(tf.keras.layers.Layer):
    def __init__(self, h, d, max_seq=1024, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = None
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = tf.keras.layers.Dense(int(self.d / 2))
        self.Wk = tf.keras.layers.Dense(int(self.d / 2))
        self.Wv = tf.keras.layers.Dense(int(self.d))
        self.fc = tf.keras.layers.Dense(d)
        self.max_seq = max_seq

    def build(self, input_shape):
        self.len_k = input_shape[1][1]
        # self.max_seq = max(input_shape[0][1], input_shape[1][1], input_shape[2][1])

    def call(self, inputs, mask=None, weight_out=False, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param weight_out: decide to get weather weight or not
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        Kt = tf.transpose(k, [0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits += (tf.cast(mask, tf.float32) * -1e9)

        attention_weights = tf.nn.softmax(logits, -1)
        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))

        out = self.fc(out)

        return out, attention_weights

class RelativeGlobalAttention(tf.keras.layers.Layer):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=128, add_emb=False, max_seq=1024, **kwargs):
        super().__init__(**kwargs)
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = tf.keras.layers.Dense(int(self.d))
        self.Wk = tf.keras.layers.Dense(int(self.d))
        self.Wv = tf.keras.layers.Dense(int(self.d))
        self.fc = tf.keras.layers.Dense(d)
        self.additional = add_emb
        if self.additional:
            self.Radd = None

    def build(self, input_shape):
        self.shape_q = input_shape[0][1]
        self.shape_k = input_shape[1][1]
        self.E = self.add_weight('emb', shape=[self.max_seq, int(self.dh)])

    def call(self, inputs, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = tf.reshape(q, (q.shape[0], q.shape[1], self.h, -1))
        q = tf.transpose(q, (0, 2, 1, 3))  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        k = tf.reshape(k, (k.shape[0], k.shape[1], self.h, -1))
        k = tf.transpose(k, (0, 2, 1, 3))

        v = inputs[2]
        v = self.Wv(v)
        v = tf.reshape(v, (v.shape[0], v.shape[1], self.h, -1))
        v = tf.transpose(v, (0, 2, 1, 3))

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]

        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = tf.einsum('bhld,md->bhlm', q, E)
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = tf.transpose(k,[0, 1, 3, 2])
        QKt = tf.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        attention_weights = tf.nn.softmax(logits, -1)
        attention = tf.matmul(attention_weights, v)

        out = tf.transpose(attention, (0, 2, 1, 3))
        out = tf.reshape(out, (out.shape[0], -1, self.d))

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    @staticmethod
    def _qe_masking(qe):
        mask = tf.sequence_mask(
            tf.range(qe.shape[-1] -1, qe.shape[-1] - qe.shape[-2] -1, -1), qe.shape[-1])
        mask = tf.logical_not(mask)
        mask = tf.cast(mask, tf.float32)
        return mask * qe

    def _skewing(self, tensor: tf.Tensor):
        padded = tf.pad(tensor, [[0, 0], [0,0], [0, 0], [1, 0]])
        reshaped = tf.reshape(padded, shape=[-1, padded.shape[1], padded.shape[-1], padded.shape[-2]])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = tf.pad(Srel, [[0,0], [0,0], [0,0], [0, self.len_k-self.len_q]])
        elif self.len_k < self.len_q:
            Srel = Srel[:,:,:,:self.len_k]
        return Srel

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, rate=0.1, h=4, additional=False, max_seq=1024):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = tf.keras.layers.Dense(self.d_model // 2, activation=tf.nn.gelu)
        self.FFN_suf = tf.keras.layers.Dense(self.d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, **kwargs):

        attn_out, w = self.rga([x,x,x])
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layernorm1(attn_out+x)
        ffn_out = self.FFN_pre(out1)
        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1+ffn_out)
        return out2, w


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, input_vocab_size, rate=0.1, max_len=None):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        #self.num_layers = 8
        self.max_len = max_len

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_embedding = tf.keras.layers.Embedding(max_len, d_model)
        self.bar_embedding = tf.keras.layers.Embedding(50, d_model)

        #if True:
        #    self.pos_encoding = DynamicPositionEmbedding(self.d_model, max_seq=max_len)

        self.enc_layers = [EncoderLayer(d_model, rate, h=self.d_model // 32, additional=False, max_seq=max_len)
                           for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, bar_x, training=False):
        weights = []
        x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        pos = np.arange(x.shape[0]*x.shape[1]).reshape(x.shape[0],x.shape[1]) % x.shape[1]
        x = x + self.pos_embedding(pos) + self.bar_embedding(bar_x)
        x = self.dropout(x, training=training)
        x = self.layernorm(x)

        for i in range(self.num_layers):
            #print(i,i,i,i,i,i,i)
            x, w = self.enc_layers[i](x, training=training)
            weights.append(w)

        return x, weights

if __name__ == '__main__':
    rga = RelativeGlobalAttention(d=9, h=1)
    q = tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32)
    k = tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32)

    import utils

    src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(q.shape[1], tf.argmax(k,-1), tf.argmax(q, -1))
    # print(src_mask.shape, trg_mask.shape, look_ahead_mask.shape)

    result = rga([
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        ], mask=trg_mask)

    print(result)

    k = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    q = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    import utils

    src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(q.shape[1], tf.argmax(k, -1),
                                                                           tf.argmax(q, -1))
    print(src_mask.shape, trg_mask.shape, look_ahead_mask.shape)
    result = rga([
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
        tf.constant([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
            dtype=tf.float32),
    ], mask=trg_mask)

    print(result)

    k = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    q = tf.constant([
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        dtype=tf.float32)
    import utils

    src_mask, trg_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(q.shape[1], tf.argmax(k, -1),
                                                                           tf.argmax(q, -1))
    print(src_mask, trg_mask, look_ahead_mask)
    result = rga([
        q,
        k,
        k
    ], mask=look_ahead_mask)

    print(result)

