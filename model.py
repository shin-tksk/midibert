from custom.layers import *
from custom.callback import *
import params as par
import sys
#from tensorflow.python import keras
import json
import tensorflow_probability as tfp
import random
import utils
from progress.bar import Bar
import os
import time
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

tf.executing_eagerly()

class MusicTransformerDecoder(tf.keras.Model):
    def __init__(self, embedding_dim=128, vocab_size=par.vocab_size, num_layer=6,
                 max_seq=1024, dropout=0.2, loader_path=None):
        super(MusicTransformerDecoder, self).__init__()

        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size

        self.Decoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = tf.keras.layers.Dense(self.vocab_size, activation=None, name='output')
        self.cla_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax, name='classification')
        self._set_metrics()

        if loader_path is not None:
            self.load_ckpt_file(loader_path)

    def call(self, inputs, training=None, eval=None, classification=None):
        bar_token = utils.bar_id(inputs)
        decoder, w = self.Decoder(inputs, bar_x=bar_token, training=training)
        fc = self.fc(decoder)
        if training:
            return fc
        elif eval:
            return fc, w
        elif classification:
            return self.cla_layer(decoder[:,0]), fc
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None):

        predictions = self.__train_step(x, y, True)
        result_metric = []

        loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)

        for metric in self.custom_metrics:
            result_metric.append(metric(y, predictions).numpy())

        return [loss.numpy()] + result_metric

    def cla_train_on_batch(self, x, y=None, real=None):

        y = tf.constant(y)
        y = tf.cast(y, tf.float32)

        predictions = self.cla_train_step(x, y, real)

        result_metric = []

        loss = tf.reduce_mean(self.loss_value)
        #loss = tf.reduce_mean(loss)

        y = y.numpy()
        predictions = predictions.numpy()
        #predictions = predictions.reshape(predictions.shape[0],-1)

        count = 0
        for i in range(y.shape[0]):
            if (y[i][0] < y[i][1]) == (predictions[i][0] < predictions[i][1]):
                count += 1
        result_metric = [count/y.shape[0]]

        return [loss.numpy()]+result_metric

    def cla_train_step(self, inp_tar, out_tar, real):
        with tf.GradientTape() as tape:
            predictions, pred = self.call(inputs=inp_tar, classification=True)
            self.loss_value = self.loss(out_tar, predictions, real, pred)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return predictions

    # @tf.function
    def __train_step(self, inp_tar, out_tar, training):

        with tf.GradientTape() as tape:
            predictions = self.call(inputs=inp_tar, training=training)
            self.loss_value = self.loss(out_tar, predictions)
        
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return predictions

    def evaluate(self, x=None, y=None):
        y = tf.constant(y)
        y = tf.cast(y, tf.float32)
        predictions, w = self.call(x, training=False, eval=True)
        loss = tf.reduce_mean(self.loss(y, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(y, tf.nn.softmax(predictions)).numpy())
        return [loss.numpy()] + result_metric, w
    
    def cla_evaluate(self, x=None, y=None, real=None):
        y = tf.constant(y)
        y = tf.cast(y, tf.float32)
        predictions, pred = self.call(inputs=x, classification=True)
        loss = tf.reduce_mean(self.loss(y, predictions, real, pred))
        result_metric = []
        predictions = predictions.numpy()
        predictions = predictions.reshape(predictions.shape[0],-1)
        count = 0
        for i in range(y.shape[0]):
            if (y[i][0] < y[i][1]) == (predictions[i][0] < predictions[i][1]):
                count += 1
        result_metric = [count/y.shape[0]]
        return [loss.numpy()] + result_metric

    def classification(self, x=None, y=None):
        predictions, _ = self.call(inputs=x, classification=True)
        predictions = predictions.numpy()
        #print(predictions)
        #print(y)
        #predictions = predictions.reshape(predictions.shape[0],-1)
        if (y[0] < y[1]) == (predictions[0][0] < predictions[0][1]):
            return 1
        else:
            return 0
    
    def get_config(self):
        config = {}
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        return config

    def save_weight(self, filepath, overwrite=True, include_optimizer=False, save_format=None):
        config_path = filepath+'/'+'config.json'
        ckpt_path = filepath+'/ckpt'

        self.save_weights(ckpt_path, save_format='tf')
        #self.save('saved_model/model')

        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        return

    def load_config_file(self, filepath):
        config_path = filepath + '/' + 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.__load_config(config)

    def load_ckpt_file(self, filepath, ckpt_name='ckpt'):
        ckpt_path = filepath + '/' + ckpt_name
        try:
            self.load_weights(ckpt_path)
        except FileNotFoundError:
            print("[Warning] model will be initialized...")

        config = {}
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        return config

    def generate_mask(self, prior: list, length=1024):
        decode_array = prior
        #for i in Bar('generating').iter(range(min(self.max_seq, length)-j)):
        for i in range(min(self.max_seq, length)):
            decode_array[i] = par.mask_token
            decode_tensor = tf.constant([decode_array])
            if decode_tensor.shape[1] > self.max_seq:
                break
            result = self.call(decode_tensor, training=False)
            pdf = tfp.distributions.Categorical(probs=result[:, i])
            result = pdf.sample(1)
            result = int(result.numpy())
            decode_array[i] = result
        return decode_array

    def generate(self, prior: list, length=1024):
        decode_array = prior
        pre_num = 3
        for i in Bar('generating').iter(range(min(self.max_seq, length)-2)):
        #for i in range(min(self.max_seq, length)-2):
            #print(decode_array)
            decode_array.append(par.mask_token)
            decode_tensor = tf.constant([decode_array])
            if decode_tensor.shape[1] > self.max_seq:
                break
            result = self.call(decode_tensor, training=False)
            pdf = tfp.distributions.Categorical(probs=result[:, -1])
            pro = result[:,-1][0].numpy()
            pre_num = utils.choice_num(pro,pre_num)
            decode_array[-1] = pre_num
        return decode_array

    def _set_metrics(self):
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    def __load_config(self, config):
        print(config)
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']

    def reset_metrics(self):
        for metric in self.custom_metrics:
            metric.reset_states()
        return

if __name__ == '__main__':
    # import utils
    print(tf.executing_eagerly())

    src = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    trg = tf.constant([utils.fill_with_placeholder([1,2,3,4],max_len=2048)])
    src_mask, trg_mask, lookup_mask = utils.get_masked_with_pad_tensor(2048, src,trg)
    print(lookup_mask)
    print(src_mask)
    mt = MusicTransformer(debug=True, embedding_dim=par.embedding_dim, vocab_size=par.vocab_size)
    mt.save_weights('my_model.h5', save_format='h5')
    mt.load_weights('my_model.h5')
    result = mt.generate([27, 186,  43, 213, 115, 131], length=100)
    print(result)
    from deprecated import sequence

    sequence.EventSeq.from_array(result[0]).to_note_seq().to_midi_file('result.midi')
    pass