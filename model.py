from custom.layers import *
from custom.callback import *
import params as par
import sys
from tensorflow.python import keras
import json
import tensorflow_probability as tfp
import random
import utils
from progress.bar import Bar
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"

tf.executing_eagerly()

class MusicTransformerDecoder(keras.Model):
    def __init__(self, embedding_dim=256, vocab_size=par.vocab_size, num_layer=6,
                 max_seq=1024, dropout=0.2, debug=False, loader_path=None, dist=False):
        super(MusicTransformerDecoder, self).__init__()

        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.Decoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq)
        self.fc = keras.layers.Dense(self.vocab_size, activation=None, name='output')
        self.cla_layer = keras.layers.Dense(2, activation=tf.nn.sigmoid, name='classification')
        self._set_metrics()

        if loader_path is not None:
            self.load_ckpt_file(loader_path)

    def call(self, inputs, training=None, eval=None, classification=None, lookup_mask=None):
        bar_token = utils.bar_id(inputs)
        decoder, w = self.Decoder(inputs, bar_x=bar_token, training=training, mask=lookup_mask)
        fc = self.fc(decoder)
        if training:
            return fc
        elif eval:
            return fc, w
        elif classification:
            return self.cla_layer(fc[:,0])
        else:
            return tf.nn.softmax(fc)

    def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        if self._debug:
            tf.print('sanity:\n', self.sanity_check(x, y, mode='d'), output_stream=sys.stdout)

        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)

        if self.dist:
            predictions = self.__dist_train_step(
                x, y, look_ahead_mask, True)
        else:
            predictions = self.__train_step(x, y, look_ahead_mask, True)

        if self._debug:
            print('train step finished')
        result_metric = []

        if self.dist:
            loss = self._distribution_strategy.reduce(tf.distribute.ReduceOp.MEAN, self.loss_value, None)
        else:
            loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        #print(predictions.shape, y.shape)
        for metric in self.custom_metrics:
            result_metric.append(metric(y, predictions).numpy())

        return [loss.numpy()]+result_metric

    def cla_train_on_batch(self, x, y=None, sample_weight=None, class_weight=None, reset_metrics=True):
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)
        y = tf.constant(y)
        y = tf.cast(y, tf.float32)
        predictions = self.cla_train_step(x, y, look_ahead_mask)
        result_metric = []
        loss = tf.reduce_mean(self.loss_value)
        loss = tf.reduce_mean(loss)
        #print(y)
        #print(predictions)
        y = y.numpy()
        predictions = predictions.numpy()
        count = 0
        for i in range(y.shape[0]):
            if (y[i][0] < y[i][1]) == (predictions[i][0] < predictions[i][1]):
                count += 1
        result_metric = [count/y.shape[0]]
        #for metric in self.custom_metrics:
        #    result_metric.append(metric(y, predictions).numpy())
        return [loss.numpy()]+result_metric

    def cla_train_step(self, inp_tar, out_tar, lookup_mask):
        #print(inp_tar)
        with tf.GradientTape() as tape:
            predictions = self.call(inputs=inp_tar, lookup_mask=lookup_mask, classification=True)
            self.loss_value = self.loss(out_tar, predictions)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return predictions

    # @tf.function
    def __train_step(self, inp_tar, out_tar, lookup_mask, training):
        with tf.GradientTape() as tape:
            predictions = self.call(
                inputs=inp_tar, lookup_mask=lookup_mask, training=training
            )
            self.loss_value = self.loss(out_tar, predictions)
            #print(self.loss_value.shape)
        gradients = tape.gradient(self.loss_value, self.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return predictions

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None,
                 max_queue_size=10, workers=1, use_multiprocessing=False):

        # x, inp_tar, out_tar = MusicTransformer.__prepare_train_data(x, y)
        _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)
        predictions, w = self.call(
                x, lookup_mask=look_ahead_mask, training=False, eval=True)
        loss = tf.reduce_mean(self.loss(y, predictions))
        result_metric = []
        for metric in self.custom_metrics:
            result_metric.append(metric(y, tf.nn.softmax(predictions)).numpy())
        return [loss.numpy()] + result_metric, w
    
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

    def sanity_check(self, x, y, mode='v', step=None):
        # mode: v -> vector, d -> dict
        # x, inp_tar, out_tar = self.__prepare_train_data(x, y)

        _, tar_mask, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x)
        predictions = self.call(
            x, lookup_mask=look_ahead_mask, training=False)

        if mode == 'v':
            tf.summary.image('vector', tf.expand_dims(predictions, -1), step)
            return predictions
        elif mode == 'd':
            dic = {}
            for row in tf.argmax(predictions, -1).numpy():
                for col in row:
                    try:
                        dic[str(col)] += 1
                    except KeyError:
                        dic[str(col)] = 1
            return dic
        else:
            tf.summary.image('tokens', tf.argmax(predictions, -1), step)
            return tf.argmax(predictions, -1)

    def get_config(self):
        config = {}
        config['debug'] = self._debug
        config['max_seq'] = self.max_seq
        config['num_layer'] = self.num_layer
        config['embedding_dim'] = self.embedding_dim
        config['vocab_size'] = self.vocab_size
        config['dist'] = self.dist
        return config

    def generate(self, prior: list, beam=None, length=2048, tf_board=False):
        decode_array = prior
        decode_array = tf.constant([decode_array])

        # TODO: add beam search
        if beam is not None:
            k = beam
            for i in range(min(self.max_seq, length)):
                if decode_array.shape[1] >= self.max_seq:
                    break
                if i % 100 == 0:
                    print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                _, _, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], decode_array, decode_array)

                result = self.call(decode_array, lookup_mask=look_ahead_mask, training=False, eval=False)
                #print(result.shape)
                if tf_board:
                    tf.summary.image('generate_vector', tf.expand_dims([result[0]], -1), i)

                result = result[:,-1,:]
                result = tf.reshape(result, (1, -1))
                result, result_idx = tf.nn.top_k(result, k)
                row = result_idx // par.vocab_size
                col = result_idx % par.vocab_size

                result_array = []
                for r, c in zip(row[0], col[0]):
                    prev_array = decode_array[r.numpy()]
                    result_unit = tf.concat([prev_array, [c.numpy()]], -1)
                    result_array.append(result_unit.numpy())
                    # result_array.append(tf.concat([decode_array[idx], result[:,idx_idx]], -1))
                decode_array = tf.constant(result_array)
                del look_ahead_mask
            decode_array = decode_array[0]

        else:
            for i in Bar('generating').iter(range(min(self.max_seq, length))):
                #print(decode_array)

                if decode_array.shape[1] >= self.max_seq:
                    break
                # if i % 100 == 0:
                #     print('generating... {}% completed'.format((i/min(self.max_seq, length))*100))
                _, _, look_ahead_mask = \
                    utils.get_masked_with_pad_tensor(decode_array.shape[1], decode_array, decode_array)
                result = self.call(decode_array, lookup_mask=look_ahead_mask, training=False)
                
                #print(result.shape)
                if tf_board:
                    tf.summary.image('generate_vector', tf.expand_dims(result, -1), i)
                # import sys
                # tf.print('[debug out:]', result, sys.stdout )
                u = random.uniform(0, 1)
                if u > 1:
                    result = tf.argmax(result[:, -1], -1)
                    result = tf.cast(result, tf.int32)
                    decode_array = tf.concat([decode_array, tf.expand_dims(result, -1)], -1)
                else:
                    pdf = tfp.distributions.Categorical(probs=result[:, -1])
                    result = pdf.sample(1)
                    result = tf.transpose(result, (1, 0))
                    result = tf.cast(result, tf.int32)
                    decode_array = tf.concat([decode_array, result], -1)
                # decode_array = tf.concat([decode_array, tf.expand_dims(result[:, -1], 0)], -1)
                del look_ahead_mask
            decode_array = decode_array[0]

        return decode_array.numpy()

    def generate_mask(self, prior: list, beam=None, length=1024, tf_board=False):
        decode_array = prior
        #print(len(decode_array))
        for i in Bar('generating').iter(range(min(self.max_seq, length))):
            decode_array[i] = par.mask_token
            decode_tensor = tf.constant([decode_array])
            #print(decode_tensor)
            if decode_tensor.shape[1] > self.max_seq:
                break
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_tensor.shape[1], decode_tensor, decode_tensor)
            result = self.call(decode_tensor, lookup_mask=look_ahead_mask, training=False)
            pdf = tfp.distributions.Categorical(probs=result[:, i])
            result = pdf.sample(1)
            result = int(result.numpy())
            decode_array[i] = result
            #result = tf.transpose(result, (1, 0))
            #result = tf.cast(result, tf.int32)
            #decode_array = tf.concat([decode_array, result], -1)
            del look_ahead_mask
        return decode_array

    def _set_metrics(self):
        accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.custom_metrics = [accuracy]

    def __load_config(self, config):
        self._debug = config['debug']
        self.max_seq = config['max_seq']
        self.num_layer = config['num_layer']
        self.embedding_dim = config['embedding_dim']
        self.vocab_size = config['vocab_size']
        self.dist = config['dist']

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