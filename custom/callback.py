#from tensorflow.python import keras
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import params as par
import sys
#from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


def transformer_dist_train_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
    mask = tf.cast(mask, tf.float32)
    #print(y_pred[0])
    y_true_vector = tf.one_hot(y_true, par.vocab_size)
    _loss = tf.nn.softmax_cross_entropy_with_logits(y_true_vector, y_pred)
    #print(_loss.shape)
    #
    # _loss = tf.reduce_mean(_loss, -1)
    _loss *= mask
    #print(_loss)
    return _loss

def classification_loss(y_true, y_pred, real, pred):
    lam1 = tf.cast(0, tf.float32)
    lam2 = tf.cast(1.0, tf.float32)

    real = tf.cast(real, tf.int32)
    mask = tf.math.logical_not(tf.math.equal(real, par.pad_token))
    mask = tf.cast(mask, tf.float32)
    real = tf.one_hot(real, par.vocab_size)
    mask_loss = tf.nn.softmax_cross_entropy_with_logits(real, pred)
    mask_loss *= mask
    mask_loss_ = tf.reduce_mean(mask_loss)

    print('mask loss :', tf.reduce_mean(mask_loss_).numpy())
    #print(y_true[:10])
    _loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    #print('cla loss :', tf.reduce_mean(_loss).numpy())
    _loss = tf.add(lam1 * tf.reduce_mean(mask_loss), lam2 * _loss)
    return _loss


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        super(CustomSchedule, self).get_config()

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


if __name__ == '__main__':
    import numpy as np
    loss = TransformerLoss()(np.array([[1],[0],[0]]), tf.constant([[0.5,0.5],[0.1,0.1],[0.1,0.1]]))
    print(loss)
