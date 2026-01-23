"""Reimplement TimeGAN-pytorch Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks," NeurIPS 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)
-----------------------------
discriminative_metric.py
Note: Use post-hoc RNN to classify original data and synthetic data
Output: discriminative score (np.abs(classification accuracy - 0.5))
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from measure_score.Utils.metric_utils import train_test_divide, extract_time


def batch_generator(data, time, batch_size):
    """Mini-batch generator."""
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)
    return X_mb, T_mb


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data"""
    # Basic Parameters
    ori_data, generated_data = np.asarray(ori_data), np.asarray(generated_data)
    no, seq_len, dim = ori_data.shape
    if generated_data.shape[-1] == 21:          # CMAPSS 21→14 通道
        generated_data = generated_data[:, :,
                            [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]]
    ori_time, ori_max_seq_len = extract_time(ori_data)
    gen_time, gen_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, gen_max_seq_len)

    hidden_dim = int(dim / 2)
    iterations = 3000
    batch_size = 128

    class Discriminator(tf.keras.Model):
        def __init__(self, hidden_dim):
            super().__init__()
            self.gru = tf.keras.layers.GRU(hidden_dim,
                                          activation='tanh',
                                          return_sequences=False,
                                          name='d_gru')
            self.logit = tf.keras.layers.Dense(1, activation=None, name='d_logit')

        def call(self, x, training=None):
            gru_out = self.gru(x, training=training)
            return self.logit(gru_out)

    discriminator = Discriminator(hidden_dim)

    bce_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    bce_fake = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, gen_time)

    @tf.function
    def train_step(x_real, x_fake):
        with tf.GradientTape() as tape:
            logit_real = discriminator(x_real, training=True)
            logit_fake = discriminator(x_fake, training=True)
            loss_real = bce_real(tf.ones_like(logit_real), logit_real)
            loss_fake = bce_fake(tf.zeros_like(logit_fake), logit_fake)
            d_loss = loss_real + loss_fake
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss

    from tqdm.auto import tqdm
    for itt in tqdm(range(iterations), desc='training', total=iterations):
        x_mb, t_mb = batch_generator(train_x, train_t, batch_size)
        x_hat_mb, t_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
        # 按原始时间长度做动态长度掩码
        x_real = tf.keras.preprocessing.sequence.pad_sequences(
            x_mb, maxlen=max_seq_len, dtype='float32', padding='post')
        x_fake = tf.keras.preprocessing.sequence.pad_sequences(
            x_hat_mb, maxlen=max_seq_len, dtype='float32', padding='post')
        step_d_loss = train_step(x_real, x_fake)
        if itt % 500 == 0:
            print('Loss: {}'.format(step_d_loss.numpy()))

    def predict(x_list):
        x_pad = tf.keras.preprocessing.sequence.pad_sequences(
            x_list, maxlen=max_seq_len, dtype='float32', padding='post')
        logits = discriminator(x_pad, training=False)
        return tf.nn.sigmoid(logits).numpy()

    y_pred_real = predict(test_x)
    y_pred_fake = predict(test_x_hat)

    y_pred_final = np.squeeze(np.concatenate([y_pred_real, y_pred_fake]))
    y_label_final = np.concatenate([
        np.ones(len(y_pred_real)),
        np.zeros(len(y_pred_fake))
    ])

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    fake_acc = accuracy_score(np.zeros(len(y_pred_fake)),
                              (y_pred_fake > 0.5).flatten())
    real_acc = accuracy_score(np.ones(len(y_pred_real)),
                              (y_pred_real > 0.5).flatten())
    print("Fake Accuracy: ", fake_acc)
    print("Real Accuracy: ", real_acc)

    discriminative_score = np.abs(0.5 - acc)
    print("Discriminative Score:", discriminative_score)
    return discriminative_score, fake_acc, real_acc
