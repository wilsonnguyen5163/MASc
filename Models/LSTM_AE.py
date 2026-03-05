import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from tensorflow import keras
import numpy as np
import pandas as pd
import functions


class LSTM_AE(keras.Model):
    def __init__(self, timesteps, n_features, latent_dim, dropout_rate=0.2, kernel_regularizer=None, batch_size=128, use_recurrent_regularizer=True):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=False, input_shape=(timesteps, n_features),
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4) if use_recurrent_regularizer else None),
        ])

        self.decoder = keras.Sequential([
            keras.layers.RepeatVector(timesteps),
            keras.layers.LSTM(128, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate, recurrent_regularizer=keras.regularizers.L2(1e-4) if use_recurrent_regularizer else None),
            keras.layers.TimeDistributed(
                keras.layers.Dense(n_features, activation='linear'))
        ])

    def compile(self, optimizer, loss_fn, clipnorm=1.0, early_stop_patience=10, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clipnorm = clipnorm
        self.early_stop_patience = early_stop_patience

    def call(self, x, training=False):
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

    def fit_manual(self, train_ds, epochs=100, verbose=0, clipping=False):
        train_losses = keras.metrics.Mean(name='train_loss')
        val_losses = keras.metrics.Mean(name='val_loss')

        card = tf.data.experimental.cardinality(train_ds).numpy()
        if card < 0 or card == tf.data.experimental.UNKNOWN_CARDINALITY:
            card = 0
            for _ in train_ds:
                card += 1
        train_size = int(0.8 * card)
        train_split = train_ds.take(train_size)
        val_split = train_ds.skip(train_size)
        val_ds = val_split.batch(self.batch_size)

        ckpt = tf.train.Checkpoint(
            model=self,
            optimizer=self.optimizer
        )
        manager = tf.train.CheckpointManager(
            ckpt, './checkpoint_states/LSTM', max_to_keep=1)

        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                reconstructed = self(x, training=True)
                loss = self.loss_fn(x, reconstructed)
                if self.losses:
                    loss += tf.add_n(self.losses)
            gradients = tape.gradient(loss, self.trainable_variables)
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(
                gradients, self.trainable_variables)]
            if clipping:
                grads, global_norm = tf.clip_by_global_norm(
                    grads, self.clipnorm)
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables))
            train_losses.update_state(loss)
            return loss

        @tf.function
        def eval_step(x):
            reconstructed = self(x, training=False)
            loss = self.loss_fn(x, reconstructed)
            val_losses.update_state(loss)
            return loss

        wait = 0
        best_loss = float('inf')
        for epoch in range(epochs):
            train_losses.reset_state()
            val_losses.reset_state()
            ds = train_split.shuffle(1000).batch(self.batch_size)
            for x in ds:
                train_step(x)
            for x in val_ds:
                eval_step(x)
            print(f"Epoch {epoch+1}")
            if verbose > 0:
                print(
                    f"\t Train Loss: {train_losses.result():.6f} | Val Loss: {val_losses.result():.6f}")
            # Early stopping logic can be added here based on val_losses
            if best_loss - val_losses.result() > 1e-4:
                best_loss = val_losses.result()
                manager.save()
                wait = 0
            else:
                wait += 1
                if wait >= self.early_stop_patience:
                    if verbose > 0:
                        print("Early stopping ... ")
                    ckpt.restore(manager.latest_checkpoint)
                    break

    def score(self, data_ds):
        anomaly_scores = []
        data_ds = data_ds.batch(1024)
        for x in data_ds:
            reconstructed = self(x, training=False)
            loss = tf.reduce_mean(tf.math.squared_difference(
                x, reconstructed), axis=[1, 2])
            anomaly_scores.extend(loss.numpy())
        return np.array(anomaly_scores)
