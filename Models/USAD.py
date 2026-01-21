import tensorflow as tf
from tensorflow import keras
import numpy as np


class USAD_MTS(keras.Model):
    def __init__(self, timesteps, n_features, latent_dim, dropout_rate=0.2, kernel_regularizer=None, batch_size=64):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.encoder = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
            keras.layers.LSTM(32, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
            keras.layers.LSTM(latent_dim, return_sequences=False,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
        ])

        self.decoder1 = keras.Sequential([
            keras.layers.RepeatVector(timesteps),
            keras.layers.LSTM(32, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate, recurrent_regularizer=keras.regularizers.L2(1e-4)),
            keras.layers.LSTM(128, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
            keras.layers.TimeDistributed(
                keras.layers.Dense(n_features, activation='linear'))
        ])

        self.decoder2 = keras.Sequential([
            keras.layers.RepeatVector(timesteps),
            keras.layers.LSTM(32, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate, recurrent_regularizer=keras.regularizers.L2(1e-4)),
            keras.layers.LSTM(128, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
            keras.layers.TimeDistributed(
                keras.layers.Dense(n_features, activation='linear'))
        ])

    def compile(self, optimizer, clipnorm=1.0, early_stop_patience=5, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.optimizer = optimizer
        self.clipnorm = clipnorm
        self.early_stop_patience = early_stop_patience

    def fit_manual(self, train_ds, epochs=100, verbose=0, early_stop_patience=5):
        train_losses = keras.metrics.Mean(name='train_loss')
        val_losses = keras.metrics.Mean(name='val_loss')
        train_ds = train_ds.cache()
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
            ckpt, './checkpoint_states/USAD', max_to_keep=1)

        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape1:
                z = self.encoder(x, training=True)
                x1_hat = self.decoder1(z, training=True)
                loss1 = tf.reduce_mean(tf.reduce_sum(
                    tf.square(x - x1_hat), axis=[1, 2]))
            vars_d1 = self.encoder.trainable_variables + self.decoder1.trainable_variables
            grads1 = tape1.gradient(loss1, vars_d1)
            # grads1, _ = tf.clip_by_global_norm(grads1, self.clipnorm)
            self.optimizer.apply_gradients(zip(grads1, vars_d1))

            with tf.GradientTape() as tape2:
                z = self.encoder(x, training=False)
                x1_hat = self.decoder1(z, training=False)
                x1_hat = tf.stop_gradient(x1_hat)
                z_hat = self.encoder(x1_hat, training=False)
                x2_hat = self.decoder2(z_hat, training=True)
                loss2 = tf.reduce_mean(tf.reduce_sum(
                    tf.square(x - x2_hat), axis=[1, 2]))
            grads2 = tape2.gradient(loss2, self.decoder2.trainable_variables)
            # grads2, _ = tf.clip_by_global_norm(grads2, self.clipnorm)
            self.optimizer.apply_gradients(
                zip(grads2, self.decoder2.trainable_variables))
            train_losses.update_state(loss1 + loss2)
            return loss1, loss2

        @tf.function
        def val_step(x):
            z = self.encoder(x, training=False)
            x1_hat = self.decoder1(z, training=False)
            z_hat = self.encoder(x1_hat, training=False)
            x2_hat = self.decoder2(z_hat, training=False)
            loss1 = tf.reduce_mean(tf.reduce_sum(
                tf.math.squared_difference(x, x1_hat), axis=[1, 2]))
            loss2 = tf.reduce_mean(tf.reduce_sum(
                tf.math.squared_difference(x, x2_hat), axis=[1, 2]))
            loss = loss1 + loss2
            val_losses.update_state(loss)
            return loss

        wait = 0
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_losses.reset_state()
            val_losses.reset_state()
            train_ds = train_split.shuffle(1000).batch(self.batch_size)
            for x in train_ds:
                train_step(x)
            for x in val_ds:
                val_step(x)
            print(f"Epoch {epoch+1}")
            if verbose > 0:
                print(
                    f"\t Train Loss: {train_losses.result():.6f} | Val Loss: {val_losses.result():.6f}")
            # Early stopping logic can be added here based on val_losses
            if best_val_loss - val_losses.result() > 1e-4:
                best_val_loss = val_losses.result()
                manager.save()
                wait = 0
            else:
                wait += 1
                if wait >= early_stop_patience:
                    if verbose > 0:
                        print("Early stopping ... ")
                    ckpt.restore(manager.latest_checkpoint)
                    break

    def score(self, data_ds):
        anomaly_scores = []
        data_ds = data_ds.batch(1024)
        for x in data_ds:
            z = self.encoder(x, training=False)
            x1_hat = self.decoder1(z, training=False)
            z_hat = self.encoder(x1_hat, training=False)
            x2_hat = self.decoder2(z_hat, training=False)
            loss = tf.reduce_mean(
                tf.math.squared_difference(x, x2_hat), axis=[1, 2])
            anomaly_scores.extend(loss.numpy())
        return np.array(anomaly_scores)
