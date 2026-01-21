import tensorflow as tf
from tensorflow import keras
import numpy as np


class VAE_sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def KL_normal(z_mean, z_log_var):
    kl_loss = -0.5 * \
        tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) -
                      tf.exp(z_log_var), axis=1)
    return kl_loss


class VAE_encoder(keras.Model):
    def __init__(self, timesteps, n_features, lstm_h_dim, hidden_dim, dropout_rate=0.2, kernel_regularizer=None):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = hidden_dim

        self.lstm_encoder = keras.Sequential([keras.layers.LSTM(lstm_h_dim, return_sequences=True,
                                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
                                              keras.layers.LSTM(int(lstm_h_dim/2), return_sequences=False,
                                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4))])
        self.z_mean = keras.layers.Dense(hidden_dim)
        self.z_log_var = keras.layers.Dense(hidden_dim)
        self.sampling = VAE_sampling()

    def call(self, x, training=False):
        x = self.lstm_encoder(x, training=training)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z, z_mean, z_log_var


class VAE_decoder(keras.Model):
    def __init__(self, timesteps, n_features, lstm_h_dim, latent_dim, dropout_rate=0.2, kernel_regularizer=None):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.repeat = keras.layers.RepeatVector(timesteps)
        self.lstm_decoder = keras.Sequential([keras.layers.LSTM(lstm_h_dim, return_sequences=True,
                                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
                                              keras.layers.LSTM(int(lstm_h_dim/2), return_sequences=True,
                                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4))])

        self.x_mean = keras.layers.TimeDistributed(
            keras.layers.Dense(n_features), name='x_mean')

    def call(self, z, training=False):
        input = self.repeat(z)
        x = self.lstm_decoder(input, training=training)
        mu = self.x_mean(x)
        return mu


class LSTM_VAE(keras.Model):
    def __init__(self, timesteps, n_features, lstm_h_dim, latent_dim, dropout_rate=0.2, kernel_regularizer=None, kl_beta=1.0, batch_size=128):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta
        self.batch_size = batch_size
        self.encoder = VAE_encoder(
            timesteps, n_features, lstm_h_dim, latent_dim, dropout_rate, kernel_regularizer)
        self.decoder = VAE_decoder(
            timesteps, n_features, lstm_h_dim, latent_dim, dropout_rate, kernel_regularizer)

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, clipnorm=1.0, early_stop_patience=5, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.clipnorm = clipnorm
        self.early_stop_patience = early_stop_patience

    def reconstruction_mse_per_sample(self, x, x_mu):
        # returns per-sample MSE (mean over timesteps and features)
        # shape: (batch,)
        return tf.reduce_mean(tf.square(x - x_mu), axis=[1, 2])

    def call(self, x, training=False):
        z, z_mu, z_log_var = self.encoder(x, training=training)
        x_mu = self.decoder(z, training=training)
        return x_mu, z_mu, z_log_var

    def kl_annealing(self, epoch):
        # KL warmup: starts at 0.0, gradually increases to 1.0
        # This allows model to learn reconstruction first, then add KL regularization
        return min(1.0, (epoch + 1) / self.epochs)

    def fit_manual(self, train_ds, epochs=100, verbose=0, clipping=False):
        train_losses = keras.metrics.Mean(name='train_loss')
        val_losses = keras.metrics.Mean(name='val_loss')
        self.epochs = epochs

        ckpt = tf.train.Checkpoint(
            model=self,
            optimizer=self.optimizer
        )

        card = tf.data.experimental.cardinality(train_ds).numpy()
        if card < 0 or card == tf.data.experimental.UNKNOWN_CARDINALITY:
            card = 0
            for _ in train_ds:
                card += 1
        train_size = int(0.8 * card)
        train_split = train_ds.take(train_size)
        val_split = train_ds.skip(train_size)
        val_ds = val_split.batch(self.batch_size)

        manager = tf.train.CheckpointManager(
            ckpt, './checkpoint_states/LSTM-VAE', max_to_keep=1)

        @tf.function
        def train_step(x, kl_anneal_factor):
            with tf.GradientTape() as tape:
                x_mu, z_mu, z_log_var = self(x, training=True)
                recon_loss_per_sample = self.reconstruction_mse_per_sample(
                    x, x_mu)
                kl_loss = tf.reduce_mean(
                    KL_normal(z_mu, z_log_var)) * self.kl_beta * kl_anneal_factor
                recon_loss = tf.reduce_mean(recon_loss_per_sample)
                loss = recon_loss + kl_loss

            gradients = tape.gradient(loss, self.trainable_variables)
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(
                gradients, self.trainable_variables)]
            if clipping:
                grads, _ = tf.clip_by_global_norm(
                    grads, self.clipnorm)
            self.optimizer.apply_gradients(
                zip(grads, self.trainable_variables))
            train_losses.update_state(loss)
            return loss

        @tf.function
        def val_step(x, kl_anneal_factor):
            x_mu, z_mu, z_log_var = self(x, training=False)
            recon_loss_per_sample = self.reconstruction_mse_per_sample(x, x_mu)
            kl_loss = tf.reduce_mean(KL_normal(z_mu, z_log_var)) * \
                self.kl_beta * kl_anneal_factor
            recon_loss = tf.reduce_mean(recon_loss_per_sample)
            loss = recon_loss + kl_loss
            val_losses.update_state(loss)
            return loss

        wait = 0
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_losses.reset_state()
            val_losses.reset_state()

            # Compute KL annealing factor for this epoch
            kl_anneal_factor = tf.constant(
                self.kl_annealing(epoch), dtype=tf.float32)

            train_ds = train_split.shuffle(1000).batch(self.batch_size)
            for x in train_ds:
                train_step(x, kl_anneal_factor)
            for x in val_ds:
                val_step(x, kl_anneal_factor)
            print(f"Epoch {epoch+1}")
            # Early stopping logic can be added here based on val_losses
            if verbose > 0:
                print(
                    f"\t Train Loss: {train_losses.result():.6f} | Val Loss: {val_losses.result():.6f}")
            if best_val_loss - val_losses.result() > 1e-5:
                best_val_loss = val_losses.result()
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
            _, z_mu, z_log_var = self(x, training=False)
            x_mu = self.decoder(z_mu, training=False)
            recon_loss_per_sample = self.reconstruction_mse_per_sample(x, x_mu)
            anomaly_scores.extend(recon_loss_per_sample.numpy())
        return np.array(anomaly_scores)
