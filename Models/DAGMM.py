import tensorflow as tf
from tensorflow import keras
import numpy as np


class DAGMM_encoder(keras.Model):
    def __init__(self, timesteps, n_features, latent_dim, hidden_unit=64, dropout_rate=0.2, kernel_regularizer=None):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.encoder = keras.Sequential([
            keras.layers.LSTM(hidden_unit, return_sequences=False,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate,
                              recurrent_regularizer=tf.keras.regularizers.L2(1e-4)),
        ])
        self.final_out = keras.layers.Dense(latent_dim)

    def call(self, x, training=False):
        x = self.encoder(x, training=training)
        z = self.final_out(x)
        return z


class DAGMM_decoder(keras.Model):
    def __init__(self, timesteps, n_features, latent_dim, hidden_unit=64, dropout_rate=0.2, kernel_regularizer=None):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.decoder = keras.Sequential([
            keras.layers.RepeatVector(timesteps),
            keras.layers.LSTM(hidden_unit, return_sequences=True,
                              kernel_regularizer=kernel_regularizer, dropout=dropout_rate, recurrent_regularizer=keras.regularizers.L2(1e-4)),
            keras.layers.TimeDistributed(
                keras.layers.Dense(n_features, activation='linear'))
        ])

    def call(self, z, training=False):
        x = self.decoder(z, training=training)
        return x


class DAGMM_estimation(keras.Model):
    def __init__(self, estimator_units, n_gmm_components, dropout_rate=0.2, kernel_regularizer=None):
        super().__init__()
        self.hidden_dim = estimator_units
        self.n_gmm_components = n_gmm_components

        self.estimation = keras.Sequential([
            keras.layers.Dense(estimator_units, activation='tanh',
                               kernel_regularizer=kernel_regularizer),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(n_gmm_components, activation='softmax',
                               kernel_regularizer=kernel_regularizer)
        ])

    def call(self, x, training=False):
        gamma = self.estimation(x, training=training)
        return gamma


class DAGMM_AE(keras.Model):
    def __init__(self, timesteps, n_features, latent_dim,
                 dropout_rate=0.2, kernel_regularizer=None, hidden_units=64, lambda_energy=1.0, lambda_cov_reg=0.005, num_gmm_components=2, epsilon=1e-3, batch_size=256):
        super().__init__()
        self.timesteps = timesteps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lambda_energy = lambda_energy
        self.lambda_cov_reg = lambda_cov_reg
        self.num_gmm_components = num_gmm_components
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.encoder = DAGMM_encoder(
            timesteps, n_features, latent_dim, hidden_units, dropout_rate, kernel_regularizer)
        self.decoder = DAGMM_decoder(
            timesteps, n_features, latent_dim, hidden_units, dropout_rate, kernel_regularizer)
        self.estimator = DAGMM_estimation(estimator_units=64, n_gmm_components=self.num_gmm_components,
                                          dropout_rate=dropout_rate, kernel_regularizer=kernel_regularizer)
        self.pi = tf.constant(np.pi, dtype=tf.float32)

    def compile(self, optimizer, clipnorm=1.0, early_stop_patience=5, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.early_stop_patience = early_stop_patience
        self.clipnorm = clipnorm

    def call(self, x, training=False):
        z = self.encoder(x, training=training)
        x_reconstructed = self.decoder(z, training=training)
        return x_reconstructed, z

    @tf.function
    def _compute_augmented_representation(self, x, x_hat, z):
        x_flat = tf.reshape(x, (tf.shape(x)[0], -1))         # (N, T*F)
        xhat_flat = tf.reshape(x_hat, (tf.shape(x_hat)[0], -1))
        # relative euclidean distance
        rec_dist = tf.norm(x_flat - xhat_flat, axis=1) / \
            (tf.norm(x_flat, axis=1) + 1e-12)  # (N,)
        rec_dist = tf.expand_dims(rec_dist, axis=1)
        dot = tf.reduce_sum(x_flat * xhat_flat, axis=1)
        cos = dot / (tf.norm(x_flat, axis=1) *
                     tf.norm(xhat_flat, axis=1) + 1e-12)
        cos_sim = tf.expand_dims(cos, axis=1)
        augmented_representation = tf.concat([z, rec_dist, cos_sim], axis=1)
        return augmented_representation

    @tf.function
    def _compute_gmm_params(self, u, gamma, num_components, eps=1e-3):
        N = tf.shape(u)[0]
        D = tf.shape(u)[1]
        K = num_components

        # mixture weights (phi)
        gamma_sum = tf.reduce_sum(gamma, axis=0) + 1e-12        # (K,)
        phi = gamma_sum / tf.reduce_sum(gamma_sum)             # (K,)

        # component means mu (K, D)
        mu = tf.matmul(gamma, u, transpose_a=True) / \
            tf.reshape(gamma_sum, (K, 1))
        # covariance per component: loop (K small)
        u_exp = tf.expand_dims(u, 1)        # (N,1,D)
        mu_exp = tf.expand_dims(mu, 0)      # (1,K,D)
        diff = u_exp - mu_exp               # (N,K,D)

        var_num = tf.einsum('nk,nkd,nkd->kd', gamma, diff, diff)  # (K,D)
        var = var_num / tf.reshape(gamma_sum, (K, 1))            # (K,D)
        # stabilize variance and ensure positive
        var = var + eps
        var_exp = tf.expand_dims(var, 0)           # (1,K,D)
        mahal = tf.reduce_sum((diff * diff) / var_exp, axis=-1)   # (N,K)

        # log determinant for diagonal cov: sum(log(var_k_d)) over D -> (K,)
        logdet = tf.reduce_sum(tf.math.log(var), axis=1)          # (K,)
        # (1,K) for broadcasting
        logdet = tf.reshape(logdet, (1, K))

        # log-probabilities: -0.5*(D*log(2π) + logdet + mahal)
        D_float = tf.cast(D, tf.float32)
        log_norm = -0.5 * \
            (D_float * tf.math.log(2.0 * self.pi) + logdet)  # (1,K)
        log_norm = tf.broadcast_to(
            log_norm, [N, K])                        # (N,K)
        log_prob = log_norm - 0.5 * \
            mahal                                   # (N,K)

        # add log mixture weights
        log_phi = tf.reshape(tf.math.log(phi + 1e-12), (1, K))  # (1,K)
        weighted_log = log_prob + log_phi                       # (N,K)

        # log-sum-exp across components -> log-likelihood per sample (N,)
        max_val = tf.reduce_max(weighted_log, axis=1, keepdims=True)   # (N,1)
        lse = max_val + tf.math.log(tf.reduce_sum(
            tf.exp(weighted_log - max_val), axis=1, keepdims=True) + 1e-12)  # (N,1)
        log_likelihood = tf.reshape(lse, (-1,))     # (N,)
        sample_energy = -log_likelihood              # (N,)

        # build cov matrices as diagonal (K, D, D) to match previous API
        covs = tf.linalg.diag(var)                   # (K, D, D)

        return phi, mu, covs, sample_energy

    def fit_manual(self, train_ds, epochs=100, verbose=0):
        train_losses = keras.metrics.Mean(name='train_loss')
        val_losses = keras.metrics.Mean(name='val_loss')

        card = tf.data.experimental.cardinality(train_ds).numpy()
        if card < 0 or card == tf.data.experimental.UNKNOWN_CARDINALITY:
            card = 0
            for _ in train_ds:
                card += 1
        train_size = int(0.8 * card)
        train_split = train_ds.take(train_size)
        val_ds = train_ds.skip(train_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        best_val_loss = float('inf')
        wait = 0

        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                x_hat, z = self(x, training=True)
                recon_loss = tf.reduce_mean(tf.reduce_sum(
                    tf.math.squared_difference(x, x_hat), axis=[1, 2]))
                u = self._compute_augmented_representation(x, x_hat, z)
                gamma = self.estimator(u, training=True)
                phi, mu, covs, energies = self._compute_gmm_params(
                    u, gamma, self.num_gmm_components, eps=self.epsilon)
                energy_loss = tf.reduce_mean(energies)
                cov_diag = tf.linalg.diag_part(covs)
                cov_reg_loss = tf.reduce_sum(1.0 / (cov_diag+1e-6))
                total_loss = recon_loss + self.lambda_energy * \
                    energy_loss + self.lambda_cov_reg * cov_reg_loss
            train_vars = self.encoder.trainable_variables + \
                self.decoder.trainable_variables + self.estimator.trainable_variables
            grads = tape.gradient(total_loss, train_vars)
            # grads, _ = tf.clip_by_global_norm(grads, self.clipnorm)
            self.optimizer.apply_gradients(zip(grads, train_vars))
            train_losses.update_state(total_loss)
            return total_loss

        @tf.function
        def val_step(x):
            x_hat, z = self(x, training=False)
            recon_loss = tf.reduce_mean(tf.reduce_sum(
                tf.math.squared_difference(x, x_hat), axis=[1, 2]))
            u = self._compute_augmented_representation(x, x_hat, z)
            gamma = self.estimator(u, training=False)
            phi, mu, covs, energies = self._compute_gmm_params(
                u, gamma, self.num_gmm_components, eps=self.epsilon)
            energy_loss = tf.reduce_mean(energies)
            cov_diag = tf.linalg.diag_part(covs)
            cov_reg_loss = tf.reduce_sum(1.0 / (cov_diag+1e-6))
            total_loss = recon_loss + self.lambda_energy * \
                energy_loss + self.lambda_cov_reg * cov_reg_loss
            val_losses.update_state(total_loss)
            return total_loss

        for epoch in range(epochs):
            train_losses.reset_state()
            val_losses.reset_state()
            train_ds = train_split.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            for x in train_ds:
                train_step(x)
            for x in val_ds:
                val_step(x)
            print(f"Epoch {epoch+1} completed.")
            if verbose > 0:
                print(
                    f"\t Train Loss: {train_losses.result():.6f} | Val Loss: {val_losses.result():.6f}")
            if best_val_loss - val_losses.result() > 1e-4:
                best_val_loss = val_losses.result()
                wait = 0
            else:
                wait += 1
                if wait >= self.early_stop_patience:
                    if verbose > 0:
                        print("Early stopping ... ")
                    break

    def score(self, data_ds):
        anomaly_scores = []
        data_ds = data_ds.batch(1024)
        for x in data_ds:
            x_hat, z = self(x, training=False)
            u = self._compute_augmented_representation(x, x_hat, z)
            gamma = self.estimator(u, training=False)
            phi, mu, covs, energies = self._compute_gmm_params(
                u, gamma, self.num_gmm_components, eps=self.epsilon)
            anomaly_scores.extend(energies.numpy())
        return np.array(anomaly_scores)
