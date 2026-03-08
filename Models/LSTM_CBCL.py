import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import functions
import pandas as pd
from PIL import Image
import ast, os, gc, shutil, math
import pickle


class MLPProjector(keras.layers.Layer):
    def __init__(self, hidden_units=32, out_dim=64, use_bias=True, leaky_alpha=0.01, **kwargs):
        super().__init__(**kwargs)

        self.model = keras.Sequential([
            keras.layers.Dense(hidden_units),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dense(out_dim),
        ])
        
    def call(self, x, training=False):
        return self.model(x, training=training)

class StackedEncoder(keras.layers.Layer):
    def __init__(self, num_hidden_layers=2, bottleneck=False, hidden_units=64, latent_dim=10, dropout_rate=0.2, kernel_regularizer=None, recurrent_regularizer=None, 
                 **kwargs):
        super().__init__(**kwargs) 
        self.latent_output_shape = latent_dim 
        self.layers = []
        for i in range(num_hidden_layers):
            hu = int(hidden_units / (i+1)) if bottleneck else hidden_units
            self.layers.append(
                keras.layers.LSTM(hu, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform(), dropout=dropout_rate,
                                  recurrent_initializer=tf.keras.initializers.Orthogonal(), kernel_regularizer=kernel_regularizer,
                                  recurrent_regularizer=recurrent_regularizer)
            )
        self.layers.append(
                keras.layers.LSTM(latent_dim, return_sequences=False, kernel_initializer=tf.keras.initializers.GlorotUniform(), dropout=dropout_rate,
                                  recurrent_initializer=tf.keras.initializers.Orthogonal(), kernel_regularizer=kernel_regularizer,
                                  recurrent_regularizer=recurrent_regularizer)
            )
        
        self.model = keras.Sequential(self.layers)
        
    def call(self, x, training=False):
        return self.model(x, training=training)
    
class StackedDecoder(keras.layers.Layer):
    def __init__(self, out_dim, timesteps, num_hidden_layers=2, bottleneck=False, hidden_units=64, dropout_rate=0.2, kernel_regularizer=None, recurrent_regularizer=None, 
            **kwargs):
        super().__init__(**kwargs) 
        assert out_dim is not None, "Must specify original non-encoded feature dimension"
        assert timesteps is not None, "Must specify window frame size"
        self.output_dim = out_dim 
        self.layers = [keras.layers.RepeatVector(timesteps)]
        
        for i in range(num_hidden_layers-1, -1, -1):
            hu = int(hidden_units / (i+1)) if bottleneck else hidden_units
            self.layers.append(
                keras.layers.LSTM(hu, return_sequences=True, kernel_initializer=tf.keras.initializers.GlorotUniform(), dropout=dropout_rate,
                                  recurrent_initializer=tf.keras.initializers.Orthogonal(), kernel_regularizer=kernel_regularizer,
                                  recurrent_regularizer=recurrent_regularizer)
            )
        self.layers.append(keras.layers.TimeDistributed(
                keras.layers.Dense(out_dim, activation='linear')))    
        self.model = keras.Sequential(self.layers)
        
    def call(self, x, training=False):
        return self.model(x, training=training)
    

class LSTM_CBCL(keras.Model):
    def __init__(self, timesteps, n_features, latent_dim, num_layers=2, bottleneck=True, hidden_units = 64,
                 dropout_rate=0.2, kernel_regularizer=False, batch_size=128, patience=5, recurrent_regularizer=True, MLP_hu=128, rep_dim=128):
        super().__init__()
        self.batch_size=batch_size
        self.encoder = StackedEncoder(num_hidden_layers=num_layers, bottleneck=bottleneck, hidden_units=hidden_units,
                                      latent_dim=latent_dim, dropout_rate=dropout_rate, 
                                      kernel_regularizer=tf.keras.regularizers.L2(1e-4) if kernel_regularizer else None, 
                                      recurrent_regularizer=tf.keras.regularizers.L2(1e-4) if recurrent_regularizer else None)
        
        self.decoder = StackedDecoder(n_features, timesteps, num_hidden_layers=num_layers, bottleneck=bottleneck, 
                                      hidden_units=hidden_units, dropout_rate=dropout_rate, 
                                      kernel_regularizer=tf.keras.regularizers.L2(1e-4) if kernel_regularizer else None, 
                                      recurrent_regularizer=tf.keras.regularizers.L2(1e-4) if recurrent_regularizer else None)
        
        self.projector_head = MLPProjector(hidden_units=MLP_hu, out_dim=rep_dim, use_bias=True, leaky_alpha=0.1)
        
        self.pos_aug1 = functions.AnomalyWindowSampler().augment_noise
        self.pos_aug2 = functions.AnomalyWindowSampler().augment_dropout_point
        self.neg_aug1 = functions.AnomalyWindowSampler().augment_additive
        self.neg_aug2 = functions.AnomalyWindowSampler().augment_amplify
        
        self.centroid = None
        self.weight_mse = 1.0
        self.weight_contrast = 0.08
        self.weight_centroid = 0.5
        self.weight_repel = 0.3
        self.ema_momentum = 0.99
        self.clip_norm=5
        self.patience = patience
        
        
    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.optimizer = optimizer
        
    def call(self, x, training=False):
        latents = self.encoder(x, training=training)
        projected = self.projector_head(latents, training=training)
        reconstructed = self.decoder(latents, training=training)
    
        return latents, projected, reconstructed
    
    def create_augments(self, batch_windows,random_state=123):
        if isinstance(batch_windows, tf.Tensor):
            batch_windows = batch_windows.numpy()
        N = batch_windows.shape[0]

        # produce two positive views (these should preserve "normal" semantics)
        pos1 = self.pos_aug1(batch_windows, nv=self.nv_pos_aug1, p=self.p_pos1,
                             sigma=self.noise_sigma, random_state=random_state)
        
        """ pos2 = self.pos_aug2(batch_windows, nv=self.nv_pos_aug2, p=self.p_pos2,
                             random_state=random_state + 1) """
                             
        pos2 = self.pos_aug1(batch_windows, nv=self.nv_pos_aug1, p=self.p_pos1,
                             sigma=self.noise_sigma, random_state=random_state+1)
        
        # produce two negative (anomalous) views (these should make "bad" examples)
        neg1 = self.neg_aug1(batch_windows, nv=self.nv_neg_aug1, p=self.p_neg1,
                             bias=self.bias, random_state=random_state + 2)
        
        neg2 = self.neg_aug2(batch_windows, nv=self.nv_neg_aug2, p=self.p_neg2,
                             amp_val=self.amp, random_state=random_state + 3) 

        pos_batch = np.concatenate([pos1, pos2], axis=0)
        neg_batch = np.concatenate([neg1, neg2], axis=0)
        return pos_batch, neg_batch

    def set_aug_params(self, nv_pos1=1, nv_pos2=1, nv_neg1=1, nv_neg2=1, p_pos1=0.5, p_pos2=0.5, p_neg1=0.5, p_neg2=0.5,
                       noise_sigma=0.05, bias=0.2, amp=1.5):
        self.nv_pos_aug1 = nv_pos1
        self.nv_pos_aug2 = nv_pos2
        self.nv_neg_aug1 = nv_neg1
        self.nv_neg_aug2 = nv_neg2
        self.p_pos1 = p_pos1
        self.p_pos2 = p_pos2
        self.p_neg1 = p_neg1
        self.p_neg2 = p_neg2
        self.noise_sigma = noise_sigma
        self.bias = bias
        self.amp = amp
        
    def linear_tau_scheduler(self, epoch, base_tau=0.5, min_tau=0.05, last_epoch=50):
        if epoch > last_epoch:
            return min_tau
        return base_tau - (base_tau - min_tau) * (epoch / last_epoch)
    
    def cosine_tau_scheduler(self, epoch, base_tau=0.5, min_tau=0.05, last_epoch=50):
        # Example: cosine decay schedule
        if epoch > last_epoch:
            return min_tau
        return min_tau + 0.5 * (base_tau - min_tau) * (1 + math.cos(math.pi * epoch / last_epoch))
    
    @tf.function
    def cosine_sim(self, Z):
        Z=tf.math.l2_normalize(Z, axis=1)
        return tf.matmul(Z, Z, transpose_b=True)
    
    @tf.function    
    def reconstruction_loss(self, orig, recons):
        loss = tf.reduce_mean(tf.square(orig - recons))
        return loss
    
    @tf.function  
    def contrastive_loss(self, orig_proj, pos_proj, neg_proj, tau=0.1):  
        ############## EXPERIMENTAL ###############
        N = tf.shape(orig_proj)[0]
        view1, view2 = tf.split(pos_proj, 2, axis=0)    # each (N, D)
        Z = tf.concat([orig_proj, view1, view2, neg_proj], axis=0)
        M = tf.shape(Z)[0]

        # normalize
        Z_norm = tf.math.l2_normalize(Z, axis=1)
        anchors = tf.math.l2_normalize(orig_proj, axis=1)    # (N, D)

        logits = tf.matmul(anchors, Z_norm, transpose_b=True) / tf.cast(tau, dtype=anchors.dtype)  # (N, M)

        # mask out self (orig[i] corresponds to column i in Z)
        large_neg = tf.constant(-1e9, dtype=logits.dtype)
        logits = tf.linalg.set_diag(logits, tf.fill([tf.shape(logits)[0]], large_neg))
        log_probs = tf.nn.log_softmax(logits, axis=1)  # (N, M)

        # compute positions of the two positive columns for each anchor i:
        # columns: orig -> [0 .. N-1], view1 -> [N .. 2N-1], view2 -> [2N .. 3N-1]
        col_view1 = tf.range(N, N + N, dtype=tf.int32)
        col_view2 = tf.range(N + N, N + N + N, dtype=tf.int32)

        # build mask_pos where mask_pos[i, col_view1[i]] = 1 and mask_pos[i, col_view2[i]] = 1
        idx1 = tf.stack([tf.range(N, dtype=tf.int32), col_view1], axis=1)
        idx2 = tf.stack([tf.range(N, dtype=tf.int32), col_view2], axis=1)
        mask_pos = tf.zeros([N, M], dtype=log_probs.dtype)
        mask_pos = tf.tensor_scatter_nd_update(mask_pos, idx1, tf.ones([N], dtype=log_probs.dtype))
        mask_pos = tf.tensor_scatter_nd_update(mask_pos, idx2, tf.ones([N], dtype=log_probs.dtype))

        log_prob_pos = tf.reduce_sum(log_probs * mask_pos, axis=1)   # (N,)
        pos_count = tf.reduce_sum(mask_pos, axis=1)                  # should be 2 for all anchors
        mean_log_prob = log_prob_pos / (pos_count + 1e-8)
        loss = -tf.reduce_mean(mean_log_prob)
        return loss

        
    @tf.function
    def centroid_loss(self, orig_proj, pos_proj):
        centroid = self.centroid.read_value()[None, :]
        orig_loss = tf.reduce_mean(tf.reduce_sum(tf.square(orig_proj - centroid), axis=1))
        pos_loss = tf.reduce_mean(tf.reduce_sum(tf.square(pos_proj - centroid), axis=1))
        return orig_loss + pos_loss
    
    @tf.function
    def hinge_repel(self, neg_proj, dist_repel=1.0):
        if neg_proj is None:
            return tf.constant(0.0)
        d = tf.norm(neg_proj - self.centroid.read_value()[None, :], axis=1)        # (Bneg,)
        hinge = tf.nn.relu(dist_repel - d)                            # positive if within radius
        return tf.reduce_mean(tf.square(hinge))
    
        
    def fit_manual(self, train_ds: tf.data.Dataset, epochs=100, verbose=0, clipping=False):
        train_losses = keras.metrics.Mean(name='train_loss')
        val_losses = keras.metrics.Mean(name='val_loss')

        card = tf.data.experimental.cardinality(train_ds).numpy()
        if card < 0 or card == tf.data.experimental.UNKNOWN_CARDINALITY:
            card = 0
            for _ in train_ds:
                card += 1
        train_size = int(0.75 * card)
        train_split = train_ds.take(train_size)
        drop = train_split.cardinality() >= self.batch_size

        val_ds = train_ds.skip(train_size).batch(
            self.batch_size, drop_remainder=drop).prefetch(tf.data.AUTOTUNE)

        best_val_loss = float('inf')
        wait = 0

        self.compute_center(train_split.shuffle(10000).batch(self.batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE))
        
        @tf.function
        def train_step(orig, pos_batch, neg_batch, clipping=False, tau=0.1):
            with tf.GradientTape() as tape:
                pos_latents, pos_projected, pos_reconstructed = self(pos_batch, training=True)
                neg_latents, neg_projected, neg_reconstructed = self(neg_batch, training=True)
                orig_latents, orig_projected, orig_reconstructed = self(orig, training=True)
                
                mse_loss = self.reconstruction_loss(orig, orig_reconstructed)
                contrast_loss = self.contrastive_loss(orig_projected, pos_projected, neg_projected, tau=tau)
                centroid_loss = self.centroid_loss(orig_projected, pos_projected)
                hinge_loss = self.hinge_repel(neg_projected, dist_repel=1)
                other_losses = tf.add_n(self.encoder.losses + self.decoder.losses + self.projector_head.losses) if (self.encoder.losses or self.decoder.losses or self.projector_head.losses) else 0.0
                total_loss = self.weight_mse * mse_loss + self.weight_contrast * contrast_loss + self.weight_centroid * centroid_loss + self.weight_repel * hinge_loss + other_losses
            
            train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables + self.projector_head.trainable_variables
            grad = tape.gradient(total_loss, train_vars)
            if clipping:
                grad, _ = tf.clip_by_global_norm(grad, self.clip_norm)
            self.optimizer.apply_gradients(zip(grad, train_vars))

            batch_mean = tf.reduce_mean(orig_projected, axis=0)
            self.centroid.assign(self.ema_momentum * self.centroid.read_value() + (1.0-self.ema_momentum) * batch_mean)
            
            train_losses.update_state(total_loss)
            return total_loss, self.weight_mse * mse_loss, self.weight_contrast * contrast_loss, self.weight_centroid * centroid_loss, self.weight_repel * hinge_loss

        @tf.function
        def val_step(orig, pos_batch, neg_batch, tau=0.1):
            pos_latents, pos_projected, pos_reconstructed = self(pos_batch, training=False)
            neg_latents, neg_projected, neg_reconstructed = self(neg_batch, training=False)
            orig_latents, orig_projected, orig_reconstructed = self(orig, training=False)
            
            mse_loss = self.reconstruction_loss(orig, orig_reconstructed)
            contrast_loss = self.contrastive_loss(orig_projected, pos_projected, neg_projected, tau=tau)
            centroid_loss = self.centroid_loss(orig_projected, pos_projected)
            hinge_loss = self.hinge_repel(neg_projected, dist_repel=1)
            other_loss = tf.add_n(self.encoder.losses + self.decoder.losses + self.projector_head.losses) if (self.encoder.losses or self.decoder.losses or self.projector_head.losses) else 0.0
            total_loss = self.weight_mse * mse_loss + self.weight_contrast * contrast_loss + self.weight_centroid * centroid_loss + self.weight_repel * hinge_loss + other_loss
            
            val_losses.update_state(total_loss)

        for e in range(epochs):
            train_losses.reset_states()
            val_losses.reset_states()
            train_ds = train_split.shuffle(10000).batch(self.batch_size, drop_remainder=drop).prefetch(tf.data.AUTOTUNE)
            for idx, batch in enumerate(train_ds):
                pos_batch, neg_batch = self.create_augments(batch.numpy())
                pos_batch_tf = tf.convert_to_tensor(
                    pos_batch, dtype=tf.float32)
                neg_batch_tf = tf.convert_to_tensor(
                    neg_batch, dtype=tf.float32)
                tau = self.cosine_tau_scheduler(e, base_tau=0.5, min_tau=0.05, last_epoch=epochs//2)
                #tau = 0.1
                total_loss, mse_loss, contrast_loss, centroid_loss, hinge_loss = train_step(batch, pos_batch_tf, neg_batch_tf, clipping=clipping, tau=tau)

            for idx, batch in enumerate(val_ds):
                pos_batch, neg_batch = self.create_augments(batch.numpy())

                pos_batch_tf = tf.convert_to_tensor(
                    pos_batch, dtype=tf.float32)
                neg_batch_tf = tf.convert_to_tensor(
                    neg_batch, dtype=tf.float32)
                tau = self.cosine_tau_scheduler(e, base_tau=0.5, min_tau=0.05, last_epoch=epochs//2)
                val_loss = val_step(batch, pos_batch_tf, neg_batch_tf, tau=tau)

            cur_val = val_losses.result().numpy()
            if verbose:
                print(f"Epoch {e+1}/{epochs} - train_loss: {train_losses.result().numpy():.6f} - val_loss: {cur_val:.6f}")
                print(f'\t Breakdown: \n\t MSE: {mse_loss:.6f}, Contrastive: {contrast_loss:.6f}, Cluster: {centroid_loss:.6f}, Hinge: {hinge_loss:.6f}')

            # checkpointing + early stopping
            if best_val_loss - cur_val > 1e-3:
                best_val_loss = cur_val
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    if verbose:
                        print("Early stopping.")
                    break
    
    def score(self, data_ds, use_mse=True):
        mse_list, cluster_list = [], []
        centroid = self.centroid.read_value()[None, :]
        for batch in data_ds.batch(1024):
            _, projected, reconstructed = self(batch, training=False)
            mse_list.append(tf.reduce_mean(tf.square(batch - reconstructed), axis=[1,2]))
            cluster_list.append(tf.reduce_sum(tf.square(projected - centroid), axis=1))
        return tf.concat(cluster_list, axis=0).numpy()
        #return mse_scores
        
    def compute_center(self, train_ds, eps=0.1):
        running_sum = None
        count = 0
        for batch in train_ds:
            _, projected, _ = self(batch, training=False)
            batch_sum = tf.reduce_sum(projected, axis=0)
            running_sum = batch_sum if running_sum is None else running_sum + batch_sum
            count += tf.shape(projected)[0]
        center = running_sum / tf.cast(count, tf.float32)
        if self.centroid is not None:
            self.centroid.assign(center)
        else:
            self.centroid = tf.Variable(center, trainable=False)
        
    
    