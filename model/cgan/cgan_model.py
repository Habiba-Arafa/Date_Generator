from __future__ import annotations

import tensorflow as tf

from ..shared.config import (
    NUM_DAYS_OF_WEEK, NUM_MONTHS, NUM_LEAP_OPTIONS, NUM_DECADES,
    EMBEDDING_DIM, CONDITION_DIM, DATE_DIM,
    CGAN_NOISE_DIM, CGAN_HIDDEN_DIM,
)


# shared condition embedding 

class ConditionEmbedder(tf.keras.layers.Layer):
 
    def __init__(self, embed_dim: int = EMBEDDING_DIM, **kwargs) -> None:
        super().__init__(**kwargs)
        self.day_embed    = tf.keras.layers.Embedding(NUM_DAYS_OF_WEEK, embed_dim)
        self.month_embed  = tf.keras.layers.Embedding(NUM_MONTHS,       embed_dim)
        self.leap_embed   = tf.keras.layers.Embedding(NUM_LEAP_OPTIONS, embed_dim)
        self.decade_embed = tf.keras.layers.Embedding(NUM_DECADES,      embed_dim)

    def call(self, conditions: tf.Tensor) -> tf.Tensor:
        d = self.day_embed   (conditions[:, 0])
        m = self.month_embed (conditions[:, 1])
        l = self.leap_embed  (conditions[:, 2])
        e = self.decade_embed(conditions[:, 3])
        return tf.concat([d, m, l, e], axis=-1)  


#  Generator 

class CGANGenerator(tf.keras.Model):
    def __init__(self, hidden_dim, noise_dim, **kwargs):
        super().__init__(**kwargs)

        self.noise_dim = noise_dim
        self.embedder = ConditionEmbedder()

        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.dense2 = tf.keras.layers.Dense(hidden_dim)
        self.dense3 = tf.keras.layers.Dense(hidden_dim // 2)
        self.out = tf.keras.layers.Dense(DATE_DIM, activation="sigmoid")

        self.act = tf.keras.layers.LeakyReLU(0.2)

    def call(self, conditions, noise, training=False):
        cond = self.embedder(conditions)
        x = tf.concat([cond, noise], axis=-1)

        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.act(self.dense3(x))

        return self.out(x)
    
    def generate_samples(self, conditions: tf.Tensor, num_samples: int) -> tf.Tensor:
       
        batch = tf.shape(conditions)[0]

        cond_rep = tf.repeat(conditions, repeats=num_samples, axis=0)
        noise = tf.random.normal((batch * num_samples, self.noise_dim))

        samples = self(cond_rep, noise, training=False)

        return tf.reshape(samples, (batch, num_samples, DATE_DIM))

# Discriminator 

class CGANDiscriminator(tf.keras.Model):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)

        self.embedder = ConditionEmbedder()

        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.dense2 = tf.keras.layers.Dense(hidden_dim)
        self.dense3 = tf.keras.layers.Dense(hidden_dim // 2)
        self.out = tf.keras.layers.Dense(1)

        self.act = tf.keras.layers.LeakyReLU(0.2)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, conditions, date_vec, training=False):
        cond = self.embedder(conditions)
        x = tf.concat([cond, date_vec], axis=-1)

        x = self.act(self.dense1(x))
        x = self.dropout(x, training=training)

        x = self.act(self.dense2(x))
        x = self.dropout(x, training=training)

        x = self.act(self.dense3(x))

        return self.out(x)