import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import keras_tuner as kt

class GRUHyperModel(kt.HyperModel):
    def build(self, hp):
        input_shape = (80, 100) #change
        inputs = tf.keras.Input(shape=input_shape)

        # parameters tuning
        hp_dropout_rate = hp.Choice('dropout', values=[0.0, 0.15, 0.25, 0.5, 0.75, 0.8])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        hp_activation = hp.Choice('activation_linear_layer', values=["relu", "sigmoid"])
        hp_gru_units = hp.Int('gru_units', min_value=32, max_value=256, step=32)

        x = tf.keras.layers.Masking(mask_value=0.)(inputs)
        x = layers.Bidirectional(layers.GRU(units=hp_gru_units, return_sequences=True))(x)
        x = layers.GlobalAveragePooling1D()(x)

        for i in range(hp.Int("num_layers", 0, 3)):
            x = layers.Dropout(rate=hp_dropout_rate)(x)
            x = layers.Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32), 
                             activation=None)(x)
            if hp.Boolean("batchnorm"):
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=hp_activation)(x)

        # Final Fully Connected
        x = layers.Dropout(rate=hp_dropout_rate)(x)
        x = layers.Dense(5, activation=None)(x)
        if hp.Boolean("batchnorm"):
            x = layers.BatchNormalization()(x)
        if hp.Boolean("relu_before_softmax"):
            x = layers.Activation("relu")(x)
        x = layers.Activation("softmax")(x)

        model=tf.keras.Model(inputs, x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=[tfa.metrics.F1Score(num_classes=5, average='micro', name='f1_score_micro'), 
                               tfa.metrics.F1Score(num_classes=5, average='macro', name='f1_score_macro')])

        return model

    def fit(self, hp, model, *args, **kwargs):
        hp_batch_size = hp.Int('batch_size', min_value=8, max_value=128, step=8)

        return model.fit(
            *args,
            batch_size=hp_batch_size,
            **kwargs,
        )