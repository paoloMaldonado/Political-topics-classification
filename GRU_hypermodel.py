import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import keras_tuner as kt

class GRUHyperModel(kt.HyperModel):
    def build(self, hp):
        input_shape = (60, 300) #change
        prev_phrase_in = tf.keras.Input(shape=input_shape, name='prev_phrase')
        phrase_in      = tf.keras.Input(shape=input_shape, name='phrase')
        party_in       = tf.keras.Input(shape=(1, 78), name='party')

        # parameters tuning
        hp_dropout_rate = hp.Choice('dropout_rate', values=[0.25, 0.5, 0.8])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_gru_units = hp.Choice('gru_units', values=[32, 64, 128, 256, 512])

        mixed_layers = []

        # for previous phrase
        x_prev = layers.Masking(mask_value=0.)(prev_phrase_in)
        x_prev = layers.Bidirectional(layers.GRU(units=hp_gru_units, return_sequences=True))(x_prev)
        x_prev = layers.GlobalAveragePooling1D()(x_prev)
        mixed_layers.append(x_prev)

        # for current phrase
        x_curr = layers.Masking(mask_value=0.)(phrase_in)
        x_curr = layers.Bidirectional(layers.GRU(units=hp_gru_units, return_sequences=True))(x_curr)
        x_curr = layers.GlobalAveragePooling1D()(x_curr)
        mixed_layers.append(x_curr)

        # Append the correspond political party
        party = layers.Flatten()(party_in)
        mixed_layers.append(party)

        x = layers.Concatenate()(mixed_layers)
        x = layers.Flatten()(x)

        for i in range(hp.Int("num_hidden_layers", 0, 1)):
            x = layers.Dropout(rate=hp_dropout_rate)(x)
            x = layers.Dense(units=hp.Int(f"units_{i}", min_value=64, max_value=128, step=32), 
                             activation=None)(x)
            if hp.Boolean("batchnorm_after_hidden_layer"):
                x = layers.BatchNormalization()(x)
            x = layers.Activation(activation='sigmoid')(x)

        # Final Fully Connected
        x = layers.Dropout(rate=hp_dropout_rate)(x)
        x = layers.Dense(7, activation=None)(x)
        if hp.Boolean("relu_before_softmax"):
            x = layers.Activation("relu")(x)

        model = tf.keras.Model(inputs={'prev_phrase' : prev_phrase_in, 'phrase' : phrase_in, 'party' : party_in}, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['categorical_accuracy', 
                                tfa.metrics.F1Score(num_classes=7, average='macro', name='f1_score_macro')])

        return model

    def fit(self, hp, model, *args, **kwargs):
        #hp_batch_size = hp.Int('batch_size', min_value=8, max_value=128, step=8)

        return model.fit(
            *args,
            #batch_size=hp_batch_size,
            **kwargs,
        )