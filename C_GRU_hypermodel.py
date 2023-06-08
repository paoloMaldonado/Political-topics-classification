import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import keras_tuner as kt

class sCGRU(kt.HyperModel):
    def build(self, hp):
        input_shape = (60, 300) #change
        prev_phrase_in = tf.keras.Input(shape=input_shape, name='prev_phrase')
        phrase_in      = tf.keras.Input(shape=input_shape, name='phrase')
        party_in       = tf.keras.Input(shape=(1, 78), name='party')

        # parameters tuning
        hp_dropout_rate = hp.Choice('dropout_rate', values=[0.25, 0.5, 0.8])
        hp_filters = hp.Choice('filters', values=[100, 150])
        hp_kernel_size = hp.Choice('kernel_size', values=[2, 3, 4])
        hp_units = hp.Choice('units', values=[32, 64, 128, 256])

        mixed_layers = []

        # for previous phrase
        prev_phrase_in_drop = layers.Dropout(rate=hp_dropout_rate)(prev_phrase_in)
        # Convolutions with current phrase
        conv = layers.Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, name="prev_conv1D", trainable=True)(prev_phrase_in_drop)
        conv = layers.Activation('relu')(conv)
        # RNN Layer (GRU)
        gru = layers.Masking(mask_value=0.)(conv)
        gru = layers.GRU(units=hp_units, name="prev_gru", trainable=True)(gru)
        mixed_layers.append(gru)

        # for previous phrase
        phrase_in_drop = layers.Dropout(rate=hp_dropout_rate)(phrase_in)
        # Convolutions with current phrase
        conv = layers.Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, name="curr_conv1D", trainable=True)(phrase_in_drop)
        conv = layers.Activation('relu')(conv)
        # RNN Layer (GRU)
        gru = layers.Masking(mask_value=0.)(conv)
        gru = layers.GRU(units=hp_units, name="curr_gru", trainable=True)(gru)
        mixed_layers.append(gru)

        # Append the correspond political party
        party = layers.Flatten()(party_in)
        mixed_layers.append(party)

        x = layers.Concatenate()(mixed_layers)

        # Final Fully Connected
        x = layers.Dropout(rate=hp_dropout_rate)(x)
        x = layers.Dense(7, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.001), name="softmax_layer")(x)

        model = tf.keras.Model(inputs={'prev_phrase' : prev_phrase_in, 'phrase' : phrase_in, 'party' : party_in}, outputs=x)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
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