import numpy as np
import tensorflow as tf
from transformers import TFBertTokenizer, TFBertModel

class BertClassifier:
    def build(self, bert_tokenizer, bert_model):
        text_in = tf.keras.Input(shape=(), dtype=tf.string, name='text')

        encoder_inputs = bert_tokenizer(text_in, padding="longest", truncation=True)
        outputs = bert_model(encoder_inputs)
        x = outputs['pooler_output']
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(7, activation=None, name='classifier')(x)

        model = tf.keras.Model(inputs={'text' : text_in}, outputs=x)
        return model



