import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertTokenizer, TFBertModel
from official.nlp import optimization

from BertClassifiers import BertClassifier, BertClassifierForTwoPhrases, BertClassifierForTwoPhrasesParty

def __instanciateModel(mode, instance):
    if mode == "single_phrase":
        instance = BertClassifier()
    elif mode == "double_phrase":
        instance = BertClassifierForTwoPhrases()
    elif mode == "double_phrase_plus_party":
        instance = BertClassifierForTwoPhrasesParty()
    elif mode == "custom":
        return instance
    return instance

def createOptimizer(X, epochs=5):
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(X).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    init_lr = 3e-5
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    return optimizer

class AutoBert:
    def __init__(self, bert_model, bert_tokenizer, mode='single_phrase', instance=None):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.mode = mode
        self.model = None
        if mode == "custom" and instance == None:
            raise Exception('When using mode=custom, a valid instance of a custom bert model is expected')
        self.instance = instance
    
    def __build__(self):
        model_instance = __instanciateModel(self.mode, self.instance)
        return model_instance.build(bert_tokenizer=self.bert_tokenizer, bert_model=self.bert_model)
        
    def fit(self, X, y=None, epochs=5, validation_split=0.2, validation_data=None, checkpoint_path=None, build_only=False, **kwargs):
        self.model = self.__build__()
        
        if self.model == None:
            raise Exception('No model was instantiated, please specify a valid mode')

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['categorical_accuracy', tfa.metrics.F1Score(num_classes=7, average='macro', name='f1_score_macro')]
        optimizer = createOptimizer(X, epochs=epochs)

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)
        if build_only:
            print("Model built, skipping training, now you can use predict() with saved weights")
            return

        tf_callbacks = []
        if checkpoint_path != None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            tf_callbacks.append(cp_callback)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=kwargs['log_dir'])
        tf_callbacks.append(tensorboard_callback)

        print('Training model with mode: {}'.format(self.mode))
        self.model.fit(x=X, validation_data=validation_data,
                       epochs=epochs,
                       callbacks=tf_callbacks)
    
    def predict(self, X, y=None, from_disk=False, **kwargs):
        if from_disk:
            print("Loading saved weights from disk")
            self.model.load_weights(kwargs['checkpoint_path'])
        print("Predicting with test data")
        eval_result = self.model.evaluate(X, y)
        print("[test loss, test f1 score,[macro]]:", eval_result)
        return    
        
