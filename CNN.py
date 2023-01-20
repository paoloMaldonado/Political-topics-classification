import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt
from CNN_hypermodel import ConvHyperModel

class AutoCNN:
    def __init__(self):
        self.tuner = None
        self.hypermodel = None
        self.objective = kt.Objective("val_f1_score_micro", direction="max")

    def fit_and_tune(self, X, y, epochs=100, validation_split=0.2, objective="val_f1_score_micro", n_trials=100, refit=True):        
        if objective == "val_f1_score_macro":
            self.objective = kt.Objective("val_f1_score_macro", direction="max")
        elif objective == "val_loss":
            self.objective = "val_loss"

        tuner = kt.BayesianOptimization(ConvHyperModel(),
                                        objective=self.objective,
                                        max_trials=n_trials,
                                        overwrite=True,
                                        seed=2023,
                                        directory="../CNN_finetune",
                                        project_name="politics")
        self.tuner = tuner
        tuner.search(X, y, epochs=epochs, validation_split=validation_split)
        
        best_hps = self.get_best_hyperparamenters()

        print("Refit with best hyperparameters")
        if not refit:
            return 

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs")
        hypermodel = ConvHyperModel()
        model = hypermodel.build(best_hps)
        history = hypermodel.fit(best_hps, model, x=X, y=y, epochs=epochs, validation_split=validation_split, callbacks=[tensorboard_callback])   

        val_per_epoch = history.history[objective]
        if objective == "val_loss":
            best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        best_epoch = val_per_epoch.index(max(val_per_epoch)) + 1
        print("*"*10)
        print('Best epoch: %d' % (best_epoch,))

    def get_best_hyperparamenters(self, verbose=1):
        if self.tuner == None:
            print("No tuner was found, please run fit_and_tune() method first")
            return
        
        best_hps=self.tuner.get_best_hyperparameters()[0]
        
        if verbose == 1:
            print("Hyperparameters")
            print("*"*10)
            print("Number of convolutional layers:", best_hps.get('num_conv_layers'))
            print("Number of filters:", best_hps.get('filters'))
            print("Dropout:", best_hps.get('dropout'))
            print("Number of linear layers:", best_hps.get('num_layers'))

            for i in range(best_hps.get('num_layers')):
                print(f"Dense_{i+1}:", best_hps.get(f"units_{i}"))

            print("Activation linear layer:", best_hps.get('activation_linear_layer'))
            print("Batch normalization:", best_hps.get('batchnorm'))
            print("Relu before softmax:", best_hps.get('relu_before_softmax'))
            print("Batch size:", best_hps.get('batch_size'))
            print("Learning rate", best_hps.get('learning_rate'))
            print("*"*10)

        return best_hps

    def fit(self, X, y, epochs, validation_split=0.0):
        best_hps = self.get_best_hyperparamenters(0)

        hypermodel = ConvHyperModel()
        model = hypermodel.build(best_hps)
        model.fit(X, y, batch_size=best_hps.get('batch_size'), epochs=epochs, validation_split=validation_split)
        #model = hypermodel.fit(best_hps, model, x=X, y=y, epochs=epochs, validation_split=validation_split)
        self.hypermodel = model
        return model
    
    def predict(self, X_test, y_test):
        print("Predicting with test data")
        eval_result = self.hypermodel.evaluate(X_test, y_test)
        print("[test loss, test f1 score,[macro]]:", eval_result)
        return 





