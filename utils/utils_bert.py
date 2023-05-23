import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import preprocessing

def getPercentagesPerClass(x, class_name, percentage_decimals=2):
    # number of rows that contain class_name divided by total of rows
    return np.round(x[x == class_name].shape[0]/x.shape[0] * 100, decimals=percentage_decimals)

def getInputDict(slice_dataset, columns, party_binarizer):
    param = {}
    for i, c in enumerate(columns):
      param[c] = slice_dataset[:, i]
      if c == 'partyname':
        param[c] = party_binarizer.transform(slice_dataset[:, i])
    
    return param

def createTFDatasetFromPandas(dataset, test_size, validation_size, columns=['prev_text', 'text', 'partyname'], batch_size=32, shuffle_buffer_size=1000, seed=2023, verbosity=1):
    print('Found {} examples in dataset'.format(len(dataset)))
    sentences_raw = dataset[columns].values
    target_raw = dataset['domain_name'].values
    
    class_names = list(np.unique(target_raw))
    
    # label binarizer
    lb = preprocessing.LabelBinarizer()
    lb.fit(target_raw)
    target_raw = lb.transform(target_raw)

    # party binarizer
    pb = preprocessing.LabelBinarizer()
    pb.fit(dataset.partyname.values)
    
    # Split dataset in train/test
    X_train, X_test, y_train, y_test = train_test_split(sentences_raw, target_raw, 
                                                        test_size=test_size, random_state=seed)
    if validation_size > 0.0:
        # With the training set already created, split it again in train/validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                          test_size=validation_size, random_state=seed)
    # percentages
    print('Using {} examples for training'.format(X_train.shape[0]))
    if verbosity > 1:
        for c in class_names:
            print('\t{} : {}%'.format(c, getPercentagesPerClass(y_train, c)))
    
    if validation_size > 0.0:
        print('Using {} examples for validation'.format(X_val.shape[0]))
        if verbosity > 1:
            for c in class_names:
                print('\t{} : {}%'.format(c, getPercentagesPerClass(y_val, c)))
    else:
        print('Using 0 examples for validation')
    print('Using {} examples for testing'.format(X_test.shape[0]))
    if verbosity > 1:
        for c in class_names:
            print('\t{} : {}%'.format(c, getPercentagesPerClass(y_test, c)))
    
    # transform numpy arrays into tf.data.Dataset format suitable for tensorflow pipeline, then
    # shuffle and batch the data (shuffle the batch) for every split
    # to notice: batch is shuffled each iteration, you can chage this for (maybe) hyperparameter tuning 
    AUTOTUNE = tf.data.AUTOTUNE

    train_param_in = getInputDict(X_train, columns, party_binarizer=pb)
    train_ds = tf.data.Dataset.from_tensor_slices((train_param_in, 
                                                   y_train))
    train_ds = train_ds.shuffle(shuffle_buffer_size).batch(batch_size)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)  
    
    if validation_size > 0.0:
        val_param_in = getInputDict(X_val, columns, party_binarizer=pb)
        val_ds   = tf.data.Dataset.from_tensor_slices((val_param_in, 
                                                    y_val))
        val_ds   = val_ds.shuffle(shuffle_buffer_size).batch(batch_size)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  
    
    test_param_in = getInputDict(X_test, columns, party_binarizer=pb)
    test_ds  = tf.data.Dataset.from_tensor_slices((test_param_in, 
                                                   y_test))
    test_ds  = test_ds.shuffle(shuffle_buffer_size).batch(batch_size)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)  
    
    if validation_size > 0.0:
        return train_ds, val_ds, test_ds
    else:
        return train_ds, test_ds