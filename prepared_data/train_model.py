from datetime import datetime
from pathlib import Path
import re
import tensorflow as tf
import os


def callbacks(model_name):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5,
                                                     verbose=1, cooldown=0, min_lr=0.0001)

    checkpoint_filepath = Path(f'saved_model/checkpoint/final_result/' + datetime.now().strftime("%Y:%m:%d-%H:%M:%S") +'_epoch{epoch:02d}-loss{val_loss:.3f}-train{loss:.3f}.hdf5')

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',)

    log_dir = Path(f'logs/fit/{model_name}')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    return early_stop, reduce_lr, tensorboard_callback, checkpoint_callback


def split_features_target_and_map_target(data):
    target = data['target']
    train_features = data.drop('target', axis=1)
    di = {'home': 2, 'draw': 1, 'away': 0}
    target = target.replace(di)
    return train_features, target

def save_results(output_path, output_string, model_summary=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:  # write end of file
        if model_summary is not None:
            model_summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(output_string + '\n')


def columns_only_first_hist(data):
    used = set()
    columns_only_first_hist = [column for column in data.columns.values if
                               re.sub('\d', '', column) not in used and (used.add(re.sub('\d', '', column)) or True)]
    return columns_only_first_hist
