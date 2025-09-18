import itertools
import time
import os
from typing import Tuple, Sequence, Optional

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPool2D, AveragePooling2D, Flatten,
    Dropout, Dense, LeakyReLU, ELU, BatchNormalization
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import Precision, Recall


def build_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    conv_activation: str = "relu",         
    dropout: float = 0.0,                  
    use_batchnorm: bool = False,
    filters: Sequence[int] = (10, 10, 10), 
    kernel_size: int = 3,
    pooling: str = "max"                   
):
    if conv_activation == "relu":
        act_layer = "relu"
    elif conv_activation == "leaky_relu":
        act_layer = LeakyReLU(alpha=0.1)
    elif conv_activation == "elu":
        act_layer = ELU(alpha=1.0)
    else:
        raise ValueError(f"Unsupported activation: {conv_activation}")

    Pool = MaxPool2D if pooling == "max" else AveragePooling2D

    def maybe_bn():
        return [BatchNormalization()] if use_batchnorm else []

    def act_block():
        return [(act_layer if isinstance(act_layer, (LeakyReLU, ELU))
                 else tf.keras.layers.Activation(act_layer))]

    def model_fn():
        layers = [Input(shape=input_shape)]
        for f in filters:
            layers += [
                Conv2D(f, kernel_size, padding="valid", activation=None),
                *maybe_bn(),
                *act_block(),
                Pool(pool_size=2)
            ]
        layers += [
            Flatten(),
            Dropout(dropout),
            Dense(1, activation="sigmoid")
        ]
        return Sequential(layers)

    return model_fn


def make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return Adam(learning_rate=lr)
    if name == "sgd":
        return SGD(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def f1_from_prec_recall(prec: float, rec: float, eps: float = 1e-8) -> float:
    return float(2.0 * (prec * rec) / (prec + rec + eps))



def train_once(
    model_fn,
    optimizer_name: str,
    lr: float,
    conv_activation: str,
    dropout: float,
    use_batchnorm: bool,
    pooling: str,
    train_gen,
    val_gen,
    epochs: int = 10,
    verbose: int = 0
):
    tf.keras.backend.clear_session()
    model = model_fn()

    opt = make_optimizer(optimizer_name, lr)
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt,
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0
        )
    ]

    t0 = time.time()
    hist = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks
    )
    t1 = time.time()

    loss, acc, prec, rec = model.evaluate(val_gen, verbose=0)
    f1 = f1_from_prec_recall(float(prec), float(rec))
    elapsed = t1 - t0

    result = {
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "conv_activation": conv_activation,
        "dropout": dropout,
        "batchnorm": use_batchnorm,
        "pooling": pooling,
        "val_loss": float(loss),
        "val_accuracy": float(acc),
        "val_precision": float(prec),
        "val_recall": float(rec),
        "val_f1": float(f1),
        "train_time_sec": float(elapsed),
        "epochs_run": len(hist.history["loss"]),
        "history": hist.history,  
    }
    return result, model, hist


def run_grid(
    train_gen,
    val_gen,
    optimizers=("adam", "sgd"),
    learning_rates=(1e-3, 3e-4, 1e-4, 3e-3, 1e-2),
    activations=("relu", "leaky_relu", "elu"),
    dropouts=(0.0, 0.1, 0.2, 0.3),
    batchnorm=(True, False),
    poolings=("max", "avg"),
    filters_list=((10, 10, 10),), 
    epochs: int = 10,
    verbose: int = 0,
    save_best_model_path: Optional[str] = None
):
    combos = list(itertools.product(
        optimizers, learning_rates, activations, dropouts, batchnorm, poolings, filters_list
    ))

    records = []
    best = None
    best_model = None

    for opt_name, lr, act, dr, bn, pool, filt in combos:
        model_fn = build_cnn(
            conv_activation=act,
            dropout=dr,
            use_batchnorm=bn,
            pooling=pool,
            filters=filt
        )

        res, model, hist = train_once(
            model_fn, opt_name, lr, act, dr, bn, pool,
            train_gen, val_gen, epochs=epochs, verbose=verbose
        )
        records.append(res)

        if (best is None) or (res["val_f1"] > best["val_f1"]):
            best = res
            best_model = model
            if save_best_model_path:
                root, _ = os.path.splitext(save_best_model_path)
                timestamp = int(time.time())
                save_path = f"{root}_{timestamp}.keras"
                model.save(save_path)
                print(f"Saved best model to {save_path}")

        print(f"[GRID] {opt_name:7s} | lr={lr:<8g} | act={act:10s} | "
              f"dr={dr:.2f} | bn={bn!s:5s} | pool={pool:3s} | filters={filt} --> "
              f"val_f1={res['val_f1']:.4f} (acc={res['val_accuracy']:.4f}, "
              f"prec={res['val_precision']:.4f}, rec={res['val_recall']:.4f})")

    df = pd.DataFrame.from_records(records)
    df.sort_values("val_f1", ascending=False, inplace=True)
    return df, best_model
