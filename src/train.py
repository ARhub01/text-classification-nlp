from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X_train, y_train, save_path, epochs=5, batch_size=64):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Model saved to {save_path}")
    return history
