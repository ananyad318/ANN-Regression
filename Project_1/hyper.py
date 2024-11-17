import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 4. Build model function to tune the hyperparameters
def build_model(hp):
    model = Sequential()

    # 5. First hidden layer
    model.add(Dense(units=hp.Int('units1', min_value=10, max_value=100, step=10),
                    activation='relu', input_dim=X_train.shape[1]))

    # 6. Add additional hidden layers based on 'num_layers' hyperparameter
    for i in range(hp.Int('num_layers', 1, 5)):  # Tune number of layers (1 to 5 layers)
        model.add(Dense(units=hp.Int(f'units_{i+2}', min_value=10, max_value=100, step=10),
                        activation='relu'))

    # 7. Output layer (sigmoid for binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # 8. Choose optimizer based on hyperparameter
    

    # 9. Compile the model with chosen optimizer and loss function for binary classification
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 10. Setup Keras Tuner with Hyperband algorithm
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,  # Maximum epochs per trial
    factor=3,  # Defines the number of models to explore in each bracket
    directory='my_dir',  # Directory for saving the results
    project_name='hyperparameter_tuning'
)

# 11. Run hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 12. Get the best hyperparameters found by the tuner
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 13. Print the best hyperparameters
print("Best hyperparameters:")
print(f"Number of layers: {best_hps['num_layers']}")
print(f"Neurons in first hidden layer: {best_hps['units1']}")
for i in range(best_hps['num_layers']):
    print(f"Neurons in layer {i+2}: {best_hps[f'units_{i+2}']}")
print(f"Optimizer: {best_hps['optimizer']}")

# 14. Build the best model based on found hyperparameters
best_model = tuner.hypermodel.build(best_hps)

# 15. Train the best model
best_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 16. Evaluate the best model
loss, accuracy = best_model.evaluate(X_val, y_val)
print(f"Final model accuracy on validation data: {accuracy}")
