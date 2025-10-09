import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# 0. Loading data sequences: each sample is a matriz (sequence) of windows which have same number of time domain features
with np.load('emg_sequences_dataset.npz') as data:
    X_sequences = data['X']
    y_sequences = data['y']

print("Total dataset lenght", X_sequences.shape[0])    
#################################################################################################################################################################

# 1. Splitting data

# 80% train and 20% test -> Stratify and Shuffle split

print("splitting data, train_val and test. ")
X_train_val, X_test, y_train_val, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True, stratify=y_sequences, random_state=42)
# X_train_val
# y_train_val
# X_test
# y_test

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=True, stratify=y_train_val, random_state=42)
# X_train
# y_train
# X_val
# y_val

print(f"Training set size: {len(X_train)} sequences")
print(f"Validation set size: {len(X_val)} sequences")
print(f"Test set size: {len(X_test)} sequences")

#################################################################################################################################################################

# 2. Standarize data

num_samples_train, seq_length, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)
print(X_train_reshaped.shape)

scaler = StandardScaler()
scaler.fit(X_train_reshaped)

# Apply the same fitted scaler to all datasets and reshape back to 3D
X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

print("Standardization complete. Data is ready for the model.")
print(f"Shape of X_train_scaled: {X_train_scaled.shape}")
print(f"Shape of X_val_scaled: {X_val_scaled.shape}")
print(f"Shape of X_test_scaled: {X_test_scaled.shape}")

#################################################################################################################################################################

# 3. Define the model architecture 
print("\nDefining the model with Keras")

# Get the dimensions from data
seq_length = X_train_scaled.shape[1] #number of windows per sequence
num_features = X_train_scaled.shape[2] # number of features per window
num_classes = len(np.unique(y_train)) # Number of unique gestures

model = Sequential([
    # The LSTM layer. The input_shape is crucial for the first layer.
    # It tells the model to expect sequences of shape (10, 48) in your case.
    # We use return_sequences=False because the next layer is a Dense layer.
    LSTM(64, input_shape=(seq_length, num_features), return_sequences=False),
    
    # A Dropout layer for regularization to help prevent overfitting.
    Dropout(0.5), 
    
    # Output layer
    Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
)

# model.summary()

#################################################################################################################################################################

# 4. and 5 Train the model, evaluate the test set

history = model.fit(
    X_train_scaled, 
    y_train, 
    epochs=1, 
    batch_size=64, # Number of samples per gradient update.
    validation_data=(X_val_scaled, y_val), 
    verbose=1 
)

print("\nTraining Complete")

print("\nEvaluating the model on test set")

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


print("\nPerforming test inference on 5 random samples")

# Get 5 random samples from the test set
indices = np.random.choice(len(X_test_scaled), 5, replace=False)
test_samples = X_test_scaled[indices]
true_labels = y_test[indices]

# Get model predictions
predictions_probabilities = model.predict(test_samples)
predicted_classes = np.argmax(predictions_probabilities, axis=1)

# Display the results
label_names = {0: 'Hand_Open', 1: 'Hand_Close'} 

for i in range(5):
    true_label_name = label_names.get(true_labels[i], "Unknown")
    pred_label_name = label_names.get(predicted_classes[i], "Unknown")
    print(f"Sample #{i+1}: True Label = {true_label_name}, Prediction = {pred_label_name}")

