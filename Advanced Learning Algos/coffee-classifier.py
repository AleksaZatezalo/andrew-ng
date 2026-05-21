"""
Description: Time and temp for ideal coffee roasting
Date: May 20, 2026
Author: Aleksa Zatezalo
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# DATA
# =============================================================================
def generate_roast_data(n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    temperature = rng.uniform(150, 300, n_samples)
    duration    = rng.uniform(1,   20,  n_samples)
    good_roast  = (
        (temperature >= 175) & (temperature <= 260) &
        (duration    >= 3)   & (duration    <= 15)  &
        (temperature + (duration * 10) >= 230)
    ).astype(int)
    return np.column_stack([temperature, duration]), good_roast.reshape(-1, 1)

def preprocess(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# =============================================================================
# MODEL
# =============================================================================
def build_model(input_dim=2):
    return Sequential([
        tf.keras.Input(shape=(input_dim,)),
        Dense(16, activation="relu",    name="layer1"),
        Dense(8,  activation="relu",    name="layer2"),
        Dense(1,  activation="sigmoid", name="output"),
    ], name="coffee_roast_classifier")

# =============================================================================
# TRAINING
# =============================================================================
def train(model, X_train, y_train, epochs=100, batch_size=32):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1,
    )
    return history

# =============================================================================
# EVALUATION & VISUALIZATION
# =============================================================================
def evaluate(model, X_test, y_test):
    y_pred = (model.predict(X_test) >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Bad Roast", "Good Roast"]))

def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["loss"],         label="train")
    ax1.plot(history.history["val_loss"],     label="val")
    ax1.set_title("Loss"); ax1.legend()
    ax2.plot(history.history["accuracy"],     label="train")
    ax2.plot(history.history["val_accuracy"], label="val")
    ax2.set_title("Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_boundary(model, scaler, resolution=300):
    T, D = np.meshgrid(np.linspace(150, 300, resolution), np.linspace(1, 20, resolution))
    grid = scaler.transform(np.column_stack([T.ravel(), D.ravel()]))
    Z    = model.predict(grid, verbose=0).reshape(T.shape)
    plt.figure(figsize=(10, 6))
    plt.contourf(T, D, Z, levels=[0, 0.5, 1], alpha=0.3, colors=["#d9534f", "#5cb85c"])
    plt.colorbar(label="P(Good Roast)")
    plt.xlabel("Temperature (°C)"); plt.ylabel("Duration (minutes)")
    plt.title("Coffee Roast Decision Boundary")
    plt.legend(handles=[
        mpatches.Patch(color="#d9534f", alpha=0.5, label="Bad Roast"),
        mpatches.Patch(color="#5cb85c", alpha=0.5, label="Good Roast"),
    ])
    plt.tight_layout(); plt.show()

# =============================================================================
# INFERENCE
# =============================================================================
def predict_roast(model, scaler, temperature, duration):
    prob  = model.predict(scaler.transform([[temperature, duration]]), verbose=0)[0][0]
    label = "Good Roast ✓" if prob >= 0.5 else "Bad Roast ✗"
    return f"{label}  (confidence: {prob:.2%})"

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    X, y = generate_roast_data()
    X_train, X_test, y_train, y_test, scaler = preprocess(X, y)

    model = build_model()
    model.summary()

    history = train(model, X_train, y_train)
    plot_training(history)
    evaluate(model, X_test, y_test)
    plot_boundary(model, scaler)

    print(predict_roast(model, scaler, temperature=220, duration=10))  # good
    print(predict_roast(model, scaler, temperature=160, duration=2))   # bad
    print(predict_roast(model, scaler, temperature=290, duration=18))  # edge