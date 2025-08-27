import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

def create_broadbean_callback(target_accuracy=0.99, monitor='accuracy'):
    class SimpleCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None and logs.get(monitor) is not None:
                if logs.get(monitor) >= target_accuracy:
                    self.model.stop_training = True
    return SimpleCallback()

def create_enhanced_training_callbacks():
    """
    Creates a comprehensive set of callbacks for efficient training.
    
    Returns:
        list: List of callback instances
    """
    callbacks = []
    
    # 1. Broadbean callback for early stopping at 99% accuracy
    broadbean_callback = create_broadbean_callback()
    callbacks.append(broadbean_callback)
    
    # 2. Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # 3. Model checkpointing to save best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # 4. Learning rate reduction when plateau is reached
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    return callbacks

def create_mnist_model():
    """
    Creates the MNIST classification model from the lesson.
    
    Returns:
        tf.keras.Model: Compiled model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28 * 28,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_mnist_data():
    """
    Loads and preprocesses MNIST data.
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) preprocessed data
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    # Reshape data
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    return x_train, y_train, x_test, y_test

def train_with_callbacks():
    """
    Demonstrates training with the broadbean callback and other callbacks.
    """
    print("🚀 Starting MNIST training with Broadbean Callback!")
    print("=" * 60)
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist_data()
    
    # Create model
    model = create_mnist_model()
    
    # Create callbacks
    callbacks = create_enhanced_training_callbacks()
    
    print("📊 Model Summary:")
    model.summary()
    print("\n" + "=" * 60)
    
    # Train with callbacks
    history = model.fit(
        x_train, y_train,
        epochs=20,  # Increased epochs since we have early stopping
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate final model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n📈 Final Test Accuracy: {test_acc:.4f}")
    
    return model, history

def plot_training_history(history):
    """
    Plots training history.
    
    Args:
        history: Training history from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_cost_savings():
    """
    Demonstrates the cost savings from using callbacks.
    """
    print("💰 COST SAVINGS ANALYSIS")
    print("=" * 40)
    
    # Without callbacks (original example)
    print("Without Callbacks:")
    print("- Fixed 5 epochs regardless of performance")
    print("- No early stopping = potential overfitting")
    print("- No model checkpointing = risk of losing progress")
    print("- No learning rate scheduling = suboptimal convergence")
    
    print("\nWith Callbacks:")
    print("- Early stopping when target reached (99% accuracy)")
    print("- Automatic overfitting prevention")
    print("- Best model automatically saved")
    print("- Learning rate optimization")
    print("- Reduced training time and computational costs")
    
    print("\n📊 Estimated Savings:")
    print("- Training time: 40-60% reduction")
    print("- Computational resources: 30-50% savings")
    print("- Storage: Automatic best model saving")
    print("- Hyperparameter tuning: Reduced manual intervention")

if __name__ == "__main__":
    # Run the demonstration
    model, history = train_with_callbacks()
    
    # Plot results
    plot_training_history(history)
    
    # Show cost savings analysis
    demonstrate_cost_savings()
