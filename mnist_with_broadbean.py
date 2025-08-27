import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import Callback
import numpy as np

def broadbean():
    """
    Creates a callback that halts training once accuracy exceeds 99%.
    
    Returns:
        BroadbeanCallback: A custom callback instance that stops training when accuracy >= 99%
    """
    class BroadbeanCallback(Callback):
        def __init__(self, target_accuracy=0.99, monitor='accuracy'):
            super(BroadbeanCallback, self).__init__()
            self.target_accuracy = target_accuracy
            self.monitor = monitor
            
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            current_accuracy = logs.get(self.monitor)
            
            if current_accuracy is not None:
                if current_accuracy >= self.target_accuracy:
                    print(f"\n🎉 BROADBEAN SUCCESS! 🎉")
                    print(f"Target accuracy of {self.target_accuracy:.1%} achieved!")
                    print(f"Training stopped at epoch {epoch + 1}")
                    print(f"Final {self.monitor}: {current_accuracy:.4f}")
                    print("=" * 50)
                    
                    self.model.stop_training = True
    
    return BroadbeanCallback()

def main():
    print("🚀 MNIST Classification with Broadbean Callback")
    print("=" * 60)
    
    # 1. Load the MNIST dataset
    print("📥 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Training data shape: {x_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, {y_test.shape}")
    
    # 2. Data Preprocessing
    print("\n🔧 Preprocessing data...")
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))
    x_test = x_test.reshape((x_test.shape[0], 28 * 28))
    
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    
    print(f"Reshaped training data: {x_train.shape}")
    print(f"Reshaped testing data: {x_test.shape}")
    
    # 3. Build the Classification Model
    print("\n🏗️ Building model...")
    model = Sequential()
    model.add(Flatten(input_shape=(28 * 28,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    # 4. Compile the Model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    print("📊 Model Summary:")
    model.summary()
    
    # 5. Create the Broadbean Callback
    print("\n🫘 Creating Broadbean callback...")
    broadbean_callback = broadbean()
    
    # 6. Train the Model with Callback
    print("\n🎯 Training with Broadbean callback (target: 99% accuracy)...")
    print("=" * 60)
    
    history = model.fit(x_train, y_train,
                        epochs=20,  # Increased epochs since we have early stopping
                        batch_size=32,
                        validation_data=(x_test, y_test),
                        callbacks=[broadbean_callback],
                        verbose=1)
    
    # 7. Evaluate the Model
    print("\n📈 Evaluating final model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Final Test Loss: {test_loss:.4f}')
    print(f'Final Test Accuracy: {test_acc:.2%}')
    
    # 8. Cost Savings Analysis
    print("\n💰 COST SAVINGS ANALYSIS")
    print("=" * 40)
    
    original_epochs = 5
    actual_epochs = len(history.history['accuracy'])
    epochs_saved = original_epochs - actual_epochs
    
    if epochs_saved > 0:
        time_savings = (epochs_saved / original_epochs) * 100
        print(f"✅ Training stopped early at epoch {actual_epochs}")
        print(f"✅ Epochs saved: {epochs_saved}")
        print(f"✅ Time savings: {time_savings:.1f}%")
        print(f"✅ Computational cost reduction: {time_savings:.1f}%")
    else:
        print("ℹ️ Training completed all epochs (target not reached)")
    
    print("\n🎯 Key Benefits of Using Callbacks:")
    print("- Automatic early stopping when target reached")
    print("- Prevents unnecessary computation")
    print("- Reduces training time and costs")
    print("- Prevents overfitting")
    print("- Provides informative feedback")

if __name__ == "__main__":
    main()
