import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def broadbean():
    """
    Creates a callback that halts training once accuracy exceeds 99%.
    
    Returns:
        BroadbeanCallback: A custom callback instance that stops training when accuracy >= 99%
    """
    class BroadbeanCallback(Callback):
        def __init__(self, target_accuracy=0.99, monitor='accuracy'):
            """
            Initialize the Broadbean callback.
            
            Args:
                target_accuracy (float): Target accuracy threshold (default: 0.99)
                monitor (str): Metric to monitor ('accuracy' or 'val_accuracy')
            """
            super(BroadbeanCallback, self).__init__()
            self.target_accuracy = target_accuracy
            self.monitor = monitor
            
        def on_epoch_end(self, epoch, logs=None):
            """
            Called at the end of each epoch.
            
            Args:
                epoch (int): Current epoch number
                logs (dict): Dictionary containing training metrics
            """
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

# Example usage:
# callbacks = [broadbean()]
# model.fit(x_train, y_train, epochs=20, callbacks=callbacks)
