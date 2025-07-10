import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class AdvancedMNISTRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        self.ensemble_models = []
        
    def load_and_preprocess_data(self, augment_data=True):
        """Enhanced data loading with optional augmentation"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape for CNN
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        
        # Create validation split
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        # Convert to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_val = keras.utils.to_categorical(y_val, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Optional data augmentation
        if augment_data:
            x_train, y_train = self.create_augmented_data(x_train, y_train)
        
        print(f"Training data: {x_train.shape}")
        print(f"Validation data: {x_val.shape}")
        print(f"Test data: {x_test.shape}")
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    def create_augmented_data(self, x_train, y_train):
        """Create additional training data with augmentation"""
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.08,
            height_shift_range=0.08,
            zoom_range=0.08,
            shear_range=0.05,
            fill_mode='nearest'
        )
        
        # Generate augmented data
        augmented_images = []
        augmented_labels = []
        
        print("Creating augmented training data...")
        for i in range(len(x_train)):
            # Original image
            augmented_images.append(x_train[i])
            augmented_labels.append(y_train[i])
            
            # Generate 1 augmented version for each original
            img = x_train[i].reshape(1, 28, 28, 1)
            aug_iter = datagen.flow(img, batch_size=1)
            aug_img = next(aug_iter)[0]
            
            augmented_images.append(aug_img)
            augmented_labels.append(y_train[i])
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def build_advanced_cnn(self):
        """Build an advanced CNN with modern techniques"""
        model = keras.Sequential([
            # Input layer with normalization
            layers.Input(shape=(28, 28, 1)),
            layers.BatchNormalization(),
            
            # First block - Feature extraction
            layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second block - Deeper features
            layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block - High-level features
            layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(128, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten
            layers.GlobalAveragePooling2D(),
            
            # Dense layers with regularization
            layers.Dense(512, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            layers.Dense(256, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def build_residual_model(self):
        """Build a ResNet-inspired model for MNIST"""
        def residual_block(x, filters, kernel_size=3, stride=1):
            # Main path
            y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
            y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(y)
            y = layers.BatchNormalization()(y)
            
            # Shortcut path
            if stride != 1 or x.shape[-1] != filters:
                x = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)
            
            # Add shortcut
            y = layers.Add()([x, y])
            y = layers.Activation('relu')(y)
            return y
        
        # Input
        inputs = layers.Input(shape=(28, 28, 1))
        x = layers.BatchNormalization()(inputs)
        
        # Initial conv
        x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Residual blocks
        x = residual_block(x, 32)
        x = residual_block(x, 64, stride=2)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        
        # Final layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        return keras.Model(inputs, outputs)
    
    def compile_model(self, model, learning_rate=0.001):
        """Compile model with advanced optimizer"""
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.0001,
            clipnorm=1.0
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def get_advanced_callbacks(self):
        """Get advanced callbacks for training"""
        callbacks = [
            # Learning rate scheduling
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            
            # Model checkpointing
            keras.callbacks.ModelCheckpoint(
                'best_advanced_model.keras',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            
            # Cosine annealing
            keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * (0.95 ** epoch)
            ),
            
            # Mixed precision training
            keras.callbacks.LossScaling() if tf.config.list_physical_devices('GPU') else None
        ]
        
        return [cb for cb in callbacks if cb is not None]
    
    def create_ensemble(self, x_train, y_train, x_val, y_val, num_models=3):
        """Create an ensemble of models"""
        print(f"Training ensemble of {num_models} models...")
        
        self.ensemble_models = []
        
        for i in range(num_models):
            print(f"\nTraining model {i+1}/{num_models}")
            
            # Use different architectures
            if i % 2 == 0:
                model = self.build_advanced_cnn()
            else:
                model = self.build_residual_model()
            
            model = self.compile_model(model, learning_rate=0.001 * (0.8 ** i))
            
            # Train with different seeds
            tf.random.set_seed(42 + i)
            np.random.seed(42 + i)
            
            history = model.fit(
                x_train, y_train,
                batch_size=64,
                epochs=30,
                validation_data=(x_val, y_val),
                callbacks=self.get_advanced_callbacks(),
                verbose=1
            )
            
            self.ensemble_models.append(model)
    
    def train_model(self, x_train, y_train, x_val, y_val, model_type='advanced'):
        """Train a single advanced model"""
        print(f"Training {model_type} model...")
        
        if model_type == 'advanced':
            self.model = self.build_advanced_cnn()
        elif model_type == 'residual':
            self.model = self.build_residual_model()
        
        self.model = self.compile_model(self.model)
        
        print("\nModel Summary:")
        self.model.summary()
        
        # Advanced data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=7,
            width_shift_range=0.07,
            height_shift_range=0.07,
            zoom_range=0.05,
            shear_range=0.05,
            fill_mode='nearest'
        )
        datagen.fit(x_train)
        
        # Train
        self.history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=96),
            epochs=50,
            validation_data=(x_val, y_val),
            callbacks=self.get_advanced_callbacks(),
            steps_per_epoch=len(x_train) // 96,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        if self.model:
            print("\nSingle Model Performance:")
            test_loss, test_accuracy, top_k_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Top-2 Accuracy: {top_k_accuracy:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
        
        if self.ensemble_models:
            print("\nEnsemble Performance:")
            ensemble_predictions = np.zeros((len(x_test), 10))
            
            for i, model in enumerate(self.ensemble_models):
                pred = model.predict(x_test, verbose=0)
                ensemble_predictions += pred
                individual_acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))
                print(f"Model {i+1} accuracy: {individual_acc:.4f}")
            
            ensemble_predictions /= len(self.ensemble_models)
            ensemble_accuracy = np.mean(
                np.argmax(ensemble_predictions, axis=1) == np.argmax(y_test, axis=1)
            )
            print(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            return ensemble_accuracy
        
        return test_accuracy if self.model else 0
    
    def predict_with_ensemble(self, x):
        """Make predictions using ensemble"""
        if not self.ensemble_models:
            return self.model.predict(x) if self.model else None
        
        predictions = np.zeros((len(x), 10))
        for model in self.ensemble_models:
            predictions += model.predict(x, verbose=0)
        
        return predictions / len(self.ensemble_models)
    
    def plot_training_history(self):
        """Plot comprehensive training history"""
        if not self.history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].grid(True)
        
        # Top-k accuracy
        if 'top_k_categorical_accuracy' in self.history.history:
            axes[1, 1].plot(self.history.history['top_k_categorical_accuracy'], label='Training')
            axes[1, 1].plot(self.history.history['val_top_k_categorical_accuracy'], label='Validation')
            axes[1, 1].set_title('Top-2 Accuracy')
            axes[1, 1].set_ylabel('Top-2 Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function with different training options"""
    print("Advanced MNIST Recognition Training")
    print("="*40)
    
    recognizer = AdvancedMNISTRecognizer()
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = recognizer.load_and_preprocess_data()
    
    # Choose training method
    print("\nTraining options:")
    print("1. Advanced CNN")
    print("2. Residual Network")
    print("3. Ensemble (3 models)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        recognizer.train_model(x_train, y_train, x_val, y_val, 'advanced')
    elif choice == '2':
        recognizer.train_model(x_train, y_train, x_val, y_val, 'residual')
    elif choice == '3':
        recognizer.create_ensemble(x_train, y_train, x_val, y_val)
    else:
        print("Invalid choice, using advanced CNN")
        recognizer.train_model(x_train, y_train, x_val, y_val, 'advanced')
    
    # Evaluate
    final_accuracy = recognizer.evaluate_model(x_test, y_test)
    
    # Plot training history
    recognizer.plot_training_history()
    
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    print("Training completed!")

if __name__ == "__main__":
    main()