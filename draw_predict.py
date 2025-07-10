import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import glob

class DigitDrawer:
    def __init__(self, model_path=None):
        """Initialize the digit drawer with enhanced preprocessing"""
        # Auto-detect model file if not specified
        if model_path is None:
            model_path = self.find_model_file()
        
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}!")
            
            # Display model info
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure the model file exists and is valid.")
            return
        
        # Canvas settings optimized for digit recognition
        self.canvas_size = 400
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        self.drawing = False
        self.last_point = None
        self.brush_size = 12  # Thinner pen for better control
        self.pen_color = 255  # White on black background
        
        # Prediction display
        self.prediction_canvas = np.zeros((200, 400, 3), dtype=np.uint8)
        
    def find_model_file(self):
        """Auto-detect available model files - updated for advanced MNIST models"""
        # Priority order for model files (matching your advanced training script)
        model_patterns = [
            'best_advanced_model.keras',  # From your advanced training script
            'advanced_mnist_model_*.keras',
            'residual_mnist_model_*.keras',
            'ensemble_model_*.keras',
            'best_enhanced_mnist_model.keras',
            'enhanced_mnist_model_final.keras',
            'enhanced_mnist_model_*.keras',
            'best_mnist_model.keras',
            'mnist_model.keras',
            '*.keras'
        ]
        
        print("Searching for advanced MNIST model files...")
        
        for pattern in model_patterns:
            files = glob.glob(pattern)
            if files:
                # Sort by modification time (newest first)
                files.sort(key=os.path.getmtime, reverse=True)
                selected_file = files[0]
                print(f"Found model file: {selected_file}")
                return selected_file
        
        # If no files found, show error
        print("No .keras model files found in current directory!")
        available_files = [f for f in os.listdir('.') if f.endswith('.keras')]
        if available_files:
            print("Available .keras files:")
            for f in available_files:
                print(f"  - {f}")
        else:
            print("No .keras files found at all!")
        
        raise FileNotFoundError("No suitable model file found")
    
    def draw_line(self, event, x, y, flags, param):
        """Handle mouse events for drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.canvas, (x, y), self.brush_size, self.pen_color, -1)
            self.last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.last_point:
                # Draw smooth lines with thinner pen
                cv2.line(self.canvas, self.last_point, (x, y), self.pen_color, self.brush_size)
                cv2.circle(self.canvas, (x, y), self.brush_size//3, self.pen_color, -1)  # Smaller circle for smoother lines
                self.last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None
    
    def preprocess_drawing(self):
        """Enhanced preprocessing matching the advanced model training"""
        # Step 1: Find the bounding box of the drawn digit
        coords = np.column_stack(np.where(self.canvas > 0))
        if len(coords) == 0:
            return np.zeros((28, 28), dtype=np.float32)
        
        # Get bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add padding
        padding = 20
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(self.canvas_size, y_max + padding)
        x_max = min(self.canvas_size, x_max + padding)
        
        # Crop the digit
        cropped = self.canvas[y_min:y_max, x_min:x_max]
        
        # Step 2: Resize to fit in 20x20 box (MNIST digits are ~20x20 in a 28x28 image)
        h, w = cropped.shape
        if h > w:
            new_h = 20
            new_w = int(w * 20 / h)
        else:
            new_w = 20
            new_h = int(h * 20 / w)
        
        # Resize with anti-aliasing
        if new_h > 0 and new_w > 0:
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cropped
        
        # Step 3: Center the digit in a 28x28 image
        digit_28x28 = np.zeros((28, 28), dtype=np.uint8)
        
        # Calculate position to center the digit
        y_offset = (28 - resized.shape[0]) // 2
        x_offset = (28 - resized.shape[1]) // 2
        
        # Place the resized digit in the center
        digit_28x28[y_offset:y_offset+resized.shape[0], 
                   x_offset:x_offset+resized.shape[1]] = resized
        
        # Step 4: Apply center of mass centering (like MNIST)
        # Calculate center of mass
        cy, cx = ndimage.center_of_mass(digit_28x28)
        
        # Shift to center
        shift_y = int(14 - cy)
        shift_x = int(14 - cx)
        
        # Apply the shift
        if abs(shift_y) < 5 and abs(shift_x) < 5:  # Only apply small shifts
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            digit_28x28 = cv2.warpAffine(digit_28x28, M, (28, 28))
        
        # Step 5: Normalize to [0, 1] range (matching the advanced model preprocessing)
        normalized = digit_28x28.astype(np.float32) / 255.0
        
        # Step 6: Apply slight Gaussian blur to smooth edges (like MNIST)
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0.5)
        
        return blurred
    
    def predict_digit(self, image):
        """Predict digit from preprocessed image - enhanced for advanced models"""
        # Reshape for model input (matching advanced CNN input shape)
        image_input = np.expand_dims(image, axis=(0, -1))  # Add batch and channel dims
        
        # Make prediction
        prediction = self.model.predict(image_input, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return digit, confidence, prediction[0]
    
    def predict_with_ensemble(self, image):
        """Enhanced prediction method for ensemble models"""
        # This method can be extended if you have multiple models
        # For now, it uses the single loaded model
        return self.predict_digit(image)
    
    def update_prediction_display(self, digit, confidence, all_predictions):
        """Update the prediction display canvas with enhanced information"""
        self.prediction_canvas.fill(0)
        
        # Display main prediction with enhanced styling
        cv2.putText(self.prediction_canvas, f"Prediction: {digit}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(self.prediction_canvas, f"Confidence: {confidence:.1%}", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add model type indicator
        cv2.putText(self.prediction_canvas, "Advanced CNN Model", 
                   (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display all predictions as a bar chart
        cv2.putText(self.prediction_canvas, "All Predictions:", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        for i, prob in enumerate(all_predictions):
            # Draw bar with enhanced colors
            bar_width = int(prob * 150)
            if i == digit:
                color = (0, 255, 0)  # Green for predicted digit
            elif prob > 0.1:
                color = (0, 165, 255)  # Orange for high probability alternatives
            else:
                color = (100, 100, 100)  # Gray for low probability
                
            cv2.rectangle(self.prediction_canvas, 
                         (50, 120 + i * 15), (50 + bar_width, 120 + i * 15 + 10), 
                         color, -1)
            
            # Draw digit label and percentage
            cv2.putText(self.prediction_canvas, f"{i}:", 
                       (10, 130 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(self.prediction_canvas, f"{prob:.1%}", 
                       (210, 130 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main application loop with enhanced features"""
        if not hasattr(self, 'model'):
            print("Cannot run: Model not loaded properly.")
            return
        
        cv2.namedWindow("Advanced MNIST Digit Drawer", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Predictions", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Model Input (28x28)", cv2.WINDOW_AUTOSIZE)
        
        cv2.setMouseCallback("Advanced MNIST Digit Drawer", self.draw_line)
        
        print("Enhanced Controls:")
        print("- Draw with mouse")
        print("- Press 'p' to predict")
        print("- Press 'c' to clear")
        print("- Press 'q' to quit")
        print("- Press 's' to save current drawing")
        print("- Press 'i' to show model info")
        
        while True:
            # Create display image with enhanced instructions
            display_img = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)
            
            # Add instructions with enhanced styling
            cv2.putText(display_img, "Draw a digit (0-9)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, "Advanced CNN Model", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display_img, "P=Predict, C=Clear, S=Save, I=Info, Q=Quit", 
                       (10, self.canvas_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show main canvas
            cv2.imshow("Advanced MNIST Digit Drawer", display_img)
            
            # Show prediction canvas
            cv2.imshow("Predictions", self.prediction_canvas)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('p'):  # Predict
                processed_img = self.preprocess_drawing()
                
                # Show what the model sees
                model_input_display = (processed_img * 255).astype(np.uint8)
                model_input_large = cv2.resize(model_input_display, (280, 280), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Model Input (28x28)", model_input_large)
                
                # Make prediction
                digit, confidence, all_predictions = self.predict_digit(processed_img)
                
                # Update displays
                self.update_prediction_display(digit, confidence, all_predictions)
                
                # Enhanced console output
                print(f"\n{'='*40}")
                print(f"PREDICTION RESULTS")
                print(f"{'='*40}")
                print(f"Predicted Digit: {digit}")
                print(f"Confidence: {confidence:.1%}")
                print(f"Top 3 predictions:")
                top_3 = sorted(enumerate(all_predictions), key=lambda x: x[1], reverse=True)[:3]
                for i, (digit_idx, prob) in enumerate(top_3, 1):
                    print(f"  {i}. Digit {digit_idx}: {prob:.1%}")
                print(f"{'='*40}")
            
            elif key == ord('c'):  # Clear
                self.canvas.fill(0)
                self.prediction_canvas.fill(0)
                print("Canvas cleared")
            
            elif key == ord('s'):  # Save
                cv2.imwrite('drawn_digit.png', self.canvas)
                processed_img = self.preprocess_drawing()
                cv2.imwrite('processed_digit.png', (processed_img * 255).astype(np.uint8))
                print("Drawing saved as 'drawn_digit.png' and 'processed_digit.png'")
            
            elif key == ord('i'):  # Model info
                print(f"\n{'='*50}")
                print("MODEL INFORMATION")
                print(f"{'='*50}")
                print(f"Model Architecture: Advanced CNN")
                print(f"Input Shape: {self.model.input_shape}")
                print(f"Output Shape: {self.model.output_shape}")
                print(f"Total Parameters: {self.model.count_params():,}")
                print(f"Trainable Parameters: {sum([np.prod(v.shape) for v in self.model.trainable_weights]):,}")
                print(f"Model Features:")
                print("  - BatchNormalization layers")
                print("  - Dropout regularization")
                print("  - Global Average Pooling")
                print("  - Advanced optimizer (AdamW)")
                print("  - Data augmentation trained")
                print(f"{'='*50}")
            
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        print("Advanced MNIST Digit Drawer closed")

def main():
    """Main function with enhanced model detection for advanced models"""
    print("Advanced MNIST Digit Drawing Application")
    print("="*50)
    print("Compatible with Advanced CNN and Residual Models")
    print("="*50)
    
    # List available model files with enhanced information
    keras_files = glob.glob('*.keras')
    if keras_files:
        print("Available model files:")
        for i, file in enumerate(keras_files, 1):
            size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
            mod_time = os.path.getmtime(file)
            model_type = "Advanced" if "advanced" in file.lower() else "Standard"
            print(f"  {i}. {file} ({size:.1f} MB) - {model_type}")
    else:
        print("No .keras model files found!")
        print("Please run the advanced training script first to create model files.")
        return
    
    try:
        # Auto-detect and load model
        drawer = DigitDrawer()
        if hasattr(drawer, 'model'):
            print("="*50)
            print("Advanced MNIST Digit Drawer ready!")
            print("Features:")
            print("  - Enhanced preprocessing pipeline")
            print("  - Real-time prediction display")
            print("  - Advanced CNN model support")
            print("  - Detailed prediction analysis")
            print("="*50)
            print("Draw digits with your mouse and press 'p' to predict")
            print("="*50)
            drawer.run()
        else:
            print("Failed to initialize digit drawer.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have run the advanced training script first")
        print("2. Check that 'best_advanced_model.keras' exists in the current directory")
        print("3. Ensure the model files are not corrupted")
        print("4. Try running the training script to generate new model files")

if __name__ == "__main__":
    main()

