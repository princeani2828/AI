import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json
import os

# Ensure reproducible results
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(csv_path='ups_synthetic.csv'):
    """Load and preprocess the UPS dataset"""
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Separate features and target
    feature_columns = ['Vin', 'Vout', 'Vbat', 'Iload', 'Ichg', 'Temp_bat', 'Temp_tx', 'mains_ok', 'mode_bat']
    X = df[feature_columns]
    y = df['fault_type']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {sorted(y.unique())}")
    
    return X, y, feature_columns

def create_lightweight_model(input_shape, num_classes):
    """Create a lightweight neural network for ESP32 deployment"""
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(input_shape,), name='dense_1'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu', name='dense_2'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Complete training pipeline"""
    # Load data
    X, y, feature_columns = load_and_preprocess_data()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Save class names mapping
    class_names = {i: name for i, name in enumerate(label_encoder.classes_)}
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class mapping: {class_names}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler parameters for ESP32
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)
    print("Scaler parameters saved")
    
    # Create and train model
    model = create_lightweight_model(X_train_scaled.shape[1], len(class_names))
    
    print("\nModel architecture:")
    model.summary()
    
    # Calculate model size
    param_count = model.count_params()
    print(f"\nTotal parameters: {param_count:,}")
    
    # Early stopping and model checkpoint
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed evaluation
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    target_names = [class_names[i] for i in range(len(class_names))]
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    
    # Save Keras model
    model.save('ups_model_keras.h5')
    print("Keras model saved as 'ups_model_keras.h5'")
    
    return model, scaler, class_names, history, test_accuracy

def convert_to_tflite(model, X_test_scaled):
    """Convert Keras model to quantized TFLite"""
    print("\nConverting to TensorFlow Lite...")
    
    # Create a representative dataset for quantization
    def representative_dataset():
        for i in range(min(100, len(X_test_scaled))):
            yield [X_test_scaled[i:i+1].astype(np.float32)]
    
    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('ups_model_quant.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"Quantized TFLite model saved as 'ups_model_quant.tflite'")
    print(f"Model size: {len(tflite_model)/1024:.1f} KB")
    
    return tflite_model

def test_tflite_model(tflite_model, X_test_scaled, y_test, scaler):
    """Test the TFLite model accuracy"""
    print("\nTesting TFLite model...")
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test on a subset of test data
    correct_predictions = 0
    total_predictions = min(1000, len(X_test_scaled))
    
    for i in range(total_predictions):
        # Prepare input
        input_data = X_test_scaled[i:i+1].astype(np.float32)
        
        # Quantize input if needed
        if input_details[0]['dtype'] == np.int8:
            input_scale = input_details[0]['quantization'][0]
            input_zero_point = input_details[0]['quantization'][1]
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if needed
        if output_details[0]['dtype'] == np.int8:
            output_scale = output_details[0]['quantization'][0]
            output_zero_point = output_details[0]['quantization'][1]
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        # Get prediction
        predicted_class = np.argmax(output_data[0])
        actual_class = y_test[i]
        
        if predicted_class == actual_class:
            correct_predictions += 1
    
    tflite_accuracy = correct_predictions / total_predictions
    print(f"TFLite model accuracy: {tflite_accuracy:.4f} ({tflite_accuracy*100:.2f}%)")
    
    return tflite_accuracy

def generate_arduino_header():
    """Generate Arduino header file with model and scaler data"""
    print("\nGenerating Arduino header file...")
    
    # Load scaler parameters
    scaler_mean = np.load('scaler_mean.npy')
    scaler_scale = np.load('scaler_scale.npy')
    
    # Load class names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Read TFLite model as bytes
    with open('ups_model_quant.tflite', 'rb') as f:
        tflite_bytes = f.read()
    
    # Generate header content
    header_content = f"""#ifndef UPS_AI_MODEL_H
#define UPS_AI_MODEL_H

// Model size: {len(tflite_bytes)} bytes ({len(tflite_bytes)/1024:.1f} KB)
// Generated automatically - do not edit

// Number of input features
#define NUM_FEATURES 9

// Number of output classes
#define NUM_CLASSES {len(class_names)}

// Scaler mean values
const float scaler_mean[NUM_FEATURES] = {{
{', '.join([f'{x:.6f}f' for x in scaler_mean])}
}};

// Scaler scale values
const float scaler_scale[NUM_FEATURES] = {{
{', '.join([f'{x:.6f}f' for x in scaler_scale])}
}};

// Class names
const char* class_names[NUM_CLASSES] = {{
{', '.join([f'"{name}"' for name in class_names.values()])}
}};

// TensorFlow Lite model data
const unsigned char ups_model_data[] = {{
"""
    
    # Add model bytes (16 per line)
    for i in range(0, len(tflite_bytes), 16):
        chunk = tflite_bytes[i:i+16]
        hex_values = ', '.join([f'0x{b:02x}' for b in chunk])
        header_content += f"  {hex_values}"
        if i + 16 < len(tflite_bytes):
            header_content += ","
        header_content += "\n"
    
    header_content += f"""
}};

const unsigned int ups_model_data_len = {len(tflite_bytes)};

#endif // UPS_AI_MODEL_H
"""
    
    # Save header file
    with open('ups_model_data.h', 'w') as f:
        f.write(header_content)
    
    print("Arduino header file saved as 'ups_model_data.h'")

def main():
    """Main training and conversion pipeline"""
    print("=== UPS AI Fault Detection Training Pipeline ===\n")
    
    # Check if dataset exists
    if not os.path.exists('ups_synthetic.csv'):
        print("Dataset not found. Please run generate_dataset.py first.")
        return
    
    # Train model
    model, scaler, class_names, history, test_accuracy = train_model()
    
    # Convert to TFLite
    X, y, _ = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, LabelEncoder().fit_transform(y), test_size=0.2, random_state=42
    )
    X_test_scaled = scaler.transform(X_test)
    
    tflite_model = convert_to_tflite(model, X_test_scaled)
    
    # Test TFLite model
    tflite_accuracy = test_tflite_model(tflite_model, X_test_scaled, y_test, scaler)
    
    # Generate Arduino header
    generate_arduino_header()
    
    print(f"\n=== Training Complete ===")
    print(f"Keras model accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"TFLite model accuracy: {tflite_accuracy:.4f} ({tflite_accuracy*100:.2f}%)")
    print(f"Model size: {len(tflite_model)/1024:.1f} KB")
    print(f"Total parameters: {model.count_params():,}")
    
    print(f"\nFiles generated:")
    print(f"  - ups_model_keras.h5 (Keras model)")
    print(f"  - ups_model_quant.tflite (Quantized TFLite)")
    print(f"  - ups_model_data.h (Arduino header)")
    print(f"  - scaler_mean.npy, scaler_scale.npy (Normalization)")
    print(f"  - class_names.json (Class mapping)")

if __name__ == "__main__":
    main()
