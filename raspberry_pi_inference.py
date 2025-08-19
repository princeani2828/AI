import numpy as np
import json
import time
import csv
from datetime import datetime

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    # Fallback to full TensorFlow if tflite_runtime is not available
    import tensorflow as tf
    tflite = tf.lite

class UPSFaultDetector:
    def __init__(self, model_path='ups_model_quant.tflite'):
        """Initialize the UPS fault detector for Raspberry Pi"""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.class_names = None
        
        self.load_model()
        self.load_scaler_and_classes()
    
    def load_model(self):
        """Load the TFLite model"""
        print(f"Loading TFLite model from {self.model_path}...")
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def load_scaler_and_classes(self):
        """Load scaler parameters and class names"""
        try:
            self.scaler_mean = np.load('scaler_mean.npy')
            self.scaler_scale = np.load('scaler_scale.npy')
            print("Scaler parameters loaded")
        except FileNotFoundError:
            print("Warning: Scaler files not found. Assuming no normalization needed.")
            self.scaler_mean = np.zeros(9)
            self.scaler_scale = np.ones(9)
        
        try:
            with open('class_names.json', 'r') as f:
                class_names_dict = json.load(f)
                self.class_names = [class_names_dict[str(i)] for i in range(len(class_names_dict))]
            print(f"Class names loaded: {self.class_names}")
        except FileNotFoundError:
            print("Warning: class_names.json not found. Using default names.")
            self.class_names = ['normal', 'battery_low', 'charging_fault', 'mains_fail', 
                              'overload', 'short_circuit', 'transformer_overheat']
    
    def normalize_inputs(self, inputs):
        """Normalize inputs using saved scaler parameters"""
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (inputs - self.scaler_mean) / self.scaler_scale
        return inputs
    
    def predict(self, sensor_readings):
        """Run inference on sensor readings"""
        # Convert to numpy array and normalize
        inputs = np.array(sensor_readings, dtype=np.float32).reshape(1, -1)
        inputs = self.normalize_inputs(inputs)
        
        # Quantize input if needed (for INT8 models)
        if self.input_details[0]['dtype'] == np.int8:
            input_scale = self.input_details[0]['quantization'][0]
            input_zero_point = self.input_details[0]['quantization'][1]
            inputs = (inputs / input_scale + input_zero_point).astype(np.int8)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Dequantize output if needed
        if self.output_details[0]['dtype'] == np.int8:
            output_scale = self.output_details[0]['quantization'][0]
            output_zero_point = self.output_details[0]['quantization'][1]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        # Get predicted class and probabilities
        predicted_class = np.argmax(output[0])
        probabilities = output[0]
        
        return predicted_class, probabilities
    
    def get_fault_name(self, class_index):
        """Get fault name from class index"""
        if 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return "unknown"
    
    def is_fault_condition(self, fault_name):
        """Check if the predicted condition is a fault"""
        return fault_name != 'normal'

def simulate_sensor_readings():
    """Simulate different UPS sensor scenarios"""
    scenarios = {
        'normal': [230.5, 225.2, 13.2, 3.5, 1.2, 25.8, 35.4, 1, 0],
        'battery_low': [232.1, 210.5, 11.2, 4.0, 0.1, 28.5, 38.2, 1, 1],
        'charging_fault': [228.7, 224.1, 12.1, 2.8, 0.0, 30.2, 42.1, 1, 0],
        'mains_fail': [85.2, 220.5, 12.5, 5.2, 0.0, 32.1, 45.8, 0, 1],
        'overload': [235.2, 205.8, 13.0, 12.5, 0.8, 38.5, 65.2, 1, 0],
        'short_circuit': [231.8, 45.2, 11.8, 18.5, 0.0, 55.2, 85.7, 1, 0],
        'transformer_overheat': [229.4, 218.5, 13.1, 7.8, 1.5, 42.5, 95.8, 1, 0]
    }
    
    scenario_name = input("Enter scenario (normal/battery_low/charging_fault/mains_fail/overload/short_circuit/transformer_overheat) or 'manual': ").strip()
    
    if scenario_name == 'manual':
        print("Enter 9 sensor values:")
        print("Format: Vin Vout Vbat Iload Ichg Temp_bat Temp_tx mains_ok mode_bat")
        manual_input = input("Values: ").strip().split()
        try:
            return [float(x) for x in manual_input]
        except (ValueError, IndexError):
            print("Invalid input. Using normal scenario.")
            return scenarios['normal']
    elif scenario_name in scenarios:
        return scenarios[scenario_name]
    else:
        print("Unknown scenario. Using normal.")
        return scenarios['normal']

def read_from_csv(csv_file):
    """Read sensor data from CSV file"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract the 9 features
                readings = [
                    float(row['Vin']), float(row['Vout']), float(row['Vbat']),
                    float(row['Iload']), float(row['Ichg']), float(row['Temp_bat']),
                    float(row['Temp_tx']), float(row['mains_ok']), float(row['mode_bat'])
                ]
                yield readings, row.get('fault_type', 'unknown')
    except FileNotFoundError:
        print(f"CSV file {csv_file} not found.")
        return

def log_prediction(sensor_readings, predicted_class, fault_name, probabilities, log_file='ups_predictions.log'):
    """Log predictions to file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp}, {', '.join(map(str, sensor_readings))}, {predicted_class}, {fault_name}, {', '.join([f'{p:.3f}' for p in probabilities])}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)

def main():
    """Main function for Raspberry Pi inference"""
    print("=== UPS Fault Detection - Raspberry Pi ===\n")
    
    # Initialize detector
    detector = UPSFaultDetector()
    
    # Main loop
    while True:
        print("\nOptions:")
        print("1. Simulate sensor readings")
        print("2. Process CSV file")
        print("3. Continuous monitoring (simulated)")
        print("4. Exit")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == '1':
            # Single prediction
            sensor_readings = simulate_sensor_readings()
            
            print(f"\nSensor readings: {sensor_readings}")
            
            # Run prediction
            predicted_class, probabilities = detector.predict(sensor_readings)
            fault_name = detector.get_fault_name(predicted_class)
            
            print(f"\nPrediction Results:")
            print(f"Predicted fault: {fault_name}")
            print(f"Confidence: {probabilities[predicted_class]:.3f}")
            
            print(f"\nAll probabilities:")
            for i, prob in enumerate(probabilities):
                print(f"  {detector.class_names[i]}: {prob:.3f}")
            
            if detector.is_fault_condition(fault_name):
                print(f"⚠️ FAULT DETECTED: {fault_name}")
                # Here you would trigger alerts, log to database, etc.
            else:
                print("✅ System operating normally")
            
            # Log prediction
            log_prediction(sensor_readings, predicted_class, fault_name, probabilities)
        
        elif choice == '2':
            # Process CSV file
            csv_file = input("Enter CSV file path (or press Enter for ups_synthetic.csv): ").strip()
            if not csv_file:
                csv_file = 'ups_synthetic.csv'
            
            print(f"Processing {csv_file}...")
            
            correct_predictions = 0
            total_predictions = 0
            
            for readings, actual_fault in read_from_csv(csv_file):
                predicted_class, probabilities = detector.predict(readings)
                fault_name = detector.get_fault_name(predicted_class)
                
                total_predictions += 1
                if fault_name == actual_fault:
                    correct_predictions += 1
                
                if total_predictions <= 10:  # Show first 10 predictions
                    print(f"Actual: {actual_fault:15} | Predicted: {fault_name:15} | Confidence: {probabilities[predicted_class]:.3f}")
                
                # Log prediction
                log_prediction(readings, predicted_class, fault_name, probabilities)
            
            if total_predictions > 0:
                accuracy = correct_predictions / total_predictions
                print(f"\nProcessed {total_predictions} samples")
                print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        elif choice == '3':
            # Continuous monitoring
            print("Starting continuous monitoring (Ctrl+C to stop)...")
            scenario_names = ['normal', 'battery_low', 'charging_fault', 'mains_fail', 
                            'overload', 'short_circuit', 'transformer_overheat']
            
            try:
                import random
                while True:
                    # Randomly select a scenario
                    scenario = random.choice(scenario_names)
                    sensor_readings = simulate_sensor_readings()
                    
                    predicted_class, probabilities = detector.predict(sensor_readings)
                    fault_name = detector.get_fault_name(predicted_class)
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"[{timestamp}] Predicted: {fault_name:15} | Confidence: {probabilities[predicted_class]:.3f}")
                    
                    if detector.is_fault_condition(fault_name):
                        print(f"          ⚠️ FAULT ALERT: {fault_name}")
                    
                    # Log prediction
                    log_prediction(sensor_readings, predicted_class, fault_name, probabilities)
                    
                    time.sleep(2)  # Wait 2 seconds
                    
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")
        
        elif choice == '4':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
