#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ups_model_data.h"

// GPIO pins
#define RELAY_PIN 25
#define LED_PIN 2

// TensorFlow Lite setup
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

// Memory allocation for model
constexpr int kTensorArenaSize = 60 * 1024; // 60KB for ESP32
uint8_t tensor_arena[kTensorArenaSize];

// Input buffer for sensor readings
float sensor_readings[NUM_FEATURES];

void setup() {
  Serial.begin(115200);
  
  // Initialize GPIO pins
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("UPS AI Fault Detection System Starting...");
  
  // Initialize TensorFlow Lite
  setupTensorFlow();
  
  Serial.println("System ready for inference!");
  Serial.println("Enter 9 sensor values (space-separated):");
  Serial.println("Format: Vin Vout Vbat Iload Ichg Temp_bat Temp_tx mains_ok mode_bat");
}

void setupTensorFlow() {
  // Load the model
  model = tflite::GetModel(ups_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return;
  }
  
  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }
  
  Serial.print("Model loaded successfully. Memory used: ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println(" bytes");
}

void normalizeInputs(float* inputs) {
  // Normalize inputs using saved scaler parameters
  for (int i = 0; i < NUM_FEATURES; i++) {
    inputs[i] = (inputs[i] - scaler_mean[i]) / scaler_scale[i];
  }
}

int runInference(float* inputs) {
  // Get input tensor
  TfLiteTensor* input = interpreter->input(0);
  
  // Normalize inputs
  normalizeInputs(inputs);
  
  // Copy normalized data to input tensor
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = inputs[i];
  }
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return -1;
  }
  
  // Get output tensor
  TfLiteTensor* output = interpreter->output(0);
  
  // Find the class with highest probability
  int predicted_class = 0;
  float max_prob = output->data.f[0];
  
  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > max_prob) {
      max_prob = output->data.f[i];
      predicted_class = i;
    }
  }
  
  // Print probabilities for debugging
  Serial.print("Probabilities: ");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(class_names[i]);
    Serial.print(": ");
    Serial.print(output->data.f[i], 3);
    Serial.print(" ");
  }
  Serial.println();
  
  return predicted_class;
}

void handleFaultDetection(int predicted_class) {
  const char* fault_name = class_names[predicted_class];
  
  Serial.print("Predicted fault: ");
  Serial.println(fault_name);
  
  // Check if it's a fault condition
  if (strcmp(fault_name, "normal") != 0) {
    // Fault detected - activate relay and LED
    digitalWrite(RELAY_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    Serial.println("⚠️ FAULT DETECTED - Relay activated!");
    
    // Send alert based on fault type
    if (strcmp(fault_name, "battery_low") == 0) {
      Serial.println("Alert: Battery voltage is low. Check battery condition.");
    } else if (strcmp(fault_name, "charging_fault") == 0) {
      Serial.println("Alert: Charging system fault. Check charger connection.");
    } else if (strcmp(fault_name, "mains_fail") == 0) {
      Serial.println("Alert: Mains power failure detected. Running on battery.");
    } else if (strcmp(fault_name, "overload") == 0) {
      Serial.println("Alert: System overload detected. Reduce load immediately.");
    } else if (strcmp(fault_name, "short_circuit") == 0) {
      Serial.println("Alert: SHORT CIRCUIT! Disconnect load immediately!");
    } else if (strcmp(fault_name, "transformer_overheat") == 0) {
      Serial.println("Alert: Transformer overheating. Check ventilation and load.");
    }
  } else {
    // Normal operation
    digitalWrite(RELAY_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    Serial.println("✅ System operating normally");
  }
}

void simulateRealisticReadings() {
  // Simulate realistic UPS sensor readings
  // You would replace this with actual sensor reading code
  
  static int simulation_mode = 0;
  simulation_mode = (simulation_mode + 1) % 7; // Cycle through different scenarios
  
  switch (simulation_mode) {
    case 0: // Normal operation
      sensor_readings[0] = 230.5;  // Vin
      sensor_readings[1] = 225.2;  // Vout
      sensor_readings[2] = 13.2;   // Vbat
      sensor_readings[3] = 3.5;    // Iload
      sensor_readings[4] = 1.2;    // Ichg
      sensor_readings[5] = 25.8;   // Temp_bat
      sensor_readings[6] = 35.4;   // Temp_tx
      sensor_readings[7] = 1;      // mains_ok
      sensor_readings[8] = 0;      // mode_bat
      break;
      
    case 1: // Battery low
      sensor_readings[0] = 232.1;
      sensor_readings[1] = 210.5;
      sensor_readings[2] = 11.2;   // Low battery
      sensor_readings[3] = 4.0;
      sensor_readings[4] = 0.1;
      sensor_readings[5] = 28.5;
      sensor_readings[6] = 38.2;
      sensor_readings[7] = 1;
      sensor_readings[8] = 1;      // On battery
      break;
      
    case 2: // Charging fault
      sensor_readings[0] = 228.7;
      sensor_readings[1] = 224.1;
      sensor_readings[2] = 12.1;
      sensor_readings[3] = 2.8;
      sensor_readings[4] = 0.0;    // No charging
      sensor_readings[5] = 30.2;
      sensor_readings[6] = 42.1;
      sensor_readings[7] = 1;
      sensor_readings[8] = 0;
      break;
      
    case 3: // Mains fail
      sensor_readings[0] = 85.2;   // Low input voltage
      sensor_readings[1] = 220.5;
      sensor_readings[2] = 12.5;
      sensor_readings[3] = 5.2;
      sensor_readings[4] = 0.0;
      sensor_readings[5] = 32.1;
      sensor_readings[6] = 45.8;
      sensor_readings[7] = 0;      // Mains failed
      sensor_readings[8] = 1;      // On battery
      break;
      
    case 4: // Overload
      sensor_readings[0] = 235.2;
      sensor_readings[1] = 205.8;  // Voltage drop
      sensor_readings[2] = 13.0;
      sensor_readings[3] = 12.5;   // High load
      sensor_readings[4] = 0.8;
      sensor_readings[5] = 38.5;   // Higher temps
      sensor_readings[6] = 65.2;
      sensor_readings[7] = 1;
      sensor_readings[8] = 0;
      break;
      
    case 5: // Short circuit
      sensor_readings[0] = 231.8;
      sensor_readings[1] = 45.2;   // Very low output
      sensor_readings[2] = 11.8;
      sensor_readings[3] = 18.5;   // Very high current
      sensor_readings[4] = 0.0;
      sensor_readings[5] = 55.2;   // High temps
      sensor_readings[6] = 85.7;
      sensor_readings[7] = 1;
      sensor_readings[8] = 0;
      break;
      
    case 6: // Transformer overheat
      sensor_readings[0] = 229.4;
      sensor_readings[1] = 218.5;
      sensor_readings[2] = 13.1;
      sensor_readings[3] = 7.8;
      sensor_readings[4] = 1.5;
      sensor_readings[5] = 42.5;
      sensor_readings[6] = 95.8;   // Overheated transformer
      sensor_readings[7] = 1;
      sensor_readings[8] = 0;
      break;
  }
  
  Serial.print("Simulated readings: ");
  for (int i = 0; i < NUM_FEATURES; i++) {
    Serial.print(sensor_readings[i], 1);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Check for serial input
  if (Serial.available()) {
    // Read manual input
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() > 0) {
      // Parse space-separated values
      int value_count = 0;
      int start_pos = 0;
      
      for (int i = 0; i <= input.length() && value_count < NUM_FEATURES; i++) {
        if (i == input.length() || input.charAt(i) == ' ') {
          if (i > start_pos) {
            sensor_readings[value_count] = input.substring(start_pos, i).toFloat();
            value_count++;
          }
          start_pos = i + 1;
        }
      }
      
      if (value_count == NUM_FEATURES) {
        Serial.print("Manual input: ");
        for (int i = 0; i < NUM_FEATURES; i++) {
          Serial.print(sensor_readings[i], 1);
          Serial.print(" ");
        }
        Serial.println();
        
        // Run inference
        int predicted_class = runInference(sensor_readings);
        if (predicted_class >= 0) {
          handleFaultDetection(predicted_class);
        }
      } else {
        Serial.println("Error: Please provide exactly 9 values");
      }
    }
  } else {
    // Auto-simulation mode
    static unsigned long last_inference = 0;
    unsigned long current_time = millis();
    
    if (current_time - last_inference > 5000) { // Every 5 seconds
      last_inference = current_time;
      
      Serial.println("\n--- Auto Simulation ---");
      simulateRealisticReadings();
      
      // Run inference
      int predicted_class = runInference(sensor_readings);
      if (predicted_class >= 0) {
        handleFaultDetection(predicted_class);
      }
      
      Serial.println("Enter manual values or wait for next simulation...");
    }
  }
  
  delay(100);
}
