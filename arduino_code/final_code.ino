/*
 * Combined Plant Monitor (Bio-signal, Environment, Soil)
 * - Merges AD623, BME280, BH1750, and Soil Moisture sensors.
 * - Outputs all data as a single JSON string for real-time ML analysis.
 * - Uses a non-blocking timer (millis()) for continuous readings.
 */

// I2C Sensor Libraries
#include <Wire.h>
#include <BH1750.h>          // For Light [cite: 22]
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h> // For Temp/Humidity/Pressure [cite: 28]

// JSON Library
#include <ArduinoJson.h>

// --- Sensor Objects ---
BH1750 lightMeter;
Adafruit_BME280 bme; // I2C [cite: 28]

// --- Pin Definitions ---
// !! CRITICAL: Use two DIFFERENT analog pins!
const int AD623_PIN = 34;         // AD623 Vout [cite: 10]
const int SOIL_MOISTURE_PIN = 35; // Soil moisture sensor (Moved from 34)

// --- AD623 Calculation Constants ---
const float ADC_RESOLUTION = 4095.0;      // ESP32 12-bit ADC
const float V_ESP32_SUPPLY = 3.3;         // ESP32 voltage [cite: 12]
const float V_AD623_SUPPLY = 5.0;
const float V_REF = V_AD623_SUPPLY / 2.0; // [cite: 13]
const float AD623_GAIN = 101.0;           // Gain for 1k resistor [cite: 13]

// --- Soil Moisture Calibration ---
// Calibrate these values for your sensor [cite: 5, 6]
const int SOIL_DRY = 2620; // [cite: 7]
const int SOIL_WET = 1085; // [cite: 8]

// --- Non-Blocking Timer ---
unsigned long lastReadTime = 0;
const int readInterval = 2000; // Read all sensors every 2 seconds

void setup() {
  Serial.begin(115200);
  Wire.begin(); // Initialize I2C bus [cite: 23]

  // --- Initialize BME280 (Temp/Humidity/Pressure) ---
  // Use address 0x76 as you specified [cite: 31]
  bool bme_status = bme.begin(0x76);  
  if (!bme_status) {
    Serial.println(F("Error: BME280 sensor not found. Check wiring.")); [cite: 32]
    while (1);
  } else {
    Serial.println(F("BME280 sensor detected."));
  }

  // --- Initialize BH1750 (Light) ---
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println(F("BH1750 sensor detected.")); [cite: 24]
  } else {
    Serial.println(F("Error: BH1750 sensor not found. Check wiring.")); [cite: 25]
    while (1);
  }

  Serial.println("Plant monitoring system initialized. Starting real-time data output...");
}

void loop() {
  // Use millis() for non-blocking sensor reads
  unsigned long currentTime = millis();
  
  if (currentTime - lastReadTime >= readInterval) {
    lastReadTime = currentTime;
    
    // --- 1. Read Bio-Signal (AD623) ---
    int rawADC_AD623 = analogRead(AD623_PIN); [cite: 15]
    float vOut_AD623 = (rawADC_AD623 / ADC_RESOLUTION) * V_ESP32_SUPPLY; [cite: 16]
    float amplifiedSignal = vOut_AD623 - V_REF;
    float bioSignal_V = amplifiedSignal / AD623_GAIN; [cite: 18]

    // --- 2. Read Soil Moisture ---
    int rawADC_Soil = analogRead(SOIL_MOISTURE_PIN);
    float moisturePercent = map(rawADC_Soil, SOIL_DRY, SOIL_WET, 0, 100);
    // Constrain values to 0-100%
    moisturePercent = constrain(moisturePercent, 0, 100);

    // --- 3. Read Environment (BME280) ---
    float tempC = bme.readTemperature(); [cite: 34]
    float humidity = bme.readHumidity(); [cite: 34]
    float pressure_hPa = bme.readPressure() / 100.0F; [cite: 34]

    // --- 4. Read Light (BH1750) ---
    float light_lux = lightMeter.readLightLevel(); [cite: 26]
    if (light_lux < 0) {
      light_lux = 0.0; // Handle read error [cite: 27]
    }

    // --- 5. Create JSON Packet ---
    StaticJsonDocument<256> doc;
    
    doc["timestamp"] = currentTime;
    // The key data for your ML model
    doc["bio_signal_V"] = bioSignal_V; 

    // Environmental data (context for your model)
    JsonObject environment = doc.createNestedObject("environment");
    environment["temp_C"] = tempC;
    environment["humidity_pct"] = humidity;
    environment["pressure_hPa"] = pressure_hPa;
    environment["light_lux"] = light_lux;
    
    // Soil data
    JsonObject soil = doc.createNestedObject("soil");
    soil["moisture_pct"] = moisturePercent;
    soil["moisture_raw"] = rawADC_Soil;

    // --- 6. Send JSON to Serial ---
    serializeJson(doc, Serial);
    Serial.println(); // Send a newline to mark the end of the packet
  }
}