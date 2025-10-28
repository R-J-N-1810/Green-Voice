/*
 * Combined Plant Monitor (Bio-signal, Environment, Soil)
 * - Merges AD623, BME280, BH1750, and Soil Moisture sensors.
 * - Outputs all data as a single JSON string for real-time ML analysis.
 * - Uses a non-blocking timer (millis()) for continuous readings.
 */

// I2C Sensor Libraries
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// JSON Library
#include <ArduinoJson.h>

// --- Sensor Objects ---
BH1750 lightMeter;
Adafruit_BME280 bme; // I2C

// --- Pin Definitions ---
const int AD623_PIN = 34;
const int SOIL_MOISTURE_PIN = 35; // Moved from 34

// --- AD623 Calculation Constants ---
const float ADC_RESOLUTION = 4095.0;
const float V_ESP32_SUPPLY = 3.3;
const float V_AD623_SUPPLY = 5.0;
const float V_REF = V_AD623_SUPPLY / 2.0;
const float AD623_GAIN = 101.0;

// --- Soil Moisture Calibration ---
const int SOIL_DRY = 2620;
const int SOIL_WET = 1085;

// --- Non-Blocking Timer ---
unsigned long lastReadTime = 0;
// Set your desired sample rate here.
// 2000ms = 1 sample every 2 seconds.
// 1000ms = 1 sample every 1 second.
const int readInterval = 2000; 

void setup() {
  Serial.begin(115200);
  Wire.begin(); 

  // --- Initialize BME280 (Temp/Humidity/Pressure) ---
  bool bme_status = bme.begin(0x76);  
  if (!bme_status) {
    Serial.println(F("Error: BME280 sensor not found."));
    while (1);
  } else {
    Serial.println(F("BME280 sensor detected."));
  }

  // --- Initialize BH1750 (Light) ---
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println(F("BH1750 sensor detected."));
  } else {
    Serial.println(F("Error: BH1750 sensor not found."));
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
    int rawADC_AD623 = analogRead(AD623_PIN);
    float vOut_AD623 = (rawADC_AD623 / ADC_RESOLUTION) * V_ESP32_SUPPLY;
    float amplifiedSignal = vOut_AD623 - V_REF;
    float bioSignal_V = amplifiedSignal / AD623_GAIN;

    // --- 2. Read Soil Moisture ---
    int rawADC_Soil = analogRead(SOIL_MOISTURE_PIN);
    float moisturePercent = map(rawADC_Soil, SOIL_DRY, SOIL_WET, 0, 100);
    moisturePercent = constrain(moisturePercent, 0, 100);

    // --- 3. Read Environment (BME280) ---
    float tempC = bme.readTemperature();
    float humidity = bme.readHumidity();
    float pressure_hPa = bme.readPressure() / 100.0F;

    // --- 4. Read Light (BH1750) ---
    float light_lux = lightMeter.readLightLevel();
    if (light_lux < 0) {
      light_lux = 0.0; // Handle read error
    }

    // --- 5. Create JSON Packet ---
    // Increase JSON doc size if you add more fields
    StaticJsonDocument<256> doc;
    
    doc["timestamp"] = currentTime;
    doc["bio_signal_V"] = bioSignal_V; 

    JsonObject environment = doc.createNestedObject("environment");
    environment["temp_C"] = tempC;
    environment["humidity_pct"] = humidity;
    environment["pressure_hPa"] = pressure_hPa;
    environment["light_lux"] = light_lux;
    
    JsonObject soil = doc.createNestedObject("soil");
    soil["moisture_pct"] = moisturePercent;
    soil["moisture_raw"] = rawADC_Soil;

    // --- 6. Send JSON to Serial ---
    serializeJson(doc, Serial);
    Serial.println(); // Send a newline to mark the end of the packet
  }
}