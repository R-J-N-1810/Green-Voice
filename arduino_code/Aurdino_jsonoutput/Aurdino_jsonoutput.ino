/*
 * GREEN VOICE: Plant Health Monitoring System - ESP32 Firmware
 * Hardware: ESP32 NodeMCU Development Board
 * Sensors: BME280 (Temp/Humidity/Pressure), BH1750 (Light), 
 *          AD623 (Bio-signal), Capacitive Soil Moisture
 * Output: JSON-formatted sensor data via Serial (115200 baud)
 * Sampling Rate: 5 seconds per measurement cycle
 */

// ========================================
// LIBRARY INCLUDES
// ========================================
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ========================================
// DISPLAY CONFIGURATION
// ========================================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ========================================
// SENSOR OBJECTS
// ========================================
BH1750 lightMeter;
Adafruit_BME280 bme;

// ========================================
// PIN ASSIGNMENTS
// ========================================
const int AD623_PIN = 34;              // Bio-signal amplifier input
const int SOIL_MOISTURE_PIN = 35;      // Capacitive soil sensor

// ========================================
// SIGNAL PROCESSING CONSTANTS
// ========================================
const float ADC_RESOLUTION = 4095.0;
const float V_ESP32_SUPPLY = 3.3;
const float V_AD623_SUPPLY = 5.0;
const float V_REF = V_AD623_SUPPLY / 2.0;
const float AD623_GAIN = 101.0;

// ========================================
// SOIL MOISTURE CALIBRATION
// ========================================
const int SOIL_DRY = 2620;             // ADC reading in dry air
const int SOIL_WET = 1085;             // ADC reading in water

// ========================================
// TIMING CONTROL
// ========================================
unsigned long lastReadTime = 0;
const int readInterval = 5000;         // 5 second sampling interval

// ========================================
// INITIALIZATION
// ========================================
void setup() {
    Serial.begin(115200);
    Wire.begin();
    
    // Initialize OLED Display
    if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { 
        Serial.println(F("Error: OLED display initialization failed"));
        while(1);
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(F("Green Voice System"));
    display.println(F("Initializing..."));
    display.display();
    delay(1000);
    
    // Initialize BME280 Sensor
    if (!bme.begin(0x76)) {
        Serial.println(F("Error: BME280 sensor not detected"));
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println(F("BME280 FAILURE"));
        display.display();
        while(1);
    }
    Serial.println(F("BME280 initialized successfully"));
    
    // Initialize BH1750 Light Sensor
    if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
        Serial.println(F("BH1750 initialized successfully"));
    } else {
        Serial.println(F("Error: BH1750 sensor not detected"));
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println(F("BH1750 FAILURE"));
        display.display();
        while(1);
    }
    
    Serial.println("Green Voice system ready - JSON output active");
}

// ========================================
// MAIN LOOP
// ========================================
void loop() {
    unsigned long currentTime = millis();
    
    // Execute measurement cycle every 5 seconds
    if (currentTime - lastReadTime >= readInterval) {
        lastReadTime = currentTime;
        
        // --- BIO-SIGNAL ACQUISITION ---
        int rawADC_AD623 = analogRead(AD623_PIN);
        float vOut_AD623 = (rawADC_AD623 / ADC_RESOLUTION) * V_ESP32_SUPPLY;
        float amplifiedSignal = vOut_AD623 - V_REF;
        float bioSignal_V = amplifiedSignal / AD623_GAIN;
        float bioSignal_mV = bioSignal_V * 1000.0;
        
        // --- SOIL MOISTURE ACQUISITION ---
        int rawADC_Soil = analogRead(SOIL_MOISTURE_PIN);
        float moisturePercent = map(rawADC_Soil, SOIL_DRY, SOIL_WET, 0, 100);
        moisturePercent = constrain(moisturePercent, 0, 100);
        
        // --- ENVIRONMENTAL SENSOR READINGS ---
        float tempC = bme.readTemperature();
        float humidity = bme.readHumidity();
        float pressure_hPa = bme.readPressure() / 100.0F;
        
        // --- LIGHT INTENSITY READING ---
        float light_lux = lightMeter.readLightLevel();
        if (light_lux < 0) {
            light_lux = 0.0;
        }
        
        // --- OLED DISPLAY UPDATE ---
        display.clearDisplay();
        display.setCursor(0, 0);
        display.print(F("Temp:   "));
        display.print(tempC, 1);
        display.println(F(" C"));
        display.print(F("Humid:  "));
        display.print(humidity, 1);
        display.println(F(" %"));
        display.print(F("Press:  "));
        display.print(pressure_hPa, 1);
        display.println(F(" hPa"));
        display.print(F("Light:  "));
        display.print(light_lux, 0);
        display.println(F(" lx"));
        display.print(F("Soil:   "));
        display.print(moisturePercent, 0);
        display.println(F(" %"));
        display.print(F("BioSig: "));
        display.print(bioSignal_mV, 3);
        display.println(F(" mV"));
        display.display();
        
        // --- JSON OUTPUT TO SERIAL ---
        Serial.print(F("{\"timestamp\":"));
        Serial.print(currentTime);
        Serial.print(F(",\"temp_c\":"));
        Serial.print(tempC, 2);
        Serial.print(F(",\"humidity_p\":"));
        Serial.print(humidity, 2);
        Serial.print(F(",\"pressure_hpa\":"));
        Serial.print(pressure_hPa, 2);
        Serial.print(F(",\"light_lux\":"));
        Serial.print(light_lux, 0);
        Serial.print(F(",\"soil_moisture_p\":"));
        Serial.print(moisturePercent, 0);
        Serial.print(F(",\"bio_signal_mv\":"));
        Serial.print(bioSignal_mV, 3);
        Serial.println(F("}"));
    }
}
