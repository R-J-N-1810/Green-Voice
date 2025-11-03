/*
 * Combined Plant Monitor (Bio-signal, Environment, Soil)
 * - Merges AD623, BME280, BH1750, and Soil Moisture sensors.
 * - Displays all data on a 0.96" I2C OLED display.
 * - Uses a non-blocking timer (millis()) for continuous readings.
 * * --- MODIFIED: Reading interval set to 5 seconds.
 * --- MODIFIED: Data output to Serial is now in JSON format.
 */

// I2C Sensor Libraries
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>

// --- ADDED: I2C Display Libraries ---
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- Display Definitions ---
#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels
#define OLED_RESET     -1 // Reset pin # (or -1 if sharing Arduino reset pin)
// Initialize the I2C display
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);


// --- Sensor Objects ---
BH1750 lightMeter;
Adafruit_BME280 bme; // I2C

// --- Pin Definitions ---
const int AD623_PIN = 34;
const int SOIL_MOISTURE_PIN = 35;

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
// *** MODIFIED: Changed from 2000 to 5000 milliseconds (5 seconds) ***
const int readInterval = 5000; // 5-second interval

void setup() {
    Serial.begin(115200);
    Wire.begin();

    // --- Initialize OLED Display ---
    // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V
    // 0x3C is the most common I2C address for 0.96" displays
    if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println(F("Error: SSD1306 OLED display failed to init."));
        while(1); // Don't proceed
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println(F("Plant Monitor Boot..."));
    display.display();
    delay(1000); // Show boot message

    // --- Initialize BME280 (Temp/Humidity/Pressure) ---
    bool bme_status = bme.begin(0x76);
    if (!bme_status) {
        Serial.println(F("Error: BME280 sensor not found."));
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println(F("BME280 NOT FOUND"));
        display.display();
        while (1);
    } else {
        Serial.println(F("BME280 sensor detected."));
    }

    // --- Initialize BH1750 (Light) ---
    if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
        Serial.println(F("BH1750 sensor detected."));
    } else {
        Serial.println(F("Error: BH1750 sensor not found."));
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println(F("BH1750 NOT FOUND"));
        display.display();
        while (1);
    }

    Serial.println("Plant monitoring system initialized. Starting JSON data output...");
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
        float bioSignal_mV = bioSignal_V * 1000.0; // Convert to millivolts for easier reading

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

        // --- 5. Display Data on OLED (No change from original) ---
        display.clearDisplay();
        display.setCursor(0, 0);

        display.print(F("Temp:   ")); display.print(tempC, 1); display.println(F(" C"));
        display.print(F("Humid:  ")); display.print(humidity, 1); display.println(F(" %"));
        display.print(F("Press:  ")); display.print(pressure_hPa, 1); display.println(F(" hPa"));
        display.print(F("Light:  ")); display.print(light_lux, 0); display.println(F(" lx"));
        display.print(F("Soil:   ")); display.print(moisturePercent, 0); display.println(F(" %"));
        display.print(F("BioSig: ")); display.print(bioSignal_mV, 3); display.println(F(" mV"));

        display.display();

        // --- 6. Print JSON Data to Serial for ML Model Ingestion ---
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
        Serial.println(F("}")); // Cleanly ends the JSON object with a newline
    }
}