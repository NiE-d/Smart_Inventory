
#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ===== WI-FI SETTINGS =====
const char* ssid = "LELE";
const char* password = "gogohaha00";

// Laptop's current IPv4 address
const char* serverURL = "http://10.15.40.117:5000/upload";

// ===== HC-SR04 SETTINGS =====
#define TRIG_PIN D1
#define ECHO_PIN D2

#define MIN_DISTANCE_CM 5.0
#define MAX_DISTANCE_CM 40.0

// ===== XIAO ESP32-S3 SENSE CAMERA PINS =====
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39

#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15

#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

// ===== UPLOAD SETTINGS =====
unsigned long lastUpload = 0;
const unsigned long uploadInterval = 500;

// ===== READ DISTANCE =====
float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, 30000);

  if (duration == 0) {
    return -1.0;
  }

  return duration * 0.0343f / 2.0f;
}

// ===== WI-FI CONNECTION =====
bool connectWiFi(unsigned long timeoutMs = 10000) {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

  Serial.println("Connecting/reconnecting to Wi-Fi...");

  WiFi.disconnect();
  delay(200);
  WiFi.begin(ssid, password);

  unsigned long start = millis();

  while (WiFi.status() != WL_CONNECTED &&
         millis() - start < timeoutMs) {
    delay(250);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWi-Fi connected");
    Serial.print("XIAO IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.printf("\nWi-Fi connection failed. Status: %d\n",
                WiFi.status());
  return false;
}

// ===== SETUP =====
void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  // Initialize camera
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Lower resolution for faster transfers
  config.frame_size = FRAMESIZE_QQVGA;
  config.jpeg_quality = 20;

  // One framebuffer reduces memory pressure
  config.fb_count = 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  esp_err_t err = esp_camera_init(&config);

  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }

  Serial.println("Camera initialized");

  // Set Wi-Fi mode and connect
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);

  if (!connectWiFi()) {
    Serial.println("Will retry Wi-Fi in loop.");
  }

  Serial.print("Upload destination: ");
  Serial.println(serverURL);
}

// ===== MAIN LOOP =====
void loop() {
  // Recover Wi-Fi without blocking forever
  if (WiFi.status() != WL_CONNECTED) {
    if (!connectWiFi(5000)) {
      delay(1000);
      return;
    }
  }

  if (millis() - lastUpload < uploadInterval) {
    delay(5);
    return;
  }

  lastUpload = millis();

  // Measure distance
  float distance = readDistance();

  if (distance < 0) {
    Serial.println("Distance unavailable; skipping image");
    return;
  }

  Serial.printf("Distance: %.1f cm\n", distance);

  // Only upload inside the permitted scan zone
  if (distance < MIN_DISTANCE_CM ||
      distance > MAX_DISTANCE_CM) {
    Serial.println("Outside scan zone; skipping image");
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();

  if (fb == nullptr) {
    Serial.println("Camera capture failed");
    return;
  }

  Serial.printf("Captured image: %u bytes\n",
                (unsigned int)fb->len);

  WiFiClient client;
  HTTPClient http;

  http.setConnectTimeout(2000);
  http.setTimeout(3000);

  bool started = http.begin(client, serverURL);

  if (!started) {
    Serial.println("HTTP connection setup failed");
    esp_camera_fb_return(fb);
    return;
  }

  http.addHeader("Content-Type", "image/jpeg");

  int response = http.POST(fb->buf, fb->len);

  if (response == 200) {
    Serial.println("Image uploaded successfully");
  } else {
    Serial.printf("Image upload failed: %d\n", response);
  }

  http.end();
  esp_camera_fb_return(fb);
}
