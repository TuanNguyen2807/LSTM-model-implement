#include <ESP8266WiFi.h>
#include <PubSubClient.h> // https://github.com/knolleary/pubsubclient/releases/tag/v2.3
#include <ArduinoJson.h> // https://github.com/bblanchon/ArduinoJson/releases/tag/v5.0.7

//-------- Customise these values -----------
const char* ssid = "Meo Meo";
const char* password = "nguyenanh0509";

#define ORG "0tj0q8"
#define DEVICE_TYPE "ESP8266"
#define DEVICE_ID "esp82661"
#define TOKEN "?zDpbJOA2E3vxEDlr*"
//-------- Customise the above values --------

char server[] = ORG ".messaging.internetofthings.ibmcloud.com";
char authMethod[] = "use-token-auth";
char token[] = TOKEN;
char clientId[] = "d:" ORG ":" DEVICE_TYPE ":" DEVICE_ID;

const char eventTopic[] = "iot-2/evt/status/fmt/json";
const char cmdTopic[] = "iot-2/cmd/led/fmt/json";



WiFiClient wifiClient;
void callback(char* topic, byte* payload, unsigned int payloadLength) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  for (int i = 0; i < payloadLength; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();

  // Switch on the LED if an 1 was received as first character
  if (payload[0] == '1') {
    digitalWrite(BUILTIN_LED, LOW);   // Turn the LED on (Note that LOW is the voltage level
    // but actually the LED is on; this is because
    // it is acive low on the ESP-01)
  } else {
    digitalWrite(BUILTIN_LED, HIGH);  // Turn the LED off by making the voltage HIGH
  }

}
PubSubClient client(server, 1883, callback, wifiClient);

int publishInterval = 5000; // 2 mins//Send adc every 5sc
long lastPublishMillis;

void setup() {
  Serial.begin(115200); Serial.println();
  pinMode(LED_BUILTIN, OUTPUT);
  wifiConnect();
  mqttConnect();
}

void loop() {
  if (millis() - lastPublishMillis > publishInterval) {
    publishData();
    lastPublishMillis = millis();
  }

  if (!client.loop()) {
    mqttConnect();
  }
}

void wifiConnect() {
  Serial.print("Connecting to "); Serial.print(ssid);  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.print("nWiFi connected, IP address: "); Serial.println(WiFi.localIP());

}

void mqttConnect() {
  if (!!!client.connected()) {
    Serial.print("Reconnecting MQTT client to "); Serial.println(server);
    while (!!!client.connect(clientId, authMethod, token)) {
      Serial.print(".");
      delay(500);
    }
    if (client.subscribe(cmdTopic)) {
      Serial.println("subscribe to responses OK");
    } else {
      Serial.println("subscribe to responses FAILED");
    }
    Serial.println();
  }
}


void publishData() {
  temp = random(20,30)
  ph = random(6.0,8.0)
  turbidity = random(30, 70)
  oxygen = random(8.0,12.0)
  String payload = "{\"time\":\"21/06/2021 00:00\",\"temp\": abc,\"ph\":8.3,\"turbidity\":36.1,\"oxygen\":5.6}";

  Serial.print("Sending payload: "); Serial.println(payload);

  if (client.publish(eventTopic, (char*) payload.c_str())) {
    Serial.println("Publish OK");
  } else {
    Serial.println("Publish FAILED");
  }
}
