#include <BluetoothSerial.h>
#include <MPU9250_asukiaaa.h>

#include <MPU9250_asukiaaa.h>
#include "BluetoothSerial.h"
BluetoothSerial SerialBT;
#ifdef _ESP32_HAL_I2C_H_
#define SDA_PIN 21
#define SCL_PIN 22
#endif
 
MPU9250_asukiaaa mySensor;
 
void setup() {
while(!Serial);
 
Serial.begin(115200);
Serial.println("started");
 
#ifdef _ESP32_HAL_I2C_H_
// for esp32
Wire.begin(SDA_PIN, SCL_PIN); //sda, scl
#else
Wire.begin();
#endif

SerialBT.begin("ESP32test"); //Bluetooth device name
Serial.println("The device started, now you can pair it with bluetooth!");
// only print once after start, may adjust pattern
SerialBT.println("accelX,accelY,accelZ,magneX,magneY,magneZ,gyrosX,gyrosY,gyrosZ,time(ms)");
 
mySensor.setWire(&Wire);
 
mySensor.beginAccel();
mySensor.beginMag();
mySensor.beginGyro();

//for special usage
// you can set your own offset for mag values
// mySensor.magXOffset = -50;
// mySensor.magYOffset = -55;
// mySensor.magZOffset = -10;
}
 
void loop() {
mySensor.accelUpdate();
mySensor.magUpdate();
mySensor.gyroUpdate();

String suffix = ",";
String aX = String(mySensor.accelX()) + suffix;
String aY = String(mySensor.accelY()) + suffix;
String aZ = String(mySensor.accelZ()) + suffix;

String mX = String(mySensor.magX()) + suffix;
String mY = String(mySensor.magY()) + suffix;
String mZ = String(mySensor.magZ()) + suffix;

String gX = String(mySensor.gyroX()) + suffix;
String gY = String(mySensor.gyroY()) + suffix;
String gZ = String(mySensor.gyroZ()) + suffix;

// String timeUnit = "(ms)";
String ptime = String(millis()); // + timeUnit;
SerialBT.println(aX + aY + aZ + mX + mY + mZ + gX + gY + gZ + ptime);

delay(8.333); // 120 data points per sec
}
