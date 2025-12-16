#include <Wire.h>
#include <Adafruit_LSM6DSOX.h>
#include <Adafruit_Sensor.h>

// PINS VALIDES (Ceux qui marchent chez toi)
#define PIN_SDA 5
#define PIN_SCL 4

// L'ADRESSE QUE TON SCANNER A TROUVEE
#define ADRESSE_CIBLE 0x6A 

Adafruit_LSM6DSOX imu;

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);

  // Démarrage I2C
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(100000); // Vitesse lente pour éviter le retour du fantôme 0x7E

  Serial.print("Connexion forcee sur 0x6B... ");
  
  if (!imu.begin_I2C(ADRESSE_CIBLE, &Wire)) {
    Serial.println("ECHEC.");
    Serial.println("Si ca rate, verifie que le fil SDO touche toujours le 3.3V.");
    while(1);
  }
  
  Serial.println("SUCCES !");
  
  // Configuration pour le mouvement
  imu.setAccelRange(LSM6DS_ACCEL_RANGE_4_G);
  imu.setGyroRange(LSM6DS_GYRO_RANGE_500_DPS);
}

void loop() {
  sensors_event_t a, g, t;
  imu.getEvent(&a, &g, &t);

  // Format JSON standard pour Python
  Serial.print("{\"accel\":{\"x\":");
  Serial.print(a.acceleration.x, 2);
  Serial.print(",\"y\":"); Serial.print(a.acceleration.y, 2);
  Serial.print(",\"z\":"); Serial.print(a.acceleration.z, 2);
  Serial.print("},\"gyro\":{\"z\":"); 
  Serial.print(g.gyro.z, 3);
  Serial.println("}}");

  delay(15);
}