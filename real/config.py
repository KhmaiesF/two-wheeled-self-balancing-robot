# === Pins (à adapter selon câblage) ===
# BTS7960: pour chaque moteur, 2 PWM (RPWM/LPWM)

LEFT_RPWM = 12
LEFT_LPWM = 13
RIGHT_RPWM = 19    # Inversé (moteur monté en miroir)
RIGHT_LPWM = 18    # Inversé (moteur monté en miroir)

# Inversion logicielle du sens moteur (sans recâbler)
LEFT_MOTOR_INVERTED = False
RIGHT_MOTOR_INVERTED = True

# Enable pins BTS7960 (via GPIO, pas besoin de 5V/3.3V séparés)
LEFT_R_EN = 24
LEFT_L_EN = 25
RIGHT_R_EN = 26
RIGHT_L_EN = 27

# Encodeurs (A/B) (optionnel)
ENC_L_A = 5
ENC_L_B = 6
ENC_R_A = 16
ENC_R_B = 20

# IMU I2C bus
I2C_BUS = 1
MPU_ADDR = 0x68

# Boucle contrôle
CONTROL_HZ = 200
PPO_HZ = 100

MAX_TORQUE_EQUIV = 0.5          # Torque max simulé (mappé vers PWM)
PWM_MAX = 255
PWM_DEADZONE = 25               # PWM min pour vaincre frottement statique

# === PID auto-tuné ===
PID_KP = 20.0
PID_KI = 3.0
PID_KD = 0.5
KPOS = 0.15
KVEL = 0.15
