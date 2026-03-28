#!/usr/bin/env python3
"""
================================================================================
🔌 TEST BTS7960 - DEUX MOTEURS - RPi.GPIO
================================================================================

CÂBLAGE:
    BTS7960 GAUCHE:          BTS7960 DROIT:
      RPWM ← GPIO 12          RPWM ← GPIO 18
      LPWM ← GPIO 13          LPWM ← GPIO 19
      R_EN ← GPIO 24          R_EN ← GPIO 26
      L_EN ← GPIO 25          L_EN ← GPIO 27

Usage:
    python3 test_bts_rpi.py
================================================================================
"""

import RPi.GPIO as GPIO
import time

from config import (
    LEFT_RPWM, LEFT_LPWM,
    RIGHT_RPWM, RIGHT_LPWM,
    LEFT_R_EN, LEFT_L_EN,
    RIGHT_R_EN, RIGHT_L_EN,
)

PWM_FREQ = 1000
DUTY_CYCLE = 50  # %

ALL_EN = [LEFT_R_EN, LEFT_L_EN, RIGHT_R_EN, RIGHT_L_EN]

print("=" * 60)
print("🔧 TEST BTS7960 - DEUX MOTEURS (RPi.GPIO)")
print("=" * 60)
print(f"Gauche: RPWM=GPIO{LEFT_RPWM}, LPWM=GPIO{LEFT_LPWM}")
print(f"Droit:  RPWM=GPIO{RIGHT_RPWM}, LPWM=GPIO{RIGHT_LPWM}")
print(f"Enable: GPIO{LEFT_R_EN}/{LEFT_L_EN} (G), GPIO{RIGHT_R_EN}/{RIGHT_L_EN} (D)")
print(f"PWM: {PWM_FREQ}Hz, {DUTY_CYCLE}%")
print("=" * 60)

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Enable pins → HIGH
for pin in ALL_EN:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH)
print("✅ Enable pins activés")

# PWM pins
GPIO.setup(LEFT_RPWM, GPIO.OUT)
GPIO.setup(LEFT_LPWM, GPIO.OUT)
GPIO.setup(RIGHT_RPWM, GPIO.OUT)
GPIO.setup(RIGHT_LPWM, GPIO.OUT)

pwm_l_fwd = GPIO.PWM(LEFT_RPWM, PWM_FREQ)
pwm_l_bwd = GPIO.PWM(LEFT_LPWM, PWM_FREQ)
pwm_r_fwd = GPIO.PWM(RIGHT_RPWM, PWM_FREQ)
pwm_r_bwd = GPIO.PWM(RIGHT_LPWM, PWM_FREQ)

pwm_l_fwd.start(0)
pwm_l_bwd.start(0)
pwm_r_fwd.start(0)
pwm_r_bwd.start(0)
print("✅ GPIO configurés")

print("\n⚠️  Robot SURÉLEVÉ ?")
input("Appuyez sur ENTRÉE pour commencer...\n")

try:
    print("▶️  TEST 1: Gauche AVANT")
    pwm_l_fwd.ChangeDutyCycle(DUTY_CYCLE)
    time.sleep(2)
    pwm_l_fwd.ChangeDutyCycle(0)
    time.sleep(1)

    print("◀️  TEST 2: Gauche ARRIÈRE")
    pwm_l_bwd.ChangeDutyCycle(DUTY_CYCLE)
    time.sleep(2)
    pwm_l_bwd.ChangeDutyCycle(0)
    time.sleep(1)

    print("▶️  TEST 3: Droit AVANT")
    pwm_r_fwd.ChangeDutyCycle(DUTY_CYCLE)
    time.sleep(2)
    pwm_r_fwd.ChangeDutyCycle(0)
    time.sleep(1)

    print("◀️  TEST 4: Droit ARRIÈRE")
    pwm_r_bwd.ChangeDutyCycle(DUTY_CYCLE)
    time.sleep(2)
    pwm_r_bwd.ChangeDutyCycle(0)
    time.sleep(1)

    print("🔄 TEST 5: Rampe gauche (0→100%→0)")
    for duty in range(0, 101, 10):
        pwm_l_fwd.ChangeDutyCycle(duty)
        time.sleep(0.2)
    for duty in range(100, -1, -10):
        pwm_l_fwd.ChangeDutyCycle(duty)
        time.sleep(0.2)
    time.sleep(1)

    print("🟡 TEST 6: Les deux AVANT")
    pwm_l_fwd.ChangeDutyCycle(DUTY_CYCLE)
    pwm_r_fwd.ChangeDutyCycle(DUTY_CYCLE)
    time.sleep(2)
    pwm_l_fwd.ChangeDutyCycle(0)
    pwm_r_fwd.ChangeDutyCycle(0)

    print("\n✅ TESTS TERMINÉS!")

except KeyboardInterrupt:
    print("\n⏹️ Interrompu")

finally:
    pwm_l_fwd.ChangeDutyCycle(0)
    pwm_l_bwd.ChangeDutyCycle(0)
    pwm_r_fwd.ChangeDutyCycle(0)
    pwm_r_bwd.ChangeDutyCycle(0)
    pwm_l_fwd.stop()
    pwm_l_bwd.stop()
    pwm_r_fwd.stop()
    pwm_r_bwd.stop()
    for pin in ALL_EN:
        GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()
    print("👋 Fin")
