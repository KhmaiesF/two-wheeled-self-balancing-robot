#!/usr/bin/env python3
"""
================================================================================
🔌 TEST BTS7960 - DEUX MOTEURS - RASPBERRY PI 4
================================================================================

CÂBLAGE:
    BTS7960 GAUCHE:          BTS7960 DROIT:
      RPWM ← GPIO 12          RPWM ← GPIO 18
      LPWM ← GPIO 13          LPWM ← GPIO 19
      R_EN ← GPIO 24          R_EN ← GPIO 26
      L_EN ← GPIO 25          L_EN ← GPIO 27
      VCC  ← 5V               VCC  ← 5V
      GND  ← GND              GND  ← GND
      M+/M-← Batterie 12V     M+/M-← Batterie 12V

    MPU6050: SDA=GPIO2, SCL=GPIO3

Usage:
    sudo pigpiod
    python3 test_bts.py
================================================================================
"""

import pigpio
import time

# === GPIO (BCM) ===
LEFT_RPWM  = 12
LEFT_LPWM  = 13
LEFT_R_EN  = 24
LEFT_L_EN  = 25

RIGHT_RPWM = 18
RIGHT_LPWM = 19
RIGHT_R_EN = 26
RIGHT_L_EN = 27

PWM_SPEED = 120  # 0-255

ALL_PWM = [LEFT_RPWM, LEFT_LPWM, RIGHT_RPWM, RIGHT_LPWM]
ALL_EN  = [LEFT_R_EN, LEFT_L_EN, RIGHT_R_EN, RIGHT_L_EN]


def main():
    print("=" * 60)
    print("🔧 TEST BTS7960 - DEUX MOTEURS")
    print("=" * 60)
    print(f"Gauche: RPWM=GPIO{LEFT_RPWM}, LPWM=GPIO{LEFT_LPWM}, EN=GPIO{LEFT_R_EN}/{LEFT_L_EN}")
    print(f"Droit:  RPWM=GPIO{RIGHT_RPWM}, LPWM=GPIO{RIGHT_LPWM}, EN=GPIO{RIGHT_R_EN}/{RIGHT_L_EN}")
    print(f"PWM: {PWM_SPEED}/255")
    print("=" * 60)

    pi = pigpio.pi()
    if not pi.connected:
        print("❌ pigpiod non démarré! Lancez: sudo pigpiod")
        return

    print("✅ Connecté à pigpiod")

    # Activer Enable pins → HIGH
    for pin in ALL_EN:
        pi.set_mode(pin, pigpio.OUTPUT)
        pi.write(pin, 1)
    print("✅ Enable pins activés (HIGH)")

    # Config PWM pins
    for pin in ALL_PWM:
        pi.set_mode(pin, pigpio.OUTPUT)
        pi.set_PWM_frequency(pin, 20000)
        pi.set_PWM_range(pin, 255)
        pi.set_PWM_dutycycle(pin, 0)
    print("✅ PWM configurés (20kHz)")

    print("\n⚠️  Robot SURÉLEVÉ ?")
    input("Appuyez sur ENTRÉE pour commencer...\n")

    try:
        # --- TEST 1 ---
        print("-" * 40)
        print("🔵 TEST 1: Moteur GAUCHE - AVANT")
        pi.set_PWM_dutycycle(LEFT_RPWM, PWM_SPEED)
        time.sleep(2)
        pi.set_PWM_dutycycle(LEFT_RPWM, 0)
        rep = input("   A tourné ? (o/n): ").lower()
        t1 = rep == 'o'
        time.sleep(0.5)

        # --- TEST 2 ---
        print("\n🔵 TEST 2: Moteur GAUCHE - ARRIÈRE")
        pi.set_PWM_dutycycle(LEFT_LPWM, PWM_SPEED)
        time.sleep(2)
        pi.set_PWM_dutycycle(LEFT_LPWM, 0)
        rep = input("   Sens inverse ? (o/n): ").lower()
        t2 = rep == 'o'
        time.sleep(0.5)

        # --- TEST 3 ---
        print("\n" + "-" * 40)
        print("🟢 TEST 3: Moteur DROIT - AVANT")
        pi.set_PWM_dutycycle(RIGHT_RPWM, PWM_SPEED)
        time.sleep(2)
        pi.set_PWM_dutycycle(RIGHT_RPWM, 0)
        rep = input("   A tourné ? (o/n): ").lower()
        t3 = rep == 'o'
        time.sleep(0.5)

        # --- TEST 4 ---
        print("\n🟢 TEST 4: Moteur DROIT - ARRIÈRE")
        pi.set_PWM_dutycycle(RIGHT_LPWM, PWM_SPEED)
        time.sleep(2)
        pi.set_PWM_dutycycle(RIGHT_LPWM, 0)
        rep = input("   Sens inverse ? (o/n): ").lower()
        t4 = rep == 'o'
        time.sleep(0.5)

        # --- TEST 5 ---
        print("\n" + "-" * 40)
        print("🟡 TEST 5: LES DEUX - AVANT")
        pi.set_PWM_dutycycle(LEFT_RPWM, PWM_SPEED)
        pi.set_PWM_dutycycle(RIGHT_RPWM, PWM_SPEED)
        time.sleep(2)
        pi.set_PWM_dutycycle(LEFT_RPWM, 0)
        pi.set_PWM_dutycycle(RIGHT_RPWM, 0)
        rep = input("   Les deux ensemble ? (o/n): ").lower()
        t5 = rep == 'o'
        time.sleep(0.5)

        # --- TEST 6: EN ON/OFF ---
        print("\n" + "-" * 40)
        print("🔴 TEST 6: ENABLE ON/OFF")
        pi.set_PWM_dutycycle(LEFT_RPWM, PWM_SPEED)
        time.sleep(1)
        print("   Désactivation EN...")
        for pin in ALL_EN:
            pi.write(pin, 0)
        time.sleep(1)
        rep1 = input("   Moteur arrêté ? (o/n): ").lower()
        print("   Réactivation EN...")
        for pin in ALL_EN:
            pi.write(pin, 1)
        time.sleep(1)
        rep2 = input("   Moteur redémarré ? (o/n): ").lower()
        pi.set_PWM_dutycycle(LEFT_RPWM, 0)
        t6 = rep1 == 'o' and rep2 == 'o'

    except KeyboardInterrupt:
        print("\n⏹️ Interrompu")
        t1 = t2 = t3 = t4 = t5 = t6 = False
    finally:
        for pin in ALL_PWM:
            pi.set_PWM_dutycycle(pin, 0)
        for pin in ALL_EN:
            pi.write(pin, 0)
        pi.stop()

    # RÉSUMÉ
    ok = lambda v: "✅ OK" if v else "❌ ÉCHEC"
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"  Gauche avant:    {ok(t1)}")
    print(f"  Gauche arrière:  {ok(t2)}")
    print(f"  Droit avant:     {ok(t3)}")
    print(f"  Droit arrière:   {ok(t4)}")
    print(f"  Les deux:        {ok(t5)}")
    print(f"  Enable on/off:   {ok(t6)}")

    if all([t1, t2, t3, t4, t5, t6]):
        print("\n🎉 TOUT EST BON!")
    else:
        print("\n⚠️  Vérifiez le câblage des tests échoués")
    print("=" * 60)


if __name__ == "__main__":
    main()
