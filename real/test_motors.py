#!/usr/bin/env python3
"""
================================================================================
🔧 TEST MOTEURS BTS7960 (RPi.GPIO)
================================================================================

Script interactif pour tester les moteurs via RPi.GPIO (même lib que test_bts_rpi.py).

Usage:
    python3 test_motors.py              # Mode interactif
    python3 test_motors.py --auto       # Test automatique complet
    python3 test_motors.py --left       # Test moteur gauche
    python3 test_motors.py --right      # Test moteur droit

ATTENTION: Surélever le robot avant de tester !
================================================================================
"""

import argparse
import time
import sys

import RPi.GPIO as GPIO

from config import (
    LEFT_RPWM, LEFT_LPWM,
    RIGHT_RPWM, RIGHT_LPWM,
    LEFT_R_EN, LEFT_L_EN,
    RIGHT_R_EN, RIGHT_L_EN,
)

ALL_EN  = [LEFT_R_EN, LEFT_L_EN, RIGHT_R_EN, RIGHT_L_EN]
PWM_FREQ = 1000   # Hz (comme test_bts_rpi.py)

# Objets PWM globaux
pwm_l_fwd = None
pwm_l_bwd = None
pwm_r_fwd = None
pwm_r_bwd = None


def init_gpio():
    """Configure GPIO et crée les objets PWM."""
    global pwm_l_fwd, pwm_l_bwd, pwm_r_fwd, pwm_r_bwd

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Enable pins → HIGH
    for pin in ALL_EN:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.HIGH)
    print(f"✅ Enable pins activés: {ALL_EN}")

    # PWM pins
    for pin in [LEFT_RPWM, LEFT_LPWM, RIGHT_RPWM, RIGHT_LPWM]:
        GPIO.setup(pin, GPIO.OUT)

    pwm_l_fwd = GPIO.PWM(LEFT_RPWM, PWM_FREQ)
    pwm_l_bwd = GPIO.PWM(LEFT_LPWM, PWM_FREQ)
    pwm_r_fwd = GPIO.PWM(RIGHT_RPWM, PWM_FREQ)
    pwm_r_bwd = GPIO.PWM(RIGHT_LPWM, PWM_FREQ)

    pwm_l_fwd.start(0)
    pwm_l_bwd.start(0)
    pwm_r_fwd.start(0)
    pwm_r_bwd.start(0)

    print(f"✅ PWM configurés ({PWM_FREQ}Hz)")
    print(f"   Gauche: RPWM=GPIO{LEFT_RPWM}, LPWM=GPIO{LEFT_LPWM}")
    print(f"   Droit:  RPWM=GPIO{RIGHT_RPWM}, LPWM=GPIO{RIGHT_LPWM}")


def cleanup():
    """Arrête moteurs et libère GPIO."""
    stop_all()
    for p in [pwm_l_fwd, pwm_l_bwd, pwm_r_fwd, pwm_r_bwd]:
        if p:
            p.stop()
    for pin in ALL_EN:
        GPIO.output(pin, GPIO.LOW)
    GPIO.cleanup()
    print("🔌 GPIO libéré")


def set_motor(fwd_pwm, bwd_pwm, duty):
    """
    Commande un moteur.
    duty > 0 : avant (0-100%)
    duty < 0 : arrière (0-100%)
    duty = 0 : arrêt
    """
    duty = max(-100, min(100, duty))
    if duty > 0:
        fwd_pwm.ChangeDutyCycle(duty)
        bwd_pwm.ChangeDutyCycle(0)
    elif duty < 0:
        fwd_pwm.ChangeDutyCycle(0)
        bwd_pwm.ChangeDutyCycle(abs(duty))
    else:
        fwd_pwm.ChangeDutyCycle(0)
        bwd_pwm.ChangeDutyCycle(0)


def set_left(duty):
    set_motor(pwm_l_fwd, pwm_l_bwd, duty)


def set_right(duty):
    set_motor(pwm_r_fwd, pwm_r_bwd, duty)


def set_both(duty):
    set_left(duty)
    set_right(duty)


def stop_all():
    for p in [pwm_l_fwd, pwm_l_bwd, pwm_r_fwd, pwm_r_bwd]:
        if p:
            p.ChangeDutyCycle(0)


# ──────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────

def test_ramp(name, fwd_pwm, bwd_pwm, max_duty=60, duration=2.0):
    """Rampe progressive 0 → max → 0 (en %)."""
    print(f"\n📈 Rampe {name}: 0% → {max_duty}% → 0%")

    steps = 50
    delay = duration / (steps * 2)

    for i in range(steps + 1):
        d = int(max_duty * i / steps)
        set_motor(fwd_pwm, bwd_pwm, d)
        print(f"  Duty: {d:3d}%", end="\r")
        time.sleep(delay)

    for i in range(steps, -1, -1):
        d = int(max_duty * i / steps)
        set_motor(fwd_pwm, bwd_pwm, d)
        print(f"  Duty: {d:3d}%", end="\r")
        time.sleep(delay)

    set_motor(fwd_pwm, bwd_pwm, 0)
    print(f"\n✅ Rampe {name} terminée")


def test_direction(name, fwd_pwm, bwd_pwm, duty=50, duration=1.5):
    """Test avant/arrière."""
    print(f"\n🔄 Direction {name}")

    print(f"  → AVANT ({duty}%)")
    set_motor(fwd_pwm, bwd_pwm, duty)
    time.sleep(duration)
    set_motor(fwd_pwm, bwd_pwm, 0)
    time.sleep(0.5)

    print(f"  ← ARRIÈRE ({duty}%)")
    set_motor(fwd_pwm, bwd_pwm, -duty)
    time.sleep(duration)
    set_motor(fwd_pwm, bwd_pwm, 0)
    print(f"✅ Direction {name} terminée")


def auto_test():
    """Test automatique complet."""
    print("\n" + "=" * 50)
    print("🤖 TEST AUTOMATIQUE COMPLET")
    print("=" * 50)
    print("⚠️  Robot SURÉLEVÉ ?")
    input("Appuyez sur ENTRÉE pour commencer...\n")

    try:
        print("-" * 40)
        print("🔵 MOTEUR GAUCHE")
        test_ramp("Gauche", pwm_l_fwd, pwm_l_bwd, max_duty=60)
        time.sleep(0.5)
        test_direction("Gauche", pwm_l_fwd, pwm_l_bwd, duty=50)
        time.sleep(1)

        print("\n" + "-" * 40)
        print("🟢 MOTEUR DROIT")
        test_ramp("Droit", pwm_r_fwd, pwm_r_bwd, max_duty=60)
        time.sleep(0.5)
        test_direction("Droit", pwm_r_fwd, pwm_r_bwd, duty=50)
        time.sleep(1)

        print("\n" + "-" * 40)
        print("🟡 LES DEUX MOTEURS")
        print("  → AVANT ensemble (50%)")
        set_both(50)
        time.sleep(1.5)
        stop_all()
        time.sleep(0.5)
        print("  ← ARRIÈRE ensemble (50%)")
        set_both(-50)
        time.sleep(1.5)
        stop_all()

        print("\n" + "=" * 50)
        print("✅ TEST AUTOMATIQUE TERMINÉ")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n⏹️ Test interrompu")
    finally:
        stop_all()


def interactive_mode():
    """Mode interactif clavier."""
    print("\n" + "=" * 50)
    print("🎮 MODE INTERACTIF")
    print("=" * 50)
    print("Commandes:")
    print("  w/s : Moteur gauche avant/arrière")
    print("  i/k : Moteur droit avant/arrière")
    print("  a   : Les deux avant")
    print("  z   : Les deux arrière")
    print("  +/- : Augmenter/diminuer duty")
    print("  0   : Stop")
    print("  q   : Quitter")
    print("=" * 50)

    duty = 50  # %

    try:
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            while True:
                char = sys.stdin.read(1)
                if char == 'q':
                    break
                elif char == 'w':
                    set_left(duty)
                    print(f"\rGauche: +{duty}%  ")
                elif char == 's':
                    set_left(-duty)
                    print(f"\rGauche: -{duty}%  ")
                elif char == 'i':
                    set_right(duty)
                    print(f"\rDroit: +{duty}%   ")
                elif char == 'k':
                    set_right(-duty)
                    print(f"\rDroit: -{duty}%   ")
                elif char == 'a':
                    set_both(duty)
                    print(f"\rDeux: +{duty}%    ")
                elif char == 'z':
                    set_both(-duty)
                    print(f"\rDeux: -{duty}%    ")
                elif char == '+' or char == '=':
                    duty = min(100, duty + 10)
                    print(f"\rDuty: {duty}%     ")
                elif char == '-':
                    duty = max(10, duty - 10)
                    print(f"\rDuty: {duty}%     ")
                elif char == '0':
                    stop_all()
                    print(f"\rSTOP            ")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except ImportError:
        print("Mode texte:")
        while True:
            cmd = input("Commande (l+/l-/r+/r-/stop/quit): ").strip().lower()
            if cmd == "quit":
                break
            elif cmd == "l+":
                set_left(duty)
            elif cmd == "l-":
                set_left(-duty)
            elif cmd == "r+":
                set_right(duty)
            elif cmd == "r-":
                set_right(-duty)
            elif cmd == "stop":
                stop_all()

    stop_all()
    print("\n👋 Mode interactif terminé")


def main():
    parser = argparse.ArgumentParser(description="Test moteurs BTS7960")
    parser.add_argument("--auto", action="store_true", help="Test automatique")
    parser.add_argument("--left", action="store_true", help="Test moteur gauche")
    parser.add_argument("--right", action="store_true", help="Test moteur droit")
    parser.add_argument("--duty", type=int, default=50, help="Duty cycle %% (0-100)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🔧 TEST DRIVERS BTS7960 & MOTEURS (RPi.GPIO)")
    print("=" * 60)

    try:
        init_gpio()

        if args.auto:
            auto_test()
        elif args.left:
            test_ramp("Gauche", pwm_l_fwd, pwm_l_bwd, max_duty=args.duty)
            test_direction("Gauche", pwm_l_fwd, pwm_l_bwd, duty=args.duty)
        elif args.right:
            test_ramp("Droit", pwm_r_fwd, pwm_r_bwd, max_duty=args.duty)
            test_direction("Droit", pwm_r_fwd, pwm_r_bwd, duty=args.duty)
        else:
            interactive_mode()
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
        print("👋 Fin du test")


if __name__ == "__main__":
    main()
