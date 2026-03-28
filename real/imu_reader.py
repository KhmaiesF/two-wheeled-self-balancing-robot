"""
Lecture IMU MPU6050 via I2C (smbus2).
Retourne accélérations (g) et vitesses angulaires (rad/s).
"""

import struct
import math
import time

try:
    import smbus2
    HAS_SMBUS = True
except ImportError:
    HAS_SMBUS = False
    print("[WARN] smbus2 non installé: pip install smbus2")

from config import I2C_BUS, MPU_ADDR


# Registres MPU6050
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
CONFIG_REG = 0x1A

# Sensibilités
ACCEL_SCALE = 16384.0   # ±2g  → LSB/g
GYRO_SCALE = 131.0      # ±250°/s → LSB/(°/s)
DEG_TO_RAD = math.pi / 180.0


class IMUReader:
    """Lecteur MPU6050 via I2C."""

    def __init__(self, bus: int = I2C_BUS, addr: int = MPU_ADDR):
        self.addr = addr
        self.gyro_offset = [0.0, 0.0, 0.0]

        if not HAS_SMBUS:
            raise RuntimeError("smbus2 requis: pip install smbus2")

        self.bus = smbus2.SMBus(bus)

        # Réveiller le MPU6050
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0x00)

        # Config accéléromètre ±2g
        self.bus.write_byte_data(self.addr, ACCEL_CONFIG, 0x00)

        # Config gyroscope ±250°/s
        self.bus.write_byte_data(self.addr, GYRO_CONFIG, 0x00)

        # Filtre passe-bas interne (DLPF ~44Hz, bon compromis bruit/latence)
        self.bus.write_byte_data(self.addr, CONFIG_REG, 0x03)

        print(f"[OK] MPU6050 initialisé (bus={bus}, addr=0x{addr:02x})")

    def _read_raw(self, reg: int) -> int:
        """Lit un mot 16 bits signé."""
        high = self.bus.read_byte_data(self.addr, reg)
        low = self.bus.read_byte_data(self.addr, reg + 1)
        val = (high << 8) | low
        if val >= 0x8000:
            val -= 0x10000
        return val

    def _reinit(self):
        """Réinitialise le MPU6050 après une erreur I2C."""
        try:
            self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0x00)
            time.sleep(0.01)
            self.bus.write_byte_data(self.addr, ACCEL_CONFIG, 0x00)
            self.bus.write_byte_data(self.addr, GYRO_CONFIG, 0x00)
            self.bus.write_byte_data(self.addr, CONFIG_REG, 0x03)
        except OSError:
            pass

    def read_acc_gyro(self, retries: int = 3):
        """
        Lit accéléromètre et gyroscope avec retry I2C.

        Returns:
            (ax, ay, az, gx, gy, gz) en unités SI:
            - ax, ay, az en g (1g ≈ 9.81 m/s²)
            - gx, gy, gz en rad/s
        """
        for attempt in range(retries):
            try:
                return self._read_acc_gyro_raw()
            except OSError:
                if attempt < retries - 1:
                    time.sleep(0.002)
                    self._reinit()
                else:
                    raise

    def _read_acc_gyro_raw(self):
        """Lecture brute sans retry."""
        # Lecture bloc 14 octets (accel + temp + gyro)
        data = self.bus.read_i2c_block_data(self.addr, ACCEL_XOUT_H, 14)

        # Accéléromètre (indices 0-5)
        ax = struct.unpack('>h', bytes(data[0:2]))[0] / ACCEL_SCALE
        ay = struct.unpack('>h', bytes(data[2:4]))[0] / ACCEL_SCALE
        az = struct.unpack('>h', bytes(data[4:6]))[0] / ACCEL_SCALE

        # Gyroscope (indices 8-13, skip température 6-7)
        gx = struct.unpack('>h', bytes(data[8:10]))[0] / GYRO_SCALE * DEG_TO_RAD
        gy = struct.unpack('>h', bytes(data[10:12]))[0] / GYRO_SCALE * DEG_TO_RAD
        gz = struct.unpack('>h', bytes(data[12:14]))[0] / GYRO_SCALE * DEG_TO_RAD

        # Soustraire offset gyro
        gx -= self.gyro_offset[0]
        gy -= self.gyro_offset[1]
        gz -= self.gyro_offset[2]

        return ax, ay, az, gx, gy, gz

    def calibrate_gyro(self, samples: int = 2000, dt: float = 0.001):
        """Calibre les offsets du gyroscope (robot immobile)."""
        print("Calibration gyro... Ne pas bouger le robot!")
        time.sleep(0.5)

        sums = [0.0, 0.0, 0.0]
        count = 0
        for _ in range(samples):
            try:
                _, _, _, gx, gy, gz = self.read_acc_gyro()
                sums[0] += gx
                sums[1] += gy
                sums[2] += gz
                count += 1
            except OSError:
                pass  # Skip failed reads
            time.sleep(dt)

        if count > 0:
            self.gyro_offset = [s / count for s in sums]
        print(f"[OK] Gyro offsets: {[f'{o:.4f}' for o in self.gyro_offset]} rad/s ({count}/{samples} reads)")
        return self.gyro_offset
