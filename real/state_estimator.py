import math
import time

class ComplementaryFilter:
    def __init__(self, alpha=0.98):
        self.alpha = float(alpha)
        self.pitch = 0.0
        self.last_t = None

    def reset(self):
        self.pitch = 0.0
        self.last_t = None

    def update(self, ax, ay, az, gy, dt):
        # UNITS:
        # - gy doit etre en rad/s
        # - dt en secondes
        # - pitch est maintenu en radians
        # pitch_acc (selon ton montage, tu devras peut-être changer axes)
        pitch_acc = math.atan2(-ax, math.sqrt(ay*ay + az*az))
        # intégration gyro
        pitch_gyro = self.pitch + gy * dt
        # fusion
        self.pitch = self.alpha * pitch_gyro + (1 - self.alpha) * pitch_acc
        return self.pitch
