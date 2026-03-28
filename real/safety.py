import time

class Safety:
    def __init__(self, max_pitch_rad: float = 0.6):
        self.max_pitch = float(max_pitch_rad)
        self.last_ok_time = time.time()

    def check(self, pitch: float) -> bool:
        if abs(float(pitch)) > self.max_pitch:
            return False
        self.last_ok_time = time.time()
        return True