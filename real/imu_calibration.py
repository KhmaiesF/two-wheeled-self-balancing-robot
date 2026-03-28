import time
import numpy as np
from imu_reader import IMUReader

def calibrate_gyro(samples=2000, dt=0.001):
    imu = IMUReader()
    buf = []
    print("Pose le robot IMMOBILE. Calibration gyro en cours...")
    time.sleep(1.0)
    for _ in range(samples):
        ax, ay, az, gx, gy, gz = imu.read_acc_gyro()
        buf.append([gx, gy, gz])
        time.sleep(dt)
    mean = np.mean(np.array(buf), axis=0)
    print("Gyro offsets:", mean.tolist())
    return mean

if __name__ == "__main__":
    calibrate_gyro()
