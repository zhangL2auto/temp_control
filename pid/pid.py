import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.pre_error = 0
        self.integral = 0
        
    def update(self, error, dt):
        self.integral += error
        derivation = error - self.pre_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivation
        self.pre_error = error
        return output
        