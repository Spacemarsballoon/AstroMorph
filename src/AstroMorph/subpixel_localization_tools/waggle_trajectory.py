import numpy as np

class KalmanDriftTracker:
    def __init__(self, dt=1.0, process_var=1e-3, meas_var=1e-1):
        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))  # initial state
        self.P = np.eye(4)         # initial covariance

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Measurement matrix (we observe x, y only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Process and measurement noise
        self.Q = process_var * np.eye(4)
        self.R = meas_var * np.eye(2)

        self.I = np.eye(4)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.reshape(z, (2, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x[:2].flatten()