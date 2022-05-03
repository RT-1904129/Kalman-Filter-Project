# Importing the required libraries & modules.
import numpy as np

# Defining a class for the KalmanFilter.
class KalmanFilter(object):

    def __init__(self, dT=1, var_X=1, var_Y=1, method="Velocity"):

        super(KalmanFilter, self).__init__()

        # Storing the parameters in the attributes of the class.
        self.var_X = var_X
        self.var_Y = var_Y
        self.dT = dT

        # U represents the acceleration.
        self.U = 0
        if (method == "Acceleration"):
            self.U = 1

        #  X represents the State Matrix.
        self.X = np.matrix([[1], [0], [1], [0]])

        # Matrix A represents the State transitions.
        self.A = np.matrix([[1, self.dT, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, self.dT],  [0, 0, 0, 1]])

        #  P represents the Process Covariance Matrix.
        self.P = np.matrix(self.var_X * np.identity(self.A.shape[0]))
        self.P = self.P

        # Matrix B applies acceleration (U) to provide values to update the position and velocity of AX.
        self.B = np.matrix(
            [[self.dT/2], [self.dT], [self.dT/2], [self.dT]])

        # Matrix H helps in transforming the matrix format of P into the format desired for the K matrix.
        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])

        # R represents the Measurement Covariance Matrix, which is the error of measurement.
        self.R = np.matrix(self.var_Y *
                           np.identity(self.H.shape[0]))

        # Matrix Q represents the error terms.
        self.Q = np.matrix([[self.dT/4, self.dT/2, 0, 0],
                            [self.dT/2, self.dT, 0, 0],
                            [0, 0, self.dT/4, self.dT/2],
                            [0, 0, self.dT/2, self.dT]])

    # Function which predicts the next state based on previous state.
    def predict(self):
        # pred_X = A*X + B*U
        self.pred_X = self.A*self.X + self.B*self.U

        # pred_P = A*P*(A') + Q
        self.pred_P = self.A*self.P*self.A.T + self.Q

        predX = np.asarray(self.pred_X)
        return predX[0], predX[2]

    # Function which updates the states based on current measurement Y.
    def update(self, Y):

        # Computing the Kalman Gain (K) as : ( pred_P' * H' )/  ( H * (pred_P') * H'  +  R)
        self.K = self.pred_P * self.H.T * np.linalg.pinv(
            self.H * self.pred_P * self.H.T + self.R)

        # Updating the state (X) as : pred_X + K*(Y - H*pred_X)
        self.X = self.pred_X + self.K * (Y - (self.H*self.pred_X))

        # Updating the process covariance matrix P as : ( I - K*H )*pred_P
        self.P = (np.identity(self.P.shape[0]) - self.K*self.H)*self.pred_P
