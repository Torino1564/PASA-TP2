import numpy as np
import scipy as sp
import pandas as pd


class SystemIdentification:
    def __init__(self, input_signal, desired_response):
        self.w = None
        self.w_prev = None
        self.mu = 0.8
        self.filter_length = 10
        self.input_signal = input_signal
        self.desired_response = desired_response
        self.sample_offset = 0


    def SetFilterLenght(self, filter_length):
        self.filter_length = filter_length


    def SetMu(self, mu):
        self.mu = mu

    def SetW(self, w_values):
        self.w_prev = self.w
        self.w = np.array(w_values)


    def MSE(self):
        x = np.array(self.input_signal[self.sample_offset:+self.filter_length])
        y = np.array(self.desired_response[self.sample_offset:+self.filter_length])
        y_var = np.var(y)

        p = sp.signal.correlate(x, np.conj(y), mode='full')[len(x):]
        if len(p) < len(x):
            p = np.append(p, 0)
        elif len(x) < len(p):
            p = p[:len(p)-1]

        r = sp.signal.correlate(x, x, mode='full')
        midpoint = len(r) // 2
        r = r[midpoint:midpoint+len(x)]

        R = sp.linalg.toeplitz(r[:len(x)])

        MSE = y_var - self.w.T.dot(p) - p.T.dot(self.w) + self.w.T.dot((R.dot(self.w)))

        return MSE