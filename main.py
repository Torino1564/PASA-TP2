import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt

import SystemIdentification

if __name__ == '__main__':
    from SystemIdentification import *
    import numpy as np
    import scipy.signal as sp
    import matplotlib.pyplot as plt

    fs = 330
    time = 10
    t = np.linspace(0, time, time * fs)
    x = np.sin(2 * np.pi * 2 * t)
    y = x + 0.05 * np.random.default_rng().normal(0, 1, time * fs)

    plt.figure()
    plt.plot(t, y, label="y")
    plt.plot(t, x, label="x")
    plt.legend()
    plt.show()

    id = SystemIdentification(x, y)

    w = np.zeros(int((time/5)*fs)) + 1
    id.SetFilterLenght(len(w))
    id.SetW(w)
    id.MSE()