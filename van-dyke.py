import numpy as np
import matplotlib.pyplot as plt

alpha = 1
gamma = 1
eps = 0.05

def y_i(t):
    return 1 - np.exp(-t/eps) + eps * (-1/2 + 1/2 * np.exp(-t/eps) + t/eps * (1 + 1/2 * t/eps) * np.exp(-t/eps))
    #return alpha + eps * (gamma/3 - gamma/3 * np.exp(-2 * t / eps))

def y_o(t):
    return 1 - eps / 2 + t - t
    #return alpha/4 * (t - 2)**2 * np.exp(t) + eps * ( alpha / 4 * np.exp(t) * (-1 + 4*(2-t) + (2-t)**2 * np.log(2-t)) + (gamma/12 - alpha/16 * (7 + 4 * np.log(2))*np.exp(t)*(2-t)**2) )
    
ts = np.linspace(0,1,100)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(ts, y_i(ts), label="inner solution")
ax.plot(ts, y_o(ts), label="outer solution")
plt.legend()
plt.show()
