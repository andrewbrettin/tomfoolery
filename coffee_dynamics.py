import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

SAVE_DIR = '/Users/Andrew/PycharmProjects/resilience/output/'

# Parameters

T_0 = 20
V_0 = 100
tau = 3

def k(V):


def f(T, t):
    return (
        -k(V)*(T - T_0),
    )

def flow(f, x_0, t_0, T, dt = 0.1):
    """ Returns the flow at points in ts, xs.
        * ts is a vector indicating the times at which the flow is evaluated.
        * xs is a two-dimensional array of shape (len(ts), 5).
        Each column represents a different state variable.
        Each row respresents the state variables at different times.
        dt is the time span between sample points ts[i-1], ts[i]
        T is the timespan of the simulation.
        """
    ts = np.arange(t_0, t_0 + T, dt)
    xs = odeint(f, x_0, ts)
    return (ts, xs)

def kick(X):
    """ Returns a vector representing the kick applied to a current state X."""
    kick_vector = np.array([
        -kappa_0/np.sqrt(2),
        kappa_0/np.sqrt(2)
    ])
    return kick_vector

def flow_and_kick(f, x_0, t_0, tau, T, dt=0.01):
    """ Returns t,x pair of the flow-kick simulation.
        tau is the time span between flow kicks
        T is the total simulation timespan. """
    ts, xs = flow(f, x_0, t_0, tau)
    num_kicks = int(T / tau)
    for i in range(num_kicks):
        new_x_0 = xs[-1,:] + kick(xs[-1,:])
        #new_x_0 = xs[-1,:] + alternatingkick(xs[-1,:], index)
        #index += 1
        new_t_0 = ts[-1] + dt
        ts_next, xs_next = flow(f, new_x_0, new_t_0, tau)
        ts = np.append(ts, ts_next)
        xs = np.append(xs, xs_next, axis=0)
    # Flow for the remaining timespan.
    new_tau = T - tau * num_kicks
    ts_next, xs_next = flow(f, xs[-1,:], ts[-1], new_tau)
    ts = np.append(ts, ts_next)
    xs = np.append(xs, xs_next, axis=0)
    return (ts, xs)


# Plotting

def all_time_plot(ts, xs, output_dir=SAVE_DIR, filename = 'time-series', file_type='.png'):
    """ Plots time series plots for each state variable. """
    plt.clf()
    dim = np.shape(xs)[1] # Dimension of state vector
    fig = plt.figure();
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]);
    ax.set_ylim(0.0, 5.0)
    #ax.vlines(60, 0, 5, colors=[0.9, 0.9, 0.9])
    colorvec = ['r', 'g']
    array = ("exotic plants", "native plants")
    for i in range(dim):
        ax.plot(ts, xs[:, i], color=colorvec[i], label=array[i]);
        ax.set_xlabel("time")
        ax.set_ylabel("value")
    ax.legend(array, loc='upper left')
    #fig.savefig(output_dir + filename + file_type)
    fig.show()

# Begin simulations

# x_0 = (.2, .5, .1, 0, 2.25)
# x_0 = flow(f, x_0, 0, 30)[1][-1,:] # (Stupid way of estimating the equilibrium values)
x_0 = (1, 0)
t_0 = 0
T = 200
#ts, xs = flow(f, x_0, t_0, T)
#ts, xs = flow_and_kick(f, x_0, t_0, tau, T)
ts, xs = flow_and_kick(f, x_0, t_0, tau, T)
all_time_plot(ts, xs)
#

########################################################################################################################
