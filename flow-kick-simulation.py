import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

SAVE_DIR = '/Users/Andrew/PycharmProjects/resilience/output/'

# Parameters

a_n = 1
a_e = 6
N_n = 1
N_e = 2
b_nn = 1
b_en = 2
b_ne = .5
b_ee = .5
m_n = .1
m_e = .1
k_n = .1
k_e = .05
sigma = 1
gamma_n = 1
gamma_e = 1
c = 2.25
kappa_0 = 0.5
delta = 0.1
seeding = .001

COUNT = 0 # Index for handling alternating kicks

def f(x, t):
    """ Function f for dx/dt = f(x,t) in the five-dimensional model.
    x[0]: Native plant mass
    x[1]: Exotic plant mass
    x[2]: Native plant litter
    x[3]: Exotic plant litter
    x[4]: Soil nitrogen content"""
    f_1 = (a_n * (x[4] - N_n) - b_nn * x[2] - b_en * x[3]) * x[0]
    f_2 = (a_e * (x[4] - N_e) - b_ee * x[3] - b_ne * x[2]) * x[1]

    return (
        f_1 - m_n * x[0] + seeding,
        f_2 - m_e * x[1],
        m_n * x[0] - k_n * x[2],
        m_e * x[1] - k_e * x[3],
        c - sigma * x[4] + gamma_n * (-f_1 + k_n * x[2]) + gamma_e * (-f_2 + k_e * x[3])
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
    const = -kappa_0
    kick_vector = np.array([
        const * delta * X[0],
        const * delta * X[1],
        const * X[2],
        const * X[3],
        0
    ])
    return kick_vector

def alternatingkick(X, index, delta_vector=[0.1, 0.4]):
    """
    Returns a kick vector representing the kick applied to the state X, but cycling through
    different magnitudes kappa_0 to represent haying at different times of the year.
    :param X: state vector
    :param delta_vector: vector representing the proportion of living plant biomass removed during haying
    :param index: cycling parameter to determine which delta parameter is used. This parameter should increment after
    each method call.
    :return: kick vector.
    """
    periodicity = len(delta_vector)
    modulus = index % periodicity
    const = -kappa_0
    kick_vector = np.array([
        const * delta_vector[modulus] * X[0],
        const * delta_vector[modulus] * X[1],
        const * X[2],
        const * X[3],
        0
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

def flow_and_alternatingkick(f, x_0, t_0, tau, T, index, dt=0.01):
    """ Returns t,x pair of the flow-kick simulation.
        tau is the time span between flow kicks
        T is the total simulation timespan. """
    ts, xs = flow(f, x_0, t_0, tau)
    num_kicks = int(T / tau)
    for i in range(num_kicks):
        new_x_0 = xs[-1,:] + alternatingkick(xs[-1,:], index)
        index += 1
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

def flowkick_lazy(f, x_0, t_0, tau, T, t_lazy, late_tau, dt=0.01):
    """ Returns t,x pair of the flow-kick simulation.
        tau is the time span between flow kicks
        T is the total simulation timespan. """
    ts, xs = flow(f, x_0, t_0, tau)
    num_kicks = int(T / tau)
    for i in range(num_kicks):
        new_x_0 = xs[-1,:] + kick(xs[-1,:])
        new_t_0 = ts[-1] + dt
        interkicktime = tau if ts[-1] < t_lazy else late_tau
        ts_next, xs_next = flow(f, new_x_0, new_t_0, interkicktime)
        ts = np.append(ts, ts_next)
        xs = np.append(xs, xs_next, axis=0)
    # Flow for the remaining timespan.
    new_tau = T - interkicktime * num_kicks
    ts_next, xs_next = flow(f, xs[-1,:], ts[-1], new_tau)
    ts = np.append(ts, ts_next)
    xs = np.append(xs, xs_next, axis=0)

    return (ts, xs)

# Plotting things

def time_plot(ts, xs, variable = 0, directory = None):
    """
        variable is the coordinate number of the variable of interest (default to 0)"""
    plt.clf()
    fig = plt.figure();
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]);
    ax.plot(ts, xs[:,variable],label=r"$P_n$");
    plt.legend()
    plt.show()

def all_time_plot(ts, xs, output_dir=SAVE_DIR, filename = 'time-series', file_type='.png'):
    """ Plots time series plots for each state variable. """
    plt.clf()
    dim = np.shape(xs)[1] # Dimension of state vector
    fig = plt.figure();
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]);
    ax.set_ylim(0.0, 5.0)
    #ax.vlines(60, 0, 5, colors=[0.9, 0.9, 0.9])
    colorvec = ['g', 'r', 'b', 'm', 'k']
    array = ("native plants", "exotic plants", "native litter", "exotic litter", "nitrogen")
    for i in range(dim):
        ax.plot(ts, xs[:, i], color=colorvec[i], label=array[i]);
        ax.set_xlabel("time")
        ax.set_ylabel("value")
    ax.legend(array, loc='upper left')
    fig.savefig(output_dir + filename + file_type)

def orbit(xs, var_1, var_2, output_dir=SAVE_DIR, filename = 'orbit', file_type='.png'):
    """ given the values of xs, plots two of the variables
        against each other (phase portrait)
        var_1: coordinate of x_axis variable
        var_2: coordinate of y_axis variable """
    plt.clf()
    fig = plt.figure();
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]);
    ax.plot(xs[0, var_1], xs[0, var_2], marker='*');
    ax.plot(xs[-1, var_1], xs[-1, var_2], marker='s');
    ax.plot(xs[:, var_1], xs[:, var_2]);
    array = ["native plants", "exotic plants", "native litter", "exotic litter", "nitrogen"]
    ax.set_xlabel(array[var_1])
    ax.set_ylabel(array[var_2])
    fig.savefig(output_dir + filename + file_type)

# Begin simulations

# x_0 = (.2, .5, .1, 0, 2.25)
# x_0 = flow(f, x_0, 0, 30)[1][-1,:] # (Stupid way of estimating the equilibrium values)
x_0 = (0.01, 1.79243, .200902, 2.18583, 2.21390)
t_0 = 0
tau = 1
T_lazy = 60
T = 60
#ts, xs = flow(f, x_0, t_0, T)
#ts, xs = flow_and_kick(f, x_0, t_0, tau, T)
ts, xs = flow_and_kick(f, x_0, t_0, tau, T)
all_time_plot(ts, xs)
#
# phase_plot(xs,0,2)

########################################################################################################################
