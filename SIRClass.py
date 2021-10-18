# Module for simulating infectious disease dynamics with the SIR (Susceptible-Infected-Recovered) model

import numpy as np, scipy.integrate, scipy.optimize, sys


class SIRsystem:

    def __init__(self, beta, gamma, S, I, R):
        """Initialize an instance of an SIRsystem, providing values for the model parameters beta and gamma,
        and the initial numbers of hosts in each of the model compartments (S = Susceptible, I = Infectious,
        R = Recovered).

        Initialize internal instance variables self.t (time), self.S, self.I, self.R, self.N=S+I+R and
        a numpy array self.trajectory to hold [S,I,R] numbers at every time step.

        This is a base class for subsequent specialization, depending on whether one wants to simulate a
        deterministic or stochastic model of SIR dynamics.
        """
        self.beta = beta
        self.gamma = gamma
        self.t = 0.
        self.S = S
        self.I = I
        self.R = R
        self.N = S + I + R
        self.trajectory = np.array([[self.S, self.I, self.R]], dtype=float) / self.N
        self.times = None

    def reset(self, S, I, R, t=0.):
        """Reset the system by setting S,I,R and t to specified values, and reinitializing a trajectory array with this
        starting configuration"""
        self.t = t
        self.S = S
        self.I = I
        self.R = R
        self.trajectory = np.array([[self.S, self.I, self.R]], dtype=float) / self.N

    def get_total_infected(self):
        """Return the total number of hosts either currently or previously infected"""
        return self.I + self.R


class DeterministicSIRsystem(SIRsystem):
    """Define a specialized subclass of the general SIRsystem for modeling SIR dynamics with a deterministic differential
    equation model"""

    def dydt(self, y, t):
        """Define the right-hand-side of the equation dy/dt = dydt(y,t) for use with odeint integrator;
        note that because this is defined as a method on the class rather than as a free-standing function, the first
        argument of the function must be the self instance rather than the current state vector y"""

        s, i, r = y
        dsdt = -self.beta * s * i
        didt = self.beta * s * i - self.gamma * i
        drdt = self.gamma * i
        return np.array([dsdt, didt, drdt])

    def run(self, T, dt=None):
        """Integrate the ODE for the deterministic model from time 0 to time T, starting with the initial values stored
        in the S,I,R state variables; story the result in self.trajectory"""

        y0 = np.array([self.S, self.I, self.R], dtype=float) / self.N
        if dt is None:
            self.times = np.linspace(0., T, int(T + 1), endpoint=True)
        else:
            self.times = np.arange(0., T, dt)
        self.trajectory = scipy.integrate.odeint(self.dydt, y0, self.times)
        # self.trajectory = np.concatenate((self.trajectory, traj), axis=0)


class StochasticSIRsystem(SIRsystem):
    """Define a specialized subclass of the general SIRsystem for modeling SIR dynamics as a stochastic, continuous
    time process, using the Gillespie method of continuous time Monte Carlo"""

    def step(self):
        """Implement one step of Gillespie's Direct Method based on current reaction rates: identify a reaction to fire
        next, as well as a time at which it will fire, and execute that reaction (similar to as was described in the
        StochasticCells exercise)."""

        transition = None
        inf_rate = (self.beta * self.S * self.I) / self.N
        rec_rate = self.gamma * self.I
        total_rate = inf_rate + rec_rate
        if total_rate == 0.:
            return transition, self.t
        ranno = np.random.random()
        if ranno < inf_rate / total_rate:
            self.S -= 1
            self.I += 1
            transition = 1
        else:
            self.I -= 1
            self.R += 1
            transition = 2
        dt = np.random.exponential(1. / total_rate, 1)[0]
        self.t += dt
        return transition, self.t

    def run(self, T=None, make_traj=True):
        """Run the Gillespie algorithm for stochastic simulation from time 0 to at least time T, starting with the initial
        values stored in the S,I,R state variables; story the result in self.trajectory if make_traj argument is
        set to True"""

        if T is None:
            T = sys.maxsize
        self.times = [0.]
        t0 = self.t
        transition = 1
        while self.t < t0 + T:
            transition, t = self.step()
            if not transition:
                return self.t
            if make_traj: self.trajectory = np.concatenate((self.trajectory, [[self.S, self.I, self.R]]), axis=0)
            self.times.append(self.t)
        return self.t


def SimulateDeterministicOutbreakSize(N, R0_range=np.arange(0., 5., 0.1)):
    """For a given population size N and an array of basic reproductive numbers R0, integrate a
    DeterministicSIRsystem model for long enough time to allow the outbreak to die out, and then record the
    final size of the outbreak as a function of R0"""

    gamma = 1.0
    beta = 0.0
    Nf = float(N)
    dsir = DeterministicSIRsystem(beta, gamma, (N - 1), 1, 0)
    sizes = []
    for R0 in R0_range:
        beta = R0 * gamma
        dsir.beta = beta
        dsir.reset((N - 1), 1, 0, 0.)
        dsir.run(100.)
        dsir.S, dsir.I, dsir.R = list(map(int, N * dsir.trajectory[-1]))
        R_inf = dsir.get_total_infected()
        sizes.append((R0, R_inf))
    return np.array(sizes)


def CalculateDeterministicOutbreakSize(N, R0_range=np.arange(0., 5., 0.1)):
    """For a given population size N and an array of basic reproductive numbers R0, solve an implicit equation
    for the final outbreak size using the fsolve root-finding routine.  Compare the simulated results found in
    SimulateDeterministicOutbreakSize with those solved for here"""

    func = lambda R_inf, R0: R_inf - (1. - np.exp(-R0 * R_inf))
    sizes = []
    for R0 in R0_range:
        R_inf = scipy.optimize.fsolve(func, 0.5, args=(R0,))[0]
        sizes.append((R0, R_inf))
    return np.array(sizes)


def StochasticOutbreakSizeDistribution(N, beta, gamma, Nruns):
    """For a given population size N and model parameters beta and gamma, execute Nruns dynamical runs of the
    StochasticSIRsystem to calculate the overall size of each outbreak (total number of hosts infected) after it has
    finally died out; store the outbreak sizes in an array, which is returned for further characterization."""

    S, I, R = N - 1, 1, 0
    ssir = StochasticSIRsystem(beta, gamma, S, I, R)
    sizes = []
    durations = []
    for n in range(Nruns):
        ssir.reset(S, I, R)
        t_end = ssir.run(make_traj=False)
        outbreak_size = ssir.get_total_infected()
        sizes.append(outbreak_size)
        durations.append(t_end)
    return np.array(sizes), np.array(durations)


def FractionLargeOutbreaks(osd, N):
    """For a given array of outbreak sizes, with total population size N, calculate the fraction of those outbreaks
    that are 'large'.  Strictly speaking, we are intended in outbreaks whose sizes constitutes a finite fraction of the
    population in the limit of infinite population size, but for a finite system size N, we can implement a heuristic
    cutoff separate 'large' from 'small' outbreaks; determining such a cutoff can be assisted by examining the
    outbreak size distribution."""

    Nthresh = 0.1 * N
    return (1. * np.sum(osd < Nthresh)) / len(osd)


def yesno():
    response = input('    Continue? (y/n) ')
    if len(response) == 0:  # [CR] returns true
        return True
    elif response[0] == 'n' or response[0] == 'N':
        return False
    else:  # Default
        return True


def demo():
    import pylab
    N = 1000
    print("SIR demo")
    print("Deterministic SIR dynamics")
    pylab.figure(1)
    pylab.clf()
    dsir = DeterministicSIRsystem(1.5, 1.0, N - 1, 1, 0)
    dsir.run(30, 0.1)
    pylab.plot(dsir.times, dsir.trajectory[:, 0], 'b-', label='S')
    pylab.plot(dsir.times, dsir.trajectory[:, 1], 'r-', label='I')
    pylab.plot(dsir.times, dsir.trajectory[:, 2], 'g-', label='R')
    pylab.legend(loc='upper right')
    if not yesno(): return
    print("Deterministic outbreak size")
    R0_range = np.arange(0., 5., 0.1)
    simulated_sizes = SimulateDeterministicOutbreakSize(N=N, R0_range=R0_range)
    theoretical_sizes = CalculateDeterministicOutbreakSize(N=N, R0_range=R0_range)
    pylab.figure(2)
    pylab.clf()
    pylab.plot(R0_range, N * theoretical_sizes[:, 1], 'b-', label='theory')
    pylab.plot(R0_range, simulated_sizes[:, 1], 'bo', label='simulations')
    pylab.xlabel('R0')
    pylab.ylabel('total outbreak size')
    if not yesno(): return
    print("Stochastic SIR dynamics")
    pylab.figure(3)
    pylab.clf()
    pylab.plot(dsir.times, N * dsir.trajectory[:, 1], 'r-', linewidth=2)
    for n in range(20):
        ssir = StochasticSIRsystem(1.5, 1.0, N - 1, 1, 0)
        tfinal = ssir.run(20)
        pylab.plot(ssir.times, ssir.trajectory[:, 1], 'b-')
    if not yesno(): return

# Copyright (C) Cornell University
# All rights reserved.
# Apache License, Version 2.0

demo()
x=90