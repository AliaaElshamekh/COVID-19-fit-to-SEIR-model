import scipy.integrate
import numpy
import matplotlib.pyplot as plt


def modelo_sir(y, t, beta, gamma):
 S, I, R = y

 dS_dt = -beta*S*I
 dI_dt = beta*S*I - gamma*I
 dR_dt = gamma*I

 return([dS_dt, dI_dt,])

# Condiciones iniciales

S0 = 0.9
I0 = 0.1
R0 = 0.0
beta = 0.3
gamma = 0.1


# Vector  tiempo
t = numpy.linspace(0, 100, 1000)

# Resultado
solucion = scipy.integrate.odeint(modelo_sir, [S0, I0, R0], t, args=(beta, gamma))
solucion = numpy.array(solucion)

# Resultado en grafica
plt.figure(figsize=[6, 4])
plt.plot(t, solucion[:, 0], label="S(t)")
plt.plot(t, solucion[:, 1], label="I(t)")
plt.plot(t, solucion[:, 2], label="R(t)")
plt.grid()
plt.legend()
plt.xlabel("Tiempo")
plt.ylabel("Proporciones")
plt.title("Modelo SIR (Epidemia)")
plt.show()

x=0