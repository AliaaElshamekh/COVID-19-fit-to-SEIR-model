# SIR model example for python 2.7
# Marisa Eisenberg (marisae@umich.edu)
# Yu-Han Kao (kaoyh@umich.edu) -7-9-17

#### Import all the packages ####
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sir_ode
import sir_cost
import minifim
import proflike

from scipy.integrate import odeint as ode




#### Load Data ####
times = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
data = [270,444,444,549,549,729,761,1052,1058,1423,1423,1423,2714,2714,3554,3554,3554,3554,4586,4903,5806,7153,11177,13522,13522,16678,16678,19665,19665,19665,22112,24953,	24953,27100,27100,29631]
#data = [97, 271, 860, 1995, 4419, 6549, 6321, 4763, 2571, 1385, 615, 302, 159, 72, 34]

#shortened version for seeing how truncated data affects the estimation
#times = times[0:7]
#data = data[0:7]

#### Set initial parameter values and initial states ####
params = [0.4, 0.25, 80000.0]#make sure all the params and inition states are float
paramnames = ['beta', 'gamma', 'k']

params_new = [0.4, 0.25,0.071, 80000.0]#make sure all the params and inition states are float
paramnames_new = ['beta', 'epsilon','gamma', 'k']

ini = sir_ode.x0fcn(params,data)
print (ini)

ini_new = sir_ode.x0fcn_new(params_new,data)
print (ini_new)
#### Simulate and plot the model ####
res = ode(sir_ode.model, ini, times, args=(params,))
print (res)

res_new = ode(sir_ode.model_new, ini_new, times, args=(params_new,))
print (res_new)
#
# sim_measure = sir_ode.yfcn(res, params)
# print (sim_measure)
# plt.plot(times, sim_measure, 'b-', linewidth=3, label='Model simulation')
# plt.plot(times, data, 'k-o', linewidth=2, label='Data')
# plt.xlabel('Time')
# plt.ylabel('Individuals')
# plt.legend()
# plt.show()

sim_measure_new = sir_ode.yfcn_new(res_new, params_new)
# rem_measure_new =sir_ode.Rfcn_new(res_new, params_new)
# sus_measure_new =sir_ode.Sfcn_new(res_new, params_new)
# ex_measure_new =sir_ode.Efcn_new(res_new, params_new)
print (sim_measure_new)
plt.plot(times, sim_measure_new, 'b-', linewidth=3, label='Model simulation I')
plt.plot(times, data, 'k-o', linewidth=2, label='Data')
# plt.plot(times,rem_measure_new,'g-',linewidth=3,label='Model simulation R')
# plt.plot(times,sus_measure_new,'r-',linewidth=3,label='Model simulation S')
# plt.plot(times,ex_measure_new,'m-',linewidth=3,label='Model simulation E')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
plt.show()
#### Parameter estimation ####
optimizer = optimize.minimize(sir_cost.NLL, params, args=(data, times), method='Nelder-Mead')
paramests = np.abs(optimizer.x)
iniests = sir_ode.x0fcn(paramests, data)


#### Parameter estimation_new ####
optimizer_new = optimize.minimize(sir_cost.NLL_new, params_new, args=(data, times), method='Nelder-Mead')
paramests_new = np.abs(optimizer_new.x)
iniests_new = sir_ode.x0fcn_new(paramests_new, data)

#### Re-simulate and plot the model with the final parameter estimates ####
# xest = ode(sir_ode.model, iniests, times, args=(paramests,))
# est_measure = sir_ode.yfcn(xest, paramests)
# plt.plot(times, est_measure, 'b-', linewidth=3, label='Model simulation')
# plt.plot(times, data, 'k-o', linewidth=2, label='Data')
# plt.xlabel('Time')
# plt.ylabel('Individuals')
# plt.legend()
# plt.show()

plt.figure()
#### Re-simulate and plot the model with the final parameter estimates NEW ####
xest_new = ode(sir_ode.model_new, iniests_new, times, args=(paramests_new,))
est_measure_new = sir_ode.yfcn_new(xest_new, paramests_new)
# est_rem_measure_new =sir_ode.Rfcn_new(xest_new, paramests_new)
# est_sus_measure_new =sir_ode.Sfcn_new(xest_new, params_new)
# est_ex_measure_new =sir_ode.Efcn_new(xest_new, params_new)
plt.plot(times, est_measure_new, 'b-', linewidth=3, label='Model simulation_new I')
# plt.plot(times, est_rem_measure_new, 'g-', linewidth=3, label='Model simulation_new R')
# plt.plot(times,sus_measure_new,'r-',linewidth=3,label='Model simulation S')
# plt.plot(times,ex_measure_new,'m-',linewidth=3,label='Model simulation E')
plt.plot(times, data, 'k-o', linewidth=2, label='Data')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.legend()
plt.show()

x=0

#### Calculate the simplified Fisher Information Matrix (FIM) ####
FIM = minifim.minifisher(times, params, data, delta = 0.001)
print (np.linalg.matrix_rank(FIM)) #calculate rank of FIM
print (FIM)

#### Generate profile likelihoods and confidence bounds ####
threshold = stats.chi2.ppf(0.95,len(paramests))/2.0 + optimizer.fun
perrange = 0.25 #percent range for profile to run across
#
profiles={}
for i in range(len(paramests)):
 	profiles[paramnames[i]] = proflike.proflike(paramests, i, sir_cost.NLL, times, data, perrange=perrange)
 	plt.figure()
 	plt.scatter(paramests[i], optimizer.fun, marker='*',label='True value', color='k',s=150, facecolors='w', edgecolors='k')
 	plt.plot(profiles[paramnames[i]]['profparam'], profiles[paramnames[i]]['fcnvals'], 'k-', linewidth=2, label='Profile likelihood')
 	plt.axhline(y=threshold, ls='--',linewidth=1.0, label='Threshold', color='k')
 	plt.xlabel(paramnames[i])
 	plt.ylabel('Negative log likelihood')
 	plt.legend(scatterpoints = 1)
 	paramnames_fit = [ n for n in paramnames if n not in [paramnames[i]]]
 	paramests_fit = [v for v in paramests if v not in [paramests[i]]]
 	print (paramnames_fit)
 	print (paramests_fit)
#
 	#plot parameter relationships
# 	#for j in range(profiles[paramnames[i]]['fitparam'].shape[1]):
# 	#	plt.figure()
# 	#	plt.plot(profiles[paramnames[i]]['profparam'],profiles[paramnames[i]]['fitparam'][:,j],'k-', linewidth=2, label=paramnames_fit[j])
# 	#	plt.scatter(paramests[i], paramests_fit[j], marker='*',label='True value', color='k',s=150, facecolors='w', edgecolors='k')
# 	#	plt.xlabel(paramnames[i])
# 	#	plt.ylabel(paramnames_fit[j])
# 	#	plt.legend(scatterpoints = 1)
print (profiles)
plt.show()



#### Calculate the simplified Fisher Information Matrix (FIM) _new ####
FIM_new = minifim.minifisher_new(times, params_new, data, delta = 0.001)
print (np.linalg.matrix_rank(FIM_new)) #calculate rank of FIM_new
print (FIM_new)

#### Generate profile likelihoods and confidence bounds ####
threshold_new = stats.chi2.ppf(0.95,len(paramests_new))/2.0 + optimizer_new.fun
perrange_new = 0.25 #percent range for profile to run across

profiles_new={}
for i in range(len(paramests_new)):
	profiles_new[paramnames_new[i]] = proflike.proflike(paramests_new, i, sir_cost.NLL_new, times, data, perrange=perrange_new)
	plt.figure()
	plt.scatter(paramests_new[i], optimizer_new.fun, marker='*',label='True value', color='k',s=150, facecolors='w', edgecolors='k')
	plt.plot(profiles_new[paramnames_new[i]]['profparam'], profiles_new[paramnames_new[i]]['fcnvals'], 'k-', linewidth=2, label='Profile likelihood')
	plt.axhline(y=threshold_new, ls='--',linewidth=1.0, label='Threshold', color='k')
	plt.xlabel(paramnames_new[i])
	plt.ylabel('Negative log likelihood')
	plt.legend(scatterpoints = 1)
	paramnames_fit_new = [ n for n in paramnames_new if n not in [paramnames_new[i]]]
	paramests_fit_new = [v for v in paramests_new if v not in [paramests_new[i]]]
	print (paramnames_fit_new)
	print (paramests_fit_new)

	#plot parameter relationships
	#for j in range(profiles[paramnames[i]]['fitparam'].shape[1]):
	#	plt.figure()
	#	plt.plot(profiles[paramnames[i]]['profparam'],profiles[paramnames[i]]['fitparam'][:,j],'k-', linewidth=2, label=paramnames_fit[j])
	#	plt.scatter(paramests[i], paramests_fit[j], marker='*',label='True value', color='k',s=150, facecolors='w', edgecolors='k')
	#	plt.xlabel(paramnames[i])
	#	plt.ylabel(paramnames_fit[j])
	#	plt.legend(scatterpoints = 1)
print (profiles_new)
plt.show()

x=0