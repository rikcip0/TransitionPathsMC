TODO:

SpinGlass:
runna con q* oltre la soglia dello stato piu esterno (mi pare 0.75=45)

Graphs:
-implement power-law graphs generation and acquisition by simulation code

Parallel tempering:
-study what are characteristic quantities to check and implement relative code
-change filesystem data structure

PathsMCAnalysis/singleRunAnalysis:
-improve linear fits to take only linear parts and all the linear parts
-see autocorrelation of energy for thermodynamic integration

PathsMCAnalysis/singleMultiRunAnalysis:
-correct the plot of Z curves in terms of rescaled betas
-plot thermodynamic integration data curve by curve to better check used data
-do a check on the Chi square of quantities to be plotted



Possible improvements:

TransitionPath:
-Study possible relevant topological quantities relative to trajectories to sample



HARD CODED PARAMETERS (not in input):
START: MC PARAMETERS
MC (mc length),
MCeq (expect equilibration duration time, or larger),
MCmeas (how many mc sweeps for measuring),
MCprint (how often to print on story file (similar to MCeq seems a good idea)),
Np (number of time points to measure quantities)
END: MC PARAMETERS

START Qstar COMPUTATION PARAMETERS
mcEQ, mc, nStories
END Qstar COMPUTATION PARAMETERS

START: THERM. CHECK QUANTITIES
mcForThermCheck (how many mcsweeps to do measures to check thermalization)
averagingMCForTherm (on how many mcsweeps to average quantities to check therm)
nIntervalsForThermCheckQuantities (how many time intervals to average quantities depending on time to check therm)
resolutionForThermCheckQuantities (how many points i ntime are considered for each time interval)
END: THERM. CHECK QUANTITIES