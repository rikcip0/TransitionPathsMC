TODO:
Vedi tutti i warning di compilazione
Compila tutto con O2
Simulations:
-RRG: rinforzi alle temperature non multiple di 0.05
-ER: rinforzi con grande tau, N=200. rifai N=160 g8797


-RFIM rinforzi con grande tau, N=180,200
-RRG SG: tutto da vedere, N=120 interessanti: 4905, 4911, 6699

PathsMCAnalysis/singleRunAnalysis:
-implement what is needed to do jackknife on TI


PathsMCAnalysis/singleMultiRunAnalysis:
-do jackknife on TI to correctly compute errors on k
-consider errors on k to fit over N to extract free energy barrier
-correct the plot of Z curves in terms of rescaled betas
-do a check on the Chi square of quantities to be plotted
-extract and use best k for each graph and thermoydnamic setting
-improve thermodynamic integration data curve by curve fucntion

Graphs:
-implement power-law graphs generation and acquisition by simulation code


Parallel tempering:
-study what are characteristic quantities to check and implement relative code
-change filesystem data structure

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