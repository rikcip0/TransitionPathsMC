TODO:
sistema i 2 fit
multiRunAnalysis




Input parameters:
N p C T beta h_in h_out fPosJ
optional:
graphID, Qstar

Hidden parameters:
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

things to do:
Fare routine per controllare termalizzazione.
Implementare l'analisi dati.
