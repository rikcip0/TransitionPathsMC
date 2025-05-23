# target: dependencies... se target è stato fatto dopo l'ultima modifica di dependencies non lo rifa.
.PHONY: help

help:
	@echo List of possible targets:
	@echo Non l ho fatto

compileField:
	@cd GraphsCode && g++ makeField.cpp -o field.exe
	@echo Program compiled in field
	@echo .

compileGraph:
	@cd GraphsCode && g++ makeGraph.cpp -o graph.exe
	@echo Program compiled in graph
	@echo .

compileStructure:
	@cd GraphsCode && g++ makeStructure.cpp -o structure.exe
	@echo Program compiled in structure
	@echo .

compilePT:
	@cd GraphsCode && g++ PT.cpp -o PT.exe
	@echo Program compiled in PT
	@echo .

compileStdInAndOut:
	@cd MCMCs && g++ inAndOutRRG.cpp -o stdInAndOut.exe -DENTERING
	@echo Program compiled in stdInAndOut.exe 
	@echo .

compileSimpleStdMC:
	@cd MCMCs && g++ -O2 simpleStdMcForTp.cpp -o stMC.exe -DFIXEDEXT
	@echo Program compiled in stMC
	@echo .

compileMC2:
	@cd MCMCs && g++ MCMC.cpp -o MCWithEnAndQ.exe -DFIXEDEXT
	@echo Program compiled in MCWithEnAndQ
	@echo .

compileMCDynWithQout:
	@cd MCMCs && \
	g++ -O3 manyMCMCsWithQout.cpp -o mcDyn_WQout2.exe -DSTARTFIXEDANDHARDEND 
	@echo "Programma compilato mcDyn_WQout.exe"


compileAllSims:
	cd MCtrajs && \
	g++ -O2 transitionPaths.cpp -o tp.exe && \
	g++ -O2 transitionPaths.cpp -o tp_QuenC.exe -DQUENCHCONFS && \
	g++ -O2 transitionPaths.cpp -o tp_TrajRan.exe -DINITRANDOM && \
	g++ -O2 transitionPaths.cpp -o tp_Anneal.exe -DINITANNEALING && \
	g++ -O2 transitionPaths.cpp -o tp_FixExt.exe -DFIXEDEXT && \
	g++ -O2 transitionPaths.cpp -o tp_TrajRan_FixExt.exe -DINITRANDOM -DFIXEDEXT && \
	g++ -O2 transitionPaths.cpp -o tp_Anneal_FixExt.exe -DINITANNEALING -DFIXEDEXT && \
	g++ -O2 transitionPaths.cpp -o tp_QuenC_TrajRan.exe -DQUENCHCONFS -DINITRANDOM && \
	g++ -O2 transitionPaths.cpp -o tp_QuenC_Anneal.exe -DQUENCHCONFS -DINITANNEALING && \
	g++ -O2 transitionPaths.cpp -o tp_QuenC_FixExt.exe -DQUENCHCONFS -DFIXEDEXT && \
	g++ -O2 transitionPaths.cpp -o tp_QuenC_TrajRan_FixExt.exe -DQUENCHCONFS -DINITRANDOM -DFIXEDEXT && \
	g++ -O2 transitionPaths.cpp -o tp_QuenC_Anneal_FixExt.exe -DQUENCHCONFS -DINITANNEALING -DFIXEDEXT
	@echo All programs compiled.

compileAllSims_Idra:
	cd MCtrajs && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp.exe && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_QuenC.exe -DQUENCHCONFS && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_TrajRan.exe -DINITRANDOM && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_Anneal.exe -DINITANNEALING && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_FixExt.exe -DFIXEDEXT && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_TrajRan_FixExt.exe -DINITRANDOM -DFIXEDEXT && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_Anneal_FixExt.exe -DINITANNEALING -DFIXEDEXT && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_QuenC_TrajRan.exe -DQUENCHCONFS -DINITRANDOM && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_QuenC_Anneal.exe -DQUENCHCONFS -DINITANNEALING && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_QuenC_FixExt.exe -DQUENCHCONFS -DFIXEDEXT && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_QuenC_TrajRan_FixExt.exe -DQUENCHCONFS -DINITRANDOM -DFIXEDEXT && \
	g++ -O2 -std=c++11 transitionPaths.cpp -o tp_QuenC_Anneal_FixExt.exe -DQUENCHCONFS -DINITANNEALING -DFIXEDEXT
	@echo All programs compiled.



cleanAll: 
	directory="../"
	string_to_match="tp"
	find "$directory" -type f -name "*.exe" -exec grep -q "$string_to_match" {} \; -exec rm -f {} +
	echo "Deletion complete."
