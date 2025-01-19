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
	@cd MCMCs && g++ simpleStdMcForTp.cpp -o stMC.exe -DFIXEDEXT
	@echo Program compiled in stMC
	@echo .

compileMC2:
	@cd standardMCs && g++ MCMC.cpp -o MCWithEnAndQ.exe -DFIXEDEXT
	@echo Program compiled in MCWithEnAndQ
	@echo .

compileAllSims:
	cd MCtrajs && \
	g++ transitionPaths.cpp -o tp.exe && \
	g++ transitionPaths.cpp -o tp_QuenC.exe -DQUENCHCONFS && \
	g++ transitionPaths.cpp -o tp_TrajRan.exe -DINITRANDOM && \
	g++ transitionPaths.cpp -o tp_Anneal.exe -DINITANNEALING && \
	g++ transitionPaths.cpp -o tp_FixExt.exe -DFIXEDEXT && \
	g++ transitionPaths.cpp -o tp_TrajRan_FixExt.exe -DINITRANDOM -DFIXEDEXT && \
	g++ transitionPaths.cpp -o tp_Anneal_FixExt.exe -DINITANNEALING -DFIXEDEXT && \
	g++ transitionPaths.cpp -o tp_QuenC_TrajRan.exe -DQUENCHCONFS -DINITRANDOM && \
	g++ transitionPaths.cpp -o tp_QuenC_Anneal.exe -DQUENCHCONFS -DINITANNEALING && \
	g++ transitionPaths.cpp -o tp_QuenC_FixExt.exe -DQUENCHCONFS -DFIXEDEXT && \
	g++ transitionPaths.cpp -o tp_QuenC_TrajRan_FixExt.exe -DQUENCHCONFS -DINITRANDOM -DFIXEDEXT && \
	g++ transitionPaths.cpp -o tp_QuenC_Anneal_FixExt.exe -DQUENCHCONFS -DINITANNEALING -DFIXEDEXT
	@echo All programs compiled.

compileAllSims_Idra:
	cd MCtrajs && \
	g++ -std=c++11 transitionPaths.cpp -o tp.exe && \
	g++ -std=c++11 transitionPaths.cpp -o tp_QuenC.exe -DQUENCHCONFS && \
	g++ -std=c++11 transitionPaths.cpp -o tp_TrajRan.exe -DINITRANDOM && \
	g++ -std=c++11 transitionPaths.cpp -o tp_Anneal.exe -DINITANNEALING && \
	g++ -std=c++11 transitionPaths.cpp -o tp_FixExt.exe -DFIXEDEXT && \
	g++ -std=c++11 transitionPaths.cpp -o tp_TrajRan_FixExt.exe -DINITRANDOM -DFIXEDEXT && \
	g++ -std=c++11 transitionPaths.cpp -o tp_Anneal_FixExt.exe -DINITANNEALING -DFIXEDEXT && \
	g++ -std=c++11 transitionPaths.cpp -o tp_QuenC_TrajRan.exe -DQUENCHCONFS -DINITRANDOM && \
	g++ -std=c++11 transitionPaths.cpp -o tp_QuenC_Anneal.exe -DQUENCHCONFS -DINITANNEALING && \
	g++ -std=c++11 transitionPaths.cpp -o tp_QuenC_FixExt.exe -DQUENCHCONFS -DFIXEDEXT && \
	g++ -std=c++11 transitionPaths.cpp -o tp_QuenC_TrajRan_FixExt.exe -DQUENCHCONFS -DINITRANDOM -DFIXEDEXT && \
	g++ -std=c++11 transitionPaths.cpp -o tp_QuenC_Anneal_FixExt.exe -DQUENCHCONFS -DINITANNEALING -DFIXEDEXT
	@echo All programs compiled.



cleanAll: 
	directory="../"
	string_to_match="tp"
	find "$directory" -type f -name "*.exe" -exec grep -q "$string_to_match" {} \; -exec rm -f {} +
	echo "Deletion complete."
