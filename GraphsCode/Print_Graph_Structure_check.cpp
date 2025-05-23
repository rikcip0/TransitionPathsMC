#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <sys/stat.h>  // mkdir

#include "../MCtrajs/MCdyn_classi/Initialization/initializeHyperGraph.h"

namespace fs = std::filesystem;


// Funzione per stampare le informazioni del grafo nei file
void printGraphInfo(const vector<vector<vector<rInteraction>>> &graph) {
    int N = graph.size();

    // Crea la cartella infoGraph se non esiste
    struct stat info;
    if (stat("infoGraph", &info) != 0) {
        mkdir("infoGraph", 0777);
    }

    for (int i = 0; i < N; ++i) {
        // File per i vicini del nodo i
        std::ostringstream nomeFileBase;
        nomeFileBase << "infoGraph/node_" << i << "_vicini.txt";
        std::ofstream outBase(nomeFileBase.str());

        for (const auto &inter : graph[i][0]) {
            outBase << inter.J;
            for (int s : inter.interS) outBase << " " << s;
            outBase << "\n";
        }
        outBase.close();

        // File per ciascun vicino
        int numVicini = graph[i].size() - 1;
        for (int j = 0; j < numVicini; ++j) {
            std::ostringstream nomeFileVicino;
            nomeFileVicino << "infoGraph/node_" << i << "_vicino" << j << ".txt";
            std::ofstream outVicino(nomeFileVicino.str());

            for (const auto &inter : graph[i][j+1]) {
                outVicino << inter.J;
                for (int s : inter.interS) outVicino << " " << s;
                outVicino << "\n";
            }
            outVicino.close();
        }
    }
}

int main() {

    vector<vector<vector<rInteraction>>> Graph;
    int pmax = 3; // Valore di default, modifica se necessario
    int N = 10;
    double fracPosJ = 0.4;
    string folder = "../Data/Graphs/ERPMAX/C2_3C3_2/N10/structure796245/fPosJ0.40/graph6270/";


    if (!initializeHyperGraph(folder, Graph, pmax, N, fracPosJ))
    {
      cout << "Error in the graph initialization" << endl;
      return 1;
    }

    // Calcola i gradi (questa parte potrebbe dover essere adattata)
    vector<vector<int>> deg(N, vector<int>(pmax-1, 0));
    // Qui dovresti calcolare deg in base alla tua struttura dati
    // Per ora lo lascio vuoto, dovrai adattarlo alla tua implementazione

    // Stampa le informazioni del grafo
    printGraphInfo(Graph);

    cout << "Graph information successfully written to infoGraph folder." << endl;

    return 0;
}