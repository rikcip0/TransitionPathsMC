/* by FRT */
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "../Generic/random.h"
#include "../MCtrajs/MCdyn_classi/Generic/fileSystemUtil.h"
#include "../MCtrajs/MCdyn_classi/thermodynamicUtilities.h"
#include "../MCtrajs/MCdyn_classi/Initialization/initializeGraph.h"

#include <sys/utsname.h>

#define p 2
#define rOpt 1.58
#define MCSperSWAP 10000
#define SwapChancesPerExtraction 1
#define swapChancesBeforeFirstExtraction 20

/* variabili globali per il generatore random */
int N, C;
vector<vector<vector<rInteraction>>> Graph;
vector<int> *s;
vector<float> Beta(0);

int main(int argc, char *argv[])
{
    int ib, is, numIC, ic;
    unsigned long long int t, MCtotal;
    FILE *file;

    if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -3 || argc != 10)
    {
        cout << "Probable desired usage: ";
        cout << "d(-3:DPRRG -2:ER -1:RRG k>0:k-dim sqLatt) N Hext C structureID fracPosJ graphID fieldType(1: Bernoulli, 2: Gauss) sigma" << endl;
        cout << "If d is a lattice, C and structureID are required but ignored." << endl;
        exit(1);
    }

    int d = std::stoi(argv[1]); // should include check on d value
    N = atoi(argv[2]);
    double Hext = atof(argv[3]);
    C = 0;
    if (d < 0)
    {
        C = atoi(argv[4]);
    }
    else
    {
        C = 2 * d;
    }

    int structureID = atoi(argv[5]);
    double fracPosJ = atof(argv[6]);
    int graphID = atoi(argv[7]);

    int fieldType = atoi(argv[8]);
    double var = atof(argv[9]);


    string admittibleGraph, folder;
    if (getGraphFromGraphAndStructureID_ForCluster(d, admittibleGraph, structureID, graphID, p, C, N, fracPosJ))
        folder = admittibleGraph + "/";
    else
        return 1;

    int myrand = init_random(0, 0);
    printf("# pairwise_PT.c  N = %i  C = %i  seed = %u\n graphID = % i  fracPosJ = % f\n", N, C, myrand, graphID, fracPosJ);

    string graphType, fileName = "../Data/BetasForPT/";

    if (d < 0)
    {
        if (d == -3)
            graphType = "DPRRG";
        else if (d == -2)
            graphType = "ER";
        else if (d == -1)
            graphType = "RRG";
        graphType += "/p" + to_string(p) + "C" + to_string(C);
    }
    else if (d == 1)
    {
        graphType = "1dChain";
    }
    else
    {
        graphType = to_string(d) + "dSqLatt";
    }

    graphType += "/fPosJ" + to_string(fracPosJ).substr(0, 4) + "/N" + to_string(N);
    fileName += graphType + ".txt";

    // Open the file for reading
    if (!initializeGraph(folder, Graph, p, C, N, fracPosJ))
    {
        cout << "NOO" << endl;
        cout << "Error in the graph initialization" << endl;
        return 1;
    }

    // trova quanti file randomField ci sono e crealo col nome corretto
    if (folderExists(folder + "randomField1"))
    {
        int i1;
        for (i1 = 2; folderExists(folder + "randomField" + to_string(i1)); i1++)
        {
        }
        folder = folder + "randomField" + to_string(i1);
    }
    else
    {
        folder = folder + "randomField";
    }
    createFolder(folder);

    vector<double> randomNumbers(N);

    if (fieldType == 1)
    {
        for (double &number : randomNumbers)
        {
            number = var * Grandom();
        }
    }
    else if (fieldType == 2)
    {
        for (double &number : randomNumbers)
        {
            number = var * (1 - 2 * (Xrandom() > 0.5));
        }
    }

    // Open a file for writing
    string nomeFile = folder + "/randomField.txt";
    std::ofstream outputFile(nomeFile);

    // Check if the file is open
    if (!outputFile.is_open())
    {
        std::cerr << "Error opening the file " << nomeFile << " for writing!" << std::endl;
        return 1;
    }

    outputFile << N << " " << p << " " << C << " " << structureID<<" "<< fracPosJ<<" "<<graphID<< " "<< myrand << "\n";

    // Write the array to the file

    for (double &number : randomNumbers)
    {
       outputFile <<  number<<endl;
    }

    // Close the file
    outputFile.close();

    std::cout << "Array has been written to file " << nomeFile << std::endl;

    return EXIT_SUCCESS;
}
