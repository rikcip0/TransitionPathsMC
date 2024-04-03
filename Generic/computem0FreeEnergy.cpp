#include <iostream>
#include <fstream>
#include <cassert>
#include <time.h>
#include <sys/utsname.h>
#include <math.h>
#include "../MCdyn_classi/generic/fileSystemUtil.h"
#include "../MCdyn_classi/interaction.h"

#include "../MCdyn_classi/Initialization/initializeGraph.h"

#include "../MCdyn_classi/thermodynamicUtilities.h"

#define p 2 // as of now, the dynamics generation (and hence the code) only works for p=2 (RC)

using namespace std;

double computeH(vector<int> conf, vector<vector<vector<rInteraction>>> *Graph, double Hext) // returns an Np (eq. spaced times) array with the system Hamiltonian value
{

    int N = conf.size();
    int currenth[N];
    double H = 0;
    // init spins and fields
    for (int i = 0; i < N; i++)
    {
        currenth[i] = Hext;
        for (int j = 0; j < (*Graph)[i][0].size(); j++)
        {
            currenth[i] += conf[(*Graph)[i][0][j]] * (*Graph)[i][0][j].J; // Maybe we should include the coupling (RC)
        }
        H -= conf[i] * (currenth[i] + Hext) / 2; // /2 on h also (!?)
    }

    return H;
}

int main(int argc, char **argv)
{

    // Two input possibilities according to d=-1,-2 (RRG), or d=1,2 (square lattice)
    // RRG: ./program.exe -1 N T beta Hext hin hout C fracPosJ
    // square lattice: ./program.exe 1,2 N T beta Hext hin Q* hout
    // TO IMPLEMENT- double planted RRG: ./program.exe -2 N T beta Hext hin hout C fracPosJ

    // START of input management
    if ((argc < 2) || (atoi(argv[1]) == 0) || abs(atoi(argv[1])) > 2 || (atoi(argv[1]) == -1) && (argc < 5 || argc > 7) || (atoi(argv[1]) == 2) && (argc != 9))
    {
        cout << "Probable desired usage: ";

        if (atoi(argv[1]) == -2)
        {
            cout << "-2 N beta Hext C graphID This is double planted RRG" << endl;
        }
        else if (atoi(argv[1]) == -1)
        {
            cout << "-1 N beta Hext C graphID  This is RRG" << endl;
        }
        else if (atoi(argv[1]) == 1)
        {
            cout << "Usage: dim(=1,2) N beta Hext  This is for Sq Latt" << endl;
        }
        else
        {
            cout << argc;
            cout << "topology[-2, -1, 1, 2] N beta Hex (C, graphID [for RRG])" << endl;
        }
        exit(1);
    }

    int d = atoi(argv[1]); // used to determine simulation type d=1: chain 1d, d=2 square lattice 2d, d=-1 RRG with connectivity C (RC)
    int N = atoi(argv[2]);
    double beta = atof(argv[3]);
    double Hext = atof(argv[4]);
    int C;
    int graphID = -1;

    if (d < 0)
    {
        C = atoi(argv[5]);
        graphID = atoi(argv[6]);
    }

    // END of input management

    // START of initialization of graph
    char buffer[200];
    string folder;

    vector<vector<vector<rInteraction>>> Graph;

    if (d == -1)
    {
        if (graphID == -1)
        {
            vector<string> admittibleGraphs;
            if (getGraphsFromPCNf(admittibleGraphs, p, C, N, 1.))
            {
                folder = admittibleGraphs[(int)(Xrandom() * (admittibleGraphs.size()) - 1)] + "/";
            }
            else
                return 1;
        }
        else
        {
            string admittibleGraph;
            if (getGraphFromGraphID(admittibleGraph, graphID, p, C, N, 1.))
                folder = admittibleGraph + "/";
            else
                return 1;
        }
        if (!initializeGraph(folder, Graph, p, C, N, 1.))
        {
            cout << "Error in the graph initialization" << endl;
            return 1;
        }
    }
    else
    {

        folder = (string)buffer;

        if (!initializeLattice(folder, Graph, d, N, 1.))
        {
            cout << "Error in the lattice initialization" << endl;
            return 1;
        }
    }

    vector<int> referenceConfig(N, 1);

    int totalIterations = 1 << N;
    vector<double> P(N + 1);
    int m = 0;
    for (int i = 0; i < totalIterations; ++i)
    {
        std::vector<int> conf;

        for (int j = 0; j < N; ++j)
        {
            int sign = (i & (1 << j)) ? 1 : -1;
            conf.push_back(sign);
        }

        m = 0;
        for (int i = 0; i < N; i++)
            m += conf[i] * referenceConfig[i];

        if (m >= 0)
            P[(m + N) / 2] += exp(-beta * computeH(conf, &Graph, Hext));
    }

    std::ofstream outputFile("freeEnergyExComputation" + to_string(graphID) + ".txt");

    if (outputFile.is_open())
    {
        outputFile << "#"<<graphID << endl;
        outputFile << "#";
        for (int i = 0; i < N; i++)
            outputFile << referenceConfig[i] << " ";
        outputFile << endl;

        for (int i = 0; i < N + 1; ++i)
        {
            outputFile << 2 * i - N << " " << P[i] << endl;
        }

        // Close the file
        outputFile.close();

        std::cout << "Array successfully written to the file." << std::endl;
    }
    else
    {
        std::cerr << "Error opening the file for writing." << std::endl;
    }
    return 0;
}
