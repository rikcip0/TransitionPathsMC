/* by FRT */
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "../MCtrajs/MCdyn_classi/Generic/fileSystemUtil.h"
#include "../MCtrajs/MCdyn_classi/thermodynamicUtilities.h"
#include "../MCtrajs/MCdyn_classi/Initialization/initializeGraph.h"
#include "../Generic/FRTGenerator.h"

#include <sys/utsname.h>

#define p 2
#define rOpt 1.58
#define MCSperSWAP 1800
#define SwapChancesPerExtraction 1
#define swapChancesBeforeFirstExtraction 20

/* variabili globali per il generatore random */
int N, C, maxDegree, numBeta, *confIndex;
double *ener;
vector<vector<vector<rInteraction>>> Graph;
vector<int> *s, *whatBetaForReplica;
int *swapRate;
vector<float> Beta(0);

void initBetas(string fileName)
{

    std::ifstream temperaturesFile(fileName); // Open the file
    if (!temperaturesFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
        cout << fileName << endl;
        exit(1);
    }

    double number;
    while (temperaturesFile >> number)
        Beta.push_back(number);

    temperaturesFile.close();

    int ib, i;

    numBeta = Beta.size();
    confIndex = (int *)calloc(numBeta, sizeof(int));
    swapRate = (int *)calloc(numBeta - 1, sizeof(int));

    printf("# %i Betas:", numBeta);

    for (ib = 0; ib < numBeta; ib++)
        printf(" %f", Beta[ib]);

    printf("\n");
}

void initSpin(vector<double> randomField)
{
    int ib, i;

    for (ib = 0; ib < numBeta; ib++)
    {
        confIndex[ib] = ib;
        for (i = 0; i < N; i++)
        {
            s[ib][i] = pm1;
        }
        ener[ib] = energy_Graph(s[ib], N, Graph, 0., randomField);
    }
}

void oneMCStep(vector<int> &s, double beta, double *pEner)
{
    int I;
    double localEnergy;
    for (int i = 0; i < N; i++)
    {
        I = (int)(FRANDOM * N);

        localEnergy = 0; // to be changed in h_ext or h_ext[I]
        for (int j = 0; j < (Graph)[I][0].size(); j++)
        {
            localEnergy += s[Graph[I][0][j]] * Graph[I][0][j].J;
        }
        localEnergy *= s[I];
        if (localEnergy <= 0 || FRANDOM < exp(-2 * beta * localEnergy))
        {
            s[I] = -s[I];
            *pEner += 2 * localEnergy;
        }
    }
}

void swapStep(void)
{
    int ib, tmp;

    for (ib = 0; ib < numBeta; ib++)
        whatBetaForReplica[confIndex[ib]].push_back(ib);

    for (ib = 1; ib < numBeta; ib++)

        if (ener[ib - 1] <= ener[ib] || FRANDOM < exp((Beta[ib] - Beta[ib - 1]) * (ener[ib] - ener[ib - 1])))
        {
            tmp = confIndex[ib];
            confIndex[ib] = confIndex[ib - 1];
            confIndex[ib - 1] = tmp;
            tmp = ener[ib];
            ener[ib] = ener[ib - 1];
            ener[ib - 1] = tmp;
            swapRate[ib - 1]++;
        }
}

int main(int argc, char *argv[])
{
    int ib, is, numIC, ic;
    unsigned long long int t, MCtotal;
    FILE *file;

    if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -3 || argc != 10)
    {
        cout << "Probable desired usage: ";
        cout << "d(-3:DPRRG -2:ER -1:RRG k>0:k-dim sqLatt) N Hext C structureID fracPosJ graphID MCtotal numIC" << endl;
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

    MCtotal = atoi(argv[8]);
    numIC = atoi(argv[9]);

    FILE *devran = fopen("/dev/random", "r");
    fread(&myrand, 4, 1, devran);
    fclose(devran);
    initRandom();
    vector<double> field(N,0);
    string admittibleGraph, folder;
    if (getGraphFromGraphAndStructureID_ForCluster(d, admittibleGraph, structureID, graphID, p, C, N, fracPosJ))
        folder = admittibleGraph + "/";
    else
        return 1;

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
    if (!initializeGraph(folder, Graph, p, C, N, fracPosJ, field, 0, 0, 0.1))
    {   cout<<"NOO"<<endl;
        cout << "Error in the graph initialization" << endl;
        return 1;
    }

    initBetas(fileName);

    // trova quanti file configurations ci sono e crealo col nome corretto
    if (folderExists(folder + "configurations"))
    {
        int i1;
        for (i1 = 2; folderExists(folder + "configurations" + to_string(i1)); i1++)
        {
        }
        folder = folder + "configurations" + to_string(i1);
    }
    else
    {
        folder = folder + "configurations";
    }
    createFolder(folder);

    s = (vector<int> *)calloc(numBeta, sizeof(vector<int>));
    for (ib = 0; ib < numBeta; ib++)
        s[ib].assign(N, 0);

    whatBetaForReplica = (vector<int> *)calloc(numBeta, sizeof(vector<int>));
    for (ib = 0; ib < numBeta; ib++)
        whatBetaForReplica[ib].assign(0, 0);

    ener = (double *)calloc(numBeta, sizeof(double));

    pair<string, string> info;
    std::vector<std::ofstream> outFileConfs(numBeta);
    struct utsname ugnm;
    uname(&ugnm);

    info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(myrand) + ", started at " + getCurrentDateTime() + "\n";
    info.second += (string)("12 1 ") + ugnm.nodename + " " + to_string(myrand) + " " + getCurrentDateTime() + "\n";

    info.second += "21 " + to_string(N) + " " + to_string((int)p) + " " + to_string(C) + " " + to_string(graphID);
    info.second += " " + to_string(fracPosJ) + " " + to_string(Hext) + "\n";

    info.first += "mcTotal=" + to_string(MCtotal) + " numIC=" + to_string(numIC) + " MCPerSwap=" + to_string(MCSperSWAP) + " swapChancesForExtraction=" + to_string(SwapChancesPerExtraction) + " swapChancesBeforeFirstExtraction=" + to_string(swapChancesBeforeFirstExtraction) + "\n\n";
    info.second += "30 " + to_string(MCtotal) + " " + to_string(numIC) + " " + to_string(MCSperSWAP) + " " + to_string(SwapChancesPerExtraction) + " " + to_string(swapChancesBeforeFirstExtraction) + "\n";

    string nomefile = folder + "/infoLong.dat";
    ofstream detFile(nomefile);
    detFile << info.first;
    detFile.close();

    nomefile = folder + "/info.dat";
    detFile.open(nomefile);
    detFile << info.second;
    detFile.close();

    cout << "Running " << graphType << " pt with N=" << N << " graphID=" << graphID << endl;
    cout << "In folder " << folder << endl;

    for (int ib = 0; ib < numBeta; ib++)
    {
        fileName = folder + "/B" + std::to_string(Beta[ib]).substr(0, 4) + "confs";
        outFileConfs[ib].open(fileName);
        if (!outFileConfs[ib].is_open())
        {
            std::cerr << "Errore nell'apertura di " << fileName << std::endl;
            return 1;
        }
        outFileConfs[ib] << std::to_string(Beta[ib]) << std::endl;
    }

    std::ofstream outFilePTData(folder + "/energies.txt", std::ios::app);

    long double meanMinEner = 0;
    long double meanEner = 0;
    long int swapChances = 0;
    int nExtractedConfs = 0;

    for (ic = 0; ic < numIC; ic++)
    {
        initSpin(field);
        t = 0;
        do
        {
            t++;
            for (ib = 0; ib < numBeta; ib++)
                oneMCStep(s[confIndex[ib]], Beta[ib], ener + ib);

            if (t % MCSperSWAP == 0)
            {

                if ((t % (SwapChancesPerExtraction * MCSperSWAP) == 0) && t / MCSperSWAP > swapChancesBeforeFirstExtraction)
                {

                    for (ib = 0; ib < numBeta; ib++)
                    {
                        for (int i = 0; i < N; i++)
                        {
                            outFileConfs[ib] << s[confIndex[ib]][i] << " ";
                        }
                        outFileConfs[ib] << std::endl;
                    }

                    nExtractedConfs++;
                }

                outFilePTData << ic << " " << t << " ";
                meanEner = 0;
                for (ib = 0; ib < numBeta; ib++)
                {

                    meanEner += ener[ib];
                    outFilePTData << ener[ib] << " ";
                }
                outFilePTData << meanEner / (double)numBeta << std::endl;

                swapStep();
                swapChances++;
            }

        } while (t < MCtotal);
        fflush(stdout);
    }

    for (int i = 0; i < numBeta; i++)
    {
        outFileConfs[i].close();
    }

    outFilePTData.close();

    outFilePTData.open(folder + "/swapRates.txt", std::ios::app);
    outFilePTData << graphID << " " << myrand << endl;
    for (int i = 0; i < numBeta - 1; i++)
        outFilePTData << Beta[i] << " " << swapRate[i] / (double)(swapChances) << std::endl;
    outFilePTData << Beta[numBeta - 1] << " nan" << std::endl;
    outFilePTData.close();

    outFilePTData.open(folder + "/permanenceInfo.txt", std::ios::app);
    for (int i = 0; i < whatBetaForReplica[0].size(); i++)
    {
        for (int j = 0; j < numBeta; j++)
            outFilePTData << whatBetaForReplica[j][i] << " ";
        outFilePTData << std::endl;
    }
    outFilePTData.close();

    int Q = 0;
    vector<int> *configurations;
    configurations = (vector<int> *)calloc(nExtractedConfs, sizeof(vector<int>));

    std::vector<std::ofstream> outFileConfOverlaps(numBeta);
    std::vector<std::ofstream> outFileConfMagnetizations(numBeta);

    for (int ib = 0; ib < numBeta; ib++)
    {
        fileName = folder + "/B" + std::to_string(Beta[ib]).substr(0, 4) + "Qs";
        outFileConfOverlaps[ib].open(fileName);
        if (!outFileConfOverlaps[ib].is_open())
        {
            std::cerr << "Errore nell'apertura di " << fileName << std::endl;
            return 1;
        }

        fileName = folder + "/B" + std::to_string(Beta[ib]).substr(0, 4) + "Ms";
        outFileConfMagnetizations[ib].open(fileName);
        if (!outFileConfMagnetizations[ib].is_open())
        {
            std::cerr << "Errore nell'apertura di " << fileName << std::endl;
            return 1;
        }

        outFileConfOverlaps[ib] << Beta[ib] << "\n";
        outFileConfMagnetizations[ib] << Beta[ib] << "\n";

        fileName = folder + "/B" + std::to_string(Beta[ib]).substr(0, 4) + "confs";
        for (int i = 0; i < nExtractedConfs; i++)
            initializeVectorFromLine(fileName, i + 1, N, configurations[i]);

        for (int i = 0; i < nExtractedConfs; i++)
        {
            int M = 0;
            for (int k = 0; k < N; k++)
                M += configurations[i][k];
            outFileConfMagnetizations[ib] << M << " ";
        }
        outFileConfMagnetizations[ib] << endl;

        for (int i = 0; i < nExtractedConfs; i++)
        {
            for (int j = 0; j < nExtractedConfs; j++)
            {
                Q = 0;
                for (int k = 0; k < N; k++)
                    Q += configurations[i][k] * configurations[j][k];
                outFileConfOverlaps[ib] << Q << " ";
            }
            outFileConfOverlaps[ib] << endl;
            ;
        }
    }

    return EXIT_SUCCESS;
}
