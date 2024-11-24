// Voglio fare che o si puo scegliere un grafo specifico o si specifica solo il modello e vengono fatte molte iterazioni senza salvare ogni volta tutto il grafo

#include <iostream>
#include <fstream>
#include <cassert>
#include <sys/utsname.h>
#include <string.h>
#include <time.h> //L'ho aggiunto io (RC) per inizializzare il generatore

#include "../Generic/random.h"
#include "../Generic/FRTGenerator.h"
#include "../MCtrajs/MCdyn_classi/Initialization/initializeGraph.h"
#include "../MCtrajs/MCdyn_classi/Initialization/initializeReferenceConfigurations.h"

#include "../MCtrajs/MCdyn_classi/MCUtilities.h"
#include "../MCtrajs/MCdyn_classi/thermodynamicUtilities.h"

#define p 2
#define nStories 20
#define epicCount 1
#define nDisorderCopies 1 // number of instances for each disorder configuration TO BE IMPLEMENTED
#define configurationsChoiceOption -1

#ifdef EXITING
#define magIncrement -2
#define signToCompare 1
#ifdef WITHLINFIELD
#define simType "ExitingWithLinField"
#elif defined(WITHDISAPPEARINGFIELD)
#define simType "ExitingWithDisappearingField"
#else
#define simType "Exiting"
#endif
#elif defined(ENTERING)
#define simType "Entering"
#define magIncrement 2
#define signToCompare -1
#else
#define simType "Transitioning"
#define magIncrement 2
#define signToCompare -1
#endif

#define pm1 ((FRANDOM > 0.5) ? 1 : -1)

#ifdef WITHDISAPPEARINGFIELD
double m_dis;
#endif

double sqrtOrZero(double toSqrt)
{
    if (toSqrt < 0. && toSqrt > -0.0001)
    {
        return 0.;
    }
    return sqrt(toSqrt);
}

int findMaxIndex(int N, int referenceConfMag, int lowerMeasuredMag, const vector<double> array, int size, int computeCorrected, double correction, int higherBoundMagnetization)
{
    int maxIndex = (N - higherBoundMagnetization) / 2;
    double maxValue = (double)array[maxIndex];
    double toCompare;

    if (computeCorrected)
    {
        if (simType == "EnteringWithLinField")
            maxValue -= correction * (N - referenceConfMag);
        else
            maxValue -= correction * (N - referenceConfMag * referenceConfMag / (double)N);
    }

    for (int i = maxIndex + 1; i < (N - lowerMeasuredMag) / 2 + 1; i++)
    {
        if (computeCorrected == 1)
        {
#ifdef WITHDISAPPEARINGFIELD
            toCompare = array[i] - correction * (N - 2 * i > m_dis * N) * (N - 2 * i - referenceConfMag);
#else
            if (simType == "EnteringWithLinField")
                toCompare = array[i] - correction * ((N - 2 * i) * (N - 2 * i) - referenceConfMag * referenceConfMag) / (double)N;
            else
                toCompare = array[i] - correction * (N - 2 * i - referenceConfMag); // the correction to the planted configuration is considered in this formula.
#endif
        }
        else
        {
            toCompare = array[i];
        }
        if (toCompare > maxValue)
        {
            maxValue = toCompare;
            maxIndex = i;
        }
    }
    return maxIndex;
}

int *whereEqual(int *a) // returns the first element of the a-array which is equal to another element of a (if not present: NULL)
{
    int i, j;

    for (i = 0; i < p - 1; i++)
        for (j = i + 1; j < p; j++)
            if (a[i] == a[j])
                return a + j;
    return NULL;
}

void initProb(double beta, double field, int C, int *prob)
{
    int i;
    i = C;
    while (i >= 0)
    {
        prob[i] = exp(-2. * beta * (i - field));
        i -= 2;
    }
    i = C - 1;
    while (i >= 0)
    {
        prob[i] = exp(-2. * beta * (i + 1 + field));
        i -= 2;
    }
}

int main(int argc, char **argv)
{
    int is = 0;

    int ener0, Qout, nextMeasMag, ener0_sum = 0;
    double *prob;

    int magnetizationArraysLength, lowerMeasuredMag, referenceConfMag;
    vector<int> startedHere;
    vector<double> logFirstTime, logFirstTimeSquared;
    vector<long long int> num, cumNum, cumNumSquared;
    vector<long double> barrierSum;
    vector<double> meanBarrier, cumFirstBarrier, firstBarrier, cumFirstBarrierSquared, cumMeanBarrier, cumMeanBarrierSquared;
    vector<double> maxMeanBarrier, maxFirstBarrier, maxCorrectedMeanBarrier, maxCorrectedFirstBarrier;
    vector<int> isArgMaxFirstBarrier, isArgMaxMeanBarrier, isArgMaxCorrectedFirstBarrier, isArgMaxCorrectedMeanBarrier;
    // TO IMPLEMENT- double planted RRG: type -3

    // START of input management

    if ((argc < 2) || (atoi(argv[1]) == 0) || atoi(argv[1]) < -2 || (argc != 10 && argc != 12 && argc != 13 && argc != 15))
    {
        cout << "Probable desired usage: ";
        cout << "d(-2:ER -1:RRG k>0:k-dim sqLatt) N beta Hext Q*(if -1 is self-computed) \nC structureID fracPosJ graphID (requiredBetaOfSExtraction requiredQif)[if not present, FM(all +) conf.is considered]\n (randomFieldType(1: Bernoulli, 2:Gaussian), realization, sigma)" << endl;
        cout << "If d is a lattice, C and structureID are required but ignored." << endl;
        exit(1);
    }

    int d = std::stoi(argv[1]); // should include check on d value
    int N = atoi(argv[2]);
    double beta = atof(argv[3]);
    double Hext = atof(argv[4]);

    int Qstar = atoi(argv[5]);
    /*
      if ((abs(Qstar) % 2 != N % 2 || abs(Qstar) > N) && Qstar != -1)
      {
        cout << "Q* value is not allowed!" << endl;
        exit(1);
      }
    */

    int C = 0;
    int structureID = -1;
    if (d < 0)
    {
        C = atoi(argv[6]);
        structureID = atoi(argv[7]);
    }
    else
    {
        C = 2 * d;
    }

    double fracPosJ = atof(argv[8]);
    int graphID = atoi(argv[9]);

    string admittibleGraph, folder;
    pair<string, string> info; // 1° entry: long info (for human read), 2°entry: short info (for machines read)
    vector<vector<vector<rInteraction>>> Graph;
    vector<double> randomField;
    vector<int> s, s_in, s_out;

    if (getGraphFromGraphAndStructureID_ForCluster(d, admittibleGraph, structureID, graphID, p, C, N, fracPosJ))
        folder = admittibleGraph + "/";
    else
        return 1;

    int randomFieldType = 0, fieldStructureRealization = 0;
    double sigma = 1.;

    if (argc == 13 || argc == 15)
    {
        randomFieldType = atoi(argv[argc - 3]);
        fieldStructureRealization = atoi(argv[argc - 2]);
        sigma = atof(argv[argc - 1]);
    }
    else
        randomField.assign(N, 0.);

    if (!initializeGraph(folder, Graph, p, C, N, fracPosJ, randomField, randomFieldType, fieldStructureRealization, sigma))
    {
        cout << "Error in the graph initialization" << endl;
        return 1;
    }

    string graphType;
    if (d < 0)
    {
        if (d == -2)
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

    graphType += +"/N" + to_string(N) + "/fPosJ" + to_string(fracPosJ).substr(0, 4);

    struct utsname ugnm;
    uname(&ugnm);

    init_random(0, 0);
    long sstart = time(NULL);

    info.first += "Simulation on " + graphType + "\n";
    if (Qstar != -1)
    {
        info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
        info.first += "Qstar given in input\n\n";
#ifdef INITANNEALING
        info.second += (string)("500 0 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
#else
        info.second += (string)("50 0 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
#endif
    }
    else
    {
        info.first += (string)("Simulation run on: ") + ugnm.nodename + ", with seed " + to_string(sstart) + ", started at " + getCurrentDateTime() + "\n\n";
        info.first += "No Qstar given in input: self-overlap around final configuration will be computed\n\n";
#ifdef INITANNEALING
        info.second += (string)("510 0 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
#else
        info.second += (string)("51 0 ") + ugnm.nodename + " " + to_string(sstart) + " " + getCurrentDateTime() + "\n";
        cout << "Simulation with extremes fixed to reference configurations.\n\n";
#endif
    }

    pair<vector<int>, vector<int>> referenceConfigurations;
    referenceConfigurations.first.assign(N, 0);
    referenceConfigurations.second.assign(N, 0);

    if (argc == 12 || argc == 15)
    {
        double requiredBetaOfSExtraction = atof(argv[10]);
        int requiredQif = atoi(argv[11]);
        if (abs(requiredQif) > N)
        {
            std::cout << "requiredQif value is not allowed!" << std::endl;
            return 1;
        }

        std::string input(argv[10]);

        if (input == "inf")
        {
            requiredBetaOfSExtraction = 100.; // just a large number
        }
        else
        {
            requiredBetaOfSExtraction = std::stod(input);
        }
        if (requiredBetaOfSExtraction < 0.0)
        {
            std::cout << "requiredBetaOfSExtraction value is not allowed!" << std::endl;
            return 1;
        }

        if (!
#ifndef QUENCHCONFS
            initializeReferenceConfigurationsFromParTemp_FirstOccurrence(folder, N, referenceConfigurations, info, requiredQif, requiredBetaOfSExtraction, configurationsChoiceOption)
        // initializeReferenceConfigurationsFromParTemp_Typical(folder, N, referenceConfigurations, info, requiredQif, requiredBetaOfSExtraction, configurationsChoiceOption)
#else
            initializeReferenceConfigurationsSeqQuenchingFromParTemp(Graph, folder, N, referenceConfigurations, info, requiredQif, requiredBetaOfSExtraction, configurationsChoiceOption)
#endif
        )
        {
            cout << "Error in the start/end configurations initialization" << endl;
            return 1;
        }
    }
    else
    {
        if (!initializeFMConfigurations(N, referenceConfigurations, info))
        {
            cout << "Error in the start/end configurations initialization" << endl;
            return 1;
        }
    }

    s_in = referenceConfigurations.first;
    s_out = referenceConfigurations.second;

    if (Qstar == -1)
    {
        if (!computeSelfOverlap_withGraph(Qstar, s_out, Graph, beta, info, N))
        {
            cout << "Error during computation of final configuration self overlap" << endl;
            return 1;
        }
        if ((Qstar + N) % 2)
            Qstar -= 1; // decrease by 1 if Qstar and N have different parity
    }
    cout << "Using Q*= " << Qstar << endl
         << endl;

    int mutualQ = 0;
    for (int i = 0; i < N; i++)
    {
        mutualQ += s_in[i] * s_out[i];
    }

    char buffer[200];
    sprintf(buffer, "%.4g_%.2g_%i", beta, Hext, Qstar);

    // folder = makeFolderNameFromBuffer(folder+"/DataForPathsMC/", string(buffer));   //Comment if on cluster
    if (argc == 13 || argc == 15)
    {
        folder = makeFolderNameFromBuffer_ForCluster(folder + "MCMCs/Exit/", string(buffer) + "_sigma" + to_string(sigma), sstart); // For Cluster
    }
    else
    {
        folder = makeFolderNameFromBuffer_ForCluster(folder + "MCMCs/Exit/", string(buffer), sstart); // For Cluster
    }

    createFolder(folder);
    cout << "Simulation is in folder " << folder << endl;
    /*

        myrand = time(NULL); // mia inizializzazione, che ho Windows
        if (argc < 7)
        {
            fprintf(stderr, "usage: %s <N> <beta_p> <beta> <nStories> <h>\n", argv[0]);
            exit(EXIT_FAILURE);
        }

        if (isdigit(*argv[2]))
        {
            beta_p = atof(argv[2]);
        }
        else if (strcmp(argv[2], "inf") == 0)
        {
            beta_p = -1; // so, in the rest of the code, if beta_p is negative it means it is to be considered infinite
        }
        else
        {
            printf("Planting temperature must be a non-negative number or infinite (inf)\n");
            return EXIT_FAILURE;
        }

        strcpy(beta_p_string, argv[2]);
        beta = atof(argv[3]);
        H = atof(argv[5]);
    */
    int index, startMag, highestMToComputeStoryMax;
    long long unsigned int t;
    double beta_p, H = 0., posJProb;
    char beta_p_string[7];
    if (strncmp(simType, "Exiting", strlen("Exiting")) == 0)
    {
        H = -H;
        magnetizationArraysLength = N / 2 + 1;
#ifdef WITHDISAPPEARINGFIELD
        m_dis = atof(argv[7]);
#endif
    }
    else if (simType == "Entering")
    {
        magnetizationArraysLength = N + 1;
    }

    if (beta <= 0.0)
        error("in beta");

    if (C % 2)
    {
        if (H < -1.0 || H > 1.0)
            error("in h");
    }
    else

    {
        if (H < -2.0 || H > 0.0)
            error("in h");
    }

    init_random(0, 0);

    firstBarrier.assign(magnetizationArraysLength, 0.);
    cumFirstBarrier.assign(magnetizationArraysLength, 0.);
    cumFirstBarrierSquared.assign(magnetizationArraysLength, 0.);
    logFirstTime.assign(magnetizationArraysLength, 0.);
    logFirstTimeSquared.assign(magnetizationArraysLength, 0.);
    startedHere.assign(magnetizationArraysLength, 0);
    num.assign(magnetizationArraysLength, 0);
    cumNum.assign(magnetizationArraysLength, 0);
    cumNumSquared.assign(magnetizationArraysLength, 0);
    barrierSum.assign(magnetizationArraysLength, 0);
    meanBarrier.assign(magnetizationArraysLength, 0.);
    cumMeanBarrier.assign(magnetizationArraysLength, 0.);
    cumMeanBarrierSquared.assign(magnetizationArraysLength, 0.);
    isArgMaxFirstBarrier.assign(magnetizationArraysLength, 0);
    isArgMaxMeanBarrier.assign(magnetizationArraysLength, 0);
    isArgMaxCorrectedFirstBarrier.assign(magnetizationArraysLength, 0);
    isArgMaxCorrectedMeanBarrier.assign(magnetizationArraysLength, 0);
    maxFirstBarrier.assign(magnetizationArraysLength, 0.);
    maxMeanBarrier.assign(magnetizationArraysLength, 0.);
    maxCorrectedFirstBarrier.assign(magnetizationArraysLength, 0.);
    maxCorrectedMeanBarrier.assign(magnetizationArraysLength, 0.);

    printf("#%s  C = %i p = %i  N = %i  beta_p = %s  beta = %f  H = %f nStories = %d seed = %d",
           simType, C, p, N, beta_p_string, beta, fabs(H), nStories, myrand);
#ifdef WITHDISAPPEARINGFIELD
    printf(" m_dis=%f", m_dis);
#endif
    printf("\n");
    printf("# 1:Qout  2:barrier  3:time\n");
    ofstream thisEpicFile(folder + "/Epic_" + to_string(epicCount) + ".txt");

    do
    {
        s = s_in;

        if (simType == "Entering")
        {
            Qout = 0;
            for (int i = 0; i < N; i++)
            {
                s[i] = pm1;
                Qout += s[i] * s_out[i];
            }
        }

        lowerMeasuredMag = Qout; // Qout if Entering, 0 otherwise

        ener0 = energy_Graph(s, N, Graph, Hext, randomField);
        ener0_sum += ener0;
        startedHere[(N - Qout) / 2]++;
        num[(N - Qout) / 2]++;
        referenceConfMag = Qout;
        nextMeasMag = Qout + magIncrement;

        // initProb(beta, H);

        printf("Epic %d, story %d: %i %f 0\n", epicCount, is, Qout, energy_Graph(s, N, Graph, Hext, randomField) - ener0);

        if (simType == "Entering")
        {
            cout << t << " " << nextMeasMag << " " << magIncrement << " " << Qstar << endl;
            for (t = 1; nextMeasMag <= Qstar; t++)
            {
                MCSweep_withGraph_variant(s, N, Graph, beta, Hext, randomField, Qout, lowerMeasuredMag,
                                          nextMeasMag, magIncrement, t, logFirstTime, logFirstTimeSquared,
                                          firstBarrier, barrierSum, num, s_out);
            }
            cout << t << " " << nextMeasMag << " " << magIncrement << " " << Qstar << endl;
        }
        else if (strncmp(simType, "Exiting", strlen("Exiting")) == 0)
        {
            cout << t << " " << nextMeasMag << " " << magIncrement << " " << Qstar << endl;
            for (t = 1; nextMeasMag <= Qstar; t++)
            {
                MCSweep_withGraph_variant(s, N, Graph, beta, Hext, randomField, Qout, lowerMeasuredMag,
                                          nextMeasMag, magIncrement, t, logFirstTime, logFirstTimeSquared,
                                          firstBarrier, barrierSum, num, s_out);
#if defined(WITHLINFIELD)
                initProb(beta, H * Qout / (double)N);
#elif defined(WITHLINFIELD)
                if (Qout <= m_dis * N)
                {
                    initProb(beta, 0.);
                }
                else
                {
                    initProb(beta, H);
                }
#endif
            }
        }

        for (int i = 0; i < magnetizationArraysLength; i++)
        {
            if (num[i] > 0)
            {
                cumFirstBarrierSquared[i] += firstBarrier[i] * firstBarrier[i];
                cumFirstBarrier[i] += firstBarrier[i];
                cumNum[i] += num[i];
                cumNumSquared[i] += num[i] * num[i];

                meanBarrier[i] = barrierSum[i] / (double)num[i];
                cumMeanBarrier[i] += meanBarrier[i];
                cumMeanBarrierSquared[i] += meanBarrier[i] * meanBarrier[i];
            }
            barrierSum[i] = 0;
            num[i] = 0;
        }

        highestMToComputeStoryMax = (simType == "Entering") ? N : referenceConfMag;
        index = findMaxIndex(N, referenceConfMag, lowerMeasuredMag, firstBarrier, magnetizationArraysLength, 0, 0., highestMToComputeStoryMax);
        isArgMaxFirstBarrier[index]++;
        maxFirstBarrier[index] += firstBarrier[index];

        index = findMaxIndex(N, referenceConfMag, lowerMeasuredMag, meanBarrier, magnetizationArraysLength, 0, 0., highestMToComputeStoryMax);
        isArgMaxMeanBarrier[index]++;
        maxMeanBarrier[index] += meanBarrier[index];

        index = findMaxIndex(N, referenceConfMag, lowerMeasuredMag, firstBarrier, magnetizationArraysLength, 1, H, highestMToComputeStoryMax);
        isArgMaxCorrectedFirstBarrier[index]++;
        maxCorrectedFirstBarrier[index] += firstBarrier[index] - H * (N - 2 * index - referenceConfMag); // N-2i is the Qout

        index = findMaxIndex(N, referenceConfMag, lowerMeasuredMag, meanBarrier, magnetizationArraysLength, 1, H, highestMToComputeStoryMax);
        isArgMaxCorrectedMeanBarrier[index]++;
        maxCorrectedMeanBarrier[index] += meanBarrier[index] - H * (N - 2 * index - referenceConfMag);

        for (int i = 0; i < magnetizationArraysLength; i += 1)
        {
            meanBarrier[i] = 0.;
            firstBarrier[i] = 0.;
        }

        is++;
    } while (is < nStories);

    for (int i = magnetizationArraysLength - 1; i >= 0; i -= 1)
    { // I could also consider a meanBarrier whose fluctuations are only considered across different stories

        if (cumNum[i] > 0)
        {
            sprintf(buffer, "%i %g %g %g %g %g %g %g %g",
                    -2 * i + N, cumMeanBarrier[i] / (double)nStories, sqrtOrZero((cumMeanBarrierSquared[i] - cumMeanBarrier[i] * cumMeanBarrier[i] / (double)nStories) / (double)(nStories * (nStories - 1))),
                    cumNum[i] / (double)nStories, sqrtOrZero((cumNumSquared[i] - cumNum[i] * cumNum[i] / (double)nStories) / (double)(nStories * (nStories - 1))),
                    cumFirstBarrier[i] / (double)nStories, sqrtOrZero((cumFirstBarrierSquared[i] - cumFirstBarrier[i] * cumFirstBarrier[i] / (double)nStories) / (double)(nStories * (nStories - 1))),
                    logFirstTime[i] / (double)nStories, sqrtOrZero((logFirstTimeSquared[i] - logFirstTime[i] * logFirstTime[i] / (double)nStories) / (double)(nStories * (nStories - 1))));
            thisEpicFile << buffer;
            sprintf(buffer, " %g %g %g %g %g %g %g %g", isArgMaxMeanBarrier[i] / (double)nStories, isArgMaxFirstBarrier[i] / (double)nStories,
                    isArgMaxCorrectedMeanBarrier[i] / (double)nStories, isArgMaxCorrectedFirstBarrier[i] / (double)nStories,
                    maxMeanBarrier[i] / nStories, maxFirstBarrier[i] / nStories, maxCorrectedMeanBarrier[i] / nStories, maxCorrectedFirstBarrier[i] / nStories);
            thisEpicFile << buffer;
            sprintf(buffer, " %g", startedHere[i] / (double)nStories);
            thisEpicFile << buffer << endl;
        }
    }
    thisEpicFile.close();
    cout << "PROGRAMMA FINITO" << endl;
    return 0;
}