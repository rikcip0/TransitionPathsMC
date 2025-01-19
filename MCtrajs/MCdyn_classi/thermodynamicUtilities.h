#include <vector>
#include <algorithm>
#include <iostream>

#include "../../Generic/FRTGenerator.h"
#include "../../Generic/random.h"
#include "interaction.h"

using namespace std;

#ifndef THERMODYNAMICUTILITIES
#define THERMODYNAMICUTILITIES

bool zeroTemperatureMCSweep_withGraph(vector<int> &s, int N, vector<vector<vector<rInteraction>>> Graph, bool sequential = false)
{
    int I;
    double locField;
    for (int i = 0; i < N; i++)
    {
        I = (int)(Xrandom() * N);
        if (sequential)
            I = i;
        locField = 0; // to be changed in h_ext or h_ext[I]
        for (int j = 0; j < (Graph)[I][0].size(); j++)
        {
            locField += s[Graph[I][0][j]] * Graph[I][0][j].J;
        }
        if (locField * s[I] < 0)
            s[I] = -s[I];
    }
    return true;
}

double energy_Graph(vector<int> s, int N, vector<vector<vector<rInteraction>>> Graph, double h_ext, vector<double> randomField)
{
    double energy = 0.;
    for (int i = 0; i < N; i++)
    {
        energy -= (h_ext + randomField[i]) * s[i];
        for (int j = 0; j < (Graph)[i][0].size(); j++)
        {
            energy -= s[Graph[i][0][j]] * Graph[i][0][j].J * s[i] / 2.;
        }
    }
    return energy;
}


bool MCSweep_withGraph(vector<int> &s, int N, vector<vector<vector<rInteraction>>> Graph, double beta, double h_ext, vector<double> randomField)
{
    int I;
    double locField;
    for (int i = 0; i < N; i++)
    {
        I = (int)(Xrandom() * N);

        locField = h_ext + randomField[I];
        for (int j = 0; j < (Graph)[I][0].size(); j++)
        {
            locField += s[Graph[I][0][j]] * Graph[I][0][j].J;
        }
        if (locField * s[I] < 0 || exp(-2 * beta * locField * s[I]) > Xrandom())
            s[I] = -s[I];
    }
    return true;
}

bool MCSweep_withGraph2(vector<int> &s, vector<int> referenceConf, int N, vector<vector<vector<rInteraction>>> Graph, double beta, int Qstar, double h_ext, vector<double> randomField)
{
    int I;
    double locField;
    int Q = 0;

    for (int i = 0; i < N; i++)
        Q += s[i] * referenceConf[i];

    for (int i = 0; i < N; i++)
    {
        I = (int)(Xrandom() * N);

        locField = h_ext + randomField[I];
        for (int j = 0; j < (Graph)[I][0].size(); j++)
        {
            locField += s[Graph[I][0][j]] * Graph[I][0][j].J;
        }
        if (Q - 2 * s[I] * referenceConf[I] >= Qstar)
            if (locField * s[I] < 0 || exp(-2 * beta * locField * s[I]) > Xrandom())
            {
                Q -= 2 * s[I] * referenceConf[I];
                s[I] = -s[I];
            }
    }
    return true;
}

bool MCSweep_withGraph_variant(vector<int> &s, int N, vector<vector<vector<rInteraction>>> Graph, double beta, double Hext, vector<double> randomField,
                                int &mag, int lowerMeasuredMag, int &nextMeasMag, int magIncrement,  long long int t,
                                vector<double> &logFirstTime, vector<double> &logFirstTimeSquared,
                                vector<double> &firstBarrier, vector<long double> &barrierSum,
                                vector<long long int> &num, vector<int> s_out)
{
    //cout<<"APRO FIUNZ"<<endl;
    int I;
    double locField;
    for (int i = 0; i < N; i++)
    {
        I = (int)(Xrandom() * N);

        locField = Hext + randomField[I];
        for (int j = 0; j < (Graph)[I][0].size(); j++)
        {
            locField += s[Graph[I][0][j]] * Graph[I][0][j].J;
        }
        if (locField * s[I] < 0 || exp(-2 * beta * locField * s[I]) > Xrandom())
        {
            //cout<<"FLIPPO"<<endl;
            s[I] = -s[I];
            mag+=2*s[I]*s_out[I];
            if (mag >= lowerMeasuredMag)
            {
            //cout<<"entro"<<endl;
                if (mag == nextMeasMag)
                {
                    //cout<<"qui 1"<<endl;
                    //printf(" %i %f %lli\n", mag, energy_Graph(s, N, Graph, Hext, randomField), t);
                    //cout<<"qui 2"<<endl;
                    // fflush(stdout);
                    logFirstTime[(N - mag) / 2] += log(t);
                    //cout<<"qui 3"<<endl;
                    logFirstTimeSquared[(N - mag) / 2] += (log(t) * log(t));
                    //cout<<"qui 4"<<endl;
                    firstBarrier[(N - mag) / 2] += (double)(energy_Graph(s, N, Graph, Hext, randomField));
                    //cout<<"qui 5"<<endl;
                    nextMeasMag = mag + magIncrement;
                    //cout<<"qui 6"<<endl;
                }
                    //cout<<"qui 7"<<endl;
                    //cout<<mag<<" "<<N-mag<<endl;
                barrierSum[(N - mag) / 2] += energy_Graph(s, N, Graph, Hext, randomField);
                    //cout<<"qui 8"<<endl;
                num[(N - mag) / 2]++;
                    //cout<<"qui 9"<<endl;
            }
        }
    }
    //cout<<"ESCO"<<endl;
    return true;
}


bool MCSweep_withJs(vector<int> s, int N, double **J, double beta)
{
    int I;
    double locField;

    for (int i = 0; i < N; i++)
    {
        I = (int)(Xrandom() * N);

        locField = 0; // to be changed in h_ext or h_ext[I]
        for (int j = 0; j < N; j++)
        {
            locField += s[j] * J[I][j];
        }
        if (locField * s[I] < 0 || exp(-2. * beta * locField * s[I]) > Xrandom())
            s[I] = -s[I];
    }

    return true;
}

bool computeSelfOverlap_withGraph(int &QToReturn, vector<int> s, vector<vector<vector<rInteraction>>> Graph, double beta, pair<string, string> &details, int N = 0)
{
    int nStories = 1000;

    int Q = 0;
    vector<int> sAtStart = s;
    vector<double> randomField(N, 0);

    if (N != 0 && (N != s.size() || N != Graph.size()))
    {
        cout << "Input N value does not match Graph and/or configuration size" << endl;
        cout << N << endl;
        cout << Graph.size() << endl;
        return false;
    }
    N = s.size();
    int measureSweeps = (int)sqrt(N);
    for (int story = 0; story < nStories; story++)
    {
        s = sAtStart;
        for (int i = 0; i < measureSweeps; i++)
        {
            MCSweep_withGraph(s, N, Graph, beta, 0, randomField);
            // cout << "at the moment Q=" << Q/ ((double)(i + 1)) << endl;
        }
        for (int j = 0; j < N; j++)
            Q += sAtStart[j] * s[j];
    }

    QToReturn = (int)ceil((double)Q / nStories);
    cout << "Self overlap of selected configuration is: " << Q / (double)nStories << endl;
    cout << "Measured with " << measureSweeps << " measureSweeps and " << nStories << " stories." << endl;

    details.first += "Self overlap of selected configuration at dynamycs beta, beta=" + to_string(beta) + " is: " + to_string(Q / (double)nStories) + "\n";
    details.first += "Measured with " + to_string(measureSweeps) + " measureSweeps and " + to_string(nStories) + " stories.\n";
    details.first += "Returning Q=" + to_string(QToReturn) + ".\n\n";

    details.second += "61 " + to_string(beta) + " " + to_string(Q / (double)nStories) + " ";
    details.second += to_string(measureSweeps) + " " + to_string(nStories) + " " + to_string(QToReturn) + "\n";
    return true;
}

bool zeroTemperatureQuench_withGraph(vector<int> &s, vector<vector<vector<rInteraction>>> Graph, pair<string, string> &details, int nSteps, int N, bool sequential = false)
{
    if (N != 0 && (N != s.size() || N != Graph.size()))
    {
        cout << "Input N value does not match Graph and/or configuration size" << endl;
        cout << N << endl;
        cout << Graph.size() << endl;
        return false;
    }
    N = s.size();
    for (int i = 0; i < nSteps; i++)
        zeroTemperatureMCSweep_withGraph(s, N, Graph, sequential);

    return true;
}

bool computeSelfOverlap_withJs(int &QToReturn, vector<int> s, double **J, double beta, int N = 0)
{
    int eqSweeps = 10;
    int nStories = 100;

    int Q = 0;
    vector<int> sAtStart = s;

    if (N != 0 && (N != s.size()))
    {
        cout << "Input N value does not match Graph and/or configuration size" << endl;
        return false;
    }
    N = s.size();
    int measureSweeps = exp(beta * N / 43.);

    for (int story = 0; story < nStories; story++)
    {
        s = sAtStart;
        for (int i = 0; i < eqSweeps; i++)
            MCSweep_withJs(s, N, J, beta);
        for (int i = 0; i < measureSweeps; i++)
        {
            MCSweep_withJs(s, N, J, beta);
            for (int j = 0; j < N; j++)
                Q += sAtStart[j] * s[j];
            cout << "at the moment Q=" << Q / ((double)(i + 1)) << endl;
        }
    }

    QToReturn = ceil((double)Q / measureSweeps / nStories); // so I am passing the ceil of the average overlap
    cout << "Self overlap of selected configuration is: " << Q / (double)measureSweeps / nStories << endl;
    cout << "Measured with " << eqSweeps << " eqSweeps, " << measureSweeps << " measureSweeps and " << nStories << " stories." << endl;
    return true;
}

#endif