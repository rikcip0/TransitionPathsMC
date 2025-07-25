#include <vector>
#include <algorithm>
#include <iostream>

#include "../interaction.h"
#include "../../../Generic/random.h"
#include "../straj.h"

#include "../field.h"
#include "../MCUtilities.h"

using namespace std;

#ifndef INITIALIZETRAJS
#define INITIALIZETRAJS

#define startingBeta 0.2
#define deltaBetaStd 0.01
#define MCeqOVMCToArrive 5
#define MCeqOvMCAnnealingStart 500

bool initializeTrajectoriesFromRefConfs(int N, double T, vector<straj> &trajsToInitialize, pair<string, string> &details, pair<vector<int>, vector<int>> refConfs, bool swapConfs = false)
{
    vector<straj> Strajs;
    vector<int> s1(N, 0);
    vector<int> s2(N, 0);

    if (swapConfs)
    {
        s1 = refConfs.second;
        s2 = refConfs.first;
    }
    else
    {
        s1 = refConfs.first;
        s2 = refConfs.second;
    }

    for (int i = 0; i < N; i++)
    {
        straj app(s1[i], T);
        app.sT = s2[i];
        if (app.s0 != app.sT)
            app.push_back(T * Xrandom());
        Strajs.push_back(app);
    }
    trajsToInitialize = Strajs;

    if (swapConfs)
    {
        details.first += "Spin values at extremes initialized eq. to provided configurations SWAPPED. 1 or 0 jumps with uniform prob over time.\n\n";
    }
    else
    {
        details.first += "Spin values at extremes initialized eq. to provided configurations. 1 or 0 jumps with uniform prob over time.\n\n";
    }

    details.second += to_string(initTrajCCode) + "\n";
    details.second += to_string(initTrajJCode) + "\n";

    return true;
}

bool initializeTrajectoriesFromRefConfs_WithAnnealing(int N, double T, vector<straj> &trajsToInitialize, double beta, double Hext,
                                                      vector<vector<vector<rInteraction>>> Graph, double hin, double hout, int Qstar,
                                                      pair<string, string> &details, pair<vector<int>, vector<int>> refConfs, const vector<double> randomField,
                                                      bool swapConfs = false)
{
    vector<straj> Strajs;
    vector<int> s1(N, 0);
    vector<int> s2(N, 0);

    if (swapConfs)
    {
        s1 = refConfs.second;
        s2 = refConfs.first;
    }
    else
    {
        s1 = refConfs.first;
        s2 = refConfs.second;
    }

    for (int i = 0; i < N; i++)
    {
        straj app(s1[i], T);
        app.sT = s2[i];
        if (app.s0 != app.sT)
            app.push_back(T * Xrandom());
        Strajs.push_back(app);
    }

    s1 = refConfs.first;
    s2 = refConfs.second;

    int Qfin = compute_Q_fin(&Strajs, s1, s2)[1]; // It should be N..., but why not to check!?
    double betaToStart = startingBeta;
    double deltaBeta=deltaBetaStd;
    if (beta-betaToStart<10*deltaBeta)
    {
        betaToStart=betaToStart-10*deltaBeta;
        if (betaToStart<0.){
            betaToStart=0.;
        }
    }
    double annealingBeta = betaToStart;
    int MCAtBetaStart = MCeq / MCeqOvMCAnnealingStart;
    if (MCAtBetaStart == 0)
    {

        MCAtBetaStart = 1;
    }

    int MCToArrive = MCeq / MCeqOVMCToArrive;
    double factor = pow((((double)MCToArrive) / MCAtBetaStart), (deltaBeta) / (double)(beta - betaToStart));
    double MCAtBetaAnn = MCAtBetaStart;

    for (; annealingBeta <= beta + 0.0001; annealingBeta += deltaBeta)
    {
        field f(T, annealingBeta, Hext, randomField);

        for (int j = 0; j < MCAtBetaAnn; j++)
        {
            for (int i = 0; i < N; i++)
            {
                int I = (int)(N * Xrandom());
                Qfin -= Strajs[I].sT * s_out[I];
                Strajs[I] = f.generate_new_traj(&Graph[I], &Strajs, hin * s_in[I], Qfin > Qstar ? 0 : hout * s_out[I], I);
                Qfin += Strajs[I].sT * s_out[I];
            }
        }
        cout << "Annealing at beta=" << annealingBeta << " for " << MCAtBetaAnn << " mcSweeps" << endl;
        MCAtBetaAnn *= factor;
    }

    trajsToInitialize = Strajs;
    details.second += to_string(initTrajCCode) + " " + to_string(betaToStart) + " " + to_string(MCAtBetaStart) + " " + to_string(MCToArrive) + " " + to_string(deltaBeta) + "\n";
    return true;
}

bool initializeTrajectoriesFromRefConfs_WithAnnealing_FixingEnd(int N, double T, vector<straj> &trajsToInitialize, double beta, double Hext,
                                                                vector<vector<vector<rInteraction>>> Graph, double hin, double hout,
                                                                pair<string, string> &details, pair<vector<int>, vector<int>> refConfs, const vector<double> randomField,
                                                                bool swapConfs = false)
{
    vector<straj> Strajs;
    vector<int> s1(N, 0);
    vector<int> s2(N, 0);

    if (swapConfs)
    {
        s1 = refConfs.second;
        s2 = refConfs.first;
    }
    else
    {
        s1 = refConfs.first;
        s2 = refConfs.second;
    }

    for (int i = 0; i < N; i++)
    {
        straj app(s1[i], T);
        app.sT = s2[i];
        if (app.s0 != app.sT)
            app.push_back(T * Xrandom());
        Strajs.push_back(app);
    }

    s1 = refConfs.first;
    s2 = refConfs.second;

    int Qfin = compute_Q_fin(&Strajs, s1, s2)[1]; // It should be N..., but why not to check!?

    double betaToStart = startingBeta;
    double deltaBeta=deltaBetaStd;
    if (beta-betaToStart<10*deltaBeta)
    {
        betaToStart=betaToStart-10*deltaBeta;
        if (betaToStart<0.){
            betaToStart=0.;
        }
    }
    double annealingBeta = betaToStart;
    int MCAtBetaStart = MCeq / MCeqOvMCAnnealingStart;
    if (MCAtBetaStart == 0)
        MCAtBetaStart = 1;
    int MCToArrive = MCeq / MCeqOVMCToArrive;
    double factor = pow((((double)MCToArrive) / MCAtBetaStart), (deltaBeta) / (double)(beta - betaToStart));
    double MCAtBetaAnn = MCAtBetaStart;

    for (; annealingBeta <= beta + 0.0001; annealingBeta += deltaBeta)
    {
        field f(T, annealingBeta, Hext, randomField);

        for (int j = 0; j < MCAtBetaAnn; j++)
        {
            for (int i = 0; i < N; i++)
            {
                int I = (int)(N * Xrandom());
                Qfin -= Strajs[I].sT * s_out[I];
                Strajs[I] = f.generate_new_traj(&Graph[I], &Strajs, hin * s_in[I], hout * s_out[I], I);
                Qfin += Strajs[I].sT * s_out[I];
            }
        }
        cout << "Annealing at beta=" << annealingBeta << " for " << MCAtBetaAnn << " mcSweeps" << endl;
        MCAtBetaAnn *= factor;
    }

    trajsToInitialize = Strajs;
    details.second += to_string(initTrajCCode) + " " + to_string(betaToStart) + " " + to_string(MCAtBetaStart) + " " + to_string(MCToArrive) + " " + to_string(deltaBeta) + "\n";
    return true;
}

bool callAnnealing(int N, double T, vector<straj> &trajsToInitialize, double beta, double Hext,
                   vector<vector<vector<rInteraction>>> Graph, double hin, double hout, int Qstar,
                   pair<string, string> &details, pair<vector<int>, vector<int>> refConfs, const vector<double> randomField,
                   bool swapConfs = false)
{
    bool allOk;
    if (initTrajCCode == 74)
    {
        allOk = initializeTrajectoriesFromRefConfs_WithAnnealing(N, T, trajsToInitialize, beta, Hext, Graph, hin, hout, Qstar, details, refConfs, randomField, swapConfs);
    }
    else if (initTrajCCode == 740)
    {
        allOk = initializeTrajectoriesFromRefConfs_WithAnnealing_FixingEnd(N, T, trajsToInitialize, beta, Hext, Graph, hin, hout, details, refConfs, randomField, swapConfs);
    }
    return allOk;
}

// if noise=0.5, is equivalent to Initialize at random
bool initializeTrajectoriesFromRefConfs_withNoise(int N, double T, vector<straj> &trajsToInitialize, pair<string, string> &details, double noise, pair<vector<int>, vector<int>> refConfs, bool swapConfs = false)
{
    if (noise < 0. || noise > 0.5)
    {
        cout << "Noise should be in the range [0.,0.5]" << endl;
        return false;
    }

    vector<straj> Strajs;
    vector<int> s1(N, 0);
    vector<int> s2(N, 0);

    if (swapConfs)
    {
        s1 = refConfs.second;
        s2 = refConfs.first;
    }
    else
    {
        s1 = refConfs.second;
        s2 = refConfs.second;
    }

    if (s1.empty())
        for (int i = 0; i < N; i++)
            Xrandom() > noise ? s1[i] : -s1[i];
    if (s2.empty())
        for (int i = 0; i < N; i++)
            Xrandom() > noise ? s2[i] : -s2[i];

    for (int i = 0; i < N; i++)
    {
        straj app(s1[i], T);
        app.sT = s2[i];
        if (app.s0 != app.sT)
            app.push_back(T * Xrandom());
        Strajs.push_back(app);
    }
    trajsToInitialize = Strajs;

    if (swapConfs)
    {
        details.first += "Spin values at extremes initialized different to provided configurations SWAPPED with prob=" + to_string(noise) + ". 1 or 0 jumps with uniform prob over time.\n\n";
        details.second += "720 " + to_string(noise) + "\n";
    }
    else
    {
        details.first += "Spin values at extremes initialized different to provided configurations with prob=" + to_string(noise) + ". 1 or 0 jumps with uniform prob over time.\n\n";
        details.second += "710 " + to_string(noise) + "\n";
    }
    details.second += "80\n"; // specifies jumps initializ choice

    return true;
}

bool initializeTrajectoriesAtRandom(int N, double T, vector<straj> &trajsToInitialize, pair<string, string> &details)
{
    vector<straj> Strajs;

    for (int i = 0; i < N; i++)
    {
        straj app(Xrandom() > 0.5, T);
        app.sT = Xrandom() > 0.5;
        if (app.s0 != app.sT)
            app.push_back(T * Xrandom());
        Strajs.push_back(app);
    }
    trajsToInitialize = Strajs;
    details.first += "Spin values at extremes initialized random. 1 or 0 jumps with uniform prob over time.\n\n";
    details.second += to_string(initTrajCCode) + "\n";
    details.second += to_string(initTrajJCode) + "\n";
    return true;
}

#endif