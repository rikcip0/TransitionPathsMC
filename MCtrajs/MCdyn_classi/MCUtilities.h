#include <vector>
#include "straj.h"

#ifndef MCUTILITIES_H
#define MCUTILITIES_H

extern vector<int> s_in, s_out;

void print_conf(vector<straj> *Strajs, double T, string nomefile)
{

    ofstream pfile(nomefile);

    int N = Strajs->size();
    int currents[N];
    int counter[N];

    // init spins
    for (int i = 0; i < N; i++)
    {
        currents[i] = (*Strajs)[i].s0;
        counter[i] = 0;
    }
    pfile << 0;
    for (int i = 0; i < N; i++)
    {
        pfile << " " << currents[i];
    }
    pfile << endl;

    // now start
    bool done = false;
    double t_next = 0;
    int i_next = -1;
    while (!done)
    {
        done = true;
        t_next = T;
        i_next = -1;
        for (int i = 0; i < N; i++)
        {
            if (counter[i] < (*Strajs)[i].size())
            {
                done = false;
                if ((*Strajs)[i][counter[i]] < t_next)
                {
                    t_next = (*Strajs)[i][counter[i]];
                    i_next = i;
                }
            }
        }
        if (!done)
        {
            counter[i_next]++;
            currents[i_next] = -currents[i_next];
            pfile << t_next;
            for (int i = 0; i < N; i++)
            {
                pfile << " " << currents[i];
            }
            pfile << endl;
        }
    }

    pfile.close();
}

double compute_L_av(vector<straj> *Strajs, vector<vector<vector<rInteraction>>> *Graph, double T, double beta, double Hext) // this computes what is colled U in the paper (RC)
{

    int N = Strajs->size();
    int currents[N];
    int currenth[N];
    int counter[N];
    double L = 0;

    // init spins and fields
    for (int i = 0; i < N; i++)
    {
        currents[i] = (*Strajs)[i].s0;
        currenth[i] = Hext;
        for (int j = 0; j < (*Graph)[i][0].size(); j++)
            currenth[i] += (*Strajs)[(*Graph)[i][0][j]].s0 * (*Graph)[i][0][j].J;
        counter[i] = 0;
    }

    // now start
    bool done = false;
    double t_next = 0;
    double t_last = 0;
    int i_next = -1;
    while (!done)
    {
        done = true;
        t_next = T;
        i_next = -1;
        for (int i = 0; i < N; i++)
        {
            if (counter[i] < (*Strajs)[i].size())
            {
                done = false;
                if ((*Strajs)[i][counter[i]] < t_next)
                {
                    t_next = (*Strajs)[i][counter[i]];
                    i_next = i; // so t_next will be the next nearest jump time, and i_next its traj index (RC)
                }
            }
        }
        if (!done)
        {
            counter[i_next]++;
            // compute the jump term of L
            L -= currenth[i_next] * currents[i_next];
        }
        // compute the diagonal term of L
        for (int j = 0; j < N; j++)
            L += (t_next - t_last) * currenth[j] * currents[j] * exp(-beta * currenth[j] * currents[j]);
        if (!done)
        {
            // update the spin and the fields on the neighbors
            currents[i_next] = -currents[i_next];
            for (int j = 0; j < (*Graph)[i_next][0].size(); j++)
                currenth[(*Graph)[i_next][0][j]] += 2 * currents[i_next] * (*Graph)[i_next][0][j].J;
            // register the time
            t_last = t_next;
        }
    }

    return L;
}

vector<double> compute_H_av(vector<straj> *Strajs, vector<vector<vector<rInteraction>>> *Graph, int Np, double T, double Hext) // returns an Np (eq. spaced times) array with the system Hamiltonian value
{

    vector<double> ris(Np, 0);
    int N = Strajs->size();
    int currents[N];
    int currenth[N];
    int counter[N];
    double H = 0;
    // init spins and fields
    for (int i = 0; i < N; i++)
    {
        currents[i] = (*Strajs)[i].s0;
        currenth[i] = Hext;
        for (int j = 0; j < (*Graph)[i][0].size(); j++)
        {
            currenth[i] += (*Strajs)[(*Graph)[i][0][j]].s0 * (*Graph)[i][0][j].J; // Maybe we should include the coupling (RC)
        }
        counter[i] = 0;
        H -= currents[i] * (currenth[i] + Hext) / 2; // /2 on h also (!?)
    }
    // now start
    bool done = false;
    double t_next = 0;
    int i_next = -1;
    int t = 0;
    while (!done)
    {
        done = true;
        t_next = T;
        i_next = -1;
        for (int i = 0; i < N; i++)
        {
            if (counter[i] < (*Strajs)[i].size())
            {
                done = false;
                if ((*Strajs)[i][counter[i]] < t_next)
                {
                    t_next = (*Strajs)[i][counter[i]];
                    i_next = i;
                }
            }
        }
        while (T * t / (Np - 1) <= t_next)
        {
            ris[t] += H;
            t++;
        }
        if (!done)
        {
            // update the spin and the fields on the neighbors
            H += 2 * currenth[i_next] * currents[i_next];
            counter[i_next]++;
            currents[i_next] = -currents[i_next];
            for (int j = 0; j < (*Graph)[i_next][0].size(); j++)
                currenth[(*Graph)[i_next][0][j]] += 2 * currents[i_next] * (*Graph)[i_next][0][j].J;
        }
    }

    for(int i=0;i<ris.size(); i++){
        
    }
    return ris;
}

vector<vector<int>> compute_Q_av(vector<straj> *Strajs, double T, int Np) // returns an Np (eq. spaced times) array with the system overlaps with: initial conf, final conf, 1111 conf
{

    vector<vector<int>> ris(3, vector(Np, 0));

    for (int i = 0; i < Strajs->size(); i++)
    {
        int currents = (*Strajs)[i].s0;
        int t = 0;
        for (int ct = 0; ct < (*Strajs)[i].size(); ct++)
        {
            while (T * t / (Np - 1) <= (*Strajs)[i][ct])
            {
                ris[0][t] += currents * s_in[i];
                ris[1][t] += currents * s_out[i];
                ris[2][t] += currents;
                t++;
            }
            currents *= -1;
        }
        while (t < Np)
        {
            ris[0][t] += currents * s_in[i];
            ris[1][t] += currents * s_out[i];
            ris[2][t] += currents;
            t++;
        }
    }

    return ris;
}

vector<int> compute_Q_fin(vector<straj> *Strajs) // gives final magnetization of a set of traejctories
{

    vector<int> qfin(3, 0);
    for (int i = 0; i < Strajs->size(); i++)
    {
        qfin[0] += (*Strajs)[i].sT * s_in[i];
        qfin[1] += (*Strajs)[i].sT * s_out[i];
        qfin[2] += (*Strajs)[i].sT;
    }
    return qfin;
}

vector<double> count_jumps(vector<straj> *Strajs) // returns a vector containing:
                                                  // #jumps of the trajectory with lowest #jumps,
                                                  // average #jumps, #jumps of the trajectory with highest #jumps (RC)
{

    int jmax = 0; // j stands for #jumps (RC)
    int jmin = 1000000;
    double jmed = 0;
    double j2med = 0;
    int J = 0;
    for (int i = 0; i < Strajs->size(); i++)
    {
        if ((*Strajs)[i].size() > jmax)
            jmax = (*Strajs)[i].size();
        if ((*Strajs)[i].size() < jmin)
            jmin = (*Strajs)[i].size();
        J = (*Strajs)[i].size();
        jmed += J;
        j2med += J * J;
    }

    jmed /= Strajs->size();
    j2med /= Strajs->size();

    vector<double> j;
    j.push_back(jmin);
    j.push_back(jmed);
    j.push_back(sqrt(j2med - jmed * jmed));
    j.push_back(jmax);
    return j;
}

#endif