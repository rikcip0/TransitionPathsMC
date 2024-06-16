#define NCUT 2
#include "Generic/diag_matrix.h"
#include "Generic/leggifile.h"
#include "straj.h"
#include "interaction.h"

using namespace std;

#ifndef FIELD
#define FIELD

struct neighjump
{
  double t;
  int i;
  int j;
};

class field
{

public:
  double T;
  double beta;
  double Hext;
  vector<double> randomField;

  vector<neighjump> fieldjumps;
  vector<vector<double>> h;
  vector<double> lGamma;
  vector<double> lB;
  //  vector<double> lM;

  vector<matrice> WM;
  vector<matrice> TM;
  vector<int> int_s;

  field(double _T, double _beta, double _Hext, vector<double> _randomField)
  {
    T = _T;
    beta = _beta;
    Hext = _Hext;
    randomField = _randomField;
  }

  straj generate_new_traj(vector<vector<rInteraction>> *neighbors, vector<straj> *traj, double start, double end)
  {
#ifdef FIXEDEXT
    construct_field_FIXEDEXT(neighbors, traj, start, end);
#else
    construct_field(neighbors, traj, start, end);
#endif

    generate_int_spin();
    return generate_traj();
  }

  void print_field(ostream *dove)
  {
    (*dove) << "Printing status" << endl;
    for (int e = 1; e < fieldjumps.size(); e++)
    {
      (*dove) << W10 << fieldjumps[e - 1].t;
      (*dove) << W10 << fieldjumps[e].t;
      for (int i = 0; i < h.size(); i++)
      {
        (*dove) << W10 << h[i][e - 1];
      }
      (*dove) << W10 << lGamma[e - 1];
      (*dove) << W10 << lB[e - 1];
      (*dove) << W10 << WM[e - 1][0][0];
      (*dove) << W10 << WM[e - 1][0][1];
      (*dove) << W10 << WM[e - 1][1][0];
      (*dove) << W10 << WM[e - 1][1][1];
      (*dove) << W10 << int_s[e - 1];
      (*dove) << W10 << int_s[e];
      (*dove) << endl;
    }
  }

  double w(double x) { return exp(-beta * x / 2); } // transition rate, w(x)=w(-x) exp(-beta x)

  void construct_field(vector<vector<rInteraction>> *neighbors, vector<straj> *traj, double hstart, double hend)
  {
    // construct the epochs
    fieldjumps.clear();
    // FZ code: vector<vector<int>> counter = (*neighbors);  //so this is a n+1 long vector of vectors of int, with n=#first nieghbours (RC)

    // RC code to copy neighbors structure
    vector<vector<int>> counter;
    counter.resize((*neighbors).size());
    for (int i = 0; i < counter.size(); i++)
      counter[i].resize((*neighbors)[i].size());
    // end of RC code, back to FZ code

    for (int i = 0; i < counter.size(); i++)
      for (int j = 0; j < counter[i].size(); j++)
        counter[i][j] = 0;

    bool done = false;
    neighjump nexttime;
    nexttime.t = 0;
    nexttime.i = -1;
    nexttime.j = -1;
    fieldjumps.push_back(nexttime);
    while (!done)
    {
      done = true;
      nexttime.t = T;
      nexttime.i = -1;
      nexttime.j = -1;
      for (int i = 0; i < counter.size(); i++)
        for (int j = 0; j < counter[i].size(); j++)
        {
          if (counter[i][j] < (*traj)[(*neighbors)[i][j]].size())
          {
            done = false;
            if ((*traj)[(*neighbors)[i][j]][counter[i][j]] < nexttime.t)
            {
              nexttime.t = (*traj)[(*neighbors)[i][j]][counter[i][j]];
              nexttime.i = i;
              nexttime.j = j;
            }
          }
        }
      if (!done)
        counter[nexttime.i][nexttime.j]++;
      fieldjumps.push_back(nexttime);
    }
    // construct the fields  (counter=currentspin)
    for (int i = 0; i < counter.size(); i++)
      for (int j = 0; j < counter[i].size(); j++)
        counter[i][j] = (*traj)[(*neighbors)[i][j]].s0;
    h.clear();
    h.resize(counter.size());
    lGamma.clear();
    lB.clear();
    //    lM.clear();
    WM.clear();
    TM.clear();
    for (int e = 1; e < fieldjumps.size(); e++)
    { // epochs loop
      double H;
      for (int i = counter.size() - 1; i >= 0; i--)
      { // in such a way that at the end H = h[0]
        H = Hext + randomField[i];
        for (int j = 0; j < counter[i].size(); j++)
          H += counter[i][j] * (*neighbors)[i][j].J; // I think we should include the coupling here (RC)
        h[i].push_back(H);
      }
      double wp = w(2 * H);
      double ex = exp(beta * H); // NDR, h[0], local field on the spin updating, is used only here (and hence in Gamma) (RC)
      double wm = wp * ex * ex;  // so for detailed balance it is now w(-2*H) (RC)
      double Gamma = wp * ex;
      lGamma.push_back(Gamma);
      double ap = -wp;
      double am = -wm;
      for (int i = 0; i < counter[0].size(); i++)
      {
        ap -= w(2 * counter[0][i] * (h[1 + i].back() + (*neighbors)[0][i].J)); // I think we should include the coupling here (RC)
        am -= w(2 * counter[0][i] * (h[1 + i].back() - (*neighbors)[0][i].J)); // I think we should include the coupling here (RC)
      }
      double B = (ap - am) / 2.;
      lB.push_back(B);
      //      double M=(ap+am)/2.;
      //      lM.push_back(M);
      // computation of transfer matrices
      matrice W;
      double wjp = 1; // p stands for plus, m for minus. It is referred to the spin sign (RC)
      double wjm = 1;
      if (fieldjumps[e].i == 0) // i.e., if it s a jump of a first neighbour (RC)
      {
        wjp = w(2 * counter[0][fieldjumps[e].j] * (h[1 + fieldjumps[e].j].back() + (*neighbors)[0][fieldjumps[e].j].J)); // I think we should include the coupling here (RC)
        wjm = w(2 * counter[0][fieldjumps[e].j] * (h[1 + fieldjumps[e].j].back() - (*neighbors)[0][fieldjumps[e].j].J)); // I think we should include the coupling here (RC)
      }
      double lambda = fieldjumps[e].t - fieldjumps[e - 1].t;
      double sqt = sqrt(B * B + Gamma * Gamma);
      double tah = tanh(lambda * sqt);
      // 0 is +, 1 is -; W[s][s'] = W(s -> s') = w(s') <s'| exp(-l L) |s>
      W[0][0] = wjp * (1 + B / sqt * tah);
      W[1][0] = wjp * Gamma / sqt * tah * ex;
      W[0][1] = wjm * Gamma / sqt * tah / ex;
      W[1][1] = wjm * (1 - B / sqt * tah);

      if (fieldjumps.size() == 2)
      {
        double wstart = exp(hstart);
        double wend = exp(hend);
        W[0][0] *= (wstart * wend);
        W[0][1] *= (wstart / wend);
        W[1][0] *= (wend / wstart);
        W[1][1] /= (wstart * wend);
        TM.push_back(W);
      }
      else if (e == 1)
      {
        double wstart = exp(hstart);
        W[0][0] *= wstart;
        W[0][1] *= wstart;
        W[1][0] /= wstart;
        W[1][1] /= wstart;
        TM.push_back(W);
        counter[fieldjumps[e].i][fieldjumps[e].j] *= -1;
      }
      else if (e == fieldjumps.size() - 1)
      {
        double wend = exp(hend);
        W[0][0] *= wend;
        W[1][0] *= wend;
        W[0][1] /= wend;
        W[1][1] /= wend;
        TM.push_back(per(TM.back(), W));
      }
      else
      {
        TM.push_back(per(TM.back(), W));
        counter[fieldjumps[e].i][fieldjumps[e].j] *= -1;
      }
      WM.push_back(W);
    }
  }

  void construct_field_FIXEDEXT(vector<vector<rInteraction>> *neighbors, vector<straj> *traj, double h_start, double h_end)
  {
    // construct the epochs
    fieldjumps.clear();
    // FZ code: vector<vector<int>> counter = (*neighbors);  //so this is a n+1 long vector of vectors of int, with n=#first nieghbours (RC)
    int s_start, s_end;
    if (fabs(h_start) == 1)
    {
      s_start = (int)((1 - h_start) / 2);
    }
    else
    {
      s_start = -10;
    }

    if (fabs(h_end) == 1)
    {
      s_end = (int)((1 - h_end) / 2);
    }
    else
    {
      s_end = -10;
    }
    // RC code to copy neighbors structure
    vector<vector<int>> counter;
    counter.resize((*neighbors).size());
    for (int i = 0; i < counter.size(); i++)
      counter[i].resize((*neighbors)[i].size());
    // end of RC code, back to FZ code

    for (int i = 0; i < counter.size(); i++)
      for (int j = 0; j < counter[i].size(); j++)
        counter[i][j] = 0;
    bool done = false;
    neighjump nexttime;
    nexttime.t = 0;
    nexttime.i = -1;
    nexttime.j = -1;
    fieldjumps.push_back(nexttime);
    while (!done)
    {
      done = true;
      nexttime.t = T;
      nexttime.i = -1;
      nexttime.j = -1;
      for (int i = 0; i < counter.size(); i++)
        for (int j = 0; j < counter[i].size(); j++)
        {
          if (counter[i][j] < (*traj)[(*neighbors)[i][j]].size())
          {
            done = false;
            if ((*traj)[(*neighbors)[i][j]][counter[i][j]] < nexttime.t)
            {
              nexttime.t = (*traj)[(*neighbors)[i][j]][counter[i][j]];
              nexttime.i = i;
              nexttime.j = j;
            }
          }
        }
      if (!done)
        counter[nexttime.i][nexttime.j]++;
      fieldjumps.push_back(nexttime);
    }
    // construct the fields  (counter=currentspin)
    for (int i = 0; i < counter.size(); i++)
      for (int j = 0; j < counter[i].size(); j++)
        counter[i][j] = (*traj)[(*neighbors)[i][j]].s0;
    h.clear();
    h.resize(counter.size());
    lGamma.clear();
    lB.clear();
    //    lM.clear();
    WM.clear();
    TM.clear();
    for (int e = 1; e < fieldjumps.size(); e++)
    { // epochs loop
      double H;
      for (int i = counter.size() - 1; i >= 0; i--)
      { // in such a way that at the end H = h[0]
        H = Hext + randomField[i];
        for (int j = 0; j < counter[i].size(); j++)
          H += counter[i][j] * (*neighbors)[i][j].J; // I think we should include the coupling here (RC)
        h[i].push_back(H);
      }

      double wp = w(2 * H);
      double ex = exp(beta * H); // NDR, h[0], local field on the spin updating, is used only here (and hence in Gamma) (RC)
      double wm = wp * ex * ex;  // so for detailed balance it is now w(-2*H) (RC)
      double Gamma = wp * ex;
      lGamma.push_back(Gamma);
      double ap = -wp;
      double am = -wm;
      for (int i = 0; i < counter[0].size(); i++)
      {
        ap -= w(2 * counter[0][i] * (h[1 + i].back() + (*neighbors)[0][i].J)); // I think we should include the coupling here (RC)
        am -= w(2 * counter[0][i] * (h[1 + i].back() - (*neighbors)[0][i].J)); // I think we should include the coupling here (RC)
      }
      double B = (ap - am) / 2.;
      lB.push_back(B);
      //      double M=(ap+am)/2.;
      //      lM.push_back(M);
      // computation of transfer matrices
      matrice W;
      double wjp = 1; // p stands for plus, m for minus. It is referred to the spin sign (RC)
      double wjm = 1;
      if (fieldjumps[e].i == 0) // i.e., if it s a jump of a first neighbour (RC)
      {
        wjp = w(2 * counter[0][fieldjumps[e].j] * (h[1 + fieldjumps[e].j].back() + (*neighbors)[0][fieldjumps[e].j].J)); // I think we should include the coupling here (RC)
        wjm = w(2 * counter[0][fieldjumps[e].j] * (h[1 + fieldjumps[e].j].back() - (*neighbors)[0][fieldjumps[e].j].J)); // I think we should include the coupling here (RC)
      }
      double lambda = fieldjumps[e].t - fieldjumps[e - 1].t;
      double sqt = sqrt(B * B + Gamma * Gamma);
      double tah = tanh(lambda * sqt);
      // 0 is +, 1 is -; W[s][s'] = W(s -> s') = w(s') <s'| exp(-l L) |s>
      W[0][0] = wjp * (1 + B / sqt * tah);
      W[1][0] = wjp * Gamma / sqt * tah * ex;
      W[0][1] = wjm * Gamma / sqt * tah / ex;
      W[1][1] = wjm * (1 - B / sqt * tah);

      if (fieldjumps.size() == 2)
      {

        double p1start;
        double p1end;
        if (s_start == 0)
        {
          p1start = 1;
        }
        else if (s_start == 1)
        {
          p1start = 0.;
        }
        else
        {
          p1start = 0.5;
        }

        if (s_end == 0)
        {
          p1end = 1;
        }
        else if (s_end == 1)
        {
          p1end = 0.;
        }
        else
        {
          p1end = 0.5;
        }

        W[0][0] *= p1start * p1end;
        W[0][1] *= p1start * (1. - p1end);
        W[1][0] *= (1. - p1start) * p1end;
        W[1][1] *= (1. - p1start) * (1. - p1end);

        TM.push_back(W);
      }
      else if (e == 1)
      {
        if (s_start != -10)
        {

          W[1 - s_start][0] *= 0;
          W[1 - s_start][1] *= 0;
        }

        TM.push_back(W);
        counter[fieldjumps[e].i][fieldjumps[e].j] *= -1;
      }
      else if (e == fieldjumps.size() - 1)
      {
        if (s_end != -10)
        {
          W[0][1 - s_end] *= 0;
          W[1][1 - s_end] *= 0;
        }
        TM.push_back(per(TM.back(), W));
      }
      else
      {
        TM.push_back(per(TM.back(), W));
        counter[fieldjumps[e].i][fieldjumps[e].j] *= -1;
      }
      WM.push_back(W);
    }
  }

  void generate_int_spin()
  {
    // generation of sigma at the boundary of epochs; remember that 0=(+) and 1=(-)
    int_s.resize(fieldjumps.size());
    // generate init and final spin
    {
      double den = TM.back()[0][0] + TM.back()[1][0] + TM.back()[0][1] + TM.back()[1][1];
      int s0;
      int sT;
      double Xr = Xrandom();
      if (Xr < TM.back()[0][0] / den)
      {
        s0 = 0;
        sT = 0;
      }
      else if (Xr < (TM.back()[0][0] + TM.back()[1][0]) / den)
      {
        s0 = 1;
        sT = 0;
      }
      else if (Xr < (TM.back()[0][0] + TM.back()[1][0] + TM.back()[0][1]) / den)
      {
        s0 = 0;
        sT = 1;
      }
      else
      {
        s0 = 1;
        sT = 1;
      }
      int_s[0] = s0;
      int_s.back() = sT;
    }
    // done
    // generate intermediate spins
    double w0, w1;
    for (int e = fieldjumps.size() - 2; e > 0; e--)
    {
      w0 = TM[e - 1][int_s[0]][0] * WM[e][0][int_s[e + 1]];
      w1 = TM[e - 1][int_s[0]][1] * WM[e][1][int_s[e + 1]];
      assert(w0 + w1 > 0);
      int_s[e] = (Xrandom() < w0 / (w0 + w1) ? 0 : 1);
    }
    // end of generation
  }

  double P0(int e, int currents, double lambda)
  {
    double hz = (1 - 2 * currents) * lB[e - 1]; // it is simply sigma*B (RC)
    double sqt = sqrt(lB[e - 1] * lB[e - 1] + lGamma[e - 1] * lGamma[e - 1]);
    //    double sih=sinh(lambda*sqt);
    //    double coh=cosh(lambda*sqt);
    //    return exp(lambda*hz)/( coh + hz/sqt*sih );
    double e1 = exp(-lambda * (sqt - hz));
    double e2 = exp(-2 * lambda * sqt);
    return 2 * e1 / ((1 + hz / sqt) + (1 - hz / sqt) * e2); // it is indeed equiv. to RHS of eq. 35 (RC)
  }

  double draw_time_ss(int e, int currents, double lambda)
  {
    double hz = lB[e - 1] * (1 - 2 * currents); // it is simply sigma*B (RC)
    double hx = lGamma[e - 1];
    double sqt = sqrt(hz * hz + hx * hx);
    //    double a=sqt*cosh(lambda*sqt)+hz*sinh(lambda*sqt);
    double a = sqt * (1 + exp(-2 * lambda * sqt)) + hz * (1 - exp(-2 * lambda * sqt));
    //    double b=exp(hz*lambda)*sqt;
    double b = 2 * sqt * exp(-lambda * (sqt - hz));
    double t_min = 0.;
    double t_max = lambda;
    double dum = Xrandom();
    double t;
    while (t_max - t_min > 0.00000001 * lambda)
    {
      t = .5 * (t_min + t_max);
      //      double bb=exp(hz*t)*(sqt*cosh((lambda-t)*sqt)+hz*sinh((lambda-t)*sqt));
      double bb = (sqt + hz) * exp(-t * (sqt - hz)) + (sqt - hz) * exp(t * (hz + sqt) - 2 * lambda * sqt);
      if ((a - bb) / (a - b) < dum)
        t_min = t;
      else
        t_max = t;
    }
    t = .5 * (t_min + t_max);
    return t;
  }

  double draw_time_sms(int e, int currents, double lambda)
  {
    double hz = lB[e - 1] * (1 - 2 * currents);
    double hx = lGamma[e - 1];
    double sqt = sqrt(hz * hz + hx * hx);
    double t_min = 0.;
    double t_max = lambda;
    double dum = Xrandom();
    double t;
    while (t_max - t_min > 0.00000001 * lambda)
    {
      t = .5 * (t_min + t_max);
      //      double f=1.-(exp(hz*t)*sinh((lambda-t)*sqt)/sinh(lambda*sqt));
      double f = 1. - (exp(-t * (sqt - hz)) - exp(t * (hz + sqt) - 2 * lambda * sqt)) / (1. - exp(-2 * lambda * sqt));
      if (f < dum)
        t_min = t;
      else
        t_max = t;
    }
    t = .5 * (t_min + t_max);
    return t;
  }

  straj generate_traj()
  {
    straj newtraj(1 - 2 * int_s[0], T);
    newtraj.sT = 1 - 2 * int_s.back();
    int currents; // this is still in the 0(+) 1(-) notation (RC)
    int targets;  // this is still in the 0(+) 1(-) notation (RC)
    double length;
    double currenttime;
    for (int e = 1; e < fieldjumps.size(); e++)
    {
      currents = int_s[e - 1];
      targets = int_s[e];
      length = fieldjumps[e].t - fieldjumps[e - 1].t;
      currenttime = fieldjumps[e - 1].t;
      bool undone = true;
      while (undone)
      {
        if (currents == targets)
        {
          double proba = P0(e, currents, length);
          if (Xrandom() < proba)
            undone = false;
          else
          {
            double tjump = draw_time_ss(e, currents, length);
            currenttime += tjump;
            length -= tjump;
            currents = 1 - currents;
            newtraj.push_back(currenttime);
          }
        }
        else
        {
          double tjump = draw_time_sms(e, currents, length);
          currenttime += tjump;
          length -= tjump;
          currents = 1 - currents;
          newtraj.push_back(currenttime);
        }
      }
    }
    return newtraj;
  }
};

#endif
