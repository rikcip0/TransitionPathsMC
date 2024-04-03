#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

#ifndef STRAJ
#define STRAJ

// I think we don t need to modify it for our purposes, except maybe for constructor methods.

class straj
{
public:
  int s0;
  int sT;
  double T;
  vector<double> jumps;

  // constructors
  straj()
  {
    s0 = 1;
    T = 1;
  }
  straj(double _T)
  {
    s0 = 1;
    T = _T;
  }
  straj(int _s0)
  {
    s0 = _s0;
    T = 1;
  }
  straj(int _s0, double _T)
  {
    s0 = _s0;
    T = _T;
  }
  // actions
  void set_s0(int _s0) { s0 = _s0; }
  void resize(int n) { jumps.resize(n); }
  void insert_front(double _tj) { jumps.insert(jumps.begin(), _tj); }
  void push_back(double _tj) { jumps.push_back(_tj); }
  void push_back(straj t)
  {
    jumps.insert(jumps.end(), t.begin(), t.end());
  }
  void clear() { jumps.clear(); }
  void Sort()
  {
    sort(jumps.begin(), jumps.end());
  }
  // output
  int size() { return jumps.size(); }
  double operator[](int i) { return jumps[i]; }
  double back() { return jumps.back(); }
  vector<double>::iterator begin() { return jumps.begin(); }
  vector<double>::iterator end() { return jumps.end(); }
  double s_av() // returns the spin average value over the trajectory (RC)
  {
    if (jumps.empty())
      return s0;
    else
    {
      double ris = s0 * jumps[0];
      int currents = s0;
      for (int i = 1; i < size(); i++)
      {
        currents *= -1;
        ris += currents * (jumps[i] - jumps[i - 1]);
      }
      currents *= -1;
      ris += currents * (T - jumps.back());
      return ris / T;
    }
  }
  friend ostream &operator<<(ostream &o, const straj &a)
  {
    int currentn = a.s0;
    o << "0 " << currentn << "\n";
    for (int i = 0; i < a.jumps.size(); i++)
    {
      o << a.jumps[i] << " " << currentn << "\n";
      currentn *= -1;
      o << a.jumps[i] << " " << currentn << "\n";
    }
    o << a.T << " " << currentn << "\n";
    return o;
  }
};

#endif
