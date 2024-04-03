#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

#ifndef INTERACT
#define INTERACT

class rInteraction // which is residual Interaction
{
public:
    int J;              // can be promoted to double
    vector<int> interS; // interacting spins to the interaction (except one)

    // constructors
    rInteraction(int _interactingSpin) // FM pw interaction
    {
        J = 1;
        interS.resize(1);
        interS[0] = _interactingSpin;
    }

    rInteraction(int _J, int _interactingSpin) // Generic pw interaction
    {
        J = _J;
        interS.resize(1);
        interS[0] = _interactingSpin;
    }

    /*
        rInteraction(int _J, vector<int> _interactingSpins) // Generic p-interaction (vector _interactingSpins is then long p-1)
        {
            J = _J;
            interS.resize(1);
            interS = _interactingSpin;
        }
    */

    // action
    operator int() const // useful to not change Zamponi Code
    {
        return interS[0];
    }
};

#endif
