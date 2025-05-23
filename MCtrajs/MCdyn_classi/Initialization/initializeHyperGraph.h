#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "../interaction.h"

using namespace std;

#ifndef INITIALIZEGRAPH
#define INITIALIZEGRAPH

#define PM1


bool initializeHyperGraph(string &sourceFolder, vector<vector<vector<rInteraction>>> &GraphToReturn, int pmax, int N, double fracPosJ)
{
    if (pmax <= 1) 
    {
        std::cerr << "Error: pmax must be greater than 1. Current value: " << pmax << std::endl;
        return false;
    }
    vector<vector<vector<rInteraction>>> Graph;
    int i, j, k, p, p_ext, l, site = 0, tempInt;
    vector<int> interSites;
    vector<vector<int>> J, neigh;
    J.resize(N);
    neigh.resize(N);
    vector<int> C(pmax-1);
    vector<int> n_int(pmax-1);
    vector<vector<int>> deg;
    deg.resize(N, vector<int>(pmax-1, 0));

    std::string fileName;
    FILE *file;

    /*START of graph acquisition*/
    fileName = sourceFolder + "graph.txt";
    file = fopen(fileName.c_str(), "r");

    if (file == NULL)
    {
        perror("Error opening the file");
        std::cout << fileName.c_str() << std::endl;
        return false;
    }
    for (i = 0; i < 2; i++)
    {
        if (fscanf(file, "%d", &tempInt) != 1)
        {
            fprintf(stderr, "Error reading from the file\n");
            fclose(file);
            return false;
        }
    }
    for (i = 0; i < pmax-1; i++)
    {
        if (fscanf(file, "%d", &C[i]) != 1)
        {
            fprintf(stderr, "Error reading from the file\n");
            fclose(file);
            return false;
        }
    }
    for (i = 0; i < pmax-1; i++)
    {
        if (fscanf(file, "%d", &n_int[i]) != 1)
        {
            fprintf(stderr, "Error reading from the file\n");
            fclose(file);
            return false;
        }
    }

    double temp;
    if (fscanf(file, "%lf", &temp) != 1)
    {
        fprintf(stderr, "Error reading from the file\n");
        fclose(file);
        return false;
    }

    for (i = 0; i < 2; i++)
    {
        if (fscanf(file, "%d", &tempInt) != 1)
        {
            fprintf(stderr, "Error reading from the file\n");
            fclose(file);
            return false;
        }
    }

    int coupl;

    for (i = 0; i < N; i++)
        for (j = 0; j < pmax-1; j++)
            deg[i][j] = 0;

    for (p = 2; p < pmax+1; p++)
    {
        interSites.resize(p);
        for (i = 0; i < n_int[p-2]; i++)
        {
            for (int j = 0; j < p; j++)
                if (fscanf(file, "%d", &interSites[j]) != 1)
                {
                    fprintf(stderr, "Error reading from the file\n");
                    fclose(file);
                    return false;
                }
            if (fscanf(file, "%d", &coupl) != 1)
            {
                fprintf(stderr, "Error reading from the file\n");
                fclose(file);
                return false;
            }
            for (int j = 0; j < p; j++) // For each site in the p-plet
            {
                site = interSites[j];
                for (k = 0; k < p; k++)
                    neigh[site].push_back(interSites[k]); // For each site, in the array neigh[site] we ll hav
                J[site].push_back(coupl);                 // in the C s + topology position of the array I ll have the coupling
                //  of the topology interaction of the s spin
                deg[site][p-2]++;
            }
        }
    }
    /*
// --- INIZIO DEBUG: stampa info per un nodo a scelta ---
int nodeToInspect = 94;  // Cambia questo valore con l'indice che ti serve

// Stampa neigh[nodeToInspect]
std::cout << "neigh[" << nodeToInspect << "]: ";
for (size_t idx = 0; idx < neigh[nodeToInspect].size(); ++idx) {
    std::cout << neigh[nodeToInspect][idx] << " ";
}
std::cout << std::endl;

// Stampa J[nodeToInspect]
std::cout << "J[" << nodeToInspect << "]: ";
for (size_t idx = 0; idx < J[nodeToInspect].size(); ++idx) {
    std::cout << J[nodeToInspect][idx] << " ";
}
std::cout << std::endl;

// Stampa deg[nodeToInspect]
std::cout << "deg[" << nodeToInspect << "]: ";
for (size_t idx = 0; idx < deg[nodeToInspect].size(); ++idx) {
    std::cout << deg[nodeToInspect][idx] << " ";
}
std::cout << std::endl;
// --- FINE DEBUG ---

*/



    int p_plet, counts;
    vector<vector<rInteraction>> app;
    vector<rInteraction> nn;
    vector<int> residuals;

    for (i = 0; i < N; i++)
    {
        app.clear();
        nn.clear();

        for (j = 0; j < 2 * deg[i][0]; j++)
        {
            if (neigh[i][j] != i)
                nn.push_back(rInteraction(J[i][j / 2], neigh[i][j]));
        }
        for (p = 3; p < pmax+1; p++)
        {
            p_plet = 0;
            for (j = deg[i][p-3] * (p-1); j < deg[i][p-3]*(p-1) + p*deg[i][p-2]; j += p)
            {
                residuals.clear();
                for (k = j; k < j + p; k++)
                {
                    if (neigh[i][k] != i)
                        residuals.push_back(neigh[i][k]);   
                }
                nn.push_back(rInteraction(J[i][deg[i][p-3] + p_plet], residuals));
                p_plet++;
            }
        }
        app.push_back(nn);

        vector<int> added_nn;

        for (j = 0; j < 2 * deg[i][0]; j++)
        {
            nn.clear();
            if (neigh[i][j] != i && find(added_nn.begin(), added_nn.end(), neigh[i][j]) == added_nn.end())
            {
                added_nn.push_back(neigh[i][j]);
                for (l = 0; l < 2 * deg[neigh[i][j]][0]; l++)
                {
                    if (neigh[neigh[i][j]][l] != i && neigh[neigh[i][j]][l] != neigh[i][j])
                        nn.push_back(rInteraction(J[neigh[i][j]][l / 2], neigh[neigh[i][j]][l]));

                }

                for (p = 3; p < pmax+1; p++)
                {
                    p_plet = 0;
                    for (l = deg[neigh[i][j]][p-3] * (p-1); l < deg[neigh[i][j]][p-3]*(p-1) + p*deg[neigh[i][j]][p-2]; l += p)
                    {
                        residuals.clear();
                        counts = 0;
                        for (k = l; k < l + p; k++)
                        {
                            if (neigh[neigh[i][j]][k] == i )
                                counts++;
                            else if (neigh[neigh[i][j]][k] != neigh[i][j])
                                residuals.push_back(neigh[neigh[i][j]][k]);    
                        }
                        if (counts == 0)
                            nn.push_back(rInteraction(J[neigh[i][j]][deg[neigh[i][j]][p-3] + p_plet], residuals));
                        p_plet++;

                    }
                } 
                app.push_back(nn);


            }
        }
        for (p_ext = 3; p_ext < pmax+1; p_ext++)
        {   
            
            for (j = deg[i][p_ext-3]*(p_ext-1); j < deg[i][p_ext-3]*(p_ext-1) + p_ext*deg[i][p_ext-2]; j ++)
            {
                
                nn.clear();
                if (neigh[i][j] != i && find(added_nn.begin(), added_nn.end(), neigh[i][j]) == added_nn.end())
                {
                    added_nn.push_back(neigh[i][j]);
                    for (l = 0; l < 2 * deg[neigh[i][j]][0]; l++)
                    {
                        if (neigh[neigh[i][j]][l] != i && neigh[neigh[i][j]][l] != neigh[i][j])
                            nn.push_back(rInteraction(J[neigh[i][j]][l / 2], neigh[neigh[i][j]][l]));
                        /*if(i==0 && j==13 && l==1)
                        {
                            std::cout << "nn for node " << i << ":" << std::endl;
                            for (const auto& inter : nn) {
                                std::cout << "J = " << inter.J << ", Spins = ";
                                for (int s : inter.interS)
                                    std::cout << s << " ";
                                std::cout << std::endl;
                            }
                        }*/
                    }
                    for (p = 3; p < pmax+1; p++){
                        p_plet = 0;
                        for (l = deg[neigh[i][j]][p-3] * (p-1); l < deg[neigh[i][j]][p-3]*(p-1) + p*deg[neigh[i][j]][p-2]; l += p)
                        {
                            residuals.clear();
                            counts = 0;
                            for (k = l; k < l + p; k++)
                            {
                                if (neigh[neigh[i][j]][k] == i )
                                    counts++;
                                else if (neigh[neigh[i][j]][k] != neigh[i][j])
                                    residuals.push_back(neigh[neigh[i][j]][k]);
                            }
                            if (counts == 0)
                                nn.push_back(rInteraction(J[neigh[i][j]][deg[neigh[i][j]][p-3] + p_plet], residuals));
                            p_plet++;
                        }
                    }
                    app.push_back(nn);
                }                 
            }           
        }
        Graph.push_back(app);
    }
    
    // for (i = 0; i < N; i++){                        //Useful to check on degrees
    // if (deg[i] != C && deg[i] != C-1)
    // printf("Weird degrees situation at site %d, with deg = %d\n",i, deg[i]);}

    GraphToReturn = Graph;

    return true;
}

bool initializeLattice(string sourceFolder, vector<vector<vector<rInteraction>>> &GraphToReturn, int dimension, int N, double fracPosJ = 1.)
{
    vector<vector<vector<rInteraction>>> Graph;

    if (dimension == 1)
    { // @ dimension=1
        for (int i = 0; i < N; i++)
        {

            vector<vector<rInteraction>> app;
            vector<rInteraction> nn;
            nn.push_back((i - 1 + N) % N);
            nn.push_back((i + 1) % N);
            app.push_back(nn);
            nn.clear();
            nn.push_back((i - 2 + N) % N);
            app.push_back(nn);
            nn.clear();
            nn.push_back((i + 2) % N);
            app.push_back(nn);
            Graph.push_back(app);
        }
        GraphToReturn = Graph;
        return true;
    }
    else if (dimension == 2)
    { // @ dimension=2
        for (int i = 0; i < N; i++)
        {
            vector<vector<rInteraction>> app;
            int L = sqrt(N);
            if (sqrt((double)N) - L != 0)
            {
                cout << "N value is not allowed!" << endl;
                exit(1);
            }
            int ix = i % L;
            int iy = i / L;
            vector<rInteraction> nn;
            // nearest neighbors
            nn.push_back(L * iy + (ix - 1 + L) % L);
            nn.push_back(L * iy + (ix + 1) % L);
            nn.push_back(L * ((iy - 1 + L) % L) + ix);
            nn.push_back(L * ((iy + 1) % L) + ix);
            app.push_back(nn);
            // nn of ix-1
            nn.clear();
            nn.push_back(L * iy + (ix - 2 + L) % L);
            nn.push_back(L * ((iy - 1 + L) % L) + (ix - 1 + L) % L);
            nn.push_back(L * ((iy + 1) % L) + (ix - 1 + L) % L);
            app.push_back(nn);
            // nn of ix+1
            nn.clear();
            nn.push_back(L * iy + (ix + 2) % L);
            nn.push_back(L * ((iy - 1 + L) % L) + (ix + 1) % L);
            nn.push_back(L * ((iy + 1) % L) + (ix + 1) % L);
            app.push_back(nn);
            // nn of iy-1
            nn.clear();
            nn.push_back(L * ((iy - 2 + L) % L) + ix);
            nn.push_back(L * ((iy - 1 + L) % L) + (ix - 1 + L) % L);
            nn.push_back(L * ((iy - 1 + L) % L) + (ix + 1) % L);
            app.push_back(nn);
            // nn of iy+1
            nn.clear();
            nn.push_back(L * ((iy + 2) % L) + ix);
            nn.push_back(L * ((iy + 1) % L) + (ix - 1 + L) % L);
            nn.push_back(L * ((iy + 1) % L) + (ix + 1) % L);
            app.push_back(nn);
            // end
            Graph.push_back(app);
        }

        GraphToReturn = Graph;
        return true;
    }
    else
    {
        cout << "d= " << dimension << " has not been implemented" << endl;
        return false;
    }
}


bool initializeRealGraph(string &sourceFolder, vector<vector<vector<rInteraction>>> &GraphToReturn, int &N, vector<double> &randomFieldToReturn, int randomFieldType, int fieldStructureRealization, double signedVar)
{
    vector<vector<vector<rInteraction>>> Graph;
    int n_int, i, j, k, site = 0, *deg, tempInt;
    vector<double> *J;
    vector<int>  *neigh;
    std::string fileName;
    FILE *file;

    int p=2;

    /*START of graph acquisition*/
    fileName = sourceFolder + "graph.txt";
    file = fopen(fileName.c_str(), "r");

    if (file == NULL)
    {
        perror("Error opening the file");
        std::cout << fileName.c_str() << std::endl;
        return false;
    }

    if (fscanf(file, "%d", &N) != 1)
    {
        fprintf(stderr, "Error reading from the file1\n");
        fclose(file);
        return false;
    }
    if (fscanf(file, "%d", &n_int) != 1)
    {
        fprintf(stderr, "Error reading from the file1\n");
        fclose(file);
        return false;
    }
    neigh = (vector<int> *)calloc(N, sizeof(vector<int>)); // one array for each spin
    J = (vector<double> *)calloc(N, sizeof(vector<double>));     // one array for each spin
    deg = (int *)calloc(N, sizeof(int));

    double temp;

    int *interSites;
    interSites = (int *)calloc(p, sizeof(int));
    double coupl;

    for (i = 0; i < N; i++)
        deg[i] = 0;

    for (i = 0; i < n_int; i++)
    {
        for (int j = 0; j < p; j++)
            if (fscanf(file, "%d", &interSites[j]) != 1)
            {
                fprintf(stderr, "Error reading from the file4\n");
                fclose(file);
                return false;
            }
        if (fscanf(file, "%lf", &coupl) != 1)
        {
            fprintf(stderr, "Error reading from the file5\n");
            fclose(file);
            return false;
        }
        for (int j = 0; j < p; j++) // For each site in the p-plet
        {
            site = interSites[j];
            for (k = 0; k < p; k++)
                neigh[site].push_back(interSites[k]); // For each site, in the array neigh[site] we ll hav
            J[site].push_back(coupl);                 // in the C s + topology position of the array I ll have the coupling
            //  of the topology interaction of the s spin
            deg[site]++;
        }
    }

    fclose(file);

    for (i = 0; i < N; i++)
    {
        vector<vector<rInteraction>> app;
        vector<rInteraction> nn;
        for (j = 0; j < p * deg[i]; j++)
        {
            if (neigh[i][j] != i)
                nn.push_back(rInteraction(J[i][j / p], neigh[i][j]));
        }
        app.push_back(nn);

        for (j = 0; j < p * deg[i]; j++)
        {
            nn.clear();
            if (neigh[i][j] != i)
            {
                for (int k = 0; k < p * deg[neigh[i][j]]; k++)
                {
                    if (neigh[neigh[i][j]][k] != neigh[i][j] && neigh[neigh[i][j]][k] != i)
                        nn.push_back(rInteraction(J[neigh[i][j]][k / p], neigh[neigh[i][j]][k]));
                }
                app.push_back(nn);
            }
        }
        Graph.push_back(app);
    }
    // for (i = 0; i < N; i++){                        //Useful to check on degrees
    // if (deg[i] != C && deg[i] != C-1)
    // printf("Weird degrees situation at site %d, with deg = %d\n",i, deg[i]);}

    GraphToReturn = Graph;

    // inizializzazione del campo random
    if (randomFieldType != 0)
    {
        sourceFolder += "randomFieldStructures/";
        if (randomFieldType == 1)
        {
            sourceFolder += "stdBernoulli/";
        }
        else if (randomFieldType == 2)
        {
            sourceFolder += "stdGaussian/";
        }
        else
        {
            cout << "randomFieldType in input not implemented" << endl;
        }
        sourceFolder += "realization" + to_string(fieldStructureRealization) + "/";
        string randomFieldFileName = sourceFolder + "field.txt";

        int tempInt;
        double temp;
        vector<double> randomField(N);
        FILE *file;
        file = fopen(randomFieldFileName.c_str(), "r");
        if (file == NULL)
        {
            perror("Error opening the file");
            std::cout << randomFieldFileName.c_str() << std::endl;
            return false;
        }

        for (int i = 0; i < 4; i++)
        {
            if (fscanf(file, "%d", &tempInt) != 1)
            {
                fprintf(stderr, "Error reading from the file6\n");
                fclose(file);
                return false;
            }
        }

        if (fscanf(file, "%lf", &temp) != 1)
        {
            fprintf(stderr, "Error reading from the file7\n");
            fclose(file);
            return false;
        }

        for (int i = 0; i < 3; i++)
        {
            if (fscanf(file, "%d", &tempInt) != 1)
            {
                fprintf(stderr, "Error reading from the file8\n");
                fclose(file);
                return false;
            }
        }

        for (int i = 0; i < N; i++)
        {
            if (fscanf(file, "%lf", &(randomField[i])) != 1)
            {
                fprintf(stderr, "Error reading from the file9\n");
                fclose(file);
                return false;
            }
            randomField[i] *= signedVar;
        }



        fclose(file);
        randomFieldToReturn = randomField;
    }
    else
        randomFieldToReturn.assign(N, 0);

    return true;
}

#endif