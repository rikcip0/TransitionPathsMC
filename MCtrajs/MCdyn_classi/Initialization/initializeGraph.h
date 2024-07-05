#include <vector>
#include <algorithm>
#include <iostream>

#include "../interaction.h"
using namespace std;

#ifndef INITIALIZEGRAPH
#define INITIALIZEGRAPH

#define PM1

bool initializeGraph(string &sourceFolder, vector<vector<vector<rInteraction>>> &GraphToReturn, int p, int C, int N, double fracPosJ, vector<double> &randomFieldToReturn, int randomFieldType, int fieldStructureRealization, double signedVar)
{
    vector<vector<vector<rInteraction>>> Graph;
    int n_int, numPosJ, i, j, k, site = 0, *deg, tempInt;
    vector<int> *J, *neigh;
    std::string fileName;
    FILE *file;


    neigh = (vector<int> *)calloc(N, sizeof(vector<int>)); // one array for each spin
    J = (vector<int> *)calloc(N, sizeof(vector<int>));     // one array for each spin
    deg = (int *)calloc(N, sizeof(int));
    numPosJ = (int)(fracPosJ * n_int);

    /*START of graph acquisition*/
    fileName = sourceFolder + "graph.txt";
    file = fopen(fileName.c_str(), "r");

    if (file == NULL)
    {
        perror("Error opening the file");
        std::cout << fileName.c_str() << std::endl;
        return false;
    }
    for (i = 0; i < 3; i++)
    {
        if (fscanf(file, "%d", &tempInt) != 1)
        {
            fprintf(stderr, "Error reading from the file\n");
            fclose(file);
            return false;
        }
    }

    if (fscanf(file, "%d", &n_int) != 1)
    {
        fprintf(stderr, "Error reading from the file\n");
        fclose(file);
        return false;
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

    int *interSites;
    interSites = (int *)calloc(p, sizeof(int));
    int coupl;

    for (i = 0; i < N; i++)
        deg[i] = 0;

    for (i = 0; i < n_int; i++)
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
        sourceFolder += "/randomFieldStructures/";
        if (randomFieldType == 1)
        {
            sourceFolder += "/stdBernoulli/";
        }
        else if (randomFieldType == 2)
        {
            sourceFolder += "/stdGaussian/";
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
                fprintf(stderr, "Error reading from the file\n");
                fclose(file);
                return false;
            }
        }

        if (fscanf(file, "%lf", &temp) != 1)
        {
            fprintf(stderr, "Error reading from the file\n");
            fclose(file);
            return false;
        }

        for (int i = 0; i < 3; i++)
        {
            if (fscanf(file, "%d", &tempInt) != 1)
            {
                fprintf(stderr, "Error reading from the file\n");
                fclose(file);
                return false;
            }
        }

        for (int i = 0; i < N; i++)
        {
            if (fscanf(file, "%lf", &(randomField[i])) != 1)
            {
                fprintf(stderr, "Error reading from the file\n");
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

#endif