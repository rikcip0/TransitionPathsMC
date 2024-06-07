#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "../Generic/random.h"
#include "../Mctrajs/MCdyn_classi/Generic/fileSystemUtil.h"

#define p 2
#define MCSperSWAP 600

/* variabili globali per il generatore random */
int N, M, C, *J;

int main(int argc, char *argv[])
{
    int structureID = 0;
    double fracPosJ = -1.;

    if (argc != 6)
    {
        fprintf(stderr, "usage: %s type(-3: DPRRG -2:ER -1:RRG k>0: k-dim sqLatt) <N> <fracPosJ> (<C> <structureID>)\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int type;
    try
    {
        type = std::stoi(argv[1]);
        if (type == -3 || type == -2 || type == -1 || type >= 0)
        {
            // Input valido
            std::cout << "Input valido: " << type << std::endl;
        }
        else
        {
            // Input non valido
            std::cerr << "Errore: Input non valido." << std::endl;
            return 1;
        }
    }
    catch (...)
    {
        std::cerr << "Errore: Input non valido." << std::endl;
        return 1;
    }

    N = atoi(argv[2]);
    fracPosJ = atof(argv[3]);

    C = atoi(argv[4]);
    structureID = atoi(argv[5]);

    int myrand = init_random(0, 0);

    // Open the file for reading
    std::string fileName;
    char buffer[200];
    if (type == -1)
        sprintf(buffer, "../Data/Graphs/RRG/p%dC%d/N%i/structure%i", p, C, N, structureID);
    else if (type == -2)
        sprintf(buffer, "../Data/Graphs/ER/p%dC%d/N%i/structure%i", p, C, N, structureID);
    else if (type == -3)
        sprintf(buffer, "../Data/Graphs/DPRRG/p%dC%d/N%i/structure%i", p, C, N, structureID);
    else
    {
        C = 2 * type;
        if (type == 1)
            sprintf(buffer, "../Data/Graphs/1dChain/N%i", N);
        else
            sprintf(buffer, "../Data/Graphs/SqLatt/%id/N%i", type, N);
    }

    std::string folder = (std::string)buffer;

    fileName = folder + "/structure.txt";
    std::vector<int> list; // Use vector instead of dynamically allocated array
    std::ifstream file(fileName);

    if (!file.is_open())
    {
        std::cerr << "Error opening the file: " << fileName << std::endl;
        return 1;
    }

    // Read seed, N, p, C (assuming these are the first 4 integers in the file)
    int num;
    for (int i = 0; i < 4; ++i)
    {
        if (!(file >> num))
        {
            std::cerr << "Error reading from the file\n";
            return 1;
        }
    }

    // Read the remaining numbers
    while (file >> num)
    {
        list.push_back(num);
    }

    file.close();

    M = list.size() / 2;

    J = (int *)calloc(M, sizeof(int));
    int numPosJ = 0;
    for (int i = 0; i < M; i++)
    {
        J[i] = Xrandom() < fracPosJ;
        numPosJ += J[i];
        J[i] = 2 * J[i] - 1;
    }

    folder += ("/fPosJ" + std::to_string(fracPosJ).substr(0, 4));
    folder += ("/graph" + std::to_string(myrand % 10000));
    createFolder(folder);

    fileName = folder + "/graph.txt";

    std::ofstream fileO(fileName);

    if (!fileO.is_open())
    {
        std::cerr << "Error opening the file: " << fileName << std::endl;
        return 1;
    }

    // Writing N, p, C, M, fracPosJ, numPosJ, myrand
    fileO << N << " " << p << " " << C << " " << M << " " << fracPosJ << " " << numPosJ << " " << myrand << std::endl;

    // Writing the list and J
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < p; j++)
            fileO << list[i * p + j] << " ";
        fileO << J[i] << std::endl;
    }

    fileO.close();

    return EXIT_SUCCESS;
}
