#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include "../Generic/random.h"
#include "../Generic/fileSystemUtil.h"

#define pfixed 2
#define MCSperSWAP 600

int N, *J;

int main(int argc, char *argv[])
{
    int structureID = 0;
    double fracPosJ = -1.;

    int pmax = 2;
    vector<int> CC;
    vector<int> NumInterac;   

    if (argc<6 && argc!=4)
    {
        fprintf(stderr, "usage: %s type(-4: ErdosRenyFixedPMax) -3: DPRRG -2:ER -1:RRG k>0: k-dim sqLatt) <N> <fracPosJ> (<C(p=2)> (<C(p=3)> ... <C(p=pmax)>) <structureID>)\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int type;
    try
    {
        type = std::stoi(argv[1]);
        if (type == -4 || type == -3 || type == -2 || type == -1 || type >= 0){}
        else{
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
    if (argc>=6) {
    pmax = argc - 4;
    structureID = atof(argv[pmax+3]);
    for (int i = 4; i < argc-1; ++i) {
        CC.push_back(atoi(argv[i]));
    }
    }

    int myrand = init_random(0, 0);

    // Open the file for reading
    std::string fileName;
    char buffer[200];
    if (type == -1)
        sprintf(buffer, "../Data/Graphs/RRG/p%dC%d/N%i/structure%i", pfixed, CC[0], N, structureID);
    else if (type == -2)
        sprintf(buffer, "../Data/Graphs/ER/p%dC%d/N%i/structure%i", pfixed, CC[0], N, structureID);
    else if (type == -3)
        sprintf(buffer, "../Data/Graphs/DPRRG/p%dC%d/N%i/structure%i", pfixed, CC[0], N, structureID);
    else if (type == -4) {
    std::ostringstream path;
    for (int p = 2; p <= pmax; ++p) {
        path << "C" << p << "_" << CC[p - 2];
    }
    sprintf(buffer, "../Data/Graphs/ERPMAX/%s/N%i/structure%i", path.str().c_str(), N, structureID);
    }
    else
    {
        CC[0] = 2 * type;
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

    int num;
    if(type>-4){
        // Read seed, N, p, C (assuming these are the first 4 integers in the file)
        for (int i = 0; i < 4; ++i)
        {
            if (!(file >> num))
            {
                std::cerr << "Error reading from the file\n";
                return 1;
            }
        }
    }
    else{
        for (int i = 0; i < pmax+1; ++i)
        {
            if (!(file >> num))
            {
                std::cerr << "Error reading from the file\n";
                return 1;
            }
        }
    }

    // Read the remaining numbers
    while (file >> num)
    {
        list.push_back(num);
    }

    file.close();

    if(type>-4){
        NumInterac.push_back(list.size() / 2);
    }
    else{
        std::string fileNameInterac;
        fileNameInterac = folder + "/structure_Num_Interactions.txt";
        std::ifstream file(fileNameInterac);
        if (!file.is_open())
        {
            std::cerr << "Error opening the file: " << fileNameInterac << std::endl;
            return 1;
        }
        while (file >> num)
        {
            NumInterac.push_back(num);
        }
        file.close();
    }


    int NumTotInterac = 0;
    for (int p = 2; p <= pmax; ++p) {
        NumTotInterac += NumInterac[p-2];
    }

    J = (int *)calloc(NumTotInterac, sizeof(int));
    int numPosJ = 0;
    for (int i = 0; i < NumTotInterac; i++)
    {
        J[i] = Xrandom() < fracPosJ;
        numPosJ += J[i];
        J[i] = 2 * J[i] - 1;
    }

    folder += ("/fPosJ" + std::to_string(fracPosJ).substr(0, 4));
    folder += ("/graph" + std::to_string(myrand % 10000));
    createFolder(folder);
    fileName = folder + "/graph.txt";
    cout<<fileName;

    std::ofstream fileO(fileName);

    if (!fileO.is_open())
    {
        std::cerr << "Error opening the file: " << fileName << std::endl;
        return 1;
    }

    // Writing N, p, C, M, fracPosJ, numPosJ, myrand
    fileO << N << " " << pmax << " ";
    for (int p = 2; p <= pmax; ++p)
        fileO << CC[p - 2] << " ";
    for (int p = 2; p <= pmax; ++p)
        fileO << NumInterac[p - 2] << " ";
    fileO << fracPosJ << " " << numPosJ << " " << myrand << std::endl;


    // Writing the list and J
    int index = 0; // indice nella lista piatta
    int interactionID = 0; // indice per accedere a J

    for (int p = 2; p <= pmax; ++p) {
        int num_p = NumInterac[p - 2]; // numero di interazioni a p corpi
        for (int i = 0; i < num_p; ++i) {
            for (int j = 0; j < p; ++j) {
                fileO << list[index++] << " ";
            }
            fileO << J[interactionID++] << std::endl;
        }
    }

    fileO.close();

    return EXIT_SUCCESS;
}