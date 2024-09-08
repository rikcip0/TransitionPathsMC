#include <vector>
#include <algorithm>
#include <iostream>

#include "../Generic/fileSystemUtil.h"
#include "../../../Generic/random.h"
#include "../interaction.h"
#include "../straj.h"
#include "../thermodynamicUtilities.h"

#ifndef INITIALIZEREFCONFS
#define INITIALIZEREFCONFS

using namespace std;

bool initializeReferenceConfigurationsFromParTemp_FirstOccurrence(const string sourceFolder, int N, pair<vector<int>, vector<int>> &refConfs, pair<string, string> &details, int desiredOverlap, double desiredBeta, int configurationsChoiceOption)
{

    vector<int> s1 = refConfs.first;
    vector<int> s2 = refConfs.second;
    // We read the Betas used in PT from PTRates.txt
    string fileName = sourceFolder + "configurations/swapRates.txt";
    std::ifstream PTFile(fileName); // Open the file
    if (!PTFile.is_open())
    {
        string fileName = sourceFolder + "configurations/PTInfo.txt";
        PTFile.open(fileName); // Open the file
        if (!PTFile.is_open())
        {
            std::cerr << "Error opening the file." << std::endl;
            cout << fileName << endl;
            exit(1);
        }
    }

    // PTFile.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    std::vector<double> betas;
    double number;
    int count = 0;
    while (PTFile >> number)
    {
        if (!(count % 2) && count > 0)
            betas.push_back(number);
        count++;
    }
    PTFile.close();

    double usedPTBeta;
    if (desiredBeta == -1) // so if there is no desiredBeta in input, the largest beta value is taken automatically
    {
        usedPTBeta = *(std::max_element(betas.begin(), betas.end()));
    }
    else // otherwise the closest one is considered
    {
        for (int i = 0; i < betas.size(); i++)
        {
            if (fabs(betas[i] - desiredBeta) < fabs(desiredBeta - usedPTBeta))
                usedPTBeta = betas[i];
        }
    }
    cout << "Closest beta value to required beta=" << (desiredBeta == 100. ? "inf" : to_string(desiredBeta)) << " in PT data is " << usedPTBeta << endl;

    // We look for configurations with an overlap as similar as possible to what we want
    fileName = sourceFolder + "configurations/B" + to_string(usedPTBeta).substr(0, 4) + "Qs";

    PTFile.open(fileName); // Open the file
    if (!PTFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
        cout << fileName << endl;
        return 1;
    }

    std::string line;
    std::getline(PTFile, line);
    std::getline(PTFile, line);

    int nConfigurations = countNumbersInRow(line);
    cout << "configurations are " << nConfigurations << endl;

    // Rewind the file pointer to the beginning
    PTFile.clear();
    PTFile.seekg(0);

    int usedOverlap = 4 * N, firstConfIndex, secondConfIndex;
    double temp;
    PTFile >> temp;
    count = 0;
    while (PTFile >> number)
    {
        if (abs(number - desiredOverlap) < abs(usedOverlap - desiredOverlap))
        {
            usedOverlap = number;
            firstConfIndex = count / nConfigurations;
            secondConfIndex = count % nConfigurations;
        }
        count++;
    }
    PTFile.close();

    fileName = sourceFolder + "configurations/B" + to_string(usedPTBeta).substr(0, 4) + "confs";

    if (configurationsChoiceOption == -1)
    {

        if (!initializeVectorFromLine(fileName, secondConfIndex + 1, N, s2))
            return false;

        if (!initializeVectorFromLine(fileName, firstConfIndex + 1, N, s1))
            return false;

        cout << "Taking configurations " << firstConfIndex << " and " << secondConfIndex << " from " << fileName << endl;
        cout << "Mutual overlap: " << usedOverlap << endl;
        cout << endl;

        details.first += "Taking configurations " + to_string(firstConfIndex) + " and " + to_string(secondConfIndex) + " from " + fileName + ".\n";
        details.first += "Mutual overlap: " + to_string(usedOverlap) + ".\n\n";
        details.second += "54 " + to_string(firstConfIndex) + " " + to_string(secondConfIndex) + " " + to_string(usedOverlap) + " " + to_string(usedPTBeta) + " " + fileName + "\n";
    }
    else if (configurationsChoiceOption == 1 || configurationsChoiceOption == 2)
    {
        int chosenConfigurationIndex;
        if (configurationsChoiceOption == 1)
            chosenConfigurationIndex = firstConfIndex;
        else
            chosenConfigurationIndex = secondConfIndex;

        if (!initializeVectorFromLine(fileName, chosenConfigurationIndex + 1, N, s1))
            return false;

        for (int i = 0; i < N; i++)
            s2[i] = -s1[i];

        cout << "Considering configurations " << firstConfIndex << " and " << secondConfIndex << " from " << fileName << endl;
        cout << "Mutual overlap: " << usedOverlap << endl;
        cout << endl;

        details.first += "Considering configurations " + to_string(firstConfIndex) + " and " + to_string(secondConfIndex) + " from " + fileName + "\n";
        details.first += "Mutual overlap: " + to_string(usedOverlap) + ".\n";
        details.first += "Taking configurations " + to_string(firstConfIndex) + " and the same one flipped from " + fileName + ".\n";
        details.first += "Mutual overlap: " + to_string(-N) + ".\n\n";
        details.second += "57 " + to_string(firstConfIndex) + " " + to_string(secondConfIndex) + " " + to_string(usedOverlap) + " " + to_string(usedPTBeta) + " " + fileName + "\n";
    }
    else
    {
        return false;
    }

    refConfs = {s1, s2};
    return true;
}

bool initializeSingleConfigurationFromParTemp(const string sourceFolder, int N, vector<int> &conf, pair<string, string> &details, double desiredBeta, int nConfigurationToCopy)
{

    vector<int> s = conf;
    // We read the Betas used in PT from PTRates.txt
    string fileName = sourceFolder + "configurations/PTInfo.txt";
    std::ifstream PTFile(fileName); // Open the file
    if (!PTFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
        cout << fileName << endl;
        exit(1);
    }

    // PTFile.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    std::vector<double> betas;
    double number;
    int count = 0;
    while (PTFile >> number)
    {
        if (!(count % 2) && count > 0)
            betas.push_back(number);
        count++;
    }
    PTFile.close();

    double usedPTBeta;
    if (desiredBeta == -1) // so if there is -1 in input, the largest beta value is taken
    {
        usedPTBeta = *(std::max_element(betas.begin(), betas.end()));
    }
    else // otherwise the closest one is considered
    {
        for (int i = 0; i < betas.size(); i++)
        {
            if (fabs(betas[i] - desiredBeta) < fabs(desiredBeta - usedPTBeta))
                usedPTBeta = betas[i];
        }
    }
    cout << "Closest beta value to required beta=" << (desiredBeta == 100. ? "inf" : to_string(desiredBeta)) << " in PT data is " << usedPTBeta << endl;

    // We look for configurations with an overlap as similar as possible to what we want
    fileName = sourceFolder + "configurations/B" + to_string(usedPTBeta).substr(0, 4) + "Qs";
    int nConfigurations;
    nConfigurations = countLines(fileName) - 1;

    fileName = sourceFolder + "configurations/B" + to_string(usedPTBeta).substr(0, 4) + "confs";
    cout << "Taking configuration " << nConfigurationToCopy << " from " << fileName << endl;
    cout << endl;

    if (!initializeVectorFromLine(fileName, nConfigurationToCopy + 1, N, s))
        return false;
    details.first += "Taking configuration " + to_string(nConfigurationToCopy) + " from " + fileName + ".\n\n";
    details.second += to_string(initRefConfCode) + " " + to_string(nConfigurationToCopy) + " " + to_string(usedPTBeta) + " " + fileName + "\n";

    conf = s;
    return true;
}

bool initializeReferenceConfigurationsFromParTemp_Random(const string sourceFolder, int N, pair<vector<int>, vector<int>> &refConfs, pair<string, string> &details, int desiredOverlap, double desiredBeta = -1.)
{

    vector<int> s1 = refConfs.first;
    vector<int> s2 = refConfs.second;
    // We read the Betas used in PT from PTRates.txt
    string fileName = sourceFolder + "configurations/PTInfo.txt";
    std::ifstream PTFile(fileName); // Open the file
    if (!PTFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
        cout << fileName << endl;
        exit(1);
    }

    // PTFile.ignore(std::numeric_limits<std::streamsize>::max(), ' ');
    std::vector<double> betas;
    double number;
    int count = 0;
    while (PTFile >> number)
    {
        if (!(count % 2) && count > 0)
            betas.push_back(number);
        count++;
    }
    PTFile.close();

    double usedPTBeta;
    if (desiredBeta == -1) // so if there is no desiredBeta in input, the largest beta value is taken automatically
    {
        usedPTBeta = *(std::max_element(betas.begin(), betas.end()));
    }
    else // otherwise the closest one is considered
    {
        for (int i = 0; i < betas.size(); i++)
        {
            if (fabs(betas[i] - desiredBeta) < fabs(desiredBeta - usedPTBeta))
                usedPTBeta = betas[i];
        }
    }
    cout << "Closest beta value to required beta=" << (desiredBeta == 100. ? "inf" : to_string(desiredBeta)) << " in PT data is " << usedPTBeta << endl;

    // We look for configurations with an overlap as similar as possible to what we want
    fileName = sourceFolder + "configurations/B" + to_string(usedPTBeta).substr(0, 4) + "Qs";
    int nConfigurations;
    nConfigurations = countLines(fileName) - 1;

    PTFile.open(fileName); // Open the file
    if (!PTFile.is_open())
    {
        std::cerr << "Error opening the file." << std::endl;
        cout << fileName << endl;
        return 1;
    }

    int usedOverlap = 4 * N, firstConfIndex, secondConfIndex;
    double temp;
    int firstRepetitionCount = 0;
    PTFile >> temp;
    count = 0;
    while (PTFile >> number)
    {
        if (abs(number - desiredOverlap) <= abs(usedOverlap - desiredOverlap))
        {

            if (abs(number - desiredOverlap) < abs(usedOverlap - desiredOverlap)) // so if overlap is repeated often (measured only on firt two appearances) a certain couple is taken less often
            {
                usedOverlap = number;
                firstConfIndex = count / nConfigurations;
                secondConfIndex = count % nConfigurations;

                firstRepetitionCount = 0;
            }
            else
            {
                if (firstRepetitionCount == 0)
                    firstRepetitionCount = count;
                if (Xrandom() < (firstRepetitionCount / (double)nConfigurations / nConfigurations))
                {
                    usedOverlap = number;
                    firstConfIndex = count / nConfigurations;
                    secondConfIndex = count % nConfigurations;
                }
            }
        }
        count++;
    }
    PTFile.close();

    fileName = sourceFolder + "configurations/B" + to_string(usedPTBeta).substr(0, 4) + "confs";
    cout << "Taking configurations " << firstConfIndex << " and " << secondConfIndex << " from " << fileName << endl;
    cout << "Mutual overlap: " << usedOverlap << endl;
    cout << endl;

    details.first += "Taking configurations " + to_string(firstConfIndex) + " and " + to_string(secondConfIndex) + " from " + fileName + ".\n";
    details.first += "Mutual overlap: " + to_string(usedOverlap) + ".\n\n";
    details.second += to_string(initRefConfCode) + " " + to_string(firstConfIndex) + " " + to_string(secondConfIndex) + " " + to_string(usedOverlap) + " " + to_string(usedPTBeta) + " " + fileName + "\n";

    if (!initializeVectorFromLine(fileName, firstConfIndex + 1, N, s1))
        return false;
    if (!initializeVectorFromLine(fileName, secondConfIndex + 1, N, s2))
        return false;

    refConfs = {s1, s2};
    return true;
}

bool initializeReferenceConfigurationsSeqQuenchingFromParTemp(vector<vector<vector<rInteraction>>> graph, const string sourceFolder, int N, pair<vector<int>, vector<int>> &refConfs, pair<string, string> &details, int desiredOverlap, double desiredBeta, int configurationsChoiceOption)
{
    initializeReferenceConfigurationsFromParTemp_FirstOccurrence(sourceFolder, N, refConfs, details, desiredOverlap, desiredBeta, configurationsChoiceOption);
    int nSteps = 200;
    vector<int> s_1 = refConfs.first;
    vector<int> s_2 = refConfs.second;
    int Q11prime = 0, Q22prime = 0, Q1prime2prime = 0;
    zeroTemperatureQuench_withGraph(s_1, graph, details, nSteps, true);
    zeroTemperatureQuench_withGraph(s_2, graph, details, nSteps, true);
    for (int i = 0; i < N; i++)
    {
        Q11prime += refConfs.first[i] * s_1[i];
        Q22prime += refConfs.second[i] * s_2[i];
        Q1prime2prime += s_1[i] * s_2[i];
    }
    cout << "Then configurations have been quenched for " << nSteps << " steps. After that, respective overlap with initial confs are respectively " << (Q11prime) << " and " << Q22prime << ".\n";
    cout << "The mutual overlap is: " << Q1prime2prime << ".\n\n";

    details.first += "Then configurations have been quenched for " + to_string(nSteps) + " steps. After that, respective overlap with initial confs are respectively " + to_string(Q11prime) + " and " + to_string(Q22prime) + ".\n";
    details.first += "The mutual overlap is: " + to_string(Q1prime2prime) + ".\n\n";
    details.second += " " + to_string(nSteps) + " " + to_string(Q11prime) + " " + to_string(Q22prime) + " " + to_string(Q1prime2prime) + "\n";
    refConfs = {s_1, s_2};
    return true;
}

bool initializeFMConfigurations(int N, pair<vector<int>, vector<int>> &refConfs, pair<string, string> &details)
{
    vector<int> s1 = refConfs.first;
    vector<int> s2 = refConfs.second;
    s1.assign(N, -1);
    s2.assign(N, 1);

    details.first += "Taking FM configurations.\n\n";
    details.second += "50\n";
    refConfs = {s1, s2};
    return true;
}

#endif