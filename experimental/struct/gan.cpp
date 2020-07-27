#include <iostream>
#include <vector>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
//#include "Header.h"

const int GeneratorLayer = 3;

class Generator {
private:
    int InitializationNeuronSum() {
        int neuron = 0;
        for (int i = 0; i < GeneratorLayer; i++)
            neuron = neuron + Generator::coat[i];
        return neuron;
    }
    int InitializationWeightSum() {
        int weight = 0;
        for (int i = 0; i < (GeneratorLayer - 1); i++)
            weight = weight + Generator::coat[i] * Generator::coat[i + 1];
        return weight;
    }
public:
    int WeightSum = 0, NeuronSum = 0;
    int* coat = new int[GeneratorLayer];
    //std::vector<int> coat;
    void CreateCoat(int data, ...) {
        int* p = &data;
        for (int i = 0; i < GeneratorLayer; i++) {
            Generator::coat[i] = *p;
            p++;
            p++;
        }
    }
    void Initialization() {
        Generator::NeuronSum = InitializationNeuronSum();
        Generator::WeightSum = InitializationWeightSum();
    }
    int* weight = new int[WeightSum];
    int* out = new int[NeuronSum];
}Generator;


int main() {
    Generator.CreateCoat(4, 3, 5);
    Generator.Initialization();

    std::cout << Generator.NeuronSum << std::endl;
    std::cout << Generator.WeightSum << std::endl;
}
