#include <iostream>
#include <vector>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <ctime>
//#include "Header.h"

const int GeneratorLayer = 3;

class Generator {
private:
    int CreateNeuronSum() {
        int neuron = 0;
        for (int i = 0; i < GeneratorLayer; i++)
            neuron = neuron + Generator::coat[i];
        return neuron;
    }
    int CreateWeightSum() {
        int weight = 0;
        for (int i = 0; i < (GeneratorLayer - 1); i++)
            weight = weight + Generator::coat[i] * Generator::coat[i + 1];
        return weight;
    }

public:
    int WeightSum, NeuronSum;
    int* coat = new int[GeneratorLayer];

    void CreateCoat(int data, ...) {
        int* p = &data;
        for (int i = 0; i < GeneratorLayer; i++) {
            Generator::coat[i] = *p;
            p++;
            p++;
        }
        Generator::WeightSum = CreateWeightSum();
        Generator::NeuronSum = CreateNeuronSum();
    }

    /*fix this*/
    float* weight = new float[WeightSum];
    float* delw = new float[WeightSum];
    float* out = new float[NeuronSum];

    void Initialization(float max, float min) {
        srand(static_cast<unsigned int>(clock()));
        for (int i = 0; i < Generator::WeightSum; i++) {
            Generator::weight[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
            Generator::delw[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
        }
    }
}Generator;


int main() {
    Generator.CreateCoat(4, 30, 5);
    std::cout << Generator.NeuronSum << std::endl;
    std::cout << Generator.WeightSum << std::endl;
    Generator.Initialization(3, -3);

    for (int i = 0; i < Generator.WeightSum; i++)
        std::cout << Generator.weight[i] << std::endl;

}
