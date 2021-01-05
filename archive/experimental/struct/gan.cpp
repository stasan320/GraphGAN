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
    void SumFunc(float* weight, float* out) {
        int Wnum = 0, Onum = 0;
        for (int i = 0; i < (GeneratorLayer - 1); i++) {
            for (int j = 0; j < Generator::coat[i + 1]; j++) {
                float net = 0;
                for (int k = 0; k < Generator::coat[i]; k++) {
                    net = net + weight[Wnum + j * Generator::coat[i] + k] * out[Onum + k];
                }
                out[Onum + Generator::coat[i] + j] = 1 / (1 + exp(-net));
                //std::cout << out[Onum + Generator::coat[i] + j] << std::endl;
            }
            Wnum = Wnum + Generator::coat[i] * Generator::coat[i + 1];
            Onum = Onum + Generator::coat[i];
        }
    }
    void Delta(float* weight, float* out, float* del) {

    }
public:
    int WeightSum = 0, NeuronSum = 0;
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

    void Initialization(float* weight, float* delw, float max, float min) {
        srand(static_cast<unsigned int>(clock()));
        for (int i = 0; i < Generator::WeightSum; i++) {
            weight[i] = (float)(rand()) / RAND_MAX * (max - min) + min;
            delw[i] = 0;
        }
    }
    void Iteration(float* weight, float* out) {
        SumFunc(weight, out);
    }
}Generator;


int main() {
    Generator.CreateCoat(2, 6, 1);
    float* weight = new float[Generator.WeightSum];
    float* delw = new float[Generator.WeightSum];
    float* out = new float[Generator.NeuronSum];
    float* del = new float[Generator.NeuronSum - Generator.coat[0]];
    out[0] = 1;
    out[1] = 0;
    Generator.Initialization(weight, delw, 3, -3);
    Generator.Iteration(weight, out);
}
