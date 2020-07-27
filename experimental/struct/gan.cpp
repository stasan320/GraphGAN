#include <iostream>
#include <vector>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
//#include "Header.h"

const int GeneratorLayer = 3;

class Generator {
private:
    
public:
    int WeightSum = 0, NeuronSum = 0;
    int* coat = new int[GeneratorLayer];
    //std::vector<int> coat;
    void InitializationCoat(int* coat, int Layer, int data, ...) {
        int sum = 0;
        int testsum = 0;
        int* p = &data;
        for (int i = 0; i < Layer; i++) {
            //std::cout << *p << std::endl;
            coat[i] = *p;
            p++;
            p++;
        }
    }
    void InitializationWeight(int Layer, int WeightSum, int NeuronSum, int* coat) {
        for (int i = 0; i < Layer; i++)
            NeuronSum = NeuronSum + coat[i];
        std::cout << NeuronSum << std::endl;
    }
}Generator;


int main() {
    Generator.InitializationCoat(Generator.coat, GeneratorLayer, 4, 30, 5);
    Generator.InitializationWeight(GeneratorLayer, Generator.WeightSum, Generator.NeuronSum, Generator.coat);

    /*for (int i = 0; i < GeneratorLayer; i++)
        std::cout << Generator.coat[i] << std::endl;*/
}
