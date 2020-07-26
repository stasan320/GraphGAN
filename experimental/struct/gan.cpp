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
    //float* weight = new float[];
    //std::vector<int> coat;
    int* coat = new int[GeneratorLayer];
    void InitializationCoat(int* Gcoat, int Layer, int coat, ...) {
        int sum = 0;
        int testsum = 0;
        int* p = &coat;
        for (int i = 0; i < Layer; i++) {
            //std::cout << *p << std::endl;
            Gcoat[i] = *p;
            p++;
            p++;  //i dnt know, but its working
        }
    }
}Generator;


int main() {
    Generator.InitializationCoat(Generator.coat, GeneratorLayer, 4, 3, 5);

    for (int i = 0; i < GeneratorLayer; i++)
        std::cout << Generator.coat[i] << std::endl;
}
