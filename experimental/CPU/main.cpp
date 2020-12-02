#include <iostream>
#include <fstream>

int main() {
	float weight1 = 0.3, weight2 = 0.9, data;

	for (int i = 0; i < 1000; i++) {
		//generator
		float input = 0.2, outO = 0.54423;

		float out = 1/(1+exp(-(input * weight1 + 1)));
		std::cout << out<< std::endl;

		float del = outO - out;
		weight1 = weight1 + del;

		//discriminator
		float out2 = 1/(1+exp(-(1 * weight2 + 1)));
		float del2 = 0.332 - out2;
		weight2 = weight2 + del2;
		std::cout << out2 << std::endl << std::endl;

	}
	return 0;
}
