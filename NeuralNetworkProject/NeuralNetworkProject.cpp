#include <iostream>
#include <time.h>
#include <random>

#include "Matrix.h"
#include "NeuralNetwork.h"

using namespace Sanae;

void View(Matrix _D) {
	std::cout << std::boolalpha;

	for (Ulong i = 0; i < _D.GetSize(); i++)
		std::cout << (_D[i] >= 0.5 ? true : false) << " ";
}
bool is_high(double In) {
	return In >= 0.5;
}

void NotCircuit() {
	NN Not= { 1,2,1,1,100 };

	for (Ulong i = 0; i < 100; i++) {
		Matrix In    = { {(double)is_high((double)(rand() % 100) / 100)} };
		Matrix Ident = { {(double)!(bool)In[0]} };

		Not.Learn(In, Ident);
	}

	int Accuracy = 0;
	for (Ulong i = 0; i < 10; i++) {
		Matrix In  = { {(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Not.Out(In);
		
		if (is_high(buf[0]) != In[0])
			Accuracy += 1;
	}

	printf("正答率は%d%%です。\n",Accuracy*10);
}

int main()
{
	NotCircuit();
}