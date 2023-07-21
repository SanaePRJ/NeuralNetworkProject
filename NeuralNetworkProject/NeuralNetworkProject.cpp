#include <iostream>
#include <time.h>
#include <random>
#include <fstream>

#include "Matrix.h"
#include "NeuralNetwork.h"

using namespace Sanae;

void View(Matrix _D) {
	std::cout << std::boolalpha;

	for (Ulong i = 0; i < _D.GetSize(); i++)
		std::cout << (_D[i] > 0.5 ? true : false) << " ";
}
bool is_high(double In) {
	return In >= 0.5;
}

void NotCircuit() {
	NN Not         = { 1,2,1,2,(unsigned int)time(0UL)};
	Not.learn_late = 0.05;

	int Accuracy = 0;
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Not.Out(In);

		if (is_high(buf[0]) != In[0])
			Accuracy += 1;
	}

	printf("学習前Not:正答率は%d%%です。\n", Accuracy);

	for (Ulong i = 0; i < 50000; i++) {
		Matrix In    = { {(double)is_high((double)(rand() % 100) / 100)} };
		Matrix Ident = { {(double)!(bool)In[0]} };

		Not.Learn(In, Ident);
	}

	Accuracy = 0;
	for (Ulong i = 0; i < 100; i++) {
		Matrix In  = { {(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Not.Out(In);
		
		if (is_high(buf[0]) != In[0])
			Accuracy += 1;
	}

	printf("Not:正答率は%d%%です。\n",Accuracy);
}
void AndCircuit() {
	NN And         = { 2,4,1,2,(unsigned int)time(0UL) };
	And.learn_late = 0.07;

	int Accuracy = 0;
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = And.Out(In);

		if ((In[0] == In[1]) && In[0] == is_high(buf[0]))
			Accuracy += 1;
	}

	printf("学習前And:正答率は%d%%です。\n", Accuracy);

	for (Ulong i = 0; i < 50000; i++) {
		Matrix In =    { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix Ident = { {(double)((In[0] == In[1])&&In[0])}};
			
		And.Learn(In, Ident);
	}

	Accuracy = 0;
	for (Ulong i = 0; i < 100; i++) {
		Matrix In  = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = And.Out(In);

		if (((In[0] == In[1]) && In[0])== is_high(buf[0]))
			Accuracy += 1;
	}

	printf("And:正答率は%d%%です。\n", Accuracy);
}
void OrCircuit() {
	NN Or         = {2,4,1,1,(unsigned int)time(0UL)};
	Or.learn_late = 0.01;

	int Accuracy = 0;
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Or.Out(In);

		if ((In[0] || In[1]) == is_high(buf[0]))
			Accuracy += 1;
	}

	printf("学習前Or:正答率は%d%%です。\n", Accuracy);

	for (Ulong i = 0; i < 50000; i++) {
		Matrix In    = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix Ident = { {(double)(In[0] || In[1])} };

		Or.Learn(In, Ident);
	}

	Accuracy = 0;
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Or.Out(In);

		if ((In[0] || In[1]) == is_high(buf[0]))
			Accuracy += 1;
	}

	printf("Or:正答率は%d%%です。\n", Accuracy);
}
int main() {
	NotCircuit();
	AndCircuit();
	OrCircuit();
}