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

int MaxAN = 0;
int MinAN = 0;
unsigned int MinAN_seed = 0;

void NotCircuit() {
	unsigned int seed_buf = 1689671061;

	NN Not = { 2,2,2,1,seed_buf };
	Not.learn_late = 0.7;

	int Accuracy = 0;
	//学習する前に出力してみる。
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Not.Out(In);

		if (In[0] != is_high(buf[0]) && In[1] != is_high(buf[1]))
			Accuracy += 1;
	}
	printf("Not学習前:正答率は%d%%です。\n", Accuracy);

	//srand((unsigned int)time(NULL));

	//データの学習
	for (Ulong i = 0; i < 10000; i++) {
		//データの作成
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix Ident = { {(double)!is_high(In[0])},{(double)!is_high(In[1])} };

		Not.Learn(In, Ident);
	}

	int old = Accuracy;

	Accuracy = 0;

	//学習結果から出力してみる
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = Not.Out(In);

		if (In[0] != is_high(buf[0]) && In[1] != is_high(buf[1]))
			Accuracy += 1;
	}

	printf("Not学習後:正答率は%d%%です。\n\n", Accuracy);

	//学習率
	Accuracy -= old;
	if (MaxAN < old)
		MaxAN = old;

	if (MinAN > Accuracy) {
		MinAN_seed = seed_buf;
		MinAN = Accuracy;
	}
}

int MaxAA = 0;
int MinAA = 0;
unsigned int MinAA_seed = 0;

void And_Circuit() {
	unsigned int buf_seed = (unsigned int)time(0UL);

	NN And = { 2,2,1,1,buf_seed };
	And.learn_late = 0.9;

	int Accuracy = 0;
	//学習結果から出力してみる
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = And.Out(In);

		if ((In[0] == In[1]) == is_high(buf[0]))
			Accuracy += 1;
	}
	printf("And学習前:正答率は%d%%です。\n", Accuracy);

	//データの学習
	for (Ulong i = 0; i < 50000; i++) {
		//データの作成
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix Ident = { {(double)(is_high(In[0]) == is_high(In[1]))} };

		And.Learn(In, Ident);
	}

	int old = Accuracy;
	Accuracy = 0;

	//学習結果から出力してみる
	for (Ulong i = 0; i < 100; i++) {
		Matrix In = { {(double)is_high((double)(rand() % 100) / 100)},{(double)is_high((double)(rand() % 100) / 100)} };
		Matrix buf = And.Out(In);

		if ((In[0] == In[1]) == is_high(buf[0]))
			Accuracy += 1;
	}

	printf("And学習後:正答率は%d%%です。\n\n", Accuracy);

	//学習率
	Accuracy -= old;
	if (MaxAA < Accuracy)
		MaxAA = Accuracy;

	if (MinAA > Accuracy) {
		MinAA_seed = buf_seed;
		MinAA = Accuracy;
	}
}

int main()
{
	srand((unsigned int)time(0UL));
	while (1) {
		//Not回路のモデル
		//NotCircuit();
		//And回路のモデル
		And_Circuit();

		printf("最大学習率:%d,最小学習率:%d,シード:%u\n", MaxAA, MinAA, MinAA_seed);
	}
}