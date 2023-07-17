#include <iostream>
#include <time.h>
#include <random>

#include "Matrix.h"
#include "NeuralNetwork.h"

using namespace Sanae;

void View(Matrix _D) {
	std::cout << std::boolalpha;

	for (Ulong i = 0; i < _D.GetSize();i++)
		std::cout << (_D[i] >= 0.5 ? true : false) << " ";
}


void NotCircuit() {
	NN Not         = { 2,2, (Ulong)time(0UL)};
	Not.learn_late = 0.7;//100%

	//学習させる
	for (Ulong i = 0; i < 50000; i++) {
		Matrix In =
		{
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 },
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 }
		};
		Matrix Ident
		{
			{(double)!In[0]},
			{(double)!In[1]}
		};

		Not.Learn(In, Ident);
	}

	double Accuracy = 0;

	//結果出力
	for (Ulong i = 0; i < 10;i++) {
		Matrix In =
		{
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 },
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 }
		};
		printf("\n入力:\n");
		View(In);
		printf("\n出力:\n");

		Matrix buf;
		View(buf = Not.Out(In));

		if (In[0] != (buf[0]>=0.5) && In[1] != (buf[1]>=0.5)) {
			printf(":正解\n");
			Accuracy += 1;
		}
		else
		{
			printf(":不正解\n");
		}
	}

	printf("正答率は%d%%でした。\n", (int)Accuracy * 10);
}
void And_Circuit() {
	NN And         = {2,2,(Ulong)time(0UL)};
	And.learn_late = 0.7;

	//学習させる。
	for (Ulong i = 0; i < 50000;i++) {
		Matrix In = 
		{
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 },
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 }
		};
		Matrix Ident
		{
			{In[0] == In[1] ? 1.0 : 0},
			{0}
		};

		And.Learn(In,Ident);
	}

	srand((unsigned int)time(NULL));
	
	double Accuracy = 0;
	//結果出力
	for (Ulong i = 0; i < 10; i++) {
		Matrix In =
		{
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 },
			{((double)(rand() % 100) / 100) >= 0.5 ? 1.0 : 0 }
		};
		printf("\n入力:\n");
		View(In);
		printf("\n出力:\n");
		Matrix buf;
		
		View(buf=And.Out(In));

		if ((In[0] == In[1] && buf[0]) || (In[0] != In[1] && !buf[0]))
		{
			printf(":正解\n");
			Accuracy += 1;
		}
		else
		{
			printf(":不正解\n");
		}
	}

	printf("正答率は%d%%でした。\n",(int)Accuracy*10);
}

int main()
{
	srand(time(0UL));

	//Not回路のモデル
	NotCircuit();
	//And回路のモデル
	And_Circuit();
}