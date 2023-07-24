#include <iostream>
#include <time.h>
#include <random>
#include <fstream>
#include <string.h>

#include "Matrix.h"
#include "NeuralNetwork.h"


using namespace Sanae;


void RNN_test() {
	RNN test = {3,6,3,(unsigned int)time(0UL)};

	Matrix None = 
	{
		{0},
		{0},
		{0}
	};
	Matrix I =
	{
		{1},
		{0},
		{0}
	};
	Matrix YO =
	{
		{0},
		{1},
		{0}
	};
	Matrix SI =
	{
		{0},
		{0},
		{1}
	};

	for (Ulong i = 0; i < 10000;i++) {
		//いよ->し
		test.In(I);
		test.Learn(YO, SI);

		//いし->よ
		test.In(I);
		test.Learn(SI, YO);

		//よい->し
		test.In(YO);
		test.Learn(I, SI);

		//よし->い
		test.In(YO);
		test.Learn(SI, I);

		//しい->よ
		test.In(SI);
		test.Learn(I, YO);

		//しよ->い
		test.In(SI);
		test.Learn(YO, I);

		//し->よ
		test.In(None);
		test.Learn(SI, YO);

		//よ->し
		test.In(None);
		test.Learn(YO, SI);
	}

	//いし->よ　をテスト
	test.In(I);
	test.Query(SI).View();

	//よ->し　をテスト
	test.In(None);
	test.Query(YO).View();
}

//MNISTデータ学習
void NN_test() {
	NN Machine = {784,392,10,(unsigned int)time(0UL)};
	Machine.Learn_rate = 0.01;

	unsigned char buf  = 0;
	std::string   read = "";

	Matrix image  = SizeT{1,784};
	Matrix Ideal  = SizeT{1,10};
	

	int    point  = -1;
	int    locate = 1;


	std::ifstream ifs;


	ifs.open("mnist_train.csv");
	while (buf != '\n' || locate <= 10000) {
		ifs.read((char*)&buf, sizeof(unsigned char));

		if (buf == ',' or buf == '\n') {
			if (point == -1) {
				for (Ulong i = 0; i < 10; i++)
					Ideal[i] = 0;

				Ideal[atoi(read.c_str())] = 1;
			}
			else {
				image[point] = (double)atoi(read.c_str())/255*0.99+0.01;
			}

			read = "";
			point += 1;
		}
		else
		{
			read += buf;
		}

		if (buf == '\n') {
			locate += 1;
			point   = -1;

			Machine.Learn(image, Ideal);
			
			system("cls");
			printf("%.1lf%%\n", ((double)(locate-1) / 10000) * 100);
		}
	}
	ifs.close();
	ifs.open("mnist_test.csv");

	point  = -1;
	locate = 1;
	printf("学習完了しました。\n");
	system("pause");

	double Accuracy = 0;
	while (buf != '\n' || locate <= 1000) {
		ifs.read((char*)&buf, sizeof(unsigned char));

		if (buf == ',' or buf == '\n') {
			if (point == -1) {
				for (Ulong i = 0; i < 10; i++)
					Ideal[i] = 0;

				Ideal[atoi(read.c_str())] = 1;
			}
			else {
				image[point] = (double)atoi(read.c_str()) / 255 * 0.99 + 0.01;
			}

			read = "";
			point += 1;
		}
		else
		{
			read += buf;
		}

		if (buf == '\n') {
			locate += 1;
			point   = -1;

			Matrix a = Machine.Query(image);

			//最大値を解とする。
			std::pair<double, Ulong> max = { 0,0 };
			for (Ulong i = 0; i < a.GetSize();i++) {
				if (a[i] > max.first)
					max = {a[i],i};
			}

			std::pair<double, Ulong> Ideal_max = { 0,0 };
			for (Ulong i = 0; i < Ideal.GetSize(); i++) {
				if (Ideal[i] > Ideal_max.first)
					Ideal_max = { Ideal[i],i };
			}

			if (max.second==Ideal_max.second) {
				Accuracy += 1;
			}
		}
	}
	printf("正答率は%lf%%です。\n",Accuracy/10);
}


int main() {
	NN_test();
	RNN_test();
}