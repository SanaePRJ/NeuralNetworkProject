#include <iostream>
#include <time.h>
#include <random>
#include <fstream>
#include <string.h>

#include "Matrix.h"
#include "NeuralNetwork.h"


using namespace Sanae;


//MNISTデータ学習
int main() {
	NN Machine = {784,784,10,2,(unsigned int)time(0UL)};
	Machine.learn_late = 0.8;

	unsigned char buf  = 0;
	std::string   read = "";

	Matrix image = SizeT{1,784};
	Matrix Idel  = SizeT{1,10};
	

	int    point  = -1;
	int    locate = 1;


	std::ifstream ifs;

	for (Ulong i = 0; i < 1;i++) {
		ifs.open("mnist_train_100.csv");
		while (buf != '\n' || locate <= 100) {
			ifs.read((char*)&buf, sizeof(unsigned char));

			if (buf == ',' or buf == '\n') {
				if (point == -1) {
					for (Ulong i = 0; i < 10; i++)
						Idel[i] = 0;

					Idel[atoi(read.c_str())] = 1;
				}
				else {
					image[point] = (double)atoi(read.c_str()) / 1000;
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
				point = -1;

				Machine.Learn(image, Idel);
			}
		}
		ifs.close();
		ifs.open("mnist_test_10.csv");
	}

	point  = -1;
	locate = 1;
	printf("学習完了しました。\n");
	system("pause");

	while (buf != '\n' || locate <= 10) {
		ifs.read((char*)&buf, sizeof(unsigned char));

		if (buf == ',' or buf == '\n') {
			if (point == -1) {
				for (Ulong i = 0; i < 10; i++)
					Idel[i] = 0;

				Idel[atoi(read.c_str())] = 1;
			}
			else {
				image[point] = (double)atoi(read.c_str()) / 1000;
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

			Matrix a = Machine.Out(image);
			printf("point:%d\n", locate - 1);
			a.View();
			printf("目標値:\n");
			Idel.View();

			system("pause");
		}
	}
}