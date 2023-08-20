#include <iostream>
#include <time.h>
#include <random>
#include <fstream>
#include <string.h>
#include <tuple>


#include "Matrix.h"
#include "ReadCSV.hpp"
#include "NeuralNetwork.hpp"


using namespace Sanae;


//MNISTデータ学習
void NN_test() {
	Ulong LearnCount = 5000; //学習回数
	Ulong testCount  = 10;   //テスト回数


	NN Machine = {784,392,10,0.03,(unsigned int)time(0UL)};

	ReadCSV<double> file = "mnist_train.csv";

	for (Ulong count = 0; count < LearnCount; count++) {
		Matrix image;
		Matrix Ideal = SizeT{ 10,1  };

		//正解データ(onehot表現)
		Ideal[file.ReadDataD()] = 1;
		
		//CSVfileからデータを読み取る。
		std::vector<double> buf; 
		file.ReadLineD(&buf,784);
		for (Ulong i = 0; i < buf.size(); i++)
			buf[i] = buf[i] / 255 * 0.99 + 0.01;

		//データをMatrix型に譲渡
		image.Move(&buf, SizeT{784,1});
		Machine.Learn(image,Ideal);

		if (count%100==0 && count!=0)
			printf("学習:%.2lf%%\n",(double)count/LearnCount*100);
	}
	printf("学習:%.2lf%%\n",(double)100);

	file.open("mnist_test.csv");
	printf("学習完了しました。\n");
	
	double _Accuracy = 0;
	for (Ulong i = 0; i < testCount; i++) {
		Matrix image;

		//正解データ(onehot表現)
		Ulong  Ideal = file.ReadDataD();

		//CSVfileからデータを読み取る。
		std::vector<double> buf;
		file.ReadLineD(&buf, 784);
		for (Ulong i = 0; i < buf.size(); i++)
			buf[i] = buf[i] / 255 * 0.99 + 0.01;

		//データをMatrix型に譲渡
		image.Move(&buf, SizeT{ 784,1 });

		Matrix _predict = Machine.Predict(image);
		
		const std::vector<double>* _buf       = _predict.GetVectorP();
		const Ulong                _Max_Index = std::distance(_buf->begin(),std::max_element(_buf->begin(),_buf->end()));

		if (_Max_Index == Ideal)
			_Accuracy += 1;
	}

	printf("正答率は%lf%%です。\n",_Accuracy/testCount * 100);
}

int main() {
	NN_test();
}