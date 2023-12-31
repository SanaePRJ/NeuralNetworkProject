﻿#include <iostream>
#include <time.h>
#include <random>
#include <fstream>
#include <string.h>
#include <tuple>


#include "Plot.hpp"
#include "Matrix.hpp"
#include "CSV.hpp"
#include "NeuralNetwork.hpp"


using namespace Sanae;


Ulong LearnCount = 6000;   //学習回数
Ulong testCount  = 10000;   //テスト回数

void   NN_Learn(NN* Machine,Ulong);
void   NN_test (NN* Machine);


void NN_Learn(NN* Machine,Ulong through = 0) {
	ReadCSV file = "mnist_train.csv";
	
	for (Ulong i = 0; i < through;i++) {
		std::vector<std::string> buf_str;
		file.ReadLine(&buf_str,785);
	}

	auto ChangeAngleLearn = [&](Matrix& image,Matrix& Ideal,double rad)
	{
		Matrix image_buf;

		//傾けた画像を生成するために28*28の画像に変換
		image.Deformation({ 28,28 });

		//傾ける。
		Change_Angle(image, image_buf, rad);
		//学習させるために元の行列に変換
		image_buf.Deformation({784,1});

		//学習
		Machine->Learn(image_buf,Ideal);

		//元の行列に変換
		image.Deformation({ 784,1 });
	};
	auto Read = [&](Matrix& image, Matrix& Ideal)
	{
		//正解データ(onehot表現)
		Ideal[std::stoull(file.ReadData())] = 1;

		//CSVfileからデータを読み取る。
		std::vector<std::string> buf_str;
		std::vector<double>      buf;

		file.ReadLine(&buf_str, 784);
		buf.resize(buf_str.size());

		for (Ulong i = 0; i < buf_str.size();i++)
			buf[i] = std::stod(buf_str[i]);

		//正規化
		for (Ulong i = 0; i < buf.size(); i++)
			buf[i] = buf[i] / 255 * 0.99 + 0.01;

		//データをMatrix型に譲渡
		image.Move(&buf, SizeT{ 784,1 });
	};

	for (Ulong count = 0; count < LearnCount; count++) {
		Matrix image;
		Matrix Ideal = SizeT{ 10,1 };

		Read(image, Ideal);
		Machine->Learn(image, Ideal);
		
		//10°傾けて学習
		ChangeAngleLearn(image, Ideal, 10);
		ChangeAngleLearn(image, Ideal, -10);
		
		if (count % 100 == 0)
			printf("学習進捗:%.2lf\n", (double)count / (double)LearnCount * 100);
	}

	printf("学習完了しました。");
}

//MNISTデータ学習
void NN_test(NN* Machine) {
	ReadCSV file = "mnist_test.csv";
	
	auto Read = [&](Matrix& image,Ulong& Ideal) {
		//正解データ
		Ideal = std::stoul(file.ReadData());

		//CSVfileからデータを読み取る。
		std::vector<std::string> buf_str;
		std::vector<double>      buf;

		file.ReadLine(&buf_str, 784);
		buf.resize(buf_str.size());

		for (Ulong i = 0; i < buf_str.size(); i++)
			buf[i] = std::stod(buf_str[i]);

		//正規化
		for (Ulong v = 0; v < buf.size(); v++)
			buf[v] = buf[v] / 255 * 0.99 + 0.01;

		//データをMatrix型に譲渡
		image.Move(&buf, SizeT{ 784,1 });
	};

	double _Accuracy = 0;
	for (Ulong i = 0; i < testCount; i++) {
		Matrix image;
		Ulong  Ideal = 0;

		//読み取る
		Read(image,Ideal);

		//予測
		Matrix _predict = Machine->Predict(image);
		
		//出力値のポインタを取得
		const std::vector<double>* _buf       = _predict.GetVectorP();
		//最大値のインデックスが答え
		const Ulong                _Max_Index = std::distance(_buf->begin(),std::max_element(_buf->begin(),_buf->end()));

		//正解の場合
		if (_Max_Index == Ideal)
			_Accuracy += 1;
	}

	printf("%.1lf%%\n",_Accuracy/testCount * 100);
	return;
}


int main() {
	Console << Color_Console{Color{0, 0, 0}, Color{ 0,255,0 }};
	printf("Copyright 2023 SanaeProject.\n");
	Console.Reset();


	NN Machine = { 784,20,10,0.03,(unsigned int)time(0UL) };

	//過去の重みを入力
	try {
		Machine.Read("MakeNum2.csv");
	}
	catch (std::exception) 
	{}

	//学習
	NN_Learn(&Machine,6000);
	//正答率
	NN_test (&Machine);

	printf("記録しますか？\nyes:y no:その他キー\n");
	if (_getwch()=='y'){
		printf("記録します。\n");
		Machine.Write("MakeNum2.csv");
	}
}