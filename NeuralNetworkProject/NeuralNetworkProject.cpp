#include <iostream>
#include <random>
#include <time.h>

#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"
#include "CSV.hpp"
#include "Plot.hpp"
#include "NeuralNetwork.hpp"


int main() {
	srand(time(NULL));

	Sanae::ReadCSV    mnist("mnist_train.csv");
	//入力ノード数:784,中間層ノード数:50,出力層ノード数:10,学習率:30%
	Sanae::NN<double> Machine(784,200,10,0.3,rand());


	//mnistデータは28*28
	const uint64_t MnistSize     = 784;
	//batchサイズ
	uint64_t       batch         = 50;

	//学習回数
	uint64_t       LearnCount    = 500;
	//予測回数
	uint64_t       PredictCount  = 500;


	//Mnistデータの答えと画像データをCSVファイルから読み取る。
	auto get_mnist = [&mnist, &Machine,&MnistSize]() -> std::pair<double, std::vector<double>>{
			//答え
			double Answer = std::stoi(mnist.ReadData());
			
			//画像データを読み込む
			std::vector<double> Question(MnistSize,0);
			for (uint64_t i = 0; i < MnistSize; i++)
				Question[i] = (std::stod(mnist.ReadData()) / 255 * 0.99 + 0.0001);   //正規化

			//答え,画像データをpairで送る。
			return std::pair<double, std::vector<double>>{Answer,Question};
		};

	//学習させる。
	auto Learn = [&get_mnist,&mnist,&Machine,&batch,&MnistSize]() {
			Sanae::Matrix<double> A = std::pair<size_t, size_t>{batch,0};   //問題
			Sanae::Matrix<double> Q = std::pair<size_t, size_t>{batch,0};   //答え

			//batchサイズ分だけ読み取る。
			for (size_t i = 0; i < batch;i++) {
				std::pair<double, std::vector<double>> MnistData = get_mnist();   //Mnistデータ読み取り

				//答えを配列にする。3->{0,0,0,1,0,0,0,0,0,0}
				std::vector<double> buf_A(10,0);
				buf_A[static_cast<uint64_t>(MnistData.first)] = 1;

				A[i] = { buf_A };
				Q[i] = { MnistData.second };
			}

			//学習->返り値として損失関数の返り値を返す。
			return Machine.Learn(Q,A);
		};

	//予測
	auto predict = [&mnist, &Machine,&get_mnist,&MnistSize]() {
			//mnistデータ読み込み
			std::pair<double, std::vector<double>> MnistData = get_mnist();
			
			//画像データを格納
			Sanae::Matrix<double> Q = std::pair<size_t, size_t>{1,0};
			Q[0] = MnistData.second;

			//予測結果をAに格納
			Sanae::Matrix<double> A = Machine.Predict(Q);

			//最大値が答えであるため最大値のあるindexを予測結果とする。
			std::vector<double>::iterator iter = std::max_element(A[0].begin(),A[0].end());
			size_t index = std::distance(A[0].begin(),iter);

			//正解であった場合1,不正解であった場合は0を返す。
			if (index != MnistData.first)
				return 0;
			else
				return 1;
		};

	//学習させる
	for (size_t i = 0; i < LearnCount; i++) {
		if (i % (LearnCount / 50) == 0)
			std::cout << i / static_cast<double>(LearnCount) * 100<<"%...学習完了"<<std::endl;

		Learn();
	}

	//testデータを開く
	mnist.close();
	mnist.open("mnist_test.csv");

	//予測
	double accuracy = 0;
	for (size_t i = 0; i < PredictCount; i++)
		accuracy+=predict();

	std::cout << "正答率:"<<accuracy / static_cast<double>(PredictCount) * 100 << "%";
}
