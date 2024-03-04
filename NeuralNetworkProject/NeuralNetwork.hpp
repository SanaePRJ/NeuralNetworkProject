/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/




#ifndef _SANAE_NEURALNETWORK_HPP_
#define _SANAE_NEURALNETWORK_HPP_


/*INCLUDE*/
#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <stdexcept>

#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"
#include "NNLayers.hpp"
#include "CSV.hpp"




namespace Sanae {


	template<typename ty,typename ActClass = Sanae::Layer_ReLU<ty>,typename ActWithLoss=Sanae::Layer_SoftMaxWithLoss<ty>>
	class NN
	{
	private:
		//レイヤのポインタを格納する。
		std::vector<Layer_Base<ty>*> Layers;

		//学習率
		double LearnRate;

		//入力ノード数,中間層ノード数,出力層ノード数,中間層(隠れ層)レイヤー数
		uint64_t Input_Nodes, Hidden_Nodes, Output_Nodes, Hidden_Layers;

	public:
		//バージョン
		const size_t version = 0;

		//入力層ノード数,中間層ノード数,出力層ノード数,学習率,シード値,中間層(隠れ層)レイヤー数
		NN(uint64_t arg_Input_Nodes, uint64_t arg_Hidden_Nodes, uint64_t arg_Output_Nodes,double arg_Learn_rate, unsigned int arg_Seed, uint64_t arg_Hidden_Layers)
			:	LearnRate(arg_Learn_rate), Input_Nodes(arg_Input_Nodes), Hidden_Nodes(arg_Hidden_Nodes), Output_Nodes(arg_Output_Nodes), Hidden_Layers(arg_Hidden_Layers)
		{
			//乱数
			std::default_random_engine engine(arg_Seed);
			
			//-----入力層-----
			static Sanae::Layer_Affine <ty> Affine1(arg_Input_Nodes, arg_Hidden_Nodes,&engine);   //Affineレイヤ
			Affine1.LearnRate = LearnRate;   //学習率を設定

			static ActClass Activation1;     //活性化レイヤ

			Layers.push_back((Sanae::Layer_Base<ty>*) & Affine1 );   //Affineレイヤ
			Layers.push_back((Sanae::Layer_Base<ty>*) & Activation1);//活性化レイヤを追加

			//-----中間層-----
			static std::vector<Sanae::Layer_Affine<ty>> Affine2    (arg_Hidden_Layers, Sanae::Layer_Affine<ty>(arg_Hidden_Nodes,arg_Hidden_Nodes,&engine));   //Affineレイヤ
			static std::vector<ActClass>                Activation2(arg_Hidden_Layers);   //活性化レイヤ

			for (uint64_t i = 0; i < arg_Hidden_Layers;i++) {
				Layers.push_back((Sanae::Layer_Base<ty>*)(&Affine2[i]));     //Affineレイヤを追加
				Layers.push_back((Sanae::Layer_Base<ty>*)(&Activation2[i])); //活性化レイヤを追加

				Affine2[i].LearnRate = LearnRate;   //Affineレイヤの学習率を設定
			}

			//-----出力層-----
			static Sanae::Layer_Affine <ty> Affine3(arg_Hidden_Nodes, arg_Output_Nodes, &engine);   //Affineレイヤ
			Affine3.LearnRate = LearnRate;   //Affineレイヤの学習率を設定

			static ActWithLoss awl;   //活性化-損失関数レイヤ

			Layers.push_back((Sanae::Layer_Base<ty>*) & Affine3);  //Affineレイヤの追加
			Layers.push_back((Sanae::Layer_Base<ty>*) & awl);      //活性化-損失関数レイヤの追加
		}

		//予測する。
		Sanae::Matrix<ty> Predict(Sanae::Matrix<ty> arg)
		{
			Sanae::Matrix<ty> buf = Layers[0]->forward(&arg);

			//伝番していく
			for (size_t i = 1; i < (Layers.size() - 1); i++)
				buf = Layers[i]->forward(&buf);
			
			return buf;
		}

		//学習する。
		double Learn(Sanae::Matrix<ty> arg_in, Sanae::Matrix<ty> arg_teach)
		{
			//一度出力する
			Sanae::Matrix<ty> buf = Layers[0]->forward(&arg_in);
			
			for (size_t i = 1; i < (Layers.size() - 1); i++)
				buf = Layers[i]->forward(&buf);

			//損失を求める。
			double loss = ((Layer_SoftMaxWithLoss<ty>*)Layers[Layers.size() - 1])->forward(&buf, &arg_teach);

			//逆伝番する。
			//一番後の処理
			buf = ((Layer_SoftMaxWithLoss<ty>*)Layers[Layers.size() - 1])->backward();

			//逆伝番
			for (int i = (int)Layers.size() - 2; i >= 0; i--)
				buf = Layers[i]->backward(&buf);

			return loss;
		}

		//重み,バイアスを保存
		NN& Save(const char* FileName) 
		{
			//CSVファイル書き込み
			Sanae::WriteCSV save(FileName,true);

			//NeuralNetwork-SanaeProjectで作られたファイルであること,バージョンの書き込み
			save << "NeuralNetwork-SanaeProject" << std::to_string(version) << "\n";

			//入力ノード数,中間層ノード数,出力層ノード数,中間層レイヤー数を書き込む
			save << Input_Nodes << Hidden_Nodes << Output_Nodes << Hidden_Layers << "\n";

			//活性化レイヤー,活性化・出力レイヤー
			save << typeid(ActClass).name() << typeid(ActWithLoss).name() << "\n";

			//すべての層の重み,BIASを保存する。
			for (size_t i = 0; i < Layers.size(); i+=2) {
				Sanae::Layer_Affine<ty>* LayerP = (Sanae::Layer_Affine<ty>*)Layers[i];

				//レイヤ番号,Weightが始まることを書き込み
				save << i / 2 << "\n" << "Weight" << "\n";
				//Weight書き込み
				for (uint64_t k = 0; k < LayerP->Weight.get_row(); k++) {
					save << LayerP->Weight[k] << "\n";
					save.flush();
				}

				//biasが始まることを書き込み
				save << "Bias" << "\n";
				//Bias書き込み
				for (uint64_t k = 0; k < LayerP->Bias.get_row(); k++) {
					save << LayerP->Bias[k] << "\n";
					save.flush();
				}
			}

			return *this;
		}

		NN& Load(const char* FileName) {
			Sanae::ReadCSV load;

			//CSVファイル読み取り
			try {
				load.open(FileName);
			}
			catch (std::exception e) {
				throw e;
			}

			//csvファイルがNeuralNetwork-SanaeProjectで作られた形式かチェック
			if (load.ReadData() != "NeuralNetwork-SanaeProject")
				throw std::invalid_argument("The format of this file is incorrect.");

			//csvファイルのバージョンを確認
			if (load.ReadData() != std::to_string(version))
				throw std::invalid_argument("The version is different.");

			//ノード詳細
			uint8_t error=
			(Input_Nodes   != std::stoull(load.ReadData())) +   //入力層ノード数
			(Hidden_Nodes  != std::stoull(load.ReadData())) +   //中間層ノード数
			(Output_Nodes  != std::stoull(load.ReadData())) +   //出力層ノード数
			(Hidden_Layers != std::stoull(load.ReadData()));    //中間レイヤー数

			//errorがあった場合
			if (error !=0)
				throw std::invalid_argument("Different size.");

			//レイヤclassがあっているかどうか
			if (typeid(ActClass).name() != load.ReadData())
				throw std::invalid_argument("The ActivationClass is different.");

			if (typeid(ActWithLoss).name() != load.ReadData())
				throw std::invalid_argument("The ActivationWithLossClass is different.");


			//すべての層の重み,BIASを読み込む。
			for (size_t lay = 0; lay < Layers.size(); lay+=2) {
				//レイヤのポインタ
				Sanae::Layer_Affine<ty>* LayerP = (Sanae::Layer_Affine<ty>*)Layers[lay];

				//レイヤ番号が異なっている場合書式が違う
				if (load.ReadData() != std::to_string((size_t)(lay / 2)))
					throw std::invalid_argument("The format of this file is incorrect.[num]");
				
				//Weightがない場合書式が違う
				if(load.ReadData() != std::string("Weight"))
					throw std::invalid_argument("The format of this file is incorrect.[weight]");

				//Weightの各要素を読み込み
				for (size_t row = 0; row < LayerP->Weight.get_row(); row++)
					for (size_t column = 0; column < LayerP->Weight.get_column(); column++)
						LayerP->Weight[row][column] = std::stod(load.ReadData());

				//Biasがない場合書式が違う
				if (load.ReadData() !="Bias")
					throw std::invalid_argument("The format of this file is incorrect.[bias]");

				//Biasの各要素を読み込み
				for (size_t row = 0; row < LayerP->Bias.get_row(); row++)
					for (size_t column = 0; column < LayerP->Bias.get_column(); column++)
						LayerP->Bias[row][column] = std::stod(load.ReadData());
			}

			return *this;
		}
	};
}




#endif