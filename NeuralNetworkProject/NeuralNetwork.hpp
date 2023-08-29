/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 1.0.0
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

#include "Matrix.h"
#include "NeuralNetworkLayers.hpp"
#include "ReadCSV.hpp"


//正常にインクルードしていること。 C++14以上
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {
	//勾配法により重みを修正する。
	void WeightUpdate_SGD
	(
		Layer_Affine* _Layer,
		double        _LearnRate
	)
	{
		_Layer->Weight -= _Layer->D_Weight * _LearnRate;
		_Layer->Bias   -= _Layer->D_Bias   * _LearnRate;

		return;
	}


	class NN 
	{
	private:
		//レイヤのポインタを格納する。
		std::vector<Layer_Base*> Layers;
		
		//学習率
		double LearnRate;
		

	public:
		NN(Ulong _Input_Nodes,Ulong _Hidden_Nodes,Ulong _Output_Nodes,double _Learn_rate,unsigned int _Seed) 
		{
			//シード値を設定。
			std::default_random_engine Engine(_Seed); 
			std::normal_distribution<> Dist  (0,1);   //平均0,標準偏差1で生成

			LearnRate = _Learn_rate;
			
			//Affineレイヤの設定
			static std::vector<Layer_Affine> Affine;

			Affine.push_back({ _Input_Nodes,_Hidden_Nodes ,&Engine,Dist });

			Affine.push_back({ _Hidden_Nodes,_Hidden_Nodes,&Engine,Dist });
			Affine.push_back({ _Hidden_Nodes,_Hidden_Nodes,&Engine,Dist });

			Affine.push_back({ _Hidden_Nodes,_Output_Nodes,&Engine,Dist });

			//シグモイドレイヤの設定
			static std::vector<Layer_Sigmoid> Sigmoid;
			Sigmoid.push_back({});
			Sigmoid.push_back({});

			//SoftMaxレイヤの設定
			static Layer_SoftMax SoftMax  = {};
			
			//レイヤに格納
			Layers.push_back(&Affine[0]);
			Layers.push_back(&Sigmoid[0]);

			Layers.push_back(&Affine[1]);
			Layers.push_back(&Sigmoid[1]);

			//Affineレイヤのラスト
			Layers.push_back(&*(Affine.end()-1));
			Layers.push_back(&SoftMax);
		}

		//予測する。
		Matrix Predict(Matrix _Data) 
		{
			Matrix buf = Layers[0]->forward(&_Data);
			
			for (Ulong i = 1; i < Layers.size(); i++)
				buf = Layers[i]->forward(&buf);
			
			return buf;
		}

		//学習する。
		void Learn(Matrix _In, Matrix _t) {
			//一度出力する
			Matrix buf = Layers[0]->forward(&_In);

			for (Ulong i = 1; i < Layers.size(); i++)
				buf = Layers[i]->forward(&buf);

			//逆伝番する。
			//一番後の処理
			buf = Layers[Layers.size() -1]->backward(&_t);

			//逆伝番
			for (int i = Layers.size() - 2; i >= 0; i--) {
				buf = Layers[i]->backward(&buf);

				//Affineレイヤだった場合Weightの修正
				if (Layers[i]->Is_Affine)
					WeightUpdate_SGD((Layer_Affine*)Layers[i], this->LearnRate);
			}
		}

		//重みを次回読み取れるようCSVで出力。
		void Write(const char* _WriteFile) {
			std::ofstream ofs(_WriteFile);
			ofs << "AffineLayerCount,"<< this->Layers.size()/2 << "\n";

			auto WriteAffine = [&](Ulong Layer_Point)
			{
				//Affineレイヤの番号を出力
				(ofs << Layer_Point) << "\n";

				//重みの列,行数を格納
				SizeT Size = ((Layer_Affine*)Layers[Layer_Point])->Weight.GetSizeWH();

				//列数と行数を書き込む
				ofs << "Weight," << Size.first<< "," << Size.second << "\n";

				for (Ulong i = 0; i < ((Layer_Affine*)Layers[Layer_Point])->Weight.GetSize(); i++) {
					//それぞれの重みを書き込んでいく
					(ofs << ((Layer_Affine*)Layers[Layer_Point])->Weight[i]);

					//改行
					if ((i + 1) % Size.first == 0)
						ofs << "\n";
					//CSVなのでコンマ区切り
					else
						ofs << ",";
				}

				//バイアスサイズの書き込み
				Size = ((Layer_Affine*)Layers[Layer_Point])->Bias.GetSizeWH();
				//バイアスの列数,行数を書き込み
				ofs << "Bias," << Size.first << "," << Size.second << "\n";

				for (Ulong i = 0; i < ((Layer_Affine*)Layers[Layer_Point])->Bias.GetSize(); i++) {
					//それぞれのバイアスの書き込み
					(ofs << ((Layer_Affine*)Layers[Layer_Point])->Bias[i]);

					if ((i + 1) % Size.first == 0)
						ofs << "\n";
					else
						ofs << ",";
				}
			};

			//各層の重み、バイアスを出力
			for (Ulong i = 0; i < this->Layers.size();i++) {
				if (Layers[i]->Is_Affine)
					WriteAffine(i);
			}
		}

		//書き込まれた重みバイアスを読み取る。
		void Read(const char* _ReadFile) {
			//ReadCSVで開く
			ReadCSV input = _ReadFile;

			Ulong Count = 0;
			for (Ulong i = 0; i < this->Layers.size();i++)
				Count += this->Layers[i]->Is_Affine ? 1: 0;
			
			//形式に合わない場合はエラー
			if(input.ReadDataStr<std::string>() != "AffineLayerCount") throw std::invalid_argument("");
			if(input.ReadDataD<double>() != Count)                     throw std::invalid_argument("");

			//重みを読み取る。
			auto ReadWeight = [&](Ulong point)
			{
				//形式似合わない場合はエラー
				if (input.ReadDataStr<std::string>() != "Weight") throw std::invalid_argument("");

				//列数,行数を格納
				SizeT Size = SizeT{ input.ReadDataI<int>(),input.ReadDataI<int>() };
				
				//重みを格納
				std::vector<double> Weight_buf;
				
				//読み取る。
				for (Ulong i = 0; i < Size.second; i++)
					input.ReadLineD<double>(&Weight_buf, Size.first);
				
				//重みを格納したvector配列をMatrix型にmoveする。
				((Layer_Affine*)Layers[point])->Weight.Move(&Weight_buf, Size);
			};

			//バイアスを読み取る。
			auto ReadBias   = [&](Ulong point)
			{
				//形式に合わない場合はエラー
				if (input.ReadDataStr<std::string>() != "Bias") throw std::invalid_argument("");

				//列数行数を格納
				SizeT Size = SizeT{ input.ReadDataI<int>(),input.ReadDataI<int>() };
				
				//バイアスの格納
				std::vector<double> Bias_buf;

				//読み取る。
				for (Ulong i = 0; i < Size.second; i++)
					input.ReadLineD<double>(&Bias_buf, Size.first);

				//重みを格納したvector配列をMatrix型にmoveする。
				((Layer_Affine*)Layers[point])->Bias.Move(&Bias_buf, Size);
			};

			for (Ulong i = 0; i < Count;i++) {
				Ulong Pos = (Ulong)input.ReadDataI<int>();

				//重みとバイアスを読み取る。
				ReadWeight(Pos);
				ReadBias(Pos);
			}
		}
	};


}


#endif
#endif