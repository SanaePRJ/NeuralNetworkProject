/*=============================================================
* NAME      : NeuralNetworkLayers.hpp
* AUTHOR    : SanaeProject
* VER       : 1.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_NEURALNETWORKLAYERS_HPP_
#define _SANAE_NEURALNETWORKLAYERS_HPP_


//正常にインクルードしていること。 C++14以上
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>


namespace Sanae {


	namespace Random {

		class Random_Base 
		{
		public:
			virtual inline double Get() { return 0; }
		};

		class Box_Muller : public Random_Base
		{
		private:
			const double  PAI     = 3.141592653589793238462643;
			double        ave     = 0, stddev = 0;
			bool          Use_Cos = false;

		public:
			Box_Muller(double _ave, double _stddev, unsigned int _Seed = time(0UL), bool _Use_Cos = true,bool _DoSrand = true)
				: ave(_ave), stddev(_stddev), Use_Cos(_Use_Cos)
			{
				if(_DoSrand)
					srand(_Seed);
			}

			double Get() override
			{
				Ulong max = 1e4+1;

				double x = (rand() % max) / 1e4;
				double y = (rand() % max) / 1e4;
				
				double ret = std::sqrt(-2 * std::log(x));

				if (Use_Cos)
					ret *= std::cos(2 * y * PAI);
				else
					ret *= std::sin(2 * y * PAI);

				return ret * stddev + ave;
			}
		};

		class Default_Random : Random_Base
		{
		private:
			int Max = 0 , Min = 0;

		public:
			Default_Random(int _Max,int _Min,unsigned int _Seed = 0UL,bool _DoSrand = true) 
				: Max(_Max) , Min(_Min)
			{
				if(_DoSrand)
					srand(_Seed);
			}
			double Get() override
			{
				return (double)(rand() % (Max+1)) + Min + ((double)(rand()%(int)1e4)/1e4);
			}
		};
	}


/*------活性化関数------*/
	//Sigmoid関数 Sigmoid(x) = 1 / {1+e^-x}
	inline const double Sigmoid
	(
		double _X
	)
	{
		return 1 / (1 + std::exp(-1 * _X));
	}

	//SoftMax関数
	inline Matrix SoftMax
	(
		Matrix* _In
	)
	{
		Matrix Output = _In->GetSizeWH();

		const std::vector<double>* VecP = _In->GetVectorP();
		const double               Max = *std::max_element(VecP->begin(), VecP->end());

		double                     Sum = 0;

		//合計の値を求める。
		for (Ulong i = 0; i < _In->GetSize(); i++)
			Sum += std::exp((*_In)[i] - Max);

		for (Ulong i  = 0; i < _In->GetSize(); i++)
			Output[i] = std::exp((*_In)[i] - Max) / Sum;

		return Output;
	}

/*------乱数設定用関数------*/
	void Set_Random_Num
	(
		Matrix*      _Pointer,
		unsigned int _Seed,
		bool         _DoSrand = true
	) 
	{
		Random::Default_Random rand(1,-1,_Seed,_DoSrand);

		for (Ulong i = 0; i < _Pointer->GetSize(); i++)
			(*_Pointer)[i] = rand.Get();

		return;
	}

/*------Layers------*/
	//レイヤーのベースクラス
	class Layer_Base
	{
	public:
		bool Is_Affine = false; //Is_Affineがtrueの場合update関数の実行

		inline virtual Matrix forward(Matrix*) { return Matrix{}; };
		inline virtual Matrix backward(Matrix*) { return Matrix{}; };
	};

	//ReLUレイヤー
	class Layer_ReLU : public Layer_Base
	{
	private:
		//出力の保存
		std::vector<bool> Mask;

	public:
		Matrix forward
		(
			Matrix* _In
		)
			override
		{
			//サイズ設定
			Matrix Output = _In->GetSizeWH();
			Mask.resize(_In->GetSize());

			for (Ulong point = 0; point < Output.GetSize(); point++) {
				Mask[point] = ((*_In)[point] <= 0);
				Output[point] = (*_In)[point] * !Mask[point];
			}
			return Output;
		};
		Matrix backward
		(
			Matrix* _dout
		)
			override
		{
			Matrix Diff = _dout->GetSizeWH();

			for (Ulong i = 0; i < _dout->GetSize(); i++)
				Diff[i] = (*_dout)[i] * !Mask[i];

			return Diff;
		};
	};

	//Tanhレイヤー
	class Layer_Tanh : public Layer_Base
	{
	private:
		//出力値を保存する。
		Matrix Output;

	public:
		Matrix forward
		(
			Matrix* _In
		)
			override
		{
			Output = *_In;

			for (Ulong i = 0; i < Output.GetSize(); i++)
				Output[i] = std::tanh(Output[i]);

			return Output;
		}
		Matrix backward
		(
			Matrix* _dout
		)
			override
		{
			Matrix Diff = _dout->GetSizeWH();

			//サイズが違う場合throwする。
			if (_dout->GetSize() != Output.GetSize())
				throw std::invalid_argument("Must be the same size.");

			//導関数 tanh'(x) = 1-tanh(x)^2
			for (Ulong point = 0; point < _dout->GetSize(); point++)
				Diff[point] = (*_dout)[point] * (1 - std::pow(this->Output[point], 2));

			return Diff;
		}
	};

	//シグモイドレイヤー
	class Layer_Sigmoid : public Layer_Base
	{
	private:
		//出力値を保存する。
		Matrix Output;

	public:
		Matrix forward
		(
			Matrix* _In
		)
			override
		{
			Output = *_In;

			for (Ulong i = 0; i < Output.GetSize(); i++)
				Output[i] = Sigmoid(Output[i]);

			return Output;
		}
		Matrix backward
		(
			Matrix* _dout
		)
			override
		{
			Matrix Diff = _dout->GetSizeWH();

			//サイズが違う場合throwする。
			if (_dout->GetSize() != Output.GetSize())
				throw std::invalid_argument("Must be the same size.");

			//導関数 Sigmoid(x)' = x' * (1-Sigmoid(x)) * Sigmoid(X)
			for (Ulong point = 0; point < _dout->GetSize(); point++)
				Diff[point] = (*_dout)[point] * (1 - this->Output[point]) * this->Output[point];

			return Diff;
		}
	};

	//SoftMaxレイヤー
	class Layer_SoftMax : public Layer_Base
	{
	private:
		Matrix Output;

	public:
		Matrix forward
		(
			Matrix* _In
		)
			override
		{
			Output = Sanae::SoftMax(_In);

			return Output;
		}

		Matrix backward
		(
			Matrix* _Train
		)
			override
		{
			Matrix Diff = _Train->GetSizeWH();

			for (Ulong i = 0; i < _Train->GetSize(); i++)
				Diff[i] = (Output[i] - (*_Train)[i]);

			return Diff;
		}
	};

	//Affineレイヤー
	class Layer_Affine : public Layer_Base
	{
	private:
		Matrix Input;
		
	public:
		//重みとバイアス
		Matrix Weight;
		Matrix Bias;

		//この変数により重みを改善する。
		Matrix D_Weight;
		Matrix D_Bias;
		
	public:
		Layer_Affine(Ulong _Input_Size, Ulong _Output_Size, unsigned int _Seed)
		{
			this->Is_Affine = true;

			Weight.SetSize(SizeT{ _Output_Size,_Input_Size });
			Bias  .SetSize(SizeT{ _Output_Size,1 });

			Set_Random_Num(&Weight,_Seed,false);
			Set_Random_Num(&Bias  ,_Seed,false);
		}
		Matrix forward
		(
			Matrix* _In
		)
			override
		{
			Input = *_In;

			Matrix Output = Input * Weight + Bias;

			return Output;
		}
		Matrix backward
		(
			Matrix* _dout
		)
			override
		{
			Matrix Diff    = (*_dout) * Weight.Transpose();

			this->D_Weight = this->Input.Transpose() * (*_dout);
			this->D_Bias   = *_dout;
			
			return Diff;
		}
	};
}


#endif


#endif