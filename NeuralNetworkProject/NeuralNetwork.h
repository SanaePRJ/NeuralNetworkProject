/*=============================================================
* NAME      : NeuralNetwork.h
* AUTHOR    : SanaeProject
* VER       : 1.0.2
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_NEURALNETWORK_H_
#define _SANAE_NEURALNETWORK_H_


#include <math.h>
#include <time.h>
#include <random>
#include "Matrix.h"


namespace Sanae {
	class NN 
	{
	//Define Variable(public)
	public:
		double learn_late = 0.5;   //学習率
		
	
	//Define Variable(private)
	private:
		Ulong In_Node     = 3;     //入力ノード数 
		Ulong Hidden_Node = 3;     //隠れ層入力ノード数
		Ulong Output_Node = 3;     //出力ノード数

		Ulong  Hidden_Layer   = 3;   //隠れ層数
		const Ulong  IO_Layer = 2;   //Input/Output用レイヤー

		std::vector<Matrix> Weights; //重み

	
	//Define Functions(private)
	private:
		//シグモイド関数
		void Sigmoid
		(
			Matrix& _In
		) 
		{
			//sigmoid(x)= 1 / (1+e^-x)
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i]   = 1 / (1 + std::exp(-1 * _In[i]));
		}

		//この関数により重みを改善する。
		Matrix& Modifled
		(
			Matrix& In,
			Matrix& W ,
			Matrix& E
		)
		{
			Matrix Output = W * In;
			Sigmoid(Output);

			Matrix buf = SizeT{ 1,E.GetSizeWH().second };
			for (Ulong i = 0; i < E.GetSizeWH().second; i++)
				buf[i] = E[i] * Output[i] * (1 - Output[i]);

			return W += ((buf *= In.Transpose()) *= learn_late);
		}


	//Define Functions(public)
	public:
		//Constructor
		NN
		(
			Ulong        _In_Node,
			Ulong        _Hidden_Node,
			Ulong        _Output_Node,

			Ulong        _HLayer,
			unsigned int _Seed
		) 
			: In_Node(_In_Node), Hidden_Node(_Hidden_Node), Output_Node(_Output_Node),
			Hidden_Layer(_HLayer)
		{
			//乱数のシード値設定
			srand(_Seed);
			//各層の重みを設定
			Weights.resize(Hidden_Layer+2);

			//入力層->隠れ層
			Weights[0]               .SetSize({In_Node    , Hidden_Node});
			//隠れ層->出力層
			Weights[Hidden_Layer + 1].SetSize({Hidden_Node, Output_Node});

			//入力層重みの初期設定(0.00~0.99)
			for (Ulong j = 0; j < Weights[0].GetSize(); j++)
				(Weights[0])[j]                = (double)(rand() % 100) / 100;

			//出力層重みの初期設定(0.00~0.99)
			for (Ulong j = 0; j < Weights[Hidden_Layer + 1].GetSize(); j++)
				(Weights[Hidden_Layer + 1])[j] = (double)(rand() % 100) / 100;

			//隠れ層重みの初期設定(0.00~0.99)
			for (Ulong i = 1; i < Hidden_Layer+1; i++)
			{
				Weights[i].SetSize({Hidden_Node,Hidden_Node});
				for (Ulong j = 0; j < Weights[i].GetSize(); j++)
					(Weights[i])[j] = (double)(rand() % 100) / 100;
			}
		}

		//出力します。
		Matrix Out
		(
			Matrix& In
		) 
		{
			//各層の出力値を求める。
			Matrix Out = Weights[0] * In;
			Sigmoid(Out);

			for (Ulong i = 1; i < this->Hidden_Layer+IO_Layer;i++) {
				Out = Weights[i] * Out;
				Sigmoid(Out);
			}

			return Out;
		}

		//重みを出力します。
		Matrix& GetWeight
		(
			Ulong Num
		)
		{
			return this->Weights[Num];
		}

		//学習します。
		void Learn
		(
			Matrix& In,
			Matrix& Ideal
		)
		{
			//各層出力格納用
			std::vector<Matrix> Output;
			Output.resize(this->Hidden_Layer+ IO_Layer);

			//各層誤差格納用
			std::vector<Matrix> Error;
			Error.resize(this->Hidden_Layer + IO_Layer);

			//出力を求める。
			Output[0] = Weights[0] * In;
			Sigmoid(Output[0]);

			for (Ulong i = 1; i < this->Hidden_Layer+ IO_Layer;i++){
				Output[i] = Weights[i] * Output[i-1];
				Sigmoid(Output[i]);
			}

			//誤差を求める。
			Error[this->Hidden_Layer + 1] = Ideal - Output[this->Hidden_Layer + 1];
			
			for (Ulong i = this->Hidden_Layer; i > 0; i--)
				Error[i] = Weights[i + 1].Transpose() * Error[i + 1];
				
			Error[0] = Weights[1].Transpose() * Error[1];

			//重みを調整する
			Modifled(In, Weights[0], Error[0]);
			for (Ulong i = 1; i < this->Hidden_Layer + IO_Layer; i++)
				Modifled(Output[i - 1], Weights[i], Error[i]);
		}
	};
}


#endif