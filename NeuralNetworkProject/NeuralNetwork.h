/*=============================================================
* NAME      : NeuralNetwork.h
* AUTHOR    : SanaeProject
* VER       : 1.0.0
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
		double learn_late = 0.5;
		
	
	//Define Variable(private)
	private:
		Ulong  Node  = 3;
		Ulong  Layer = 3;

		std::vector<Matrix> Weights;

	
	//Define Functions(private)
	private:
		void Sigmoid
		(
			Matrix& _In
		) 
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = 1 / (1 + std::exp(-1 * _In[i]));
		}
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

			Matrix Delta = (buf *= In.Transpose()) *= learn_late;
			return W += Delta;
		}


	//Define Functions(public)
	public:
		//Constructor
		NN
		(
			Ulong In_Node,
			Ulong In_Layer,
			Ulong In_Seed
		) 
			: Node(In_Node),Layer(In_Layer)
		{
			srand(In_Seed);
			Weights.resize(In_Layer);

			for (Ulong i = 0; i < In_Layer; i++)
			{
				Weights[i].SetSize({In_Node,In_Node});
				for (Ulong j = 0; j < Weights[i].GetSize(); j++)
					(Weights[i])[j] = (double)(rand() % 100) / 10;
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

			for (Ulong i = 1; i < this->Layer;i++) {
				Out = Weights[i] * Out;
				Sigmoid(Out);
			}

			return Out;
		}

		//重みを出力します。
		Matrix GetWeight
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
			//各層の出力値を求める。
			std::vector<Matrix> Output;
			Output.resize(this->Layer);

			std::vector<Matrix> Error;
			Error.resize(this->Layer);

			Output[0] = Weights[0] * In;
			Sigmoid(Output[0]);

			for (Ulong i = 1; i < this->Layer;i++){
				Output[i] = Weights[i] * Output[i-1];
				Sigmoid(Output[i]);
			}
			
			//誤差を求める。
			Error[this->Layer - 1] = Ideal - Output[this->Layer - 1];
			
			for (Ulong i = this->Layer >= 2 ? this->Layer - 2:0; i > 0;i--)
				Error[i] = Weights[i+1].Transpose() * Error[i + 1];
			
			if(this->Layer >= 2)
				Error[0] = Weights[1].Transpose() * Error[1];
			
			//重みを調整する
			Modifled(In, Weights[0], Error[0]);
			for (Ulong i = 1; i < this->Layer;i++)
				Modifled(Output[i - 1], Weights[i], Error[i]);
		}
	};
}


#endif