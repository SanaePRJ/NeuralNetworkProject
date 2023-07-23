/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_NEURALNETWORK_H_
#define _SANAE_NEURALNETWORK_H_


/*INCLUDE*/
#include <random>
#include <math.h>

#include "Matrix.h"



#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)

namespace Sanae {
	class NN {
	public:
		double Learn_rate = 0.5;

		Matrix I_Weights;	//入力層の重み
		Matrix O_Weights;	//出力層の重み


	private:
		//シグモイド関数を活性化関数とする。
		void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = 1 / (1 + std::exp(-1 * _In[i]));
		}
		void Modifled
		(
			Matrix& Input,
			Matrix& Output,
			Matrix& Weights,
			Matrix& Error
		)
		{
			Matrix buf = SizeT{ 1,Error.GetSizeWH().second };

			for (Ulong i = 0; i < Error.GetSizeWH().second; i++)
				buf[i] = Error[i] * Output[i] * (1 - Output[i]);

			Weights += (buf *= Input.Transpose()) *= this->Learn_rate;

			return;
		}

	public:
		NN
		(
			Ulong _Input_Nodes,
			Ulong _Hidden_Nodes,
			Ulong _Output_Nodes,
			unsigned int _Seed
		)
		{
			//シード値設定
			srand(_Seed);

			//各層の重みを設定
			I_Weights.SetSize({ _Input_Nodes,_Hidden_Nodes });
			O_Weights.SetSize({ _Hidden_Nodes,_Output_Nodes });

			//入力層の重み
			for (Ulong i = 0; i < I_Weights.GetSize(); i++)
				I_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//出力層の重み
			for (Ulong i = 0; i < O_Weights.GetSize(); i++)
				O_Weights[i] = (double)(rand() % 100) / 100 - 0.5;
		}

		Matrix Query
		(
			Matrix& _Input
		)
		{
			Matrix Output_IH = I_Weights * _Input;
			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			return Output_HO;
		}

		void Learn
		(
			Matrix& _Input,
			Matrix& _Ideal
		)
		{
			Matrix Output_IH = I_Weights * _Input;
			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);


			Matrix Output_Error = _Ideal - this->Query(_Input);
			Matrix Hidden_Error = O_Weights.Transpose() * Output_Error;

			Modifled(_Input, Output_IH, I_Weights, Hidden_Error);
			Modifled(Output_IH, Output_HO, O_Weights, Output_Error);

			return;
		}
	};
}

#endif


#endif