/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 2.1.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_NEURALNETWORK_H_
#define _SANAE_NEURALNETWORK_H_


/*INCLUDE*/
#include <random>
#include <math.h>

#include "Matrix.h"


//正常にインクルードしていること。 C++14以上
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {


	//Neural Network
	class NN {
	public:
		double Learn_rate = 0.5;

		Matrix I_Weights;	//入力層の重み
		Matrix O_Weights;	//出力層の重み


	protected:
		//シグモイド関数を活性化関数とする。
		inline void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = 1 / (1 + std::exp(-1 * _In[i]));
		}

		//Weights(重み)を修正します。
		inline void Modifled
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
		//コンストラクタ
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

		//照会します。
		virtual Matrix Query
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

		//学習させます。
		virtual void Learn
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


	class RNN : public NN {
	public:
		Matrix P_Weights;
		Matrix P_Data;


	public:
		//コンストラクタ
		RNN
		(
			Ulong _Input_Nodes,
			Ulong _Hidden_Nodes,
			Ulong _Output_Nodes,
			unsigned int _Seed
		)
		:
		NN
		(
			_Input_Nodes,
			_Hidden_Nodes,
			_Output_Nodes,
			_Seed
		)
		{
			//シード値設定
			srand(_Seed);

			//各層の重みを設定
			I_Weights.SetSize({ _Input_Nodes,_Hidden_Nodes });
			O_Weights.SetSize({ _Hidden_Nodes,_Output_Nodes });

			P_Weights.SetSize({ _Hidden_Nodes ,1 });
			P_Data.SetSize({ 1             ,_Hidden_Nodes });

			//入力層の重み
			for (Ulong i = 0; i < I_Weights.GetSize(); i++)
				I_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//出力層の重み
			for (Ulong i = 0; i < O_Weights.GetSize(); i++)
				O_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			for (Ulong i = 0; i < P_Weights.GetSize(); i++)
				P_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//重み
			for (Ulong i = 0; i < P_Weights.GetSize(); i++)
				P_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//初期値を0とする。
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				P_Data[i] = 0;
		}

		//照会します。
		Matrix Query
		(
			Matrix& _Input
		) override
		{
			Matrix Output_IH = I_Weights * _Input;
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				Output_IH[i] += P_Weights[i] * P_Data[i];

			this->Activation(Output_IH);
			P_Data = Output_IH;

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			return Output_HO;
		}

		void In
		(
			Matrix& _Input
		)
		{
			Matrix Output_IH = I_Weights * _Input;
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				Output_IH[i] += P_Weights[i] * P_Data[i];

			this->Activation(Output_IH);
			P_Data = Output_IH;
		}

		//学習させます。
		void Learn
		(
			Matrix& _Input,
			Matrix& _Ideal
		) override
		{
			Matrix Output_IH = I_Weights * _Input;
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				Output_IH[i] += P_Weights[i] * P_Data[i];

			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			Matrix Output_Error = _Ideal - this->Query(_Input);
			Matrix Hidden_Error = O_Weights.Transpose() * Output_Error;

			Modifled(_Input, Output_IH, I_Weights, Hidden_Error);
			Modifled(Output_IH, Output_HO, O_Weights, Output_Error);

			for (Ulong i = 0; i < P_Weights.GetSize(); i++) {
				double P_Error = P_Weights[i] * Hidden_Error[i];
				double Out     = P_Weights[i] * P_Data[i];
				P_Weights[i]   += Learn_rate*P_Error * Out * (1 - Out) * P_Data[i];
			}

			P_Data = Output_IH;

			return;
		}
	};
}


#endif


#endif