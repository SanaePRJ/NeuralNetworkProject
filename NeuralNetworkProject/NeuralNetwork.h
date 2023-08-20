/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 2.1.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_NEURALNETWORK_H_
#define _SANAE_NEURALNETWORK


/*INCLUDE*/
#include <vector>
#include <random>
#include <math.h>

#include "Matrix.h"


//正常にインクルードしていること。 C++14以上
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {
/*======活性化関数======*/
	//シグモイド関数(活性化関数)
	inline double Sigmoid(double _In)
	{
		return 1 / (1 + std::exp(-1 * _In));
	}
	//ReLU関数(活性化関数)
	inline double ReLU(double _In)
	{
		return _In * (_In > 0);
	}

/*======導関数======*/
	//数値微分
	double (*Num_Diff_Func)(double) = ReLU;
	inline double Num_Diff(double _In) 
	{
		return (Num_Diff_Func(_In + 1e-4) - Num_Diff_Func(_In - 1e-4)) / (2 * 1e-4);
	}


	//シグモイド関数の導関数
	inline double diff_Sigmoid(double _In)
	{
		return Sanae::Sigmoid(_In) * (1 - Sanae::Sigmoid(_In));
	}
	//ReLUの導関数
	inline double diff_ReLU(double _In)
	{
		return 1 * (_In > 0);
	}
	//tanhの導関数
	inline double diff_tanh(double _In)
	{
		return 4/std::pow(std::exp(_In)+std::exp(-1*_In), 2);
	}


	//Neural Network
	class NN {
	public:
		double (*Activation_Func)(double) = Sanae::Sigmoid;       //活性化関数
		double (*Activation_Diff)(double) = Sanae::diff_Sigmoid;  //導関数

		double Learn_rate = 1;

		Matrix I_Weights;	//入力層の重み
		Matrix O_Weights;	//出力層の重み


	protected:
		//シグモイド関数を活性化関数とする。
		inline void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = Activation_Func(_In[i]);
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
				buf[i] = Error[i] * Activation_Diff(Output[i]);

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
			//各出力を保存
			Matrix Output_IH = I_Weights * _Input;
			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			//誤差逆伝番
			Matrix Output_Error = _Ideal - this->Query(_Input);
			Matrix Hidden_Error = O_Weights.Transpose() * Output_Error;

			Modifled(_Input, Output_IH, I_Weights, Hidden_Error);
			Modifled(Output_IH, Output_HO, O_Weights, Output_Error);

			return;
		}
	};


	//Reccurent Neural Network
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
			//出力保存
			Matrix Output_IH = I_Weights * _Input;
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				Output_IH[i] += P_Weights[i] * P_Data[i];
			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			//誤差逆伝番
			Matrix Output_Error = _Ideal - this->Query(_Input);
			Matrix Hidden_Error = O_Weights.Transpose() * Output_Error;

			//重みの修正
			Modifled(_Input, Output_IH, I_Weights, Hidden_Error);
			Modifled(Output_IH, Output_HO, O_Weights, Output_Error);

			//再帰の重みの修正
			for (Ulong i = 0; i < P_Weights.GetSize(); i++) {
				double P_Error = P_Weights[i] * Hidden_Error[i];
				double Out = P_Weights[i] * P_Data[i];
				P_Weights[i] += Learn_rate * P_Error * Out * (1 - Out) * P_Data[i];
			}

			//以前の出力を保存
			P_Data = Output_IH;

			return;
		}


	};


	//Deep Neural Network
	class DNN {
	protected:
		const Ulong IO_Layer = 2;

		Ulong Input_Layer = 0;
		Ulong Output_Layer = 0;


	public:
		double (*Activation_Func)(double) = Sanae::ReLU;      //活性化関数            
		double (*Activation_Diff)(double) = Sanae::diff_ReLU; //導関数

		double              Learn_rate = 0.001;
		std::vector<Matrix> Weights;


	private:
		virtual inline void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = Activation_Func(_In[i]);

			return;
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
				buf[i] = Error[i] * (Activation_Diff(Output[i]));

			Weights += (buf *= Input.Transpose()) *= this->Learn_rate;

			return;
		}

		virtual inline void Init_Weight
		(
			Matrix* _Weight
		)
		{
			for (Ulong i = 0; i < _Weight->GetSize(); i++)
				(*_Weight)[i] = (double)(rand() % 100) / 100 - 0.5;

			return;
		}


	public:
		DNN
		(
			Ulong _Input_Node,
			Ulong _Hidden_Node,
			Ulong _Output_Node,
			Ulong _Middle_Layer,
			unsigned int _Seed
		)
		{
			const Ulong Sum_Layer = IO_Layer + _Middle_Layer;

			//サイズ設定
			Weights.resize(Sum_Layer - 1);
			Output_Layer = Weights.size() - 1;

			//入力層の重みを設定
			Weights[Input_Layer].SetSize({ _Input_Node,_Hidden_Node });
			Init_Weight(&Weights[Input_Layer]);

			//出力層の重みを設定
			Weights[Output_Layer].SetSize({ _Hidden_Node,_Output_Node });
			Init_Weight(&Weights[Output_Layer]);

			//中間層の重みを設定
			for (Ulong Layer = 1; Layer < (Sum_Layer - IO_Layer); Layer++)
			{
				Weights[Layer].SetSize({ _Hidden_Node,_Hidden_Node });
				Init_Weight(&Weights[Layer]);
			}
		}

		//照会します。
		virtual Matrix Query
		(
			Matrix& _Input
		)
		{
			Matrix Output = Weights[Input_Layer] * _Input;
			this->Activation(Output);

			for (Ulong i = 1; i <= Output_Layer; i++) {
				Output = Weights[i] * Output;
				this->Activation(Output);
			}

			return Output;
		}

		//学習させます。
		virtual void Learn
		(
			Matrix& _Input,
			Matrix& _Ideal
		)
		{
			std::vector<Matrix> _Outputs;
			_Outputs.resize(Weights.size());

			std::vector<Matrix> _Errors;
			_Errors.resize(Weights.size());

			//それぞれの出力を保存する。
			_Outputs[0] = (Weights[Input_Layer] * _Input);
			this->Activation(_Outputs[0]);

			for (Ulong i = 1; i <= Output_Layer; i++) {
				_Outputs[i] = (Weights[i] * _Outputs[i - 1]);
				this->Activation(_Outputs[i]);
			}

			//誤差逆伝番
			_Errors[_Errors.size() - 1] = _Ideal - _Outputs[Output_Layer];

			for (Ulong i = _Errors.size() - 2; i > 0; i--)
				_Errors[i] = Weights[i + 1].Transpose() * _Errors[i + 1];
			_Errors[0] = Weights[1].Transpose() * _Errors[1];

			this->Modifled(_Input, _Outputs[0], Weights[0], _Errors[0]);

			for (Ulong i = 1; i < _Outputs.size(); i++)
				this->Modifled(_Outputs[i - 1], _Outputs[i], Weights[i], _Errors[i]);
		}
	};
}


#endif


#endif