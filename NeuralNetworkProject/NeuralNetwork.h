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


//����ɃC���N���[�h���Ă��邱�ƁB C++14�ȏ�
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {
/*======�������֐�======*/
	//�V�O���C�h�֐�(�������֐�)
	inline double Sigmoid(double _In)
	{
		return 1 / (1 + std::exp(-1 * _In));
	}
	//ReLU�֐�(�������֐�)
	inline double ReLU(double _In)
	{
		return _In * (_In > 0);
	}

/*======���֐�======*/
	//���l����
	double (*Num_Diff_Func)(double) = ReLU;
	inline double Num_Diff(double _In) 
	{
		return (Num_Diff_Func(_In + 1e-4) - Num_Diff_Func(_In - 1e-4)) / (2 * 1e-4);
	}


	//�V�O���C�h�֐��̓��֐�
	inline double diff_Sigmoid(double _In)
	{
		return Sanae::Sigmoid(_In) * (1 - Sanae::Sigmoid(_In));
	}
	//ReLU�̓��֐�
	inline double diff_ReLU(double _In)
	{
		return 1 * (_In > 0);
	}
	//tanh�̓��֐�
	inline double diff_tanh(double _In)
	{
		return 4/std::pow(std::exp(_In)+std::exp(-1*_In), 2);
	}


	//Neural Network
	class NN {
	public:
		double (*Activation_Func)(double) = Sanae::Sigmoid;       //�������֐�
		double (*Activation_Diff)(double) = Sanae::diff_Sigmoid;  //���֐�

		double Learn_rate = 1;

		Matrix I_Weights;	//���͑w�̏d��
		Matrix O_Weights;	//�o�͑w�̏d��


	protected:
		//�V�O���C�h�֐����������֐��Ƃ���B
		inline void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = Activation_Func(_In[i]);
		}

		//Weights(�d��)���C�����܂��B
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
		//�R���X�g���N�^
		NN
		(
			Ulong _Input_Nodes,
			Ulong _Hidden_Nodes,
			Ulong _Output_Nodes,
			unsigned int _Seed
		)
		{
			//�V�[�h�l�ݒ�
			srand(_Seed);

			//�e�w�̏d�݂�ݒ�
			I_Weights.SetSize({ _Input_Nodes,_Hidden_Nodes });
			O_Weights.SetSize({ _Hidden_Nodes,_Output_Nodes });

			//���͑w�̏d��
			for (Ulong i = 0; i < I_Weights.GetSize(); i++)
				I_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//�o�͑w�̏d��
			for (Ulong i = 0; i < O_Weights.GetSize(); i++)
				O_Weights[i] = (double)(rand() % 100) / 100 - 0.5;
		}

		//�Ɖ�܂��B
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

		//�w�K�����܂��B
		virtual void Learn
		(
			Matrix& _Input,
			Matrix& _Ideal
		)
		{
			//�e�o�͂�ۑ�
			Matrix Output_IH = I_Weights * _Input;
			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			//�덷�t�`��
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
		//�R���X�g���N�^
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
			//�V�[�h�l�ݒ�
			srand(_Seed);

			//�e�w�̏d�݂�ݒ�
			I_Weights.SetSize({ _Input_Nodes,_Hidden_Nodes });
			O_Weights.SetSize({ _Hidden_Nodes,_Output_Nodes });

			P_Weights.SetSize({ _Hidden_Nodes ,1 });
			P_Data.SetSize({ 1             ,_Hidden_Nodes });

			//���͑w�̏d��
			for (Ulong i = 0; i < I_Weights.GetSize(); i++)
				I_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//�o�͑w�̏d��
			for (Ulong i = 0; i < O_Weights.GetSize(); i++)
				O_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			for (Ulong i = 0; i < P_Weights.GetSize(); i++)
				P_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//�d��
			for (Ulong i = 0; i < P_Weights.GetSize(); i++)
				P_Weights[i] = (double)(rand() % 100) / 100 - 0.5;

			//�����l��0�Ƃ���B
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				P_Data[i] = 0;
		}

		//�Ɖ�܂��B
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

		//�w�K�����܂��B
		void Learn
		(
			Matrix& _Input,
			Matrix& _Ideal
		) override
		{
			//�o�͕ۑ�
			Matrix Output_IH = I_Weights * _Input;
			for (Ulong i = 0; i < P_Data.GetSize(); i++)
				Output_IH[i] += P_Weights[i] * P_Data[i];
			this->Activation(Output_IH);

			Matrix Output_HO = O_Weights * Output_IH;
			this->Activation(Output_HO);

			//�덷�t�`��
			Matrix Output_Error = _Ideal - this->Query(_Input);
			Matrix Hidden_Error = O_Weights.Transpose() * Output_Error;

			//�d�݂̏C��
			Modifled(_Input, Output_IH, I_Weights, Hidden_Error);
			Modifled(Output_IH, Output_HO, O_Weights, Output_Error);

			//�ċA�̏d�݂̏C��
			for (Ulong i = 0; i < P_Weights.GetSize(); i++) {
				double P_Error = P_Weights[i] * Hidden_Error[i];
				double Out = P_Weights[i] * P_Data[i];
				P_Weights[i] += Learn_rate * P_Error * Out * (1 - Out) * P_Data[i];
			}

			//�ȑO�̏o�͂�ۑ�
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
		double (*Activation_Func)(double) = Sanae::ReLU;      //�������֐�            
		double (*Activation_Diff)(double) = Sanae::diff_ReLU; //���֐�

		double              Learn_rate = 0.001;
		std::vector<Matrix> Weights;


	private:
		virtual inline void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = Activation_Func(_In[i]);

			return;
		}

		//Weights(�d��)���C�����܂��B
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

			//�T�C�Y�ݒ�
			Weights.resize(Sum_Layer - 1);
			Output_Layer = Weights.size() - 1;

			//���͑w�̏d�݂�ݒ�
			Weights[Input_Layer].SetSize({ _Input_Node,_Hidden_Node });
			Init_Weight(&Weights[Input_Layer]);

			//�o�͑w�̏d�݂�ݒ�
			Weights[Output_Layer].SetSize({ _Hidden_Node,_Output_Node });
			Init_Weight(&Weights[Output_Layer]);

			//���ԑw�̏d�݂�ݒ�
			for (Ulong Layer = 1; Layer < (Sum_Layer - IO_Layer); Layer++)
			{
				Weights[Layer].SetSize({ _Hidden_Node,_Hidden_Node });
				Init_Weight(&Weights[Layer]);
			}
		}

		//�Ɖ�܂��B
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

		//�w�K�����܂��B
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

			//���ꂼ��̏o�͂�ۑ�����B
			_Outputs[0] = (Weights[Input_Layer] * _Input);
			this->Activation(_Outputs[0]);

			for (Ulong i = 1; i <= Output_Layer; i++) {
				_Outputs[i] = (Weights[i] * _Outputs[i - 1]);
				this->Activation(_Outputs[i]);
			}

			//�덷�t�`��
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