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


//����ɃC���N���[�h���Ă��邱�ƁB C++14�ȏ�
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {


	//Neural Network
	class NN {
	public:
		double Learn_rate = 0.5;

		Matrix I_Weights;	//���͑w�̏d��
		Matrix O_Weights;	//�o�͑w�̏d��


	protected:
		//�V�O���C�h�֐����������֐��Ƃ���B
		inline void Activation(Matrix& _In)
		{
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i] = 1 / (1 + std::exp(-1 * _In[i]));
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
				buf[i] = Error[i] * Output[i] * (1 - Output[i]);

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