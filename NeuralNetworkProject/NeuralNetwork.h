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
		double learn_late = 0.5;   //�w�K��
		
	
	//Define Variable(private)
	private:
		Ulong In_Node     = 3;     //���̓m�[�h�� 
		Ulong Hidden_Node = 3;     //�B��w���̓m�[�h��
		Ulong Output_Node = 3;     //�o�̓m�[�h��

		Ulong  Hidden_Layer   = 3;   //�B��w��
		const Ulong  IO_Layer = 2;   //Input/Output�p���C���[

		std::vector<Matrix> Weights; //�d��

	
	//Define Functions(private)
	private:
		//�V�O���C�h�֐�
		void Sigmoid
		(
			Matrix& _In
		) 
		{
			//sigmoid(x)= 1 / (1+e^-x)
			for (Ulong i = 0; i < _In.GetSize(); i++)
				_In[i]   = 1 / (1 + std::exp(-1 * _In[i]));
		}

		//���̊֐��ɂ��d�݂����P����B
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
			//�����̃V�[�h�l�ݒ�
			srand(_Seed);
			//�e�w�̏d�݂�ݒ�
			Weights.resize(Hidden_Layer+2);

			//���͑w->�B��w
			Weights[0]               .SetSize({In_Node    , Hidden_Node});
			//�B��w->�o�͑w
			Weights[Hidden_Layer + 1].SetSize({Hidden_Node, Output_Node});

			//���͑w�d�݂̏����ݒ�(0.00~0.99)
			for (Ulong j = 0; j < Weights[0].GetSize(); j++)
				(Weights[0])[j]                = (double)(rand() % 100) / 100;

			//�o�͑w�d�݂̏����ݒ�(0.00~0.99)
			for (Ulong j = 0; j < Weights[Hidden_Layer + 1].GetSize(); j++)
				(Weights[Hidden_Layer + 1])[j] = (double)(rand() % 100) / 100;

			//�B��w�d�݂̏����ݒ�(0.00~0.99)
			for (Ulong i = 1; i < Hidden_Layer+1; i++)
			{
				Weights[i].SetSize({Hidden_Node,Hidden_Node});
				for (Ulong j = 0; j < Weights[i].GetSize(); j++)
					(Weights[i])[j] = (double)(rand() % 100) / 100;
			}
		}

		//�o�͂��܂��B
		Matrix Out
		(
			Matrix& In
		) 
		{
			//�e�w�̏o�͒l�����߂�B
			Matrix Out = Weights[0] * In;
			Sigmoid(Out);

			for (Ulong i = 1; i < this->Hidden_Layer+IO_Layer;i++) {
				Out = Weights[i] * Out;
				Sigmoid(Out);
			}

			return Out;
		}

		//�d�݂��o�͂��܂��B
		Matrix& GetWeight
		(
			Ulong Num
		)
		{
			return this->Weights[Num];
		}

		//�w�K���܂��B
		void Learn
		(
			Matrix& In,
			Matrix& Ideal
		)
		{
			//�e�w�o�͊i�[�p
			std::vector<Matrix> Output;
			Output.resize(this->Hidden_Layer+ IO_Layer);

			//�e�w�덷�i�[�p
			std::vector<Matrix> Error;
			Error.resize(this->Hidden_Layer + IO_Layer);

			//�o�͂����߂�B
			Output[0] = Weights[0] * In;
			Sigmoid(Output[0]);

			for (Ulong i = 1; i < this->Hidden_Layer+ IO_Layer;i++){
				Output[i] = Weights[i] * Output[i-1];
				Sigmoid(Output[i]);
			}

			//�덷�����߂�B
			Error[this->Hidden_Layer + 1] = Ideal - Output[this->Hidden_Layer + 1];
			
			for (Ulong i = this->Hidden_Layer; i > 0; i--)
				Error[i] = Weights[i + 1].Transpose() * Error[i + 1];
				
			Error[0] = Weights[1].Transpose() * Error[1];

			//�d�݂𒲐�����
			Modifled(In, Weights[0], Error[0]);
			for (Ulong i = 1; i < this->Hidden_Layer + IO_Layer; i++)
				Modifled(Output[i - 1], Weights[i], Error[i]);
		}
	};
}


#endif