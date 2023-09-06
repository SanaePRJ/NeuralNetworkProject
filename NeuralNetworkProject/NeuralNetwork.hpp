/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
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
#include "CSV.hpp"
#include "NeuralNetworkDoc.hpp"


//����ɃC���N���[�h���Ă��邱�ƁB C++14�ȏ�
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {


	//���z�@�ɂ��d�݂��C������B
	void WeightUpdate_SGD
	(
		Layer_Affine* _Layer,
		double        _LearnRate
	)
	{
		_Layer->Weight -= _Layer->D_Weight * _LearnRate;
		_Layer->Bias -= _Layer->D_Bias * _LearnRate;

		return;
	}
	void WeightUpdate_SGD
	(
		Layer_Recurrent_Affine* _Layer,
		double                  _LearnRate
	)
	{
		_Layer->Recurrent -= _Layer->D_Recurrent * _LearnRate;
		_Layer->Weight    -= _Layer->D_Weight    * _LearnRate;
		_Layer->Bias      -= _Layer->D_Bias      * _LearnRate;

		return;
	}


	class NN 
	{
	private:
		//���C���̃|�C���^���i�[����B
		std::vector<Layer_Base*> Layers;
		
		//�w�K��
		double LearnRate;
		

	public:
		NN(Ulong _Input_Nodes,Ulong _Hidden_Nodes,Ulong _Output_Nodes,double _Learn_rate,unsigned int _Seed) 
		{
			//�V�[�h�l��ݒ�B
			std::default_random_engine Engine(_Seed); 
			LearnRate = _Learn_rate;
			
			//����
			std::normal_distribution<> DistIn    (0, (1 / std::sqrt(_Input_Nodes)));     //����0,�W���΍�(1/����͐�)�Ő���
			std::normal_distribution<> DistHidden(0, (1 / std::sqrt(_Hidden_Nodes)));    //����0,�W���΍�(1/����͐�)�Ő���
			std::normal_distribution<> DistOut   (0, (1 / std::sqrt(_Hidden_Nodes)));    //����0,�W���΍�(1/����͐�)�Ő���
			
			//Affine���C���̐ݒ�
			static std::vector<Layer_Affine> Affine;
			Affine.push_back({ _Input_Nodes,_Hidden_Nodes ,&Engine,DistIn });

			Affine.push_back({ _Hidden_Nodes,_Hidden_Nodes,&Engine,DistHidden });
			
			Affine.push_back({ _Hidden_Nodes,_Output_Nodes,&Engine,DistOut });

			//�V�O���C�h���C���̐ݒ�
			static std::vector<Layer_Sigmoid> Sigmoid;
			Sigmoid.push_back({});
			Sigmoid.push_back({});
			
			//SoftMax���C���̐ݒ�
			static Layer_SoftMax SoftMax  = {};
			
			//���C���Ɋi�[
			Layers.push_back(&Affine[0]);
			Layers.push_back(&Sigmoid[0]);

			Layers.push_back(&Affine[1]);
			Layers.push_back(&Sigmoid[1]);

			//Affine���C���̃��X�g
			Layers.push_back(&Affine[2]);
			Layers.push_back(&SoftMax);
		}

		//�\������B
		Matrix Predict(Matrix _Data) 
		{
			Matrix buf = Layers[0]->forward(&_Data);
			
			for (Ulong i = 1; i < Layers.size(); i++)
				buf = Layers[i]->forward(&buf);
			
			return buf;
		}

		//�w�K����B
		void Learn(Matrix _In, Matrix _t) {
			//��x�o�͂���
			Matrix buf = Layers[0]->forward(&_In);

			for (Ulong i = 1; i < Layers.size(); i++)
				buf = Layers[i]->forward(&buf);

			//�t�`�Ԃ���B
			//��Ԍ�̏���
			buf = Layers[Layers.size() -1]->backward(&_t);

			//�t�`��
			for (int i = (int)Layers.size() - 2; i >= 0; i--) {
				buf = Layers[i]->backward(&buf);

				//Affine���C���������ꍇWeight�̏C��
				if (Layers[i]->Is_Affine)
					WeightUpdate_SGD((Layer_Affine*)Layers[i], this->LearnRate);
			}
		}

		//�d�݂�����ǂݎ���悤CSV�ŏo�́B
		void Write(const char* _WriteFile) {
			WriteCSV write(_WriteFile,true);

			for (Ulong i = 0; i < this->Layers.size();i++) {
				if (this->Layers[i]->Is_Affine && i == 0) {
					Layer_Affine* affine          = (Layer_Affine*)this->Layers[i];
					_NeuralNetwork_DOC_Write_ Buf = {&write,i/2,&affine->Weight,&affine->Bias,NULL};
					
					NeuralNetwork_Write(Buf,true, this->Layers.size()/2);
				}else if (this->Layers[i]->Is_Affine) {
					Layer_Affine* affine = (Layer_Affine*)this->Layers[i];
					_NeuralNetwork_DOC_Write_ Buf = { &write,i / 2,&affine->Weight,&affine->Bias,NULL };

					NeuralNetwork_Write(Buf);
				}
			}
		}

		//�������܂ꂽ�d�݃o�C�A�X��ǂݎ��B
		void Read(const char* _ReadFile) {
			ReadCSV write(_ReadFile);

			for (Ulong i = 0; i < this->Layers.size(); i++) {
				if (this->Layers[i]->Is_Affine && i == 0) {
					Layer_Affine* affine         = (Layer_Affine*)this->Layers[i];
					_NeuralNetwork_DOC_Read_ Buf = { &write,i / 2,&affine->Weight,&affine->Bias,NULL };

					NeuralNetwork_Read(Buf, true, this->Layers.size() / 2);
				}
				else if (this->Layers[i]->Is_Affine) {
					Layer_Affine* affine         = (Layer_Affine*)this->Layers[i];
					_NeuralNetwork_DOC_Read_ Buf = { &write,i / 2,&affine->Weight,&affine->Bias,NULL };

					NeuralNetwork_Read(Buf);
				}
			}
		}
	};


}


#endif
#endif