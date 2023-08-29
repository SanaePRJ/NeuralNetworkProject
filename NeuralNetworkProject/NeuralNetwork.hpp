/*=============================================================
* NAME      : NeuralNetwork.hpp
* AUTHOR    : SanaeProject
* VER       : 1.0.0
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
#include "ReadCSV.hpp"


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
		_Layer->Bias   -= _Layer->D_Bias   * _LearnRate;

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
			std::normal_distribution<> Dist  (0,1);   //����0,�W���΍�1�Ő���

			LearnRate = _Learn_rate;
			
			//Affine���C���̐ݒ�
			static std::vector<Layer_Affine> Affine;

			Affine.push_back({ _Input_Nodes,_Hidden_Nodes ,&Engine,Dist });

			Affine.push_back({ _Hidden_Nodes,_Hidden_Nodes,&Engine,Dist });
			Affine.push_back({ _Hidden_Nodes,_Hidden_Nodes,&Engine,Dist });

			Affine.push_back({ _Hidden_Nodes,_Output_Nodes,&Engine,Dist });

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
			Layers.push_back(&*(Affine.end()-1));
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
			for (int i = Layers.size() - 2; i >= 0; i--) {
				buf = Layers[i]->backward(&buf);

				//Affine���C���������ꍇWeight�̏C��
				if (Layers[i]->Is_Affine)
					WeightUpdate_SGD((Layer_Affine*)Layers[i], this->LearnRate);
			}
		}

		//�d�݂�����ǂݎ���悤CSV�ŏo�́B
		void Write(const char* _WriteFile) {
			std::ofstream ofs(_WriteFile);
			ofs << "AffineLayerCount,"<< this->Layers.size()/2 << "\n";

			auto WriteAffine = [&](Ulong Layer_Point)
			{
				//Affine���C���̔ԍ����o��
				(ofs << Layer_Point) << "\n";

				//�d�݂̗�,�s�����i�[
				SizeT Size = ((Layer_Affine*)Layers[Layer_Point])->Weight.GetSizeWH();

				//�񐔂ƍs������������
				ofs << "Weight," << Size.first<< "," << Size.second << "\n";

				for (Ulong i = 0; i < ((Layer_Affine*)Layers[Layer_Point])->Weight.GetSize(); i++) {
					//���ꂼ��̏d�݂���������ł���
					(ofs << ((Layer_Affine*)Layers[Layer_Point])->Weight[i]);

					//���s
					if ((i + 1) % Size.first == 0)
						ofs << "\n";
					//CSV�Ȃ̂ŃR���}��؂�
					else
						ofs << ",";
				}

				//�o�C�A�X�T�C�Y�̏�������
				Size = ((Layer_Affine*)Layers[Layer_Point])->Bias.GetSizeWH();
				//�o�C�A�X�̗�,�s������������
				ofs << "Bias," << Size.first << "," << Size.second << "\n";

				for (Ulong i = 0; i < ((Layer_Affine*)Layers[Layer_Point])->Bias.GetSize(); i++) {
					//���ꂼ��̃o�C�A�X�̏�������
					(ofs << ((Layer_Affine*)Layers[Layer_Point])->Bias[i]);

					if ((i + 1) % Size.first == 0)
						ofs << "\n";
					else
						ofs << ",";
				}
			};

			//�e�w�̏d�݁A�o�C�A�X���o��
			for (Ulong i = 0; i < this->Layers.size();i++) {
				if (Layers[i]->Is_Affine)
					WriteAffine(i);
			}
		}

		//�������܂ꂽ�d�݃o�C�A�X��ǂݎ��B
		void Read(const char* _ReadFile) {
			//ReadCSV�ŊJ��
			ReadCSV input = _ReadFile;

			Ulong Count = 0;
			for (Ulong i = 0; i < this->Layers.size();i++)
				Count += this->Layers[i]->Is_Affine ? 1: 0;
			
			//�`���ɍ���Ȃ��ꍇ�̓G���[
			if(input.ReadDataStr<std::string>() != "AffineLayerCount") throw std::invalid_argument("");
			if(input.ReadDataD<double>() != Count)                     throw std::invalid_argument("");

			//�d�݂�ǂݎ��B
			auto ReadWeight = [&](Ulong point)
			{
				//�`��������Ȃ��ꍇ�̓G���[
				if (input.ReadDataStr<std::string>() != "Weight") throw std::invalid_argument("");

				//��,�s�����i�[
				SizeT Size = SizeT{ input.ReadDataI<int>(),input.ReadDataI<int>() };
				
				//�d�݂��i�[
				std::vector<double> Weight_buf;
				
				//�ǂݎ��B
				for (Ulong i = 0; i < Size.second; i++)
					input.ReadLineD<double>(&Weight_buf, Size.first);
				
				//�d�݂��i�[����vector�z���Matrix�^��move����B
				((Layer_Affine*)Layers[point])->Weight.Move(&Weight_buf, Size);
			};

			//�o�C�A�X��ǂݎ��B
			auto ReadBias   = [&](Ulong point)
			{
				//�`���ɍ���Ȃ��ꍇ�̓G���[
				if (input.ReadDataStr<std::string>() != "Bias") throw std::invalid_argument("");

				//�񐔍s�����i�[
				SizeT Size = SizeT{ input.ReadDataI<int>(),input.ReadDataI<int>() };
				
				//�o�C�A�X�̊i�[
				std::vector<double> Bias_buf;

				//�ǂݎ��B
				for (Ulong i = 0; i < Size.second; i++)
					input.ReadLineD<double>(&Bias_buf, Size.first);

				//�d�݂��i�[����vector�z���Matrix�^��move����B
				((Layer_Affine*)Layers[point])->Bias.Move(&Bias_buf, Size);
			};

			for (Ulong i = 0; i < Count;i++) {
				Ulong Pos = (Ulong)input.ReadDataI<int>();

				//�d�݂ƃo�C�A�X��ǂݎ��B
				ReadWeight(Pos);
				ReadBias(Pos);
			}
		}
	};


}


#endif
#endif