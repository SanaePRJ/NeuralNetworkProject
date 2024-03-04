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

#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"
#include "NNLayers.hpp"
#include "CSV.hpp"




namespace Sanae {


	template<typename ty,typename ActClass = Sanae::Layer_ReLU<ty>,typename ActWithLoss=Sanae::Layer_SoftMaxWithLoss<ty>>
	class NN
	{
	private:
		//���C���̃|�C���^���i�[����B
		std::vector<Layer_Base<ty>*> Layers;

		//�w�K��
		double LearnRate;

		//���̓m�[�h��,���ԑw�m�[�h��,�o�͑w�m�[�h��,���ԑw(�B��w)���C���[��
		uint64_t Input_Nodes, Hidden_Nodes, Output_Nodes, Hidden_Layers;

	public:
		//�o�[�W����
		const size_t version = 0;

		//���͑w�m�[�h��,���ԑw�m�[�h��,�o�͑w�m�[�h��,�w�K��,�V�[�h�l,���ԑw(�B��w)���C���[��
		NN(uint64_t arg_Input_Nodes, uint64_t arg_Hidden_Nodes, uint64_t arg_Output_Nodes,double arg_Learn_rate, unsigned int arg_Seed, uint64_t arg_Hidden_Layers)
			:	LearnRate(arg_Learn_rate), Input_Nodes(arg_Input_Nodes), Hidden_Nodes(arg_Hidden_Nodes), Output_Nodes(arg_Output_Nodes), Hidden_Layers(arg_Hidden_Layers)
		{
			//����
			std::default_random_engine engine(arg_Seed);
			
			//-----���͑w-----
			static Sanae::Layer_Affine <ty> Affine1(arg_Input_Nodes, arg_Hidden_Nodes,&engine);   //Affine���C��
			Affine1.LearnRate = LearnRate;   //�w�K����ݒ�

			static ActClass Activation1;     //���������C��

			Layers.push_back((Sanae::Layer_Base<ty>*) & Affine1 );   //Affine���C��
			Layers.push_back((Sanae::Layer_Base<ty>*) & Activation1);//���������C����ǉ�

			//-----���ԑw-----
			static std::vector<Sanae::Layer_Affine<ty>> Affine2    (arg_Hidden_Layers, Sanae::Layer_Affine<ty>(arg_Hidden_Nodes,arg_Hidden_Nodes,&engine));   //Affine���C��
			static std::vector<ActClass>                Activation2(arg_Hidden_Layers);   //���������C��

			for (uint64_t i = 0; i < arg_Hidden_Layers;i++) {
				Layers.push_back((Sanae::Layer_Base<ty>*)(&Affine2[i]));     //Affine���C����ǉ�
				Layers.push_back((Sanae::Layer_Base<ty>*)(&Activation2[i])); //���������C����ǉ�

				Affine2[i].LearnRate = LearnRate;   //Affine���C���̊w�K����ݒ�
			}

			//-----�o�͑w-----
			static Sanae::Layer_Affine <ty> Affine3(arg_Hidden_Nodes, arg_Output_Nodes, &engine);   //Affine���C��
			Affine3.LearnRate = LearnRate;   //Affine���C���̊w�K����ݒ�

			static ActWithLoss awl;   //������-�����֐����C��

			Layers.push_back((Sanae::Layer_Base<ty>*) & Affine3);  //Affine���C���̒ǉ�
			Layers.push_back((Sanae::Layer_Base<ty>*) & awl);      //������-�����֐����C���̒ǉ�
		}

		//�\������B
		Sanae::Matrix<ty> Predict(Sanae::Matrix<ty> arg)
		{
			Sanae::Matrix<ty> buf = Layers[0]->forward(&arg);

			//�`�Ԃ��Ă���
			for (size_t i = 1; i < (Layers.size() - 1); i++)
				buf = Layers[i]->forward(&buf);
			
			return buf;
		}

		//�w�K����B
		double Learn(Sanae::Matrix<ty> arg_in, Sanae::Matrix<ty> arg_teach)
		{
			//��x�o�͂���
			Sanae::Matrix<ty> buf = Layers[0]->forward(&arg_in);
			
			for (size_t i = 1; i < (Layers.size() - 1); i++)
				buf = Layers[i]->forward(&buf);

			//���������߂�B
			double loss = ((Layer_SoftMaxWithLoss<ty>*)Layers[Layers.size() - 1])->forward(&buf, &arg_teach);

			//�t�`�Ԃ���B
			//��Ԍ�̏���
			buf = ((Layer_SoftMaxWithLoss<ty>*)Layers[Layers.size() - 1])->backward();

			//�t�`��
			for (int i = (int)Layers.size() - 2; i >= 0; i--)
				buf = Layers[i]->backward(&buf);

			return loss;
		}

		//�d��,�o�C�A�X��ۑ�
		NN& Save(const char* FileName) 
		{
			//CSV�t�@�C����������
			Sanae::WriteCSV save(FileName,true);

			//NeuralNetwork-SanaeProject�ō��ꂽ�t�@�C���ł��邱��,�o�[�W�����̏�������
			save << "NeuralNetwork-SanaeProject" << std::to_string(version) << "\n";

			//���̓m�[�h��,���ԑw�m�[�h��,�o�͑w�m�[�h��,���ԑw���C���[������������
			save << Input_Nodes << Hidden_Nodes << Output_Nodes << Hidden_Layers << "\n";

			//���������C���[,�������E�o�̓��C���[
			save << typeid(ActClass).name() << typeid(ActWithLoss).name() << "\n";

			//���ׂĂ̑w�̏d��,BIAS��ۑ�����B
			for (size_t i = 0; i < Layers.size(); i+=2) {
				Sanae::Layer_Affine<ty>* LayerP = (Sanae::Layer_Affine<ty>*)Layers[i];

				//���C���ԍ�,Weight���n�܂邱�Ƃ���������
				save << i / 2 << "\n" << "Weight" << "\n";
				//Weight��������
				for (uint64_t k = 0; k < LayerP->Weight.get_row(); k++) {
					save << LayerP->Weight[k] << "\n";
					save.flush();
				}

				//bias���n�܂邱�Ƃ���������
				save << "Bias" << "\n";
				//Bias��������
				for (uint64_t k = 0; k < LayerP->Bias.get_row(); k++) {
					save << LayerP->Bias[k] << "\n";
					save.flush();
				}
			}

			return *this;
		}

		NN& Load(const char* FileName) {
			Sanae::ReadCSV load;

			//CSV�t�@�C���ǂݎ��
			try {
				load.open(FileName);
			}
			catch (std::exception e) {
				throw e;
			}

			//csv�t�@�C����NeuralNetwork-SanaeProject�ō��ꂽ�`�����`�F�b�N
			if (load.ReadData() != "NeuralNetwork-SanaeProject")
				throw std::invalid_argument("The format of this file is incorrect.");

			//csv�t�@�C���̃o�[�W�������m�F
			if (load.ReadData() != std::to_string(version))
				throw std::invalid_argument("The version is different.");

			//�m�[�h�ڍ�
			uint8_t error=
			(Input_Nodes   != std::stoull(load.ReadData())) +   //���͑w�m�[�h��
			(Hidden_Nodes  != std::stoull(load.ReadData())) +   //���ԑw�m�[�h��
			(Output_Nodes  != std::stoull(load.ReadData())) +   //�o�͑w�m�[�h��
			(Hidden_Layers != std::stoull(load.ReadData()));    //���ԃ��C���[��

			//error���������ꍇ
			if (error !=0)
				throw std::invalid_argument("Different size.");

			//���C��class�������Ă��邩�ǂ���
			if (typeid(ActClass).name() != load.ReadData())
				throw std::invalid_argument("The ActivationClass is different.");

			if (typeid(ActWithLoss).name() != load.ReadData())
				throw std::invalid_argument("The ActivationWithLossClass is different.");


			//���ׂĂ̑w�̏d��,BIAS��ǂݍ��ށB
			for (size_t lay = 0; lay < Layers.size(); lay+=2) {
				//���C���̃|�C���^
				Sanae::Layer_Affine<ty>* LayerP = (Sanae::Layer_Affine<ty>*)Layers[lay];

				//���C���ԍ����قȂ��Ă���ꍇ�������Ⴄ
				if (load.ReadData() != std::to_string((size_t)(lay / 2)))
					throw std::invalid_argument("The format of this file is incorrect.[num]");
				
				//Weight���Ȃ��ꍇ�������Ⴄ
				if(load.ReadData() != std::string("Weight"))
					throw std::invalid_argument("The format of this file is incorrect.[weight]");

				//Weight�̊e�v�f��ǂݍ���
				for (size_t row = 0; row < LayerP->Weight.get_row(); row++)
					for (size_t column = 0; column < LayerP->Weight.get_column(); column++)
						LayerP->Weight[row][column] = std::stod(load.ReadData());

				//Bias���Ȃ��ꍇ�������Ⴄ
				if (load.ReadData() !="Bias")
					throw std::invalid_argument("The format of this file is incorrect.[bias]");

				//Bias�̊e�v�f��ǂݍ���
				for (size_t row = 0; row < LayerP->Bias.get_row(); row++)
					for (size_t column = 0; column < LayerP->Bias.get_column(); column++)
						LayerP->Bias[row][column] = std::stod(load.ReadData());
			}

			return *this;
		}
	};
}




#endif