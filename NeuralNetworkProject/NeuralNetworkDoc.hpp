/*=============================================================
* NAME      : NeuralNetworkDoc.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_NEURALNETWORKDOC_HPP_
#define _SANAE_NEURALNETWORKDOC_HPP_


#include <fstream>
#include <string>

#include "Matrix.hpp"
#include "CSV.hpp"


namespace Sanae {
	//�������ލۂ̏����i�[����B
	struct _NeuralNetwork_DOC_Write_
	{
		WriteCSV* _Write;
		Ulong     _LayerCount;
		
		Matrix* _Weight;
		Matrix* _Bias  ;
		Matrix* _Recurrent = NULL;
	};

	//�ǂݍ��ލۂ̏����i�[����B
	struct _NeuralNetwork_DOC_Read_
	{
		ReadCSV* _Read;
		Ulong    _LayerCount;

		Matrix* _Weight;
		Matrix* _Bias;
		Matrix* _Recurrent = NULL;
	};


	//�w�K�����d�݂�o�C�A�X���w�肵���t�@�C���ɏ������݂܂��B
	void NeuralNetwork_Write(_NeuralNetwork_DOC_Write_ _Data,bool _Is_First = false,Ulong _AllLayerCount = 0) 
	{
		WriteCSV& Write = *_Data._Write;
		auto write_lambda = [&](Matrix* _DataP)
		{
			for (Ulong i = 0; i < _DataP->GetSize(); i++) {
				Write << (*_DataP)[i];

				if ((i + 1) % _DataP->GetSizeWH().first == 0)
					Write << "\n";
			}

			return;
		};

		if(_Is_First)
			Write << "LayerCount" << _AllLayerCount << "\n";

		Write << _Data._LayerCount << "\n";

		//Weight��������
		Write << "Weight" << _Data._Weight->GetSizeWH().first << _Data._Weight->GetSizeWH().second << "\n";
		write_lambda(_Data._Weight);

		//Bias��������
		Write << "Bias" << _Data._Bias->GetSizeWH().first << _Data._Bias->GetSizeWH().second << "\n";
		write_lambda(_Data._Bias);

		if (_Data._Recurrent != NULL) {
			//Recurrent��������
			Write << "Recurrent" << _Data._Recurrent->GetSizeWH().first << _Data._Recurrent->GetSizeWH().second << "\n";
			write_lambda(_Data._Recurrent);
		}
	}

	//�������܂ꂽ�d�݂�o�C�A�X�Ȃǂ�ǂݍ��݂܂��B
	void NeuralNetwork_Read(_NeuralNetwork_DOC_Read_ _Data, bool _Is_First=false, Ulong _AllLayerCount=0)
	{
		ReadCSV& Read = *_Data._Read;
		if (_Is_First) {
			if (Read.ReadData() != "LayerCount")
				throw std::runtime_error("Different save format");

			if (Read.ReadData() != std::to_string(_AllLayerCount))
				throw std::runtime_error("Different LayerCount");
		}

		if (Read.ReadData() != std::to_string(_Data._LayerCount))
			throw std::runtime_error("Different LayerCount");

		auto read_lambda = [&](const char* name, Matrix* _DataP)
		{
			if(Read.ReadData()!=(std::string)name)
				throw std::runtime_error("Different save format");
			
			Ulong x, y;
			try {
				x = std::stoull(Read.ReadData());
				y = std::stoull(Read.ReadData());
			}
			catch (std::exception e) 
			{
				throw std::runtime_error("Different save format");
			}

			(*_DataP) = SizeT{x,y};

			for (Ulong i = 0; i < _DataP->GetSize(); i++)
				(*_DataP)[i] = std::stod(Read.ReadData());
		};

		read_lambda("Weight",_Data._Weight);
		read_lambda("Bias", _Data._Bias);

		if (_Data._Recurrent != NULL) {
			//Recurrent�ǂݍ���
			read_lambda("Recurrent",_Data._Recurrent);
		}
	}


}


#endif