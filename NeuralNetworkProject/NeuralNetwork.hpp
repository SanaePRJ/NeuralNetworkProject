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

#include "Matrix.h"
#include "NeuralNetworkLayers.hpp"


//正常にインクルードしていること。 C++14以上
#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {
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
		std::vector<Layer_Base*> Layers;

		double LearnRate;
		

	public:
		NN(Ulong _Input_Nodes,Ulong _Hidden_Nodes,Ulong _Output_Nodes,double _Learn_rate,unsigned int _Seed) 
		{
			srand(_Seed);

			LearnRate = _Learn_rate;

			static Layer_Affine Affine1 = {_Input_Nodes,_Hidden_Nodes,_Seed};
			static Layer_Affine Affine2 = {_Hidden_Nodes,_Hidden_Nodes,_Seed};
			static Layer_Affine Affine3 = {_Hidden_Nodes,_Output_Nodes,_Seed};
			
			static Layer_Sigmoid Sigmoid1 = {};
			static Layer_Sigmoid Sigmoid2 = {};

			static Layer_SoftMax SoftMax  = {};
			
			Layers.push_back(&Affine1);
			Layers.push_back(&Sigmoid1);

			Layers.push_back(&Affine2);
			Layers.push_back(&Sigmoid2);

			Layers.push_back(&Affine3);
			Layers.push_back(&SoftMax);
		}

		Matrix Predict(Matrix _Data) 
		{
			Matrix buf = Layers[0]->forward(&_Data);
			
			for (Ulong i = 1; i < Layers.size(); i++)
				buf = Layers[i]->forward(&buf);
			
			return buf;
		}

		void Learn(Matrix _In, Matrix _t) {
			Matrix buf = Layers[0]->forward(&_In);

			for (Ulong i = 1; i < Layers.size(); i++)
				buf = Layers[i]->forward(&buf);

			buf = Layers[Layers.size() -1]->backward(&_t);
			for (int i = Layers.size() - 2; i >= 0; i--) {
				buf = Layers[i]->backward(&buf);

				if (Layers[i]->Is_Affine)
					WeightUpdate_SGD((Layer_Affine*)Layers[i], this->LearnRate);
			}
		}
	};


}


#endif
#endif