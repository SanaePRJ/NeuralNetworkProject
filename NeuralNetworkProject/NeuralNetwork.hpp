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


	template<typename ty>
	class NN
	{
	private:
		//ƒŒƒCƒ„‚Ìƒ|ƒCƒ“ƒ^‚ğŠi”[‚·‚éB
		std::vector<Layer_Base<ty>*> Layers;

		//ŠwK—¦
		double LearnRate;


	public:
		NN(uint64_t arg_Input_Nodes, uint64_t arg_Hidden_Nodes, uint64_t arg_Output_Nodes,double arg_Learn_rate, unsigned int arg_Seed)
			:	LearnRate(arg_Learn_rate)
		{
			std::default_random_engine engine(arg_Seed);

			static Sanae::Layer_Affine <ty> Affine1(arg_Input_Nodes, arg_Hidden_Nodes,&engine);
			Affine1.LearnRate = LearnRate;

			static Sanae::Layer_ReLU<ty> Sigmoid1;
			static Sanae::Layer_Affine <ty> Affine2 (arg_Hidden_Nodes, arg_Hidden_Nodes, &engine);
			Affine2.LearnRate = LearnRate;

			static Sanae::Layer_ReLU<ty> Sigmoid2;
			static Sanae::Layer_Affine <ty> Affine3(arg_Hidden_Nodes, arg_Output_Nodes, &engine);
			Affine3.LearnRate = LearnRate;

			static Sanae::Layer_SoftMaxWithLoss<double> swl;

			Layers.push_back((Sanae::Layer_Base<ty>*) &Affine1);
			Layers.push_back((Sanae::Layer_Base<ty>*) &Sigmoid1);
			Layers.push_back((Sanae::Layer_Base<ty>*) &Affine2);
			Layers.push_back((Sanae::Layer_Base<ty>*) & Sigmoid2);
			Layers.push_back((Sanae::Layer_Base<ty>*) & Affine3);
			Layers.push_back((Sanae::Layer_Base<ty>*) &swl);
		}

		//—\‘ª‚·‚éB
		Sanae::Matrix<ty> Predict(Sanae::Matrix<ty> arg)
		{
			Sanae::Matrix<ty> buf = Layers[0]->forward(&arg);

			for (size_t i = 1; i < (Layers.size() - 1); i++)
				buf = Layers[i]->forward(&buf);
			
			return buf;
		}

		//ŠwK‚·‚éB
		double Learn(Sanae::Matrix<ty> arg_in, Sanae::Matrix<ty> arg_teach)
		{
			//ˆê“xo—Í‚·‚é
			Sanae::Matrix<ty> buf = Layers[0]->forward(&arg_in);
			
			for (size_t i = 1; i < (Layers.size() - 1); i++)
				buf = Layers[i]->forward(&buf);

			double loss = ((Layer_SoftMaxWithLoss<ty>*)Layers[Layers.size() - 1])->forward(&buf, &arg_teach);

			//‹t“`”Ô‚·‚éB
			//ˆê”ÔŒã‚Ìˆ—
			buf = ((Layer_SoftMaxWithLoss<ty>*)Layers[Layers.size() - 1])->backward();

			//‹t“`”Ô
			for (int i = (int)Layers.size() - 2; i >= 0; i--)
				buf = Layers[i]->backward(&buf);

			return loss;
		}
	};
}




#endif