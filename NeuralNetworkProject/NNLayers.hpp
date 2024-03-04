/*-------------------------------------------------------------
* Name    : NNLayers.hpp
* Version : 2.0.0
* * Author: SanaePRJ
* Description:
*  ���C���[�̎���
* �[��������DeepLearning�ŏ�����Ă���python�̃R�[�h���Q�l��
* �쐬���Ă��܂��B
-------------------------------------------------------------*/




#ifndef SANAE_NNLAYERS_HPP
#define SANAE_NNLAYERS_HPP




#include <stdint.h>
#include <random>
#include <algorithm>
#include <stdexcept>

#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"




namespace Sanae {


/*-----�����֐�-----*/
	

	//�����G���g���s�[�덷 
	//E=-(1/N)*��[n=1->N](��[k=1,K](Teach[n][k]log(Out[n][k])))
	template<typename ty>
	double Cross_Entropy_Error(Sanae::Matrix<ty>* Out, Sanae::Matrix<ty>* Teach)
	{
		//���ۂ̏o�͒l�Ƌ��t�f�[�^�̃T�C�Y���Ⴄ�ꍇ
		if (Out->get_row() != Teach->get_row() || Out->get_column() != Teach->get_column())
			throw std::invalid_argument("Must be same size.");

		//�f�[�^�͈�s�œn����邽�ߍs����batch����
		size_t batch_size = Out->get_row();

		double ans        = 0;    //�덷
		double delta      = 1e-7; //log(0)�ɂȂ�̂�h������

		//��[n=1->N]
		for (size_t row = 0; row < batch_size; row++)
		{
			//-��[k=1,K](Teach[n][k]log(Out[n][k]))
			for (size_t column = 0; column < Out->get_column(); column++)
			{
				double predicted = (*Out  )[row][column];
				double target    = (*Teach)[row][column];
				 
				ans -= target * std::log(predicted + delta);
			}
		}

		//(1/N)
		return ans / static_cast<double>(batch_size);
	}

	//���a�덷
	//E = (1/(2N))*��[n->N](��[k->K]((Out[n][k]-Teach[n][k])^2))
	template<typename ty>
	double Mean_Squared_Error(Sanae::Matrix<ty>* Out, Sanae::Matrix<ty>* Teach)
	{
		//���ۂ̏o�͒l�Ƌ��t�f�[�^�̃T�C�Y���Ⴄ�ꍇ
		if (Out->get_row() != Teach->get_row() || Out->get_column() != Teach->get_column())
			throw std::invalid_argument("Must be same size.");

		
		size_t batch_size = Out->get_row(); //�f�[�^�͈�s�œn����邽�ߍs����batch����
		double ans        = 0;              //�덷

		//n���󂯎��,��[k->K]((Out[n][k]-Teach[n][k])^2)
		auto sum_error = [&Out, &Teach](size_t arg_row) {
				//�Ԃ�l
				double ans = 0;

				for (size_t column = 0; column < Out->get_column(); column++)
					ans += (double)std::pow(((*Out)[arg_row][column] - (*Teach)[arg_row][column]), 2);

				return ans;
			};

		//��[n->N]
		for (size_t row = 0; row < batch_size; row++)
			ans += sum_error(row);

		//(1/(2N))
		return ans / (2 * batch_size);
	}
	

/*-----�������֐�-----*/


	//SoftMax�֐�
	//O[r][k] = {e^a[r][k]-Max[r]} / {��[i=1->n](e^a[r][i]-Max[r])}
	template<typename ty>
	Sanae::Matrix<ty> SoftMax(Sanae::Matrix<ty>* arg) 
	{
		//���̍ő�l��Ԃ��BMax[r]
		auto get_max = [&arg](size_t arg_row)->ty{
				return *std::max_element((*arg)[arg_row].begin(), (*arg)[arg_row].end());
			};

		//���v�l���o�� ��[i=1->n](e^a[r][i]-Max[r])
		auto get_sum = [&arg](size_t arg_row,ty arg_max)->ty{
				ty ans = 0;

				for (size_t column = 0; column < arg->get_column(); column++)
					ans += std::exp((*arg)[arg_row][column] - arg_max);

				return ans;
			};

		size_t            batch_size = arg->get_row();   //�f�[�^�͈�s�œn����邽�ߍs����batch����
		Sanae::Matrix<ty> ret        = std::pair<size_t,size_t>{arg->get_row(),arg->get_column()};  //�T�C�Y��ݒ�
		
		//O[r]
		for (size_t row = 0; row < batch_size; row++) {
			ty max = get_max(row);
			ty sum = get_sum(row,max);

			//{e^a[r][k]-Max[r]} / {��[i=1->n](e^a[r][i]-Max[r])}
			for (size_t column = 0; column < arg->get_column(); column++)
				ret[row][column] = std::exp((*arg)[row][column] - max) / sum;
		}

		return ret;
	}

	//�P���֐� In=Out
	template<typename ty>
	Sanae::Matrix<ty> Identity(Sanae::Matrix<ty>* arg) {
		return *arg;
	}

	//�V�O���C�h�֐� Sigmoid(x) = 1/(1+e^-gain * x)
	template<typename ty>
	ty Sigmoid(ty arg,ty gain = 1) {
		return 1 / (1 + std::exp(-1 * gain * arg));
	}


/*-----�d�݁E�o�C�A�X�̍X�V�֐�-----*/


	//W1 = W0 - (��E/��W0)*��
	template<typename ty>
	void SGD(Sanae::Matrix<ty>* arg,Sanae::Matrix<ty>* darg,double learn_rate) {
		*arg = (*arg) - (*darg) * learn_rate;
	}


/*-----���C���[-----*/


	//�x�[�X���C���[
	template<typename ty>
	class Layer_Base {
	public:
		bool is_affine = false;  //Affine���C���[�̏ꍇ�C���^�[�t�F�[�X���قȂ邽��

		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>*)  { return { {} }; }  //���̑w��
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>*) { return { {} }; }  //�t�`��
	};

	//Sigmoid���C���[
	template<typename ty>
	class Layer_Sigmoid : public Layer_Base<ty> {
	private:
		Sanae::Matrix<ty> out;  //�o�͒l��ۑ����Ă���

	public:
		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>* arg) override
		{
			//�T�C�Y�ݒ�
			out = std::pair<size_t,size_t>{arg->get_row(),arg->get_column()};

			//���ׂĂ̗v�f��Sigmoid�֐��ɒʂ��B
			for (size_t row = 0; row < arg->get_row();row++) {
				for (size_t column = 0; column < arg->get_column();column++) {
					out[row][column] = Sigmoid<ty>((*arg)[row][column]);
				}
			}

			return out;
		}
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>* arg) override 
		{
			//�t�`�Ԃ̃T�C�Y���Ԉ���Ă���ꍇ
			if (out.get_row() != arg->get_row() || out.get_column() != arg->get_column())
				throw std::invalid_argument("Must be same size.");

			//�d�݉��P�̒l
			Sanae::Matrix<ty> dx = std::pair<size_t, size_t>{out.get_row(),out.get_column()};

			//���ׂĂ̗v�f���v�Z
			for (size_t row = 0; row < arg->get_row(); row++) {
				for (size_t column = 0; column < arg->get_column(); column++) {
					//Sigmoid'(x) = Sigmoid(x)*(1-Sigmoid(x))
					dx[row][column] = (*arg)[row][column] * (1 - out[row][column]) * out[row][column];
				}
			}

			return dx;
		}
	};

	//ReLU���C���[
	template<typename ty>
	class Layer_ReLU : public Layer_Base<ty> {
	private:
		Sanae::Matrix<ty> Mask;  //�l��0�𒴂���Ƃ��t�`�Ԃ̒l��ʂ����ߕۑ�

	public:
		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>* arg) override
		{
			
			Sanae::Matrix<ty> out = *arg;  //�Ԃ�l
			Mask = std::pair<size_t, size_t>{ arg->get_row(),arg->get_column() };  //Mask�̃T�C�Y�ݒ�

			//���ׂĂ̗v�f�ɑ΂���
			for (size_t row = 0; row < arg->get_row(); row++) {
				for (size_t column = 0; column < arg->get_column(); column++) {
					//0�𒴂���Ƃ�1�ɂ��邱�ƂŎ��̑w,�O�̑w�֑���
					if ((*arg)[row][column] > 0) {
						Mask[row][column] = 1;
					}
					else {
						Mask[row][column] = 0;
						out[row][column]  = 0;  //0�ȉ��̏ꍇ���̑w�֓n���Ȃ��B
					}
				}
			}

			return out;
		}
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>* arg) override
		{
			return Mask^*arg;  //���͂̍ۂ�0�𒴂��Ă���ꍇ�O�̑w�֋t�`��
		}
	};

	//SoftMaxWithLoss���C���[
	template<typename ty>
	class Layer_SoftMaxWithLoss 
	{
	private:
		Sanae::Matrix<ty> Out;   //�o�͒l
		Sanae::Matrix<ty> Teach; //���t�f�[�^

	public:
		inline virtual double forward(Sanae::Matrix<ty>* arg_Out,Sanae::Matrix<ty>* arg_Teach)
		{
			Out   = Sanae::SoftMax(arg_Out);  //SoftMax�֐��ɒʂ����l���o��
			Teach = *arg_Teach;               //���t�f�[�^��ۑ�

			return Sanae::Cross_Entropy_Error(&Out,&Teach); //�����G���g���s�[�덷���o��
		}
		inline virtual Sanae::Matrix<ty> backward()
		{
			size_t bachsize = Out.get_row();  //�f�[�^�͈�s�œn����邽�ߍs����batch����

			//batch�T�C�Y�Ƌ��t�f�[�^��batch�T�C�Y���قȂ�ꍇ
			if (bachsize != Teach.get_row())
				throw std::invalid_argument("Must be same bachsize.");

			//��E/��W = Out-Teach
			Sanae::Matrix<ty> Dout = Out - Teach;
			
			for (size_t row = 0; row < Dout.get_row(); row++) {
				for (size_t column = 0; column < Dout.get_column(); column++) {
					Dout[row][column] /= bachsize;
				}
			}

			return Dout;
		}
	};

	//IdentityWithLoss���C���[
	template<typename ty>
	class Layer_IdentityWithLoss
	{
	private:
		Sanae::Matrix<ty> Out;
		Sanae::Matrix<ty> Teach;

	public:
		inline virtual double forward(Sanae::Matrix<ty>* arg_Out, Sanae::Matrix<ty>* arg_Teach)
		{
			Out   = Sanae::Identity(arg_Out);  //�P���֐��֒ʂ����l���o��
			Teach = *arg_Teach;                //���t�f�[�^

			return Sanae::Mean_Squared_Error(&Out, &Teach);  //���a�덷���o��
		}
		inline virtual Sanae::Matrix<ty> backward()
		{
			size_t bachsize = Out.get_row();  //�f�[�^�͈�s�œn����邽�ߍs����batch����

			//batch�ŉE�Ƌ��t�f�[�^��batch�T�C�Y���قȂ��Ă���ꍇ
			if (bachsize != Teach.get_row())
				throw std::invalid_argument("Must be same bachsize.");

			//��E/��W=Out-Teach
			Sanae::Matrix<ty> Dout = Out - Teach;

			for (size_t row = 0; row < Dout.get_row(); row++) {
				for (size_t column = 0; column < Dout.get_column(); column++) {
					Dout[row][column] /= bachsize;
				}
			}

			return Dout;
		}
	};

	//Affine���C���[
	template<typename ty>
	class Layer_Affine :public Layer_Base<ty> {
	public:
		Sanae::Matrix<ty> Weight,Bias;

	private:
		Sanae::Matrix<ty> DWeight;  //(��E/��W)
		Sanae::Matrix<ty> DBias;    //(��E/��B)

		Sanae::Matrix<ty> Arg_Input;//���͒l

	public:
		bool   is_affine = true;  //Affine���C���[
		double LearnRate = 0.03;  //�w�K���̏����l:3%

		void (*UpdateFunc)(Sanae::Matrix<ty>*, Sanae::Matrix<ty>*, double) = Sanae::SGD;//�d�݁E�o�C�A�X�̍X�V�֐�

		//�����Ƃ��ē��̓T�C�Y,�o�̓T�C�Y,default_random_engine�̃|�C���^���󂯎��B
		Layer_Affine(size_t arg_input,size_t arg_output,std::default_random_engine* Engine) 
		{
			Weight   = std::pair<size_t, size_t>{arg_input,arg_output};  //�s��:���̓T�C�Y,��:�o�̓T�C�Y

			std::normal_distribution<> Dist(0, (1 / std::sqrt(arg_input)));  //����:0,�W���΍�1/(���͐�)^0.5
			Weight.Setter([&Engine,&Dist]()->ty {return Dist(*Engine);});    //�d�݂̏����l�𗐐��Őݒ�

			Bias = std::pair<size_t, size_t>{1,arg_output};              //�s��:1,��:�o�̓T�C�Y
			Bias.Setter([&Engine,&Dist]()->ty {return Dist(*Engine); }); //�o�C�A�X�̏d�݂̏����l�𗐐��Őݒ�
		}

		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>* arg) override {
			Arg_Input = *arg;  //���͂�ۑ�

			std::vector<ty>   buf     = Bias[0];  //�o�C�A�X�����擾
			Sanae::Matrix<ty> BiasBuf = std::pair<size_t,size_t>{arg->get_row(),buf.size()};  //batch�T�C�Y�����₷

			for (size_t i = 0; i < arg->get_row(); i++)
				BiasBuf[i] = buf;

			//Input*Weight+Bias
			return (*arg) * Weight + BiasBuf;
		}
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>* arg) override{
			Sanae::Matrix<ty> ret = (*arg) * Weight.Transpose();    //�덷�t�`��
			DWeight               = Arg_Input.Transpose() * (*arg); //��E/��W

			//�o�C�A�X�̌X����batch�T�C�Y�����߂�
			DBias = std::pair<size_t, size_t>{1,arg->get_column()};

			for (size_t column = 0; column < arg->get_column();column++) {
				for (size_t row = 0; row < arg->get_row();row++) {
					DBias[0][column] += (*arg)[row][column];
				}
			}
			
			//�d�݂ƃo�C�A�X�̒l���X�V����
			UpdateFunc(&Weight,&DWeight, LearnRate);
			UpdateFunc(&Bias  ,&DBias  , LearnRate);

			//�t�`��
			return ret;
		}
	};


}

#endif