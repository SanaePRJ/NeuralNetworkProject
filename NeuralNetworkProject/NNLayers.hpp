/*-------------------------------------------------------------
* Name    : NNLayers.hpp
* Version : 2.0.0
* * Author: SanaePRJ
* Description:
*  レイヤーの実装
* ゼロから作るDeepLearningで書かれていたpythonのコードを参考に
* 作成しています。
-------------------------------------------------------------*/




#ifndef SANAE_NNLAYERS_HPP
#define SANAE_NNLAYERS_HPP




#include <stdint.h>
#include <random>
#include <algorithm>
#include <stdexcept>

#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"




namespace Sanae {


/*-----損失関数-----*/
	

	//交差エントロピー誤差 
	//E=-(1/N)*∑[n=1->N](∑[k=1,K](Teach[n][k]log(Out[n][k])))
	template<typename ty>
	double Cross_Entropy_Error(Sanae::Matrix<ty>* Out, Sanae::Matrix<ty>* Teach)
	{
		//実際の出力値と教師データのサイズが違う場合
		if (Out->get_row() != Teach->get_row() || Out->get_column() != Teach->get_column())
			throw std::invalid_argument("Must be same size.");

		//データは一行で渡されるため行数はbatch数数
		size_t batch_size = Out->get_row();

		double ans        = 0;    //誤差
		double delta      = 1e-7; //log(0)になるのを防ぐため

		//∑[n=1->N]
		for (size_t row = 0; row < batch_size; row++)
		{
			//-∑[k=1,K](Teach[n][k]log(Out[n][k]))
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

	//二乗和誤差
	//E = (1/(2N))*∑[n->N](∑[k->K]((Out[n][k]-Teach[n][k])^2))
	template<typename ty>
	double Mean_Squared_Error(Sanae::Matrix<ty>* Out, Sanae::Matrix<ty>* Teach)
	{
		//実際の出力値と教師データのサイズが違う場合
		if (Out->get_row() != Teach->get_row() || Out->get_column() != Teach->get_column())
			throw std::invalid_argument("Must be same size.");

		
		size_t batch_size = Out->get_row(); //データは一行で渡されるため行数はbatch数数
		double ans        = 0;              //誤差

		//nを受け取る,∑[k->K]((Out[n][k]-Teach[n][k])^2)
		auto sum_error = [&Out, &Teach](size_t arg_row) {
				//返り値
				double ans = 0;

				for (size_t column = 0; column < Out->get_column(); column++)
					ans += (double)std::pow(((*Out)[arg_row][column] - (*Teach)[arg_row][column]), 2);

				return ans;
			};

		//∑[n->N]
		for (size_t row = 0; row < batch_size; row++)
			ans += sum_error(row);

		//(1/(2N))
		return ans / (2 * batch_size);
	}
	

/*-----活性化関数-----*/


	//SoftMax関数
	//O[r][k] = {e^a[r][k]-Max[r]} / {∑[i=1->n](e^a[r][i]-Max[r])}
	template<typename ty>
	Sanae::Matrix<ty> SoftMax(Sanae::Matrix<ty>* arg) 
	{
		//一列の最大値を返す。Max[r]
		auto get_max = [&arg](size_t arg_row)->ty{
				return *std::max_element((*arg)[arg_row].begin(), (*arg)[arg_row].end());
			};

		//合計値を出力 ∑[i=1->n](e^a[r][i]-Max[r])
		auto get_sum = [&arg](size_t arg_row,ty arg_max)->ty{
				ty ans = 0;

				for (size_t column = 0; column < arg->get_column(); column++)
					ans += std::exp((*arg)[arg_row][column] - arg_max);

				return ans;
			};

		size_t            batch_size = arg->get_row();   //データは一行で渡されるため行数はbatch数数
		Sanae::Matrix<ty> ret        = std::pair<size_t,size_t>{arg->get_row(),arg->get_column()};  //サイズを設定
		
		//O[r]
		for (size_t row = 0; row < batch_size; row++) {
			ty max = get_max(row);
			ty sum = get_sum(row,max);

			//{e^a[r][k]-Max[r]} / {∑[i=1->n](e^a[r][i]-Max[r])}
			for (size_t column = 0; column < arg->get_column(); column++)
				ret[row][column] = std::exp((*arg)[row][column] - max) / sum;
		}

		return ret;
	}

	//恒等関数 In=Out
	template<typename ty>
	Sanae::Matrix<ty> Identity(Sanae::Matrix<ty>* arg) {
		return *arg;
	}

	//シグモイド関数 Sigmoid(x) = 1/(1+e^-gain * x)
	template<typename ty>
	ty Sigmoid(ty arg,ty gain = 1) {
		return 1 / (1 + std::exp(-1 * gain * arg));
	}


/*-----重み・バイアスの更新関数-----*/


	//W1 = W0 - (δE/δW0)*η
	template<typename ty>
	void SGD(Sanae::Matrix<ty>* arg,Sanae::Matrix<ty>* darg,double learn_rate) {
		*arg = (*arg) - (*darg) * learn_rate;
	}


/*-----レイヤー-----*/


	//ベースレイヤー
	template<typename ty>
	class Layer_Base {
	public:
		bool is_affine = false;  //Affineレイヤーの場合インターフェースが異なるため

		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>*)  { return { {} }; }  //次の層へ
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>*) { return { {} }; }  //逆伝番
	};

	//Sigmoidレイヤー
	template<typename ty>
	class Layer_Sigmoid : public Layer_Base<ty> {
	private:
		Sanae::Matrix<ty> out;  //出力値を保存しておく

	public:
		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>* arg) override
		{
			//サイズ設定
			out = std::pair<size_t,size_t>{arg->get_row(),arg->get_column()};

			//すべての要素をSigmoid関数に通す。
			for (size_t row = 0; row < arg->get_row();row++) {
				for (size_t column = 0; column < arg->get_column();column++) {
					out[row][column] = Sigmoid<ty>((*arg)[row][column]);
				}
			}

			return out;
		}
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>* arg) override 
		{
			//逆伝番のサイズが間違っている場合
			if (out.get_row() != arg->get_row() || out.get_column() != arg->get_column())
				throw std::invalid_argument("Must be same size.");

			//重み改善の値
			Sanae::Matrix<ty> dx = std::pair<size_t, size_t>{out.get_row(),out.get_column()};

			//すべての要素を計算
			for (size_t row = 0; row < arg->get_row(); row++) {
				for (size_t column = 0; column < arg->get_column(); column++) {
					//Sigmoid'(x) = Sigmoid(x)*(1-Sigmoid(x))
					dx[row][column] = (*arg)[row][column] * (1 - out[row][column]) * out[row][column];
				}
			}

			return dx;
		}
	};

	//ReLUレイヤー
	template<typename ty>
	class Layer_ReLU : public Layer_Base<ty> {
	private:
		Sanae::Matrix<ty> Mask;  //値が0を超えるとき逆伝番の値を通すため保存

	public:
		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>* arg) override
		{
			
			Sanae::Matrix<ty> out = *arg;  //返り値
			Mask = std::pair<size_t, size_t>{ arg->get_row(),arg->get_column() };  //Maskのサイズ設定

			//すべての要素に対して
			for (size_t row = 0; row < arg->get_row(); row++) {
				for (size_t column = 0; column < arg->get_column(); column++) {
					//0を超えるとき1にすることで次の層,前の層へ送る
					if ((*arg)[row][column] > 0) {
						Mask[row][column] = 1;
					}
					else {
						Mask[row][column] = 0;
						out[row][column]  = 0;  //0以下の場合次の層へ渡さない。
					}
				}
			}

			return out;
		}
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>* arg) override
		{
			return Mask^*arg;  //入力の際に0を超えている場合前の層へ逆伝番
		}
	};

	//SoftMaxWithLossレイヤー
	template<typename ty>
	class Layer_SoftMaxWithLoss 
	{
	private:
		Sanae::Matrix<ty> Out;   //出力値
		Sanae::Matrix<ty> Teach; //教師データ

	public:
		inline virtual double forward(Sanae::Matrix<ty>* arg_Out,Sanae::Matrix<ty>* arg_Teach)
		{
			Out   = Sanae::SoftMax(arg_Out);  //SoftMax関数に通した値を出力
			Teach = *arg_Teach;               //教師データを保存

			return Sanae::Cross_Entropy_Error(&Out,&Teach); //交差エントロピー誤差を出力
		}
		inline virtual Sanae::Matrix<ty> backward()
		{
			size_t bachsize = Out.get_row();  //データは一行で渡されるため行数はbatch数数

			//batchサイズと教師データのbatchサイズが異なる場合
			if (bachsize != Teach.get_row())
				throw std::invalid_argument("Must be same bachsize.");

			//δE/δW = Out-Teach
			Sanae::Matrix<ty> Dout = Out - Teach;
			
			for (size_t row = 0; row < Dout.get_row(); row++) {
				for (size_t column = 0; column < Dout.get_column(); column++) {
					Dout[row][column] /= bachsize;
				}
			}

			return Dout;
		}
	};

	//IdentityWithLossレイヤー
	template<typename ty>
	class Layer_IdentityWithLoss
	{
	private:
		Sanae::Matrix<ty> Out;
		Sanae::Matrix<ty> Teach;

	public:
		inline virtual double forward(Sanae::Matrix<ty>* arg_Out, Sanae::Matrix<ty>* arg_Teach)
		{
			Out   = Sanae::Identity(arg_Out);  //恒等関数へ通した値を出力
			Teach = *arg_Teach;                //教師データ

			return Sanae::Mean_Squared_Error(&Out, &Teach);  //二乗和誤差を出力
		}
		inline virtual Sanae::Matrix<ty> backward()
		{
			size_t bachsize = Out.get_row();  //データは一行で渡されるため行数はbatch数数

			//batch最右と教師データのbatchサイズが異なっている場合
			if (bachsize != Teach.get_row())
				throw std::invalid_argument("Must be same bachsize.");

			//δE/δW=Out-Teach
			Sanae::Matrix<ty> Dout = Out - Teach;

			for (size_t row = 0; row < Dout.get_row(); row++) {
				for (size_t column = 0; column < Dout.get_column(); column++) {
					Dout[row][column] /= bachsize;
				}
			}

			return Dout;
		}
	};

	//Affineレイヤー
	template<typename ty>
	class Layer_Affine :public Layer_Base<ty> {
	public:
		Sanae::Matrix<ty> Weight,Bias;

	private:
		Sanae::Matrix<ty> DWeight;  //(δE/δW)
		Sanae::Matrix<ty> DBias;    //(δE/δB)

		Sanae::Matrix<ty> Arg_Input;//入力値

	public:
		bool   is_affine = true;  //Affineレイヤー
		double LearnRate = 0.03;  //学習率の初期値:3%

		void (*UpdateFunc)(Sanae::Matrix<ty>*, Sanae::Matrix<ty>*, double) = Sanae::SGD;//重み・バイアスの更新関数

		//引数として入力サイズ,出力サイズ,default_random_engineのポインタを受け取る。
		Layer_Affine(size_t arg_input,size_t arg_output,std::default_random_engine* Engine) 
		{
			Weight   = std::pair<size_t, size_t>{arg_input,arg_output};  //行数:入力サイズ,列数:出力サイズ

			std::normal_distribution<> Dist(0, (1 / std::sqrt(arg_input)));  //平均:0,標準偏差1/(入力数)^0.5
			Weight.Setter([&Engine,&Dist]()->ty {return Dist(*Engine);});    //重みの初期値を乱数で設定

			Bias = std::pair<size_t, size_t>{1,arg_output};              //行数:1,列数:出力サイズ
			Bias.Setter([&Engine,&Dist]()->ty {return Dist(*Engine); }); //バイアスの重みの初期値を乱数で設定
		}

		inline virtual Sanae::Matrix<ty> forward(Sanae::Matrix<ty>* arg) override {
			Arg_Input = *arg;  //入力を保存

			std::vector<ty>   buf     = Bias[0];  //バイアスを一列取得
			Sanae::Matrix<ty> BiasBuf = std::pair<size_t,size_t>{arg->get_row(),buf.size()};  //batchサイズ分増やす

			for (size_t i = 0; i < arg->get_row(); i++)
				BiasBuf[i] = buf;

			//Input*Weight+Bias
			return (*arg) * Weight + BiasBuf;
		}
		inline virtual Sanae::Matrix<ty> backward(Sanae::Matrix<ty>* arg) override{
			Sanae::Matrix<ty> ret = (*arg) * Weight.Transpose();    //誤差逆伝番
			DWeight               = Arg_Input.Transpose() * (*arg); //δE/δW

			//バイアスの傾きをbatchサイズ分求める
			DBias = std::pair<size_t, size_t>{1,arg->get_column()};

			for (size_t column = 0; column < arg->get_column();column++) {
				for (size_t row = 0; row < arg->get_row();row++) {
					DBias[0][column] += (*arg)[row][column];
				}
			}
			
			//重みとバイアスの値を更新する
			UpdateFunc(&Weight,&DWeight, LearnRate);
			UpdateFunc(&Bias  ,&DBias  , LearnRate);

			//逆伝番
			return ret;
		}
	};


}

#endif