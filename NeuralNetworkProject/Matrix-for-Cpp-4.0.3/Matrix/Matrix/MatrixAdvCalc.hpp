/*-------------------------------------------------------------
* Name    : MatrixDet.hpp
* Version : 4.0.3
* * Author: SanaePRJ
* Description:
*  MatrixBase�^�̍s�񎮂Ƌt�s������߂�֐��̎���
-------------------------------------------------------------*/




#ifndef SANAE_MATRIXADV_HPP
#define SANAE_MATRIXADV_HPP

#include "Matrix.h"




//�|���o���@
template<typename ty>
inline void Sanae::Matrix<ty>::m_sweepout(MatrixT* arg_from, MatrixT* arg_store)
{
	if (m_GetRowSize(arg_from) != m_GetColumnSize(arg_from))
		throw std::invalid_argument("It must be a square matrix.");

	size_t size = m_GetColumnSize(arg_from);

	//�m��
	arg_store->resize(size, std::vector<ty>(size, 0));
	//�P�ʍs���
	this->m_to_identity(arg_store);

	//����s�̒萔�{���ق��̍s�։��Z����B
	auto Operation = [arg_from, arg_store, this, size](size_t from, size_t to, ty num)
		{
			for (size_t Column = 0; Column < m_GetColumnSize(arg_from); Column++) {
				(*arg_from )[to][Column] += (*arg_from )[from][Column] * num;
				(*arg_store)[to][Column] += (*arg_store)[from][Column] * num;
			}
		};

	for (size_t Column = 0; Column < size; Column++) {
		for (size_t Row = 0; Row < size; Row++) {
			//�萔
			ty num = 0;

			if (Row == Column)
				num = (1 - 1 * (*arg_from)[Row][Column]) / (*arg_from)[Column][Column]; //�Ίp������1�ɂ���B
			else
				num = -1 * (*arg_from)[Row][Column] / (*arg_from)[Column][Column];      //���̐�����0�ɂ���B

			//����s�̒萔�{���ق��̍s�։��Z����B
			Operation(Column, Row, num);
		}
	}
}


//�t�s������߂�B
template<typename ty>
Sanae::Matrix<ty> Sanae::Matrix<ty>::Inverse() 
{
	//�񐔂͓������Ȃ���΂Ȃ�Ȃ��B
	if (this->m_CheckColumn(&this->matrix))
		throw std::invalid_argument("All the columns must be equal.");

	if (this->det() == 0)
		throw std::invalid_argument("It is not a regular matrix.");

	MatrixT buf;
	std::copy(this->matrix.begin(),this->matrix.end(),std::back_inserter(buf));

	MatrixT Inv;
	this->m_sweepout(&buf,&Inv);
	
	return Inv;
}


//�s��:���C�v�j�b�c�̍s��
template<typename ty>
inline ty Sanae::Matrix<ty>::m_det(MatrixT* arg)
{
	if (this->get_row() != this->get_column())
		throw std::invalid_argument("It must be a square matrix.");

	//�T���X�̕����ŉ����܂��B
	auto det_2 = [](MatrixT& arg_lambda)
		{
			return arg_lambda[0][0] * arg_lambda[1][1] - arg_lambda[1][0] * arg_lambda[0][1];
		};

	//[Pos][0]-[0][Pos]���܂܂Ȃ��s��𔲂����܂��B
	auto pull = [](MatrixT& arg_lambda, MatrixT& storage, size_t Pos)
		{
			const size_t size = arg_lambda.size();

			for (size_t row = 0, storage_row = 0; row < size; row++) {
				//0�̗����Ƃ��Ă���B
				for (size_t column = 1; column < size; column++) {
					if (row != Pos)
						storage[storage_row][column - 1] = arg_lambda[row][column];
				}
				//��̍����ȊO�̏ꍇ�͎���
				if (Pos != row)
					storage_row++;
			}

			return;
		};

	//�����𗎂Ƃ��܂���B
	auto dec_dim = [this, pull, det_2](MatrixT& arg_lambda, ty coeff, auto f)
		{
			const size_t size     = arg_lambda.size(); //���̃T�C�Y
			const size_t new_size = size - 1;          //���Ƃ�����̃T�C�Y

			ty ret = 0; //�Ԃ�l

			//2*2�s��ɂ܂ŗ�������T���X�̕����ŉ����ĕԂ��B
			if (size == 2)
				return coeff * det_2(arg_lambda);

			//[0][0]~[n][0]�܂�
			for (size_t Pos = 0; Pos < size; Pos++)
			{
				MatrixT buf(new_size, std::vector<ty>(new_size, 0)); //�i�[�p
				pull(arg_lambda, buf, Pos);                          //�k�������s����擾

				ty coeff_buf = arg_lambda[Pos][0] * (Pos % 2 == 0 ? 1 : -1); //�V�����W��

				ret += f(buf, coeff_buf * coeff, f); //�ċA�������ʂ����ׂĉ��Z
			}

			return ret;
		};
	
	//���̌W��
	const ty From_Coff = 1;

	return dec_dim(*arg, From_Coff, dec_dim);
}


//�s�񎮂����߂�B
template<typename ty>
ty Sanae::Matrix<ty>::det() 
{
	//�񐔂͓������Ȃ���΂Ȃ�Ȃ��B
	if (this->m_CheckColumn(&this->matrix))
		throw std::invalid_argument("All the columns must be equal.");

	return this->m_det(&this->matrix);
}




#endif