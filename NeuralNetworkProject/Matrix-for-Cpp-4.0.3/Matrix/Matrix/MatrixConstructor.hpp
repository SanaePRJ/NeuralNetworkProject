/*-------------------------------------------------------------
* Name    : MatrixConstructor.hpp
* Version : 4.0.3
* * Author: SanaePRJ
* Description:
*  MatrixBase�^�̃R���X�g���N�^�̎���
-------------------------------------------------------------*/




#ifndef SANAE_MATRIXCONSTRUCTOR_HPP
#define SANAE_MATRIXCONSTRUCTOR_HPP

#include "Matrix.h"




//�R���X�g���N�^
template<typename ty>
Sanae::Matrix<ty>::Matrix() {}


template<typename ty>
Sanae::Matrix<ty>::Matrix(std::pair<size_t,size_t> arg_Size) 
	:matrix(arg_Size.first, std::vector<ty>(arg_Size.second, 0))
{}


/*------------------------------------------
{
	{0, 0},
	{0, 0},
	{0, 0}
}
3 * 2�s��̂悤�ɒ�`���邱�Ƃ��ł���B
------------------------------------------*/
template<typename ty>
Sanae::Matrix<ty>::Matrix(MatrixInitT arg_InitValue)
	:matrix(arg_InitValue.begin(), arg_InitValue.end())
{
	//�񐔂͓������Ȃ���΂Ȃ�Ȃ��B
	if (this->m_CheckColumn(&this->matrix))
		throw std::invalid_argument("All the columns must be equal.");
}


//std::vector<std::vector<ty>>�œ����ꂽ���p
template<typename ty>
Sanae::Matrix<ty>::Matrix(const MatrixT& arg_InitValue)
	:matrix(arg_InitValue.begin(), arg_InitValue.end())
{
	//�񐔂͓������Ȃ���΂Ȃ�Ȃ��B
	if (this->m_CheckColumn(&this->matrix))
		throw std::invalid_argument("All the columns must be equal.");
}


//�R�s�[�R���X�g���N�^
template<typename ty>
Sanae::Matrix<ty>::Matrix(const Matrix& arg)
{
	std::move(arg.matrix.begin(), arg.matrix.end(), std::back_inserter(matrix));
}


//�f�X�g���N�^
template<typename ty>
inline Sanae::Matrix<ty>::~Matrix()
{
}




#endif