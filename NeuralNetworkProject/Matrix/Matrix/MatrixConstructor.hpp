/*-------------------------------------------------------------
* Name    : MatrixConstructor.hpp
* Version : 4.0.5
* * Author: SanaePRJ
* Description:
*  MatrixBase�^�̃R���X�g���N�^�̎���
-------------------------------------------------------------*/




#ifndef _INCLUDEGUARD_MATRIXCONSTRUCTOR_HPP_
#define _INCLUDEGUARD_MATRIXCONSTRUCTOR_HPP_




#include <algorithm>
#include "Matrix.h"




//�R���X�g���N�^
template<typename ty>
Sanae::Matrix<ty>::Matrix() {}


//�T�C�Y���w�肵��0�s��𐶐�����B
template<typename ty>
Sanae::Matrix<ty>::Matrix
(
	std::pair<size_t,size_t> ArgSize
) 
	:matrix(ArgSize.first, std::vector<ty>(ArgSize.second, 0))
{}


//�ȉ��̂悤�ɒ�`���邱�Ƃ��ł���B
//{
//	{0, 0},
//	{0, 0},
//	{0, 0}
//}
template<typename ty>
Sanae::Matrix<ty>::Matrix
(
	MatrixInit_t ArgInitValue
)
	:matrix(ArgInitValue.begin(), ArgInitValue.end())
{
	//�񐔂͓������Ȃ���΂Ȃ�Ȃ��B
	this->m_ValidateMatrix(this->matrix);
}


//std::vector<std::vector<ty>>
template<typename ty>
Sanae::Matrix<ty>::Matrix
(
	const Matrix_t& ArgInitValue
)
	:matrix(ArgInitValue.begin(), ArgInitValue.end())
{
	//�񐔂͓������Ȃ���΂Ȃ�Ȃ��B
	this->m_ValidateMatrix(this->matrix);
}


//�R�s�[�R���X�g���N�^
template<typename ty>
Sanae::Matrix<ty>::Matrix
(
	const Matrix& ArgMatrix
)
{
	//�T�C�Y�ݒ�
	matrix.resize(ArgMatrix.matrix.size());
	//�R�s�[
	std::copy(ArgMatrix.matrix.begin(),ArgMatrix.matrix.end(),matrix.begin());
}


//�f�X�g���N�^
template<typename ty>
inline Sanae::Matrix<ty>::~Matrix()
{
}




#endif