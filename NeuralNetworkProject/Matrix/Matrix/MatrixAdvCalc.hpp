/*-------------------------------------------------------------
* Name    : MatrixDet.hpp
* Version : 4.0.5
* * Author: SanaePRJ
* Description:
*  MatrixBase�^�̍s�񎮂Ƌt�s������߂�֐��̎���
-------------------------------------------------------------*/




#ifndef _INCLUDEGUARD_MATRIXADV_HPP_
#define _INCLUDEGUARD_MATRIXADV_HPP_




#include "Matrix.h"




/*------------------------------------------------------
* �v���C�x�[�g���\�b�h
------------------------------------------------------*/




/*
* ���݂��̃��\�b�h�͎g�p����LU�����ɂ��t�s������߂�悤�ɂ��Ă��܂��B
*/
// �|���o���@�ɂ�苁�߂����ʂ� arg_store �Ɋi�[���܂��B
// * ArgStore ���m�ۂ��Ă���K�v�͂���܂���B
template<typename ty>
inline void Sanae::Matrix<ty>::m_SweepOut
(
	Matrix_t& ArgFrom, 
	Matrix_t& ArgStore
)
{
	//���͂��ꂽ�s�񂪐����s��łȂ���΂Ȃ�Ȃ��B
	if (!m_IsSquareMatrix(ArgFrom))
		throw InvalidMatrix("Must be a square matrix.");

	//arg_from���R�s�[����B
	Matrix_t from_copy(m_GetRowSize(ArgFrom),std::vector<ty>(m_GetColumnSize(ArgFrom),0));
	//�R�s�[
	std::copy(ArgFrom.begin(), ArgFrom.end(),from_copy.begin());

	//�T�C�Y���i�[
	const size_t size = m_GetRowSize(from_copy);

	//�m��
	ArgStore = Identity(size);

	//����s�̒萔�{���ق��̍s�։��Z����B
	auto Operation = [&from_copy, &ArgStore, this, size](size_t from, size_t to, ty num)
		{
			for (size_t Column = 0; Column < m_GetColumnSize(from_copy); Column++) {
				from_copy[to][Column] += from_copy[from][Column] * num;
				ArgStore [to][Column] += ArgStore [from][Column] * num;
			}
		};

	for (size_t Column = 0; Column < size; Column++) {
		for (size_t Row = 0; Row < size; Row++) {
			//�萔
			ty num = 0;

			if (Row == Column)
				num = (1 - 1 * ( from_copy)[Row][Column]) / ( from_copy)[Column][Column]; //�Ίp������1�ɂ���B
			else
				num = -1 * ( from_copy)[Row][Column] / ( from_copy)[Column][Column];      //���̐�����0�ɂ���B

			//����s�̒萔�{���ق��̍s�։��Z����B
			Operation(Column, Row, num);
		}
	}
}


/*
* ���݂��̃��\�b�h�͎g�p����LU�����ɂ��t�s������߂�悤�ɂ��Ă��܂��B
*/
// �s��:���C�v�j�b�c�̍s��
template<typename ty>
inline ty Sanae::Matrix<ty>::m_Det
(
	Matrix_t& Arg
)
{
	//���͂��ꂽ�s�񂪐����s��łȂ���΂Ȃ�Ȃ��B
	if (!m_IsSquareMatrix(Arg))
		throw InvalidMatrix("Must be a square matrix.");

	//�T���X�̕����ŉ����܂��B
	const auto DetBy2D = [](const Matrix_t& Matrix2D)
		{
			return Matrix2D[0][0] * Matrix2D[1][1] - Matrix2D[1][0] * Matrix2D[0][1];
		};

	//[Pos][0]-[0][Pos]���܂܂Ȃ��s��𔲂����܂��B
	const auto Extraction = [](const Matrix_t& From, Matrix_t& Storage, size_t Pos)
		{
			//�����s��i�m�͊m�肵�Ă���̂Ŏ��������߂�B
			const size_t Dim = From.size();

			for (size_t Row = 0, StorageRow = 0; Row < Dim; Row++) {
				//0�̗����Ƃ��Ă���B
				for (size_t Column = 1; Column < Dim; Column++) {
					if (Row != Pos)
						Storage[StorageRow][Column - 1] = From[Row][Column];
				}
				//��̍����ȊO�̏ꍇ�͎���
				if (Pos != Row)
					StorageRow++;
			}

			return;
		};

	//�����𗎂Ƃ��܂���B
	const auto DecDim = [this, Extraction, DetBy2D](const Matrix_t& From, ty Coeff, auto Func)
		{
			const size_t Size    = From.size(); //���̃T�C�Y
			const size_t NewSize = Size - 1;    //���Ƃ�����̃T�C�Y

			ty Result = 0; //�Ԃ�l

			//2*2�s��ɂ܂ŗ�������T���X�̕����ŉ����ĕԂ��B
			if (Size == 2)
				return Coeff * DetBy2D(From);

			//[0][0]~[n][0]�܂�
			for (size_t Position = 0; Position < Size; Position++)
			{
				Matrix_t Buffer(NewSize, std::vector<ty>(NewSize, 0)); //�i�[�p
				Extraction(From, Buffer, Position);                    //�k�������s����擾

				ty CoeffBuffer = From[Position][0] * (Position % 2 == 0 ? 1 : -1); //�V�����W��

				Result += Func(Buffer, CoeffBuffer * Coeff, Func); //�ċA�������ʂ����ׂĉ��Z
			}

			return Result;
		};
	
	//���̌W��
	const ty FromCoeff = 1;

	return DecDim(Arg, FromCoeff, DecDim);
}


// ArgMatrix �� MatrixL �� MatrixU �֕������܂��B
template<typename ty>
inline void Sanae::Matrix<ty>::m_LUDecomposition
(
	const Matrix_t& ArgMatrix, 
	Matrix_t&       MatrixL, 
	Matrix_t&       MatrixU
)
	const
{
	//���͂��ꂽ�s�񂪐����s��łȂ���΂Ȃ�Ȃ��B
	if (!m_IsSquareMatrix(ArgMatrix))
		throw InvalidMatrix("Must be a square matrix.");

	//���ׂĂ̗�ɑ΂��� LRowFrom �s�� LNum �{�������̂� LRowTo �s�։��Z����B
	auto Operation = [this](Matrix_t& LambdaMatrix,size_t LambdaRowFrom,size_t LambdaRowTo,ty LambdaNum)
		{
			//���ׂĂ̗�Ŏ��s
			for (size_t Column = 0; Column < m_GetColumnSize(LambdaMatrix); Column++) {
				// LRowFrom �s�� LNum �{�������̂� LRowTo �s�։��Z����B
				LambdaMatrix[LambdaRowTo][Column] += LambdaNum * LambdaMatrix[LambdaRowFrom][Column];
			}
		};

	const size_t Size = m_GetRowSize(ArgMatrix);

	//�P�ʍs��擾
	MatrixL = Identity(Size).matrix;
	MatrixU = ArgMatrix;

	for (size_t Column = 0; Column < Size;Column++) {

		//BaseRow�s�𑼂̍s�։��Z��0�ɂ���B
		const size_t BaseRow = Column;

		for (size_t Row = Column + 1; Row < Size; Row++) {
			// A[BaseRow][Column]�ɉ����|������A[Row][Column]��0�ɂȂ邩�����߂�B
			// 
			// 0 = A[Row][Column] + A[BaseRow][Column]*MulNum
			// MulNum = -A[Row][Column] / A[BaseRow][Column]

			ty MulNum = (-1 * MatrixU[Row][Column]) / MatrixU[BaseRow][Column];

			// MatrixU�̍X�V
			Operation(MatrixU, BaseRow, Row, MulNum);

			// MatrixL�̍X�V
			MatrixL[Row][Column] = -MulNum;
		}
	}
}


template<typename ty>
inline ty Sanae::Matrix<ty>::m_DetByU
(
	const Matrix_t& MatrixU
)
	const
{
	//���͂��ꂽ�s�񂪐����s��łȂ���΂Ȃ�Ȃ��B
	if (!m_IsSquareMatrix(MatrixU))
		throw InvalidMatrix("Must be a square matrix.");

	ty Det = 1;
	
	//LU��������U�̑̊i�v�f�̐ςŋ��߂邱�Ƃ��o����B
	for (size_t Pos = 0; Pos < m_GetRowSize(MatrixU); ++Pos)
		Det *= MatrixU[Pos][Pos];
	
	return Det;
}


//�t�s������߂܂��B
template<typename ty>
inline void Sanae::Matrix<ty>::m_Inverse
(
	const Matrix_t& ArgMatrix, 
	Matrix_t&       Store, 
	ty              Epsilon
)
	const
{
	//���͂��ꂽ�s�񂪐����s��łȂ���΂Ȃ�Ȃ��B
	if (!m_IsSquareMatrix(ArgMatrix))
		throw InvalidMatrix("Must be a square matrix.");

	// �s��̃T�C�Y���擾
	const size_t Size = m_GetRowSize(ArgMatrix);

	// LU�������s��
	Matrix_t MatrixL, MatrixU;
	m_LUDecomposition(ArgMatrix, MatrixL, MatrixU);
	
	/* �s�񎮂�0�̏ꍇ0�Ɍ���Ȃ��߂����邱�Ƃŋߎ������邽�߂���Ȃ�
	if (m_DetByU(MatrixU) == 0)
		throw InvalidMatrix("It is not a regular matrix.");
	*/

	//�s�񎮂�0�̏ꍇ0�Ɍ���Ȃ��߂��������邱�Ƃŋߎ�����B
	for (size_t Pos = 0; Pos < Size;Pos++)
		MatrixU[Pos][Pos] += MatrixU[Pos][Pos] == 0 ? Epsilon : 0;

	// �P�ʍs����쐬
	Matrix_t IdentityMatrix = Identity(Size).matrix;

	// �t�s���������
	Store.resize(Size, std::vector<ty>(Size, 0));

	// �e�񂲂Ƃɋt�s������߂�
	for (size_t Column = 0; Column < Size; Column++) {
		// �P�ʍs��̗�x�N�g�������o��
		std::vector<ty> e(Size, 0);
		e[Column] = 1;

		// �O�i������s��
		std::vector<ty> MatrixY(Size, 0);
		for (size_t i = 0; i < Size; i++) {
			MatrixY[i] = e[i];
			
			for (size_t j = 0; j < i; j++)
				MatrixY[i] -= MatrixL[i][j] * MatrixY[j];

			MatrixY[i] /= MatrixL[i][i];
		}

		// ��ޑ�����s��
		std::vector<ty> MatrixX(Size, 0);
		for (long long i = Size - 1; i >= 0; i--) {
			MatrixX[i] = MatrixY[i];

			for (size_t j = i + 1; j < Size; j++) 
				MatrixX[i] -= MatrixU[i][j] * MatrixX[j];

			MatrixX[i] /= MatrixU[i][i];
		}

		// ���ʂ�Store�s��̗�ɃR�s�[����
		for (size_t i = 0; i < Size; i++)
			Store[i][Column] = MatrixX[i];
	}
}




/*------------------------------------------------------
* �p�u���b�N���\�b�h
------------------------------------------------------*/




// �s�񎮂����߂�B
template<typename ty>
inline ty Sanae::Matrix<ty>::Det()
{
	return this->m_Det(this->matrix);
}


// �t�s������߂�B
template<typename ty>
inline Sanae::Matrix<ty> Sanae::Matrix<ty>::Inverse(ty Epsilon)
{
	Matrix_t Inv;
	m_Inverse(this->matrix,Inv,Epsilon);

	return Inv;
}




#endif