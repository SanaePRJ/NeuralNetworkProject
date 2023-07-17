/*=============================================================
* NAME      : Matrix.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
* 
* �v���g�^�C�v�錾
* 
* ����:
* MatrixImpl.hpp���C���N���[�h���Ă��������B
=============================================================*/


#ifndef _SANAE_MATRIX_HPP_
#define _SANAE_MATRIX_HPP_


#include "MatrixBase/MatrixBase.hpp"
#include "MatrixBase/MatrixBaseImpl.hpp"


#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


namespace Sanae {
	class Matrix :public _MatrixBase<double>
	{
		//Define Functions(private)
	private:
		//�����l��std::vector<_T>�֕ϊ�����B
		void _Convert_to_Array
		(
			std::initializer_list<std::vector<double>>&,
			std::vector<double>&
		);
		//�s���\������B
		void _View
		(
			std::vector<double>*,
			SizeT
		);
		//�P�ʍs��ɂ��܂��B
		void _To_Identity_Matrix
		(
			std::vector<double>*,
			SizeT
		);
		//�T���X�̕����ŉ����܂��B(2����)
		double _Det_2
		(
			std::vector<double>*,
			double,
			SizeT
		);
		//�]���q�W�J�����čs����ꎟ�������܂��B�������l��_Storage�Ɋi�[����W����_Coeff�Ɋi�[����܂��B
		void _Confactor_expansion_1
		(
			std::vector<double>&,
			SizeT,
			std::vector<std::vector<double>>*,
			std::vector<double>*
		);
		//2�����ɂȂ�܂ŗ]���q�W�J���s���܂��B
		std::pair<std::vector<std::vector<double>>, std::vector<double>>
			_Confactor_expansion_to_2
			(
				std::vector<double>&,
				SizeT,
				double
			);
		//�s�񎮂����߂܂��B
		double _Det
		(
			std::vector<double>&,
			SizeT
		);
		//�t�s������߂�
		void _Inverse_matrix
		(
			std::vector<double>*,
			std::vector<double>*,
			SizeT
		);
		//Define Functions(public)
	public:
		//Constructor
		using _MatrixBase::_MatrixBase;

		//���
		Matrix& operator=
		(
			const Matrix& _Data
		);
		Matrix& operator=
		(
			std::initializer_list<std::vector<double>> _In
		);

		//�j��I����
		Matrix& operator+=
		(
			const Matrix& _Data
		);
		Matrix& operator-=
		(
			const Matrix& _Data
		);
		Matrix& operator*=
		(
			const Matrix& _Data
		);
		Matrix& operator*=
		(
			double _Num
		);

		//��j��I����
		Matrix  operator+
		(
			const Matrix& _Data
		);
		Matrix  operator-
		(
			const Matrix& _Data
		);
		Matrix  operator*
		(
			const Matrix& _Data
		);
		Matrix  operator*
		(
			double _Num
		);

		//(�j��I����)�T�C�Y�ύX(�f�[�^�͂��ׂč폜����܂��B)
		Matrix& SetSize
		(
			SizeT _Data
		);

		//(�j��I����)�s�����ւ���B
		Matrix& Swap_Line
		(
			Ulong _Line1,
			Ulong _Line2
		);
		//(�j��I����)������ւ���B
		Matrix& Swap_Column
		(
			Ulong _Column1,
			Ulong _Column2
		);

		//(�j��I����)�P�ʍs��ɂ���B
		Matrix& Ident();
		//(��j��I����)�s�񎮂����߂܂��B 
		double Det();
		//(��j��I����)�s��̓]�u���s���܂��B
		Matrix Transpose();
		//(��j��I����)�t�s���Ԃ��܂��B
		Matrix Inverse();
		
		//�s���\�����܂��B
		Matrix& View();

		//�傫����Ԃ��܂��B
		Ulong GetSize();
		//�傫����Ԃ��܂��B first:�� second:�s
		SizeT GetSizeWH();
	};

}

#endif
#endif