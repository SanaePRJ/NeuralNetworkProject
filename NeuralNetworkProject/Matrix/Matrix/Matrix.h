/*-------------------------------------------------------------
* Name    : Matrix.h
* Version : 4.0.5
* Author  : SanaePRJ
* Description:
*  MatrixBase�^�̒�`�t�@�C��
-------------------------------------------------------------*/




#ifndef _INCLUDEGUARD_MATRIX_H_
#define _INCLUDEGUARD_MATRIX_H_




#include <iostream>
#include <vector>
#include <stdexcept>
#include <functional>
#include <thread>




namespace Sanae{


	template<typename ty> class Matrix{
	private:


		template<typename RowType    > using Row_t     = std::vector<RowType>;
		template<typename RowInitType> using RowInit_t = std::initializer_list<RowInitType>;

		using Matrix_t     = std::vector          <Row_t<ty>>;
		using MatrixInit_t = std::initializer_list<RowInit_t<ty>>;

	
		//Variables
		Matrix_t matrix;


		//MatrixUtil.hpp
		// �R���X�g���N�^��=���Z�q�ȂǂŎ��s����B
		inline void   m_ValidateMatrix(const Matrix_t&) const; //���ׂĂ̍s�ɉ����ė񐔂����������m�F����B@�񐔂��������Ȃ��ꍇ throw ���܂��B
		
		inline bool   m_IsSquareMatrix(const Matrix_t&) const; //�����s�񂩂ǂ������擾����B
		inline bool   m_IsEmpty       (const Matrix_t&) const; //��̍s�񂩂ǂ�����Ԃ��B
		inline bool   m_IsSameSize    (const Matrix_t&,const Matrix_t&) const; //�����傫�����ǂ�����Ԃ��B

		inline size_t m_GetRowSize   (const Matrix_t&) const; //�s�����擾
		inline size_t m_GetColumnSize(const Matrix_t&) const; //�񐔂��擾
		

		//MatrixCalc.hpp
		template<typename FuncType>
		inline void m_Calc       (Matrix_t&, const Matrix_t&,FuncType) const; //���ׂĂ̗v�f�ő�����=FuncType(������,������)�����s����B

		inline void m_Add        (Matrix_t&, const Matrix_t&) const; //���Z        :������ += ������
		inline void m_Sub        (Matrix_t&, const Matrix_t&) const; //���Z        :������ -= ������
		inline void m_ScalarMul  (Matrix_t&, ty             ) const; //�X�J���[�{  :������ *= ������
		inline void m_HadamardMul(Matrix_t&, const Matrix_t&) const; //�A�_�}�[����:������ ^= ������

		inline void m_Mul        (Matrix_t&, const Matrix_t&); //�ρ@        :������ *= ������


		//MatrixAdvCalc.hpp
		//�폜�\��
		inline void  m_SweepOut(Matrix_t& arg_from, Matrix_t& arg_store); //�|���o���@�ɂ��t�s������߂�B
		inline ty    m_Det     (Matrix_t& arg);                           //�s�񎮂����߂�B

		inline void  m_LUDecomposition(const Matrix_t&, Matrix_t&, Matrix_t&) const; //LU�������s��������L,��O����U�Ɋi�[����B
		
		inline ty    m_DetByU (const Matrix_t&) const;                   //LU�����ɂ���ċ��߂�U(��O�p�s��)�ɂ���čs�񎮂����߂�B
		inline void  m_Inverse(const Matrix_t&,Matrix_t&,ty=1e-5) const; //�t�s������߂�B
	

	public:


		//�s��ςŎg�p����X���b�h��(�����l�͍ő�̃X���b�h���̔���)
		size_t thread = std::thread::hardware_concurrency()/2;


		//MatrixConstructor.hpp
		Matrix ();
		Matrix (std::pair<size_t,size_t>); //�T�C�Y�w��
		Matrix (MatrixInit_t);             //{{0,0},{0,0}}...2*2�s��
		Matrix (const Matrix_t&);          //std::vector<std::vector<ty>>
		Matrix (const Matrix&);            //�R�s�[�R���X�g���N�^
		~Matrix();                         //�f�X�g���N�^


		//MatrixOperator.hpp
		inline Matrix& operator =(MatrixInit_t);
		inline Matrix& operator =(const Matrix&);
		inline Matrix& operator =(Matrix&&);

		inline Matrix& operator +=(const Matrix&); //���Z
		inline Matrix& operator -=(const Matrix&); //���Z
		inline Matrix& operator ^=(const Matrix&); //����
		inline Matrix& operator *=(const Matrix&); //�s���
		inline Matrix& operator *=(ty);            //�X�J���[�{

		inline Matrix  operator + (const Matrix&); //���Z
		inline Matrix  operator - (const Matrix&); //���Z
		inline Matrix  operator ^ (const Matrix&); //����
		inline Matrix  operator * (const Matrix&); //�s���
		inline Matrix  operator * (ty);            //�X�J���[�{

		template<typename CastTy> operator Sanae::Matrix<CastTy>(); //�L���X�g

		inline Matrix& operator <<(Matrix&);         //���n
		inline std::vector<ty>& operator [](size_t); //�v�f�փA�N�Z�X

		inline bool    operator ==(const Matrix&); //��r==
		inline bool    operator !=(const Matrix&); //��r!=


		//MatrixUtil.hpp
		inline size_t GetRow   (); //�s�����擾
		inline size_t GetColumn(); //�񐔂��擾

		inline Matrix& SwapRow(size_t, size_t); //�������s�Ƒ������s�����ւ���B
		inline Matrix& SwapColumn(size_t, size_t); //��������Ƒ�����������ւ���B

		inline Matrix& ReSize(std::pair<size_t, size_t>); //�T�C�Y��ύX����B

		inline Matrix& Setter     (std::function<ty()>);                  //�����̊֐����Ăяo���e�v�f�֕Ԃ�l��������B
		inline Matrix& Setter     (std::function<ty(size_t,size_t,ty&)>); //�����̊֐����Ăяo���e�v�f�֕Ԃ�l��������B(�s��,��,���̍s��[�s��][��])�������Ƃ��Ď󂯎��B

		inline Matrix  Transpose  (); //�]�u����B

		static inline Matrix<ty> Identity(size_t); //�w�肳�ꂽ�s,�񐔂̒P�ʍs����쐬���܂��B
		static inline Matrix<ty> Zero    (size_t); //�w�肳�ꂽ�s,�񐔂�0�s����쐬���܂��B


		//MatrixAdvCalc.hpp
		inline ty     Det    ();        //�s�񎮂����߂�B
		inline Matrix Inverse(ty=1e-5); //�t�s������߂�B


 	};


}




/*�����ȍs���������O�N���X*/
class InvalidMatrix : std::exception{
protected:
	std::string m_ErrorMessage = "It is an invalid matrix.";

public:
	InvalidMatrix() {}
	
	//���b�Z�[�W�����
	InvalidMatrix(const char* ArgMessage) : m_ErrorMessage(ArgMessage) {}

	//�G���[���b�Z�[�W��Ԃ��B
	const char* what() const noexcept override {return m_ErrorMessage.c_str();}
};




#endif