/*-------------------------------------------------------------
* Name    : MatrixCalc.hpp
* Version : 4.0.3
* * Author: SanaePRJ
* Description:
*  MatrixBase�^�̌v�Z���\�b�h������
-------------------------------------------------------------*/




#ifndef SANAE_MATRIXCALC_HPP
#define SANAE_MATRIXCALC_HPP

#include "Matrix.h"




//�n���ꂽ�֐������ׂĂ̗v�f�ɑ΂��Ď��s���܂��Barg_data1 = Function(arg_data1,arg_data2)
template<typename ty>
inline void Sanae::Matrix<ty>::m_calc(MatrixT* arg_data1,const MatrixT* arg_data2, std::function<ty(ty,ty)> Func) const
{
	//�T�C�Y��0�̏ꍇ�v�Z�ł��Ȃ�...
	if (arg_data1->size() == 0 || arg_data2->size() == 0)
		throw std::invalid_argument("You must have some data.");

	//���������̍s�Ɨ񐔂��Ⴄ�ꍇ�G���[
	if (this->m_GetRowSize(arg_data1) != this->m_GetRowSize((MatrixT*)arg_data2) || this->m_GetColumnSize(arg_data1) != this->m_GetColumnSize((MatrixT*)arg_data2))
		throw std::invalid_argument("The number of rows and columns of data1 and data2 must be equal.");

	const size_t Row    = this->m_GetRowSize   (arg_data1);  //�v�Z����s��
	const size_t Column = this->m_GetColumnSize(arg_data1);  //�v�Z�����

	for (size_t pos_row = 0; pos_row < Row; pos_row++)
	{
		for (size_t pos_column = 0; pos_column < Column; pos_column++)
		{
			//arg_data1�Ɍv�Z���ʂ����Ă���
			(*arg_data1)[pos_row][pos_column] = Func((*arg_data1)[pos_row][pos_column], (*arg_data2)[pos_row][pos_column]);
		}
	}

	return;
}




//���Z
template<typename ty>
inline void Sanae::Matrix<ty>::m_add(MatrixT* arg_data1, const MatrixT* arg_data2) const
{
	this->m_calc(arg_data1, arg_data2, [](ty d1, ty d2)->ty {return d1 + d2; });
	return;
}


//���Z
template<typename ty>
inline void Sanae::Matrix<ty>::m_sub(MatrixT* arg_data1, const MatrixT* arg_data2) const
{
	this->m_calc(arg_data1, arg_data2, [](ty d1, ty d2)->ty {return d1 - d2; });
	return;
}


//dot
template<typename ty>
inline void Sanae::Matrix<ty>::m_dotmul(MatrixT* arg_data1, const MatrixT* arg_data2) const
{
	this->m_calc(arg_data1, arg_data2, [](ty d1, ty d2)->ty {return d1 * d2; });
	return;
}


//�X�J���[�{
template<typename ty>
inline void Sanae::Matrix<ty>::m_scalarmul(MatrixT* arg_data1, ty arg_data2) const
{
	this->m_calc(arg_data1, (const MatrixT*)arg_data1, [&arg_data2](ty d1, ty d2)->ty {return d1 * arg_data2; });

	return;
}




#ifndef SANAE_MATRIX_NOTHREADS


#include <thread>




//�s��̊|���Z(Threads)
template<typename ty>
inline void Sanae::Matrix<ty>::m_mul(MatrixT* arg_data1, const MatrixT* arg_data2) const
{
	//�������̗񐔂Ƒ������̍s���͓����łȂ���΂Ȃ�Ȃ��B
	if (this->m_GetColumnSize(arg_data1) != this->m_GetRowSize((MatrixT*)arg_data2))
		throw std::invalid_argument("The number of columns in data1 must be the same as the number of rows in data2.");


	//�������̍s���Ƒ������̗񐔂��m��,0�ŏ�����
	MatrixT buf(this->m_GetRowSize(arg_data1), std::vector<ty>(this->m_GetColumnSize((MatrixT*)arg_data2), 0));

	//�s��T�C�Y���擾
	const size_t Row    = this->m_GetRowSize   (&buf); //�s��
	const size_t Column = this->m_GetColumnSize(&buf); //��
	
	//�v�Z��
	const size_t alltaskcount = Row * Column;
	const size_t taskcount    = alltaskcount / this->thread;


	//�X���b�h����������!
	if (taskcount == 0)
		throw std::invalid_argument("There are too many threads.");


	//l[i][j] = ��k=0,n (m[i][k] * n[k][j])���v�Z�����郉���_��
	auto mul_lambda = [arg_data1, arg_data2, this](size_t arg_Row, size_t arg_Column)
		{
			const size_t size = this->m_GetColumnSize(arg_data1);
			ty           sum  = 0;

			//��k = 0, n(m[i][k] * n[k][j])
			for (size_t Pos = 0; Pos < size; Pos++)
				sum += (*arg_data1)[arg_Row][Pos] * (*arg_data2)[Pos][arg_Column];

			return sum;
		};

	//thread�ŕ������邽�߂̃����_��
	auto mul_thread = [&Row,&Column,&buf,&mul_lambda](size_t from,size_t to) {
			for (size_t pos = from; pos < to; pos++) 
				buf[(pos / Column)][(pos % Column)] = mul_lambda((pos / Column),( pos % Column));
		};
	

	//�X���b�h�Ǘ�
	std::vector<std::thread> threads;
	for (size_t pos = 0; pos < alltaskcount;)
	{
		size_t from = pos;
		size_t to   = pos + taskcount;
		
		if (to > alltaskcount)
			to = alltaskcount;

		//�X���b�h���쐬
		threads.push_back(std::thread(mul_thread,from,to));

		pos = to;
	}

	//�^�X�N�����܂ő҂�
	for (std::thread& th:threads)
		th.join();

	arg_data1->clear();

	//buf��������ɏ��n
	*arg_data1 = std::move(buf);


	return;
}




#else




//�s��̊|���Z(NoThreads)
template<typename ty>
inline void Sanae::Matrix<ty>::m_mul(MatrixT* arg_data1, const MatrixT* arg_data2) const
{
	//�������̗񐔂Ƒ������̍s���͓����łȂ���΂Ȃ�Ȃ��B
	if (this->m_GetColumnSize(arg_data1) != this->m_GetRowSize((MatrixT*)arg_data2))
		throw std::invalid_argument("The number of columns in data1 must be the same as the number of rows in data2.");

	//�������̍s���Ƒ������̗񐔂��m��,0�ŏ�����
	MatrixT buf(this->m_GetRowSize(arg_data1), std::vector<ty>(this->m_GetColumnSize((MatrixT*)arg_data2), 0));

	const size_t Row    = this->m_GetRowSize(&buf);    //�s��
	const size_t Column = this->m_GetColumnSize(&buf); //��

	//l[i][j] = ��k=0,n (m[i][k] * n[k][j])���v�Z�����郉���_��
	auto mul_lambda = [arg_data1, arg_data2, this](size_t arg_Row, size_t arg_Column)
		{

			const size_t size = this->m_GetColumnSize(arg_data1);
			ty           sum = 0;

			for (size_t Pos = 0; Pos < size; Pos++)
				sum += (*arg_data1)[arg_Row][Pos] * (*arg_data2)[Pos][arg_Column];

			return sum;
		};

	for (size_t pos_row = 0; pos_row < Row; pos_row++)
	{
		for (size_t pos_column = 0; pos_column < Column; pos_column++)
		{
			buf[pos_row][pos_column] = mul_lambda(pos_row, pos_column);
		}
	}

	arg_data1->clear();
	//buf��������ɏ��n
	std::move(buf.begin(), buf.end(), std::back_inserter(*arg_data1));

	return;
}




#endif




#endif