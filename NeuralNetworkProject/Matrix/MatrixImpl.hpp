/*=============================================================
* NAME      : MatrixImpl.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.2
* COPYRIGHGT: Copyright 2023 SanaeProject.
* 
* �������܂��B
* 
* ����:
* ���̃t�@�C�����C���N���[�h����O��Matrix.hpp���C���N���[�h���Ă��������B
=============================================================*/


#ifndef _SANAE_MATRIXIMPL_HPP_
#define _SANAE_MATRIXIMPL_HPP_


#if defined(_SANAE_MATRIXBASE_HPP_ALL_INCLUDE_)


#include "MatrixDet.hpp"


namespace Sanae{
	/*------Function private------*/
	//�����l��std::vector<double>�֕ϊ����܂��B
	void Matrix::_Convert_to_Array
	(
		std::initializer_list<std::vector<double>>& _Input,
		std::vector<double>& _Storage
	)
	{
		std::vector<std::vector<double>> _bufs = _Input;

		for (Ulong i = 0; i < _bufs.size(); i++)
			_Storage.insert(_Storage.end(), _bufs[i].begin(), _bufs[i].end());
	}
	//�\�����܂��B
	void Matrix::_View
	(
		std::vector<double>* _DataP,
		SizeT                _Size
	)
	{
		for (Ulong i = 0; i < _DataP->size(); i++) {
			if (i != 0 && i % _Size.first == 0) std::cout << "\n";

			std::cout << (*_DataP)[i] << " ";
		}
		std::cout << "\n";

		return;
	}
	//�P�ʍs��ɂ��܂��B
	void Matrix::_To_Identity_Matrix
	(
		std::vector<double>* _Data,
		SizeT                _Size
	)
	{
		_Data->erase (_Data->begin(), _Data->end());  //�S�f�[�^�̍폜
		_Data->resize(_Size.first * _Size.second);    //�T�C�Y�̕ύX

		for (Ulong i = 0; i < _Size.first; i++)
			for (Ulong j = 0; j < _Size.second; j++)
				(*_Data)[this->_Convert_to_ArrayNum(_Size.first, { i,j })] = i == j ? 1 : 0; //�s�Ɨ�̔ԍ��������ꍇ1����ȊO�̏ꍇ0

		return;
	}

	//�t�s������߂܂��B
	void Matrix::_Inverse_matrix
	(
		std::vector<double>* _Data,
		std::vector<double>* _Storage,
		SizeT                _Size
	)
	{
		if (this->_Det(this->_Main, this->_Size) == 0)
			throw std::runtime_error("Inverse does not exist.");

		//�P�ʍs��ɂ���B
		this->_To_Identity_Matrix(_Storage, _Size);

		//_pos:�
		for (Ulong _pos = 0; _pos < _Size.first; _pos++) {
			for (Ulong y = 0; y < _Size.second; y++) {
				if (_pos != y) {
					double div = 0;

					//���0�̏ꍇ�s�����ւ���B
					if ((*_Data)[this->_Convert_to_ArrayNum(_Size.first, { _pos,_pos })] == 0) {
						for (Ulong findy = 0; findy < _Size.second; findy++) {
							if ((*_Data)[this->_Convert_to_ArrayNum(_Size.first, { _pos,findy })] != 0) {
								this->_Swap_Line(_pos, findy, _Data, _Size);
								this->_Swap_Line(_pos, findy, _Storage, _Size);
							}
						}
					}

					//ax+b=0 x -= b/a  
					div = ((*_Data)[this->_Convert_to_ArrayNum(_Size.first, { _pos,y })] / (*_Data)[this->_Convert_to_ArrayNum(_Size.first, { _pos,_pos })]);

					//���̑���ւ̓K�p
					for (Ulong x = 0; x < _Size.first; x++) {
						(*_Data)[this->_Convert_to_ArrayNum(_Size.first, { x,y })] -= div * ((*_Data)[this->_Convert_to_ArrayNum(_Size.first, { x,_pos })]);
						(*_Storage)[this->_Convert_to_ArrayNum(_Size.first, { x,y })] -= div * ((*_Storage)[this->_Convert_to_ArrayNum(_Size.first, { x,_pos })]);
					}
				}
			}
		}

		//���1�ɂȂ��Ă��Ȃ��ꍇ���̍s�֓K�p��1�ɂ���B
		for (Ulong _pos = 0; _pos < _Size.first; _pos++) {
			for (Ulong _x = 0; _x < _Size.first; _x++) {
				double _is_zero_buf = (*_Data)[this->_Convert_to_ArrayNum(_Size.first, { _pos,_pos })];
				if (_is_zero_buf != 0)
					(*_Storage)[this->_Convert_to_ArrayNum(_Size.first, { _x,_pos })] /= _is_zero_buf;
			}
		}

		return;
	}

	//���ς����߂܂��B
	void Matrix::_Inner_Product
	(
		std::vector<double>* _Data1,
		SizeT*               _Size1,

		std::vector<double>* _Data2,
		SizeT*               _Size2,

		std::vector<double>* _Store,

		bool                 _Clear
	) 
	{
		if (_Size1 != _Size2)
			throw std::invalid_argument("Must be same size.");

		Ulong Size = _Size1->first * _Size1->second;

		if (_Clear) {
			_Store->erase(_Store->begin(), _Store->end());
			_Store->resize(Size);
		}

		for (Ulong i = 0; i < Size; i++)
			(*_Store)[i] = (*_Data1)[i] * (*_Data2)[i];
	}

	/*------Function public------*/
	//���
	Matrix& Matrix::operator=
	(
		const Matrix& _Data
	)
	{
		this->_Size = _Data._Size;
		this->_Main = _Data._Main;

		return *this;
	}
	Matrix& Matrix::operator=
	(
		std::initializer_list<std::vector<double>> _In
	)
	{
		this->_Main.erase(this->_Main.begin(), this->_Main.end());
		std::vector<std::vector<double>> _buf = _In;

		this->_Size.second = _In.size();
		this->_Size.first  = _buf[0].size();

		for (Ulong line = 0; line < this->_Size.second; line++)
			this->_Main.insert(this->_Main.end(), _buf[line].begin(), _buf[line].end());

		return *this;
	}

	/*�j��I����*/
	//���Z
	Matrix& Matrix::operator+=
	(
		const Matrix& _Data
	)
	{
		this->_Add(_Data, &this->_Main, this->_Size);

		return *this;
	}
	Matrix& Matrix::operator-=
	(
		const Matrix& _Data
	)
	{
		this->_Sub(_Data, &this->_Main, this->_Size);

		return *this;
	}
	Matrix& Matrix::operator*=
	(
		const Matrix& _Data
	)
	{
		std::vector<double> _ret;
		SizeT           _ret_size;

		std::vector<double> _buf = this->_Main;
		SizeT           _buf_size = this->_Size;

		this->_Mul(_buf, _buf_size, (std::vector<double>&)_Data._Main, _Data._Size, &this->_Main, &this->_Size);

		return *this;
	}
	Matrix& Matrix::operator*=
	(
		double _Num
	)
	{
		for (Ulong i = 0; i < this->_Main.size(); i++)
			this->_Main[i] *= _Num;

		return *this;
	}

	/*��j��I����*/
	//���Z
	Matrix  Matrix::operator+
	(
		const Matrix& _Data
	)
	{
		std::vector<double> _ret = this->_Main;
		this->_Add(_Data, &_ret, this->_Size);

		return std::pair<SizeT, std::vector<double>>{this->_Size, _ret};
	}
	Matrix  Matrix::operator-
	(
		const Matrix& _Data
	)
	{
		std::vector<double> _ret = this->_Main;
		this->_Sub(_Data, &_ret, this->_Size);

		return std::pair<SizeT, std::vector<double>>{this->_Size, _ret};
	}
	Matrix  Matrix::operator*
	(
		const Matrix& _Data
	)
	{
		std::vector<double> _ret;
		SizeT               _ret_size;

		std::vector<double> _buf      = this->_Main;
		SizeT               _buf_size = this->_Size;

		this->_Mul(_buf, _buf_size, (std::vector<double>&)_Data._Main, _Data._Size, &_ret, &_ret_size);

		return std::pair<SizeT, std::vector<double>>{_ret_size, _ret};
	}
	Matrix  Matrix::operator*
	(
		double _Num
	)
	{
		std::vector<double> _data = this->_Main;

		for (Ulong i = 0; i < _data.size(); i++)
			_data[i] *= _Num;

		return std::pair<SizeT, std::vector<double>>{this->_Size, _data};
	}

	//�T�C�Y��ύX���܂��B
	Matrix& Matrix::SetSize
	(
		SizeT _Data,
		bool  _Clear = true
	)
	{
		if (_Clear) 
		{
			this->_Main.erase(this->_Main.begin(), this->_Main.end());
		}
		else 
		{
			//�T�C�Y����
			if (_Data.first > this->_Size.first) {
				for (Ulong i = 1; i < _Data.second; i++)
					this->_Main.insert(this->_Main.begin() + (this->_Size.first * i) + (i - 1), 0);
			}

			//�T�C�Y����
			if (_Data.first < this->_Size.first) {
				for (Ulong i = 1; i < _Data.second; i++)
					this->_Main.erase(this->_Main.begin() + _Data.first, this->_Main.begin() + _Data.first + (this->_Size.first - _Data.first));
			}
		}

		this->_Main.resize(_Data.first * _Data.second);
		this->_Size = _Data;

		return *this;
	}
	//(�j��)�s�����ւ���B
	Matrix& Matrix::Swap_Line
	(
		Ulong _Line1,
		Ulong _Line2
	)
	{
		this->_Swap_Line(_Line1, _Line2, &this->_Main, this->_Size);

		return *this;
	}
	//(�j��)������ւ���B
	Matrix& Matrix::Swap_Column
	(
		Ulong _Column1,
		Ulong _Column2
	)
	{
		this->_Swap_Column(_Column1, _Column2, &this->_Main, this->_Size);

		return *this;
	}
	//�P�ʍs��ɂ���B
	Matrix& Matrix::Ident()
	{
		this->_To_Identity_Matrix(&this->_Main, this->_Size);

		return *this;
	}

	//(��j��)�s�񎮂����߂܂��B 
	double Matrix::Det()
	{
		return this->_Det(this->_Main, this->_Size);
	}
	//(��j��)�s��̓]�u���s���܂��B
	Matrix Matrix::Transpose()
	{
		std::vector<double> _Data;
		_Data.resize(this->_Size.first * this->_Size.second);

		for (Ulong x = 0; x < this->_Size.first; x++)
			for (Ulong y = 0; y < this->_Size.second; y++)
				_Data[this->_Convert_to_ArrayNum(this->_Size.second, { y,x })] = this->_Main[this->_Convert_to_ArrayNum(this->_Size.first, { x,y })];

		return std::pair<SizeT, std::vector<double>>{ SizeT{this->_Size.second, this->_Size.first}, _Data };
	}
	//(��j��)�t�s���Ԃ��܂��B
	Matrix Matrix::Inverse()
	{
		std::vector<double> _to;
		std::vector<double> _buf = this->_Main;

		this->_Inverse_matrix(&_buf, &_to, this->_Size);

		return std::pair<SizeT, std::vector<double>>{ this->_Size, _to };
	}
	//�s���\�����܂��B
	Matrix& Matrix::View()
	{
		this->_View(&this->_Main, this->_Size);

		return *this;
	}

	//�傫����Ԃ��܂��B
	Ulong Matrix::GetSize() {
		return this->_Main.size();
	}
	//�傫����Ԃ��܂��B first:�� second:�s
	SizeT Matrix::GetSizeWH() {
		return this->_Size;
	}

	//(��j��I����)���ς����߂܂��B
	Matrix& Matrix::Inner_Product
	(
		Matrix& _Data
	)
	{
		this->_Inner_Product(&this->_Main,&this->_Size,&_Data._Main,&_Data._Size,&this->_Main,false);
		return *this;
	}
	//(�j��I����)���ς����߂܂��B
	Matrix Matrix::Inner_Product_Destructive
	(
		Matrix& _Data
	) 
	{
		Matrix Ret;
		this->_Inner_Product(&this->_Main, &this->_Size, &_Data._Main, &_Data._Size, &Ret._Main, true);
		Ret._Size = this->_Size;

		return Ret;
	}

	Matrix& Matrix::Move
	(
		std::vector<double>* _Data,
		SizeT                _Size
	)
	{
		this->_Main = std::move(*_Data);
		this->_Size = _Size;

		return *this;
	}

	//_Main�̃|�C���^��Ԃ��܂��B(std::max�Ȃǂ̊֐��Ŏg�p����p)
	const std::vector<double>* Matrix::GetVectorP() 
	{
		return &this->_Main;
	}
}


#endif
#endif