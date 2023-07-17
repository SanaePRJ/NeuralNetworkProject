/*=============================================================
* NAME      : MatrixBase.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
* 
* �v���g�^�C�v�錾
* 
* ����:
* MatrixBaseImpl.hpp���C���N���[�h���Ă��������B
=============================================================*/


#ifndef _SANAE_MATRIXBASE_HPP_
#define _SANAE_MATRIXBASE_HPP_


//Check Version C++14�ȏ�
#if (_cplusplus >= 201402L) || (defined(_MSVC_LANG)&&_MSVC_LANG >= 201402L)

//Is existing
#if __has_include(<vector>) && __has_include(<utility>) && __has_include(<stdexcept>) && __has_include(<iostream>)
#define _SANAE_MATRIXBASE_HPP_ALL_INCLUDE_


/*INCLUDE*/
#include <iostream>

#include <vector>
#include <utility>
#include <stdexcept>


/*DEFINE CLASS*/
using  Ulong = unsigned _int64;
using  Slong = signed   _int64;
using  SizeT = std::pair<Ulong, Ulong>;


//��Z��10^-6�ȉ��̌덷�͖�������B
constexpr auto _ERROR_SANAE = 1e-12;


namespace Sanae {
	template<typename _T = double> class _MatrixBase {
		//Define Variables
	protected:
		std::vector<_T>   _Main;                         //�s����i�[����B
		SizeT             _Size = std::make_pair(0, 0);  //�s��̃T�C�Y���i�[����B first:�� second:�s


		//Define Functions(protected)
	protected:
		//{��,�s}��z��ԍ��ɕϊ�����B
		Ulong _Convert_to_ArrayNum
		(
			Ulong, SizeT
		);

		//�s��ԍ���{��,�s}�ɕϊ�����B
		SizeT _Convert_to_Position
		(
			Ulong, Ulong
		);

		//�w�肳�ꂽ�s�����ւ��܂��B
		void _Swap_Line
		(
			Ulong, Ulong, std::vector<_T>*, SizeT
		);

		//�w�肵��������ւ��܂��B
		void _Swap_Column
		(
			Ulong, Ulong, std::vector<_T>*, SizeT
		);

		//�����Z���s���܂��B
		void _Add
		(
			const _MatrixBase&, std::vector<_T>*, SizeT
		);

		//�����Z���s���܂��B
		void _Sub
		(
			const _MatrixBase&, std::vector<_T>*, SizeT
		);

		//�X�J���[�{���s���܂��B
		void _Scalar_mul
		(
			std::vector<_T>&, _T
		);

		//�s��̏�Z���s���܂��B
		void _Mul
		(
			std::vector<_T>&, SizeT,
			std::vector<_T>&, SizeT,
			std::vector<_T>*, SizeT*
		);


		//Define Functions(public)
	public:
		//Constructor
		_MatrixBase();
		_MatrixBase
		(
			std::initializer_list<std::vector<_T>>
		);
		_MatrixBase
		(
			SizeT
		);
		_MatrixBase
		(
			std::pair<SizeT, std::vector<_T>> _In
		);

		//Operator
		//��r
		bool operator==
		(
			const _MatrixBase<_T>&
		);

		//�z��Q��
		_T& operator[]
		(
			Ulong
		);
		//{��,�s}�Q��
		_T& operator[]
		(
			SizeT
		);

	};
}


#endif //Is existing
#endif //Check Version
#endif //Include Guard