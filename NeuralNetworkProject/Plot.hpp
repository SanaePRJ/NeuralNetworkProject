/*=============================================================
* NAME      : Plot.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_PLOT_HPP_
#define _SANAE_PLOT_HPP_


#include <iostream>
#include <vector>
#include <string>

#include "Matrix.h"

#if __has_include("windows.h")
	#include <windows.h>
#endif


namespace Sanae {


	//RGB���i�[
	struct Color {
		unsigned char R = 0, G = 0, B = 0;
	};

	//�����F�Ɣw�i��RGB���i�[
	struct Color_Console {
		Color Letter, BackGround;
	};
	const Color_Console Default_Color = { Color{255,255,255},Color{0,0,0} };
	

	//Console�̐F��ύX���܂��B
	class _CONSOLE_ {
	private:
		void Set_Color
		(
			Color _Letter_Color,
			Color _Background_Color
		)
		{
			std::string buf;

			//�����F
			buf = "\x1b[38;2;";
			(buf += std::to_string((int)_Letter_Color.R)) += ";";
			(buf += std::to_string((int)_Letter_Color.G)) += ";";
			(buf += std::to_string((int)_Letter_Color.B)) += "m";
			std::cout << buf;

			//�w�i�F
			buf = "\x1b[48;2;";
			(buf += std::to_string((int)_Background_Color.R)) += ";";
			(buf += std::to_string((int)_Background_Color.G)) += ";";
			(buf += std::to_string((int)_Background_Color.B)) += "m";
			std::cout << buf;
		}
		void Reset_Color()
		{
			std::cout << "\x1b[m";
		}

	public:
		_CONSOLE_()
		{
		//�G�X�P�[�v�V�[�P���X
		#if __has_include("windows.h")
			HANDLE stdHandle = GetStdHandle(STD_OUTPUT_HANDLE);
			DWORD  mode = 0;

			GetConsoleMode(stdHandle, &mode);
			SetConsoleMode(stdHandle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
		#endif
		}

		//Console�̐F��ݒ�
		_CONSOLE_& operator << (const Color_Console _Console_Color)
		{
			this->Set_Color(_Console_Color.Letter, _Console_Color.BackGround);

			return *this;
		}

		_CONSOLE_& SetColor(const Color_Console _Console_Color) 
		{
			this->Set_Color(_Console_Color.Letter, _Console_Color.BackGround);

			return *this;
		}
		_CONSOLE_& Reset()
		{
			this->Reset_Color();

			return *this;
		}
	};
	_CONSOLE_ Console;


	//MNIST�f�[�^�Z�b�g�̕������o�͂��邽�߂̌^
	class _PLOT_ {
	private:
		//Plot����T�C�Y
		size_t Weight;

	public:
		_PLOT_
		(
			size_t _Weight
		)
			: Weight(_Weight)
		{
		//�G�X�P�[�v�V�[�P���X
		#if __has_include("windows.h")
			HANDLE stdHandle = GetStdHandle(STD_OUTPUT_HANDLE);
			DWORD  mode      = 0;

			GetConsoleMode(stdHandle, &mode);
			SetConsoleMode(stdHandle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
		#endif
		}
		
		//�\��
		template<typename Type> _PLOT_& operator << (const std::vector<Type>& _View_Data)
		{
			size_t _point = 0;

			for (Type View : _View_Data) {
				//�w�i��ύX
				Console << Color_Console{Color{ 0, 0, 0 }, { (unsigned char)(std::abs(View)),(unsigned char)(std::abs(View)),(unsigned char)(std::abs(View)) }};
				
				//�󔒂��o��
				std::cout << "�@";

				//�w�i�F�����Z�b�g
				Console.Reset_Color();
				_point++;

				//���s���s��
				if(_point == this->Weight){
					_point = 0;
					std::cout << std::endl;
				}
			}
			std::cout << std::flush;

			return *this;
		}
	};
	//MNIST�f�[�^�Z�b�g�̕������o�͂���B
	_PLOT_ PLOT = 28;


	//�T�C�Y��ύX���܂��B
	void Change_Size(Matrix& _Data, Matrix& _Store,double _Ratio,SizeT _Center = { 14,14 }) {
		auto Pos = [_Ratio,_Center](SizeT _Data) {
			Ulong x = (Ulong)((long long)_Center.first  + (1/_Ratio) * ((long long)_Data.first  - (long long)_Center.first));
			Ulong y = (Ulong)((long long)_Center.second + (1/_Ratio) * ((long long)_Data.second - (long long)_Center.second));

			return SizeT{x,y};
		};

		//���S�_���T�C�Y���������ꍇ��throw����B
		if (_Data.GetSizeWH().first <= _Center.first || _Data.GetSizeWH().second <= _Center.second)
			throw std::invalid_argument("_DataSize must be greater than the _Center.");

		//_Store�̃T�C�Y��_Data�Ɠ����ɂ���B
		_Store.SetSize(_Data.GetSizeWH(), true);

		for (Ulong y = 0; y < _Data.GetSizeWH().second; y++) {
			for (Ulong x = 0; x < _Data.GetSizeWH().first; x++) {
				//�z��ԍ�
				SizeT pos = Pos({ x,y });

				//�͈͊O�͖�������B
				if (pos.first < _Data.GetSizeWH().first && pos.second < _Data.GetSizeWH().second)
					_Store[{x, y}] += _Data[pos];
			}
		}

		return;
	}

	//�p�x��ύX���܂��B
	void Change_Angle(Matrix& _Data, Matrix& _Store, double _Deg, std::pair<int, int> _Center = {14,14}) {
		const double _PI  = 3.141592653589793; //�~����
		const double _Rad = _PI * _Deg / 180;  //���W�A���ɕϊ�
		
		//���̓f�[�^��_Rad[rad]�X�����z��ԍ���Ԃ��B
		auto Angle = [_Rad, _Center](SizeT _Data)
		{
			double x = _Center.first  + (std::cos(_Rad) * ((long long)_Data.first - _Center.first) - std::sin(_Rad) * ((long long)_Data.second - _Center.second));
			double y = _Center.second + (std::sin(_Rad) * ((long long)_Data.first - _Center.first) + std::cos(_Rad) * ((long long)_Data.second - _Center.second));
			
			return SizeT{ static_cast<Ulong>(std::nearbyint(x)), static_cast<Ulong>(std::nearbyint(y))};
		};

		//���S�_���T�C�Y���������ꍇ��throw����B
		if (_Data.GetSizeWH().first <= _Center.first || _Data.GetSizeWH().second <= _Center.second)
			throw std::invalid_argument("_DataSize must be greater than the _Center.");

		//_Store�̃T�C�Y��_Data�Ɠ����ɂ���B
		_Store.SetSize(_Data.GetSizeWH(), true);

		for (Ulong y = 0; y < _Data.GetSizeWH().second; y++) {
			for (Ulong x = 0; x < _Data.GetSizeWH().first; x++) {
				//�X������̔z��ԍ�
				SizeT pos = Angle({ x,y });

				//�͈͊O�͖�������B
				if (pos.first < _Data.GetSizeWH().first && pos.second < _Data.GetSizeWH().second)
					_Store[{x, y}] += _Data[pos];
			}
		}

		return;
	}


}


#endif