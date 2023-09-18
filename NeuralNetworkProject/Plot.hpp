/*=============================================================
* NAME      : Plot.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.1
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/




/*-----Include Guard-----*/
#ifndef _SANAE_PLOT_HPP_
#define _SANAE_PLOT_HPP_




/*-----Include-----*/
#include "Matrix.hpp"


/*-----Check Version C++14 or more and Is existing-----*/
#if defined(SANAE_CPP_14_OR_MORE) && defined(SANAE_MATRIXBASE_HPP_ALL_INCLUDE)


/*-----Include-----*/
#include <iostream>
#include <vector>
#include <string>


#if __has_include("windows.h")
	#include <windows.h>
#endif




/*-----Define Class-----*/
namespace Sanae {
	/*-----RGBを格納-----*/
	struct Color {
		unsigned char R = 0, G = 0, B = 0;
	};
	/*-----文字色と背景のRGBを格納-----*/
	struct Color_Console {
		Color Letter, BackGround;
	};

	class CONSOLE;
	class _PLOT_;

	CONSOLE Console;
	_PLOT_ PLOT = 28;
	const Color_Console Default_Color = { Color{255,255,255},Color{0,0,0} };
}




/*-----コンソールの色を変更します。-----*/
class Sanae::CONSOLE 
{


	/*-----Define Function(private)-----*/
private:


	/*----------------------------------------------
	* Change console color.
	* 色を変更する。
	----------------------------------------------*/
	void Set_Color
	(
		Color _Letter_Color,
		Color _Background_Color
	)
	{
		std::string buf;

		//文字色
		buf = "\x1b[38;2;";
		(buf += std::to_string((int)_Letter_Color.R)) += ";";
		(buf += std::to_string((int)_Letter_Color.G)) += ";";
		(buf += std::to_string((int)_Letter_Color.B)) += "m";
		std::cout << buf;

		//背景色
		buf = "\x1b[48;2;";
		(buf += std::to_string((int)_Background_Color.R)) += ";";
		(buf += std::to_string((int)_Background_Color.G)) += ";";
		(buf += std::to_string((int)_Background_Color.B)) += "m";
		std::cout << buf;
	}

	/*----------------------------------------------
	* Reset color.
	* 色を初期値に変更。
	----------------------------------------------*/
	void Reset_Color()
	{
		std::cout << "\x1b[m";
	}


	/*-----Define Functions(public)-----*/
public:


	/*-----Constructor-----*/
	CONSOLE()
	{
	//エスケープシーケンス
	#if __has_include("windows.h") && defined(ENABLE_VIRTUAL_TERMINAL_PROCESSING)
		HANDLE stdHandle = GetStdHandle(STD_OUTPUT_HANDLE);
		DWORD  mode = 0;

		GetConsoleMode(stdHandle, &mode);
		SetConsoleMode(stdHandle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
	#endif
	}

	/*----------------------------------------------
	* Change console color.
	* 色を変更する。
	----------------------------------------------*/
	CONSOLE& operator << (const Color_Console _Console_Color)
	{
		this->Set_Color(_Console_Color.Letter, _Console_Color.BackGround);

		return *this;
	}

	/*----------------------------------------------
	* Reset console color.
	* 色をリセットする。
	----------------------------------------------*/
	CONSOLE& Reset()
	{
		this->Reset_Color();

		return *this;
	}


};




/*-----MNISTデータセットの文字を出力するための型-----*/
class Sanae::_PLOT_ {


	/*-----Define variables(private)-----*/
private:
	//Plotするサイズ
	size_t Weight;


	/*-----Define Functions(public)-----*/
public:


	/*-----Constructor-----*/
	_PLOT_
	(
		size_t _Weight
	)
		: Weight(_Weight)
	{}
		
	/*----------------------------------------------
	* Show Matrix.
	* 行列を表示します。
	----------------------------------------------*/
	template<typename Type> _PLOT_& operator << (const std::vector<Type>& _View_Data)
	{
		size_t _point = 0;

		for (Type View : _View_Data) {
			//背景を変更
			Console << Color_Console{Color{ 0, 0, 0 }, { (unsigned char)(std::abs(View)),(unsigned char)(std::abs(View)),(unsigned char)(std::abs(View)) }};
				
			//空白を出力
			std::cout << "　";

			//背景色をリセット
			Console.Reset_Color();
			_point++;

			//改行を行う
			if(_point == this->Weight){
				_point = 0;
				std::cout << std::endl;
			}
		}
		std::cout << std::flush;

		return *this;
	}
};




namespace Sanae {
	//サイズを変更します。
	void Change_Size(Matrix& _Data, Matrix& _Store, double _Ratio, SizeT _Center = { 14,14 }) {
		auto Pos = [_Ratio, _Center](SizeT _Data) {
			Ulong x = (Ulong)((long long)_Center.first + (1 / _Ratio) * ((long long)_Data.first - (long long)_Center.first));
			Ulong y = (Ulong)((long long)_Center.second + (1 / _Ratio) * ((long long)_Data.second - (long long)_Center.second));

			return SizeT{ x,y };
			};

		//中心点よりサイズが小さい場合はthrowする。
		if (_Data.GetSizeWH().first <= _Center.first || _Data.GetSizeWH().second <= _Center.second)
			throw std::invalid_argument("_DataSize must be greater than the _Center.");

		//_Storeのサイズを_Dataと同じにする。
		_Store.SetSize(_Data.GetSizeWH(), true);

		for (Ulong y = 0; y < _Data.GetSizeWH().second; y++) {
			for (Ulong x = 0; x < _Data.GetSizeWH().first; x++) {
				//配列番号
				SizeT pos = Pos({ x,y });

				//範囲外は無視する。
				if (pos.first < _Data.GetSizeWH().first && pos.second < _Data.GetSizeWH().second)
					_Store[{x, y}] += _Data[pos];
			}
		}

		return;
	}

	//角度を変更します。
	void Change_Angle(Matrix& _Data, Matrix& _Store, double _Deg, std::pair<int, int> _Center = { 14,14 }) {
		const double _PI = 3.141592653589793; //円周率
		const double _Rad = _PI * _Deg / 180;  //ラジアンに変換

		//入力データを_Rad[rad]傾けた配列番号を返す。
		auto Angle = [_Rad, _Center](SizeT _Data)
			{
				double x = _Center.first + (std::cos(_Rad) * ((long long)_Data.first - _Center.first) - std::sin(_Rad) * ((long long)_Data.second - _Center.second));
				double y = _Center.second + (std::sin(_Rad) * ((long long)_Data.first - _Center.first) + std::cos(_Rad) * ((long long)_Data.second - _Center.second));

				return SizeT{ static_cast<Ulong>(std::nearbyint(x)), static_cast<Ulong>(std::nearbyint(y)) };
			};

		//中心点よりサイズが小さい場合はthrowする。
		if (_Data.GetSizeWH().first <= _Center.first || _Data.GetSizeWH().second <= _Center.second)
			throw std::invalid_argument("_DataSize must be greater than the _Center.");

		//_Storeのサイズを_Dataと同じにする。
		_Store.SetSize(_Data.GetSizeWH(), true);

		for (Ulong y = 0; y < _Data.GetSizeWH().second; y++) {
			for (Ulong x = 0; x < _Data.GetSizeWH().first; x++) {
				//傾けた先の配列番号
				SizeT pos = Angle({ x,y });

				//範囲外は無視する。
				if (pos.first < _Data.GetSizeWH().first && pos.second < _Data.GetSizeWH().second)
					_Store[{x, y}] += _Data[pos];
			}
		}

		return;
	}
}




#endif
#endif