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
#include <iostream>
#include <vector>
#include <string>
#include <math.h>


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
namespace Sanae {
	CONSOLE Console;
}



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
			volatile unsigned char color_buf = (unsigned char)(255 - std::abs(View));

			//背景を変更
			Console << Color_Console{ Color{ 0, 0, 0 }, Color{color_buf,color_buf,color_buf} };
			//空白を出力
			std::cout << "　";

			//背景色をリセット
			Console.Reset();
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
	_PLOT_ PLOT = 28;
}



#endif