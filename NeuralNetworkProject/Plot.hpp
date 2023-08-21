#ifndef _SANAE_PLOT_HPP_
#define _SANAE_PLOT_HPP_


#include <iostream>
#include <vector>
#include <string>

#if __has_include("windows.h")
	#include <windows.h>
#endif


namespace Sanae {

	struct Color {
		unsigned char R = 0, G = 0, B = 0;
	};
	struct Color_Console {
		Color Letter, BackGround;
	};
	const Color_Console Default_Color = { Color{255,255,255},Color{0,0,0} };

	class Plot {
	private:

		size_t Weight                  = 1;
		Color  Normal_Letter_Color     = {255,255,255};
		Color  Normal_Background_Color = { 0 , 0 , 0 };
		
		void Set_Color
		(
			Color _Letter_Color,
			Color _Background_Color
		) 
		{
			std::string buf;

			buf = "\x1b[38;2;";
			(buf += std::to_string((int)_Letter_Color.R)) += ";";
			(buf += std::to_string((int)_Letter_Color.G)) += ";";
			(buf += std::to_string((int)_Letter_Color.B)) += "m";
			std::cout << buf;

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
		Plot
		(
			size_t _Weight
		)
			: Weight(_Weight)
		{
		//エスケープシーケンス
		#if __has_include("windows.h")
			HANDLE stdHandle = GetStdHandle(STD_OUTPUT_HANDLE);
			DWORD  mode = 0;

			GetConsoleMode(stdHandle, &mode);
			SetConsoleMode(stdHandle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
		#endif
		}
		
		Plot& operator << (Color_Console _Console_Color) {
			this->Set_Color(_Console_Color.Letter,_Console_Color.BackGround);
		}
		template<typename Type>
		Plot& operator << (std::vector<Type>& _View_Data) {
			size_t _point = 1;

			for (Type View : _View_Data) {
				this->Set_Color(Color{ 0,0,0 }, {(unsigned char)View,(unsigned char)View,(unsigned char)View});
				std::cout << ' ';
				this->Reset_Color();
				_point++;

				if(_point == this->Weight){
					_point = 0;
					std::cout << std::endl;
				}
			}

		return *this;
		}
	};
}


#endif