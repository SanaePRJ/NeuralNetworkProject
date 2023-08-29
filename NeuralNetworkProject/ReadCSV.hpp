/*=============================================================
* NAME      : ReadCSV.hpp
* AUTHOR    : SanaeProject
* VER       : 1.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_READCSV_HPP_
#define _SANAE_READCSV_HPP_


/*INCLUDE*/
#include <string>

#include <iostream>
#include <fstream>
#include <vector>


namespace Sanae {


	class ReadCSV {
	private:
		std::ifstream Ifs;
		bool          Is_NewLine = false;


	private:
		void Read
		(
			std::string* _data
		)
		{
			if (!Ifs.is_open())
				throw std::runtime_error("Must be open.");

			while (true) {
				unsigned char buffer = 0;
				this->Ifs.read((char*)&buffer, sizeof(unsigned char));

				if (buffer == '\n')
					this->Is_NewLine = true;
				else
					this->Is_NewLine = false;

				if (buffer == ',' or buffer == '\n' or Ifs.eof())
					break;

				*_data += buffer;
			}
		}


	public:
		ReadCSV() {}
		ReadCSV
		(
			const char* _FileName
		)
		{
			Ifs.open(_FileName,std::ios_base::in);
			if (!Ifs)
				throw std::runtime_error("Can't open file.");
		}
		~ReadCSV() 
		{
			Ifs.close();
		}

		ReadCSV& operator =
		(
				const char* _FileName
		) 
		{
			Ifs.close();
			Ifs.open(_FileName, std::ios_base::in);
			if (!Ifs)
				throw std::runtime_error("Can't open file.");
		}

		ReadCSV& open
		(
			const char* _FileName
		)
		{
			Ifs.close();
			Ifs.open(_FileName, std::ios_base::in);
			if (!Ifs)
				throw std::runtime_error("Can't open file.");

			return *this;
		}
		ReadCSV& close()
		{
			Ifs.close();
			
			return *this;
		}

		//êÆêîå^Çì«Ç›éÊÇÈ
		template<typename _ReadType> _ReadType ReadDataI
		(
			_ReadType (*_Conv)(const std::string&,std::size_t*,int) = std::stoi
		)
		{
			std::string _data;
			this->Read(&_data);

			return _Conv(_data,nullptr,10);
		}
		//é¿êîå^Çì«Ç›éÊÇÈ
		template<typename _ReadType> _ReadType ReadDataD
		(
			_ReadType(*_Conv)(const std::string&, std::size_t*) = std::stod
		)
		{
			std::string _data;
			this->Read(&_data);
			
			return _Conv(_data.c_str(), nullptr);
		}
		//ï∂éöóÒÇì«Ç›éÊÇÈ
		template<typename _ReadType> _ReadType ReadDataStr()
		{
			std::string _data;
			this->Read(&_data);
			
			return _data;
		}

		template<typename _ReadType> void ReadLineI
		(
			std::vector<_ReadType>* _Store,
			Ulong                   _Surplus = 10,

			_ReadType(*_Conv)(const std::string&, std::size_t*, int) = std::stoi
		)
		{
			this->Is_NewLine = false;
			while (!this->Is_NewLine) {
				if (_Store->size() + 1 >= _Store->capacity())
					_Store->reserve(_Surplus);

				_Store->push_back(this->ReadDataI(_Conv));
			}	
			this->Is_NewLine = false;

			return;
		}
		template<typename _ReadType> void ReadLineD
		(
			std::vector<_ReadType>* _Store,
			Ulong                   _Surplus = 10,

			_ReadType(*_Conv)(const std::string&, std::size_t*) = std::stod
		)
		{
			this->Is_NewLine = false;
			while (!this->Is_NewLine) {
				if (_Store->size() + 1 >= _Store->capacity())
					_Store->reserve(_Surplus);

				_Store->push_back(this->ReadDataD(_Conv));
			}
			this->Is_NewLine = false;
			
			return;
		}
		template<typename _ReadType> void ReadLineStr
		(
			std::vector<_ReadType>* _Store,
			Ulong                   _Surplus = 10
		)
		{
			this->Is_NewLine = false;
			while (!this->Is_NewLine) {
				if (_Store->size() + 1 >= _Store->capacity())
					_Store->reserve(_Surplus);

				_Store->push_back(this->ReadDataStr());
			}
			this->Is_NewLine = false;

			return;
		}
	};


}


#endif