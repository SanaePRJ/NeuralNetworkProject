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


	template<typename _ReadType> class ReadCSV {
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

			for (unsigned char i = 0; i < sizeof(_ReadType); i++) {
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

		//®”Œ^‚ð“Ç‚ÝŽæ‚é
		_ReadType ReadDataI
		(
			_ReadType (*_Conv)(const std::string&,std::size_t*,int) = std::stoi
		)
		{
			std::string _data;
			_data.reserve(sizeof(_ReadType));

			this->Read(&_data);

			return _Conv(_data,nullptr,10);
		}
		//ŽÀ”Œ^‚ð“Ç‚ÝŽæ‚é
		_ReadType ReadDataD
		(
			_ReadType(*_Conv)(const std::string&, std::size_t*) = std::stod
		)
		{
			std::string _data;
			_data.reserve(sizeof(_ReadType));

			this->Read(&_data);
			
			return _Conv(_data, nullptr);
		}

		void ReadLineI
		(
			std::vector<_ReadType>* _Store,
			Ulong                   _Surplus = 10,

			_ReadType(*_Conv)(const std::string&, std::size_t*, int) = std::stoi
		)
		{
			while (!this->Is_NewLine) {
				if (_Store->size() + 1 >= _Store->capacity())
					_Store->reserve(_Surplus);

				_Store->push_back(this->ReadDataI(_Conv));
			}	

			return;
		}
		void ReadLineD
		(
			std::vector<_ReadType>* _Store,
			Ulong                   _Surplus = 10,

			_ReadType(*_Conv)(const std::string&, std::size_t*) = std::stod
		)
		{
			while (!this->Is_NewLine) {
				if (_Store->size() + 1 >= _Store->capacity())
					_Store->reserve(_Surplus);

				_Store->push_back(this->ReadDataD(_Conv));
			}

			return;
		}
	};


}


#endif