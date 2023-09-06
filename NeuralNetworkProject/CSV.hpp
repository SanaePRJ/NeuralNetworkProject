/*=============================================================
* NAME      : ReadCSV.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.0
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/


#ifndef _SANAE_CSV_HPP_
#define _SANAE_CSV_HPP_


/*INCLUDE*/
#include "Matrix.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>


namespace Sanae {


/*CSVファイル読み取り用*/
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
			Ifs.open(_FileName, std::ios_base::in);
			if (!Ifs)
				throw std::runtime_error("Can't open file.");
		}
		~ReadCSV()
		{
			Ifs.close();
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

		ReadCSV& operator >> (std::string& _Data) {
			_Data = this->ReadData();

			return *this;
		}
		ReadCSV& operator >> (std::vector<std::string>& _Data) {
			this->ReadLine(&_Data);

			return *this;
		}

		//文字列を読み取る
		std::string ReadData()
		{
			std::string _data;
			this->Read(&_data);

			return _data;
		}
		//一行読み取る
		void ReadLine
		(
			std::vector<std::string>* _Store,
			Ulong                     _Surplus = 10
		)
		{
			this->Is_NewLine = false;
			while (!this->Is_NewLine) {
				if (_Store->size() + 1 >= _Store->capacity())
					_Store->reserve(_Surplus);

				_Store->push_back(this->ReadData());
			}
			this->Is_NewLine = false;

			return;
		}
	};


/*CSVファイル書き込み用*/
	class WriteCSV {
	private:
		std::ofstream ofs;
		std::string  Write_Buffer;


	private:
		//区切りを挿入する。
		void Insert() {
			if (this->Write_Buffer.size() != 0)
			{
				char buf = *(this->Write_Buffer.end() - 1);

				if (buf != ',' && buf != '\n')
					this->Write_Buffer += ',';
			}
		}

		//改行を挿入する。
		template<typename _T> bool NewLine(_T _Data) {
			std::string buf;
			buf = _Data;

			bool Is_Char =
			{
				typeid(_T) == typeid(std::string)
				or
				typeid(_T) == typeid(const char*)
				or
				typeid(_T) == typeid(const char)
				or
				typeid(_T) == typeid(char*)
				or
				typeid(_T) == typeid(char)
			};

			if (Is_Char and buf == (std::string)"\n") {
				Write_Buffer += '\n';
				this->ofs.write(Write_Buffer.c_str(), Write_Buffer.size());
				Write_Buffer.erase(Write_Buffer.begin(), Write_Buffer.end());

				return true;
			}

			return false;
		}

		//バッファに書き込む
		template<typename _T> void Buffer_Write(_T _Data) {
			if (NewLine(_Data))
				return;

			Insert();
			this->Write_Buffer += std::to_string(_Data);

			return;
		}
		template<> void Buffer_Write(const char* _Data) {
			if (NewLine(_Data))
				return;

			Insert();
			this->Write_Buffer += _Data;

			return;
		}
		template<> void Buffer_Write(std::string _Data) {
			if (NewLine(_Data))
				return;

			Insert();
			this->Write_Buffer += _Data;

			return;
		}
	
	
	public:
		WriteCSV(const char* _FileName, bool _DoInit)
		{
			if (_DoInit)
				this->ofs.open(_FileName,std::ios::out);
			else
				this->ofs.open(_FileName,std::ios::app);

			if (!this->ofs.is_open())
				throw std::runtime_error("Can't open file.");
		}
		~WriteCSV()
		{
			this->close();
		}

		WriteCSV& open(const char* _FileName, bool _DoInit)
		{
			if (_DoInit)
				this->ofs.open(_FileName, std::ios::out);
			else
				this->ofs.open(_FileName, std::ios::app);

			if (!this->ofs.is_open())
				throw std::runtime_error("Can't open file.");
		}
		WriteCSV& close()
		{
			ofs << this->Write_Buffer.c_str();
			this->Write_Buffer.erase(this->Write_Buffer.begin(), this->Write_Buffer.end());

			this->ofs.close();

			return *this;
		}

		template<typename _DataType> WriteCSV& operator << (_DataType _Data)
		{
			this->Buffer_Write(_Data);

			return *this;
		}
		template<typename _DataType> WriteCSV& operator << (std::initializer_list<_DataType> _DataList)
		{
			for (_DataType i : _DataList)
				this->Buffer_Write(i);

			return *this;
		}

		WriteCSV& flush() {
			if (Write_Buffer.size() >= 1)
			{
				this->ofs << std::string(Write_Buffer.begin(), Write_Buffer.end() - 1);
				Write_Buffer.erase(Write_Buffer.begin(), Write_Buffer.end() - 1);
			}

			this->ofs.flush();
			return *this;
		}
	};


}


#endif
