/*=============================================================
* NAME      : ReadCSV.hpp
* AUTHOR    : SanaeProject
* VER       : 2.0.1
* COPYRIGHGT: Copyright 2023 SanaeProject.
=============================================================*/




/*-----Include Guard-----*/
#ifndef _SANAE_CSV_HPP_
#define _SANAE_CSV_HPP_


/*-----Include-----*/
#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>


/*-----Check Version C++14 or more and Is existing-----*/




/*-----Define Class-----*/
namespace Sanae {
	class ReadCSV ;
	class WriteCSV;
}




/*-----CSVファイル読み取り用-----*/
class Sanae::ReadCSV {


/*-----Define Variable(private)*/
private:


	/*----------------------------------------------
	* For file editing.
	* ファイル編集用。
	----------------------------------------------*/
	std::ifstream Ifs;

	/*----------------------------------------------
	* Used at line break.
	* 改行時使用。
	----------------------------------------------*/
	bool          Is_NewLine = false;


/*-----Define Functions(private)*/
private:


	/*----------------------------------------------
	* Take out an element.
	* 一要素取り出す。
	----------------------------------------------*/
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


/*-----Define Functions(public)-----*/
public:


	/*-----Constructor-----*/
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

	/*----------------------------------------------
	* Open File.
	* ファイルを開く。
	----------------------------------------------*/
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

	/*----------------------------------------------
	* Close File..
	* ファイルを閉じる。
	----------------------------------------------*/
	ReadCSV& close()
	{
		Ifs.close();

		return *this;
	}

	/*----------------------------------------------
	* Write Read Data.
	* 読み取ったデータを書き込む。
	----------------------------------------------*/
	ReadCSV& operator >> (std::string& _Data) {
		_Data = this->ReadData();

		return *this;
	}
	/*----------------------------------------------
	* Write ReadLine Data.
	* 読み取ったデータ一行を書き込む。
	----------------------------------------------*/
	ReadCSV& operator >> (std::vector<std::string>& _Data) {
		this->ReadLine(&_Data);

		return *this;
	}

	/*----------------------------------------------
	* Read one element.
	* 一要素を読み取る。
	----------------------------------------------*/
	std::string ReadData()
	{
		std::string _data;
		this->Read(&_data);

		return _data;
	}

	/*----------------------------------------------
	* Read one line element.
	* 一行要素を読み取る。
	----------------------------------------------*/
	void ReadLine
	(
		std::vector<std::string>* _Store,
		size_t                    _Surplus = 10
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

	bool MoveToLine
	(
		size_t Line
	) 
	{
		bool eof = false;

		this->Ifs.seekg(0);
		size_t NewLineCount = 0;

		char buf=0;
		for (uint64_t i = 0; NewLineCount < Line;i++,this->Ifs.get(buf)) {
			if (this->Ifs.eof()) {
				this->Ifs.seekg(0);
				eof = true;
			}
			
			if (buf == '\n')
				NewLineCount++;
		}
		this->Ifs.get(buf);

		return eof;
	}
};




/*-----CSVファイル書き込み用-----*/
class Sanae::WriteCSV {


/*-----Define Variable(private)-----*/
private:


	/*----------------------------------------------
	* For file editing.
	* ファイル編集用。
	----------------------------------------------*/
	std::ofstream ofs;

	/*----------------------------------------------
	* Buffer.
	* バッファ
	----------------------------------------------*/
	std::string  Write_Buffer;


/*-----Define Functions(private)-----*/
private:


	/*----------------------------------------------
	* Insert a brake.
	* 区切りを挿入する。
	----------------------------------------------*/
	void Insert() {
		if (this->Write_Buffer.size() != 0)
		{
			char buf = *(this->Write_Buffer.end() - 1);

			if (buf != ',' && buf != '\n')
				this->Write_Buffer += ',';
		}
	}

	/*----------------------------------------------
	* Insert line brake.
	* 改行を挿入する。
	----------------------------------------------*/
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

	/*----------------------------------------------
	* Write to buffer.
	* バッファに書き込む。
	----------------------------------------------*/
	template<typename _T,class Dummy = void> void Buffer_Write(_T _Data) {
		if (NewLine(_Data))
			return;

		Insert();
		this->Write_Buffer += std::to_string(_Data);

		return;
	}
	template<class Dummy = void> void Buffer_Write(const char* _Data) {
		if (NewLine(_Data))
			return;

		Insert();
		this->Write_Buffer += _Data;

		return;
	}
	template<class Dummy = void> void Buffer_Write(std::string _Data) {
		if (NewLine(_Data))
			return;

		Insert();
		this->Write_Buffer += _Data;

		return;
	}
	
	
/*-----Define Functions(public)-----*/
public:


	/*-----Constructor-----*/
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

	/*----------------------------------------------
	* Open File.
	* ファイルを開く。
	----------------------------------------------*/
	WriteCSV& open(const char* _FileName, bool _DoInit)
	{
		if (_DoInit)
			this->ofs.open(_FileName, std::ios::out);
		else
			this->ofs.open(_FileName, std::ios::app);

		if (!this->ofs.is_open())
			throw std::runtime_error("Can't open file.");
	}

	/*----------------------------------------------
	* Close File..
	* ファイルを閉じる。
	----------------------------------------------*/
	WriteCSV& close()
	{
		ofs << this->Write_Buffer.c_str();
		this->Write_Buffer.erase(this->Write_Buffer.begin(), this->Write_Buffer.end());

		this->ofs.close();

		return *this;
	}

	/*----------------------------------------------
	* Write data to buffer.
	* データを書き込む。
	----------------------------------------------*/
	template<typename _DataType> WriteCSV& operator << (_DataType _Data)
	{
		this->Buffer_Write(_Data);

		return *this;
	}

	/*----------------------------------------------
	* Write line data to buffer.
	* データを書き込む。
	----------------------------------------------*/
	template<typename _DataType> WriteCSV& operator << (std::initializer_list<_DataType> _DataList)
	{
		for (_DataType i : _DataList)
			this->Buffer_Write(i);

		return *this;
	}

	/*----------------------------------------------
	* Write line data to buffer.
	* データを書き込む。
	----------------------------------------------*/
	template<typename _DataType> WriteCSV& operator << (std::vector<_DataType> _DataList)
	{
		for (_DataType i : _DataList)
			this->Buffer_Write(i);

		return *this;
	}

	/*----------------------------------------------
	* Write buffer data to file.
	* バッファをファイルに書き込む。
	----------------------------------------------*/
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




#endif