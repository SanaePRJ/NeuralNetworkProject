#pragma once

#include <functional>
#include <iostream>
#include <random>
#include <time.h>
#include <chrono>
#include <thread>

#include "Matrix/Matrix"




//�s��ύ������e�X�g�v���O����
/*
*
* ��:Core i5 8250U,DDR4(2400Mhz)
* ��������:
*
�v�Z��  :�����������ł������X���b�h��
1000000   :3,4,5,6�X���b�h
125000000 :8�X���b�h
343000000 :7�X���b�h
1000000000:4�X���b�h
3375000000:4�X���b�h
8000000000:4�X���b�h
*/
static void MulTest()
{
	using namespace std::chrono;

	std::default_random_engine       engine;      //�G���W��
	std::normal_distribution<double> dist(0, 1);  //����0,�W���΍�1

	size_t n = 2500;

	//n*n�s����쐬
	Sanae::Matrix<double> buf0 = std::pair<size_t, size_t>{ n,n };
	Sanae::Matrix<double> buf1 = std::pair<size_t, size_t>{ n,n };

	buf0.Setter([&engine, &dist]()->double {return dist(engine); });
	buf1.Setter([&engine, &dist]()->double {return dist(engine); });

	auto calc = [&buf0, &buf1, &engine, &dist]() {
		auto buf_time0 = system_clock::now();

		buf0* buf1;

		auto buf_time1 = system_clock::now();

		return duration_cast<milliseconds>(buf_time1 - buf_time0).count();
		};

	std::cout << n * n * n << "��v�Z����܂��B" << std::endl;

	//1�X���b�h�ł̌v�Z
	buf0.thread = 1;
	std::cout << buf0.thread << "�X���b�h�ł̌v�Z:";
	double time1 = static_cast<double>(calc());
	std::cout << time1 << "�~���b������܂����B" << std::endl;

	//�����X���b�h�ł̌v�Z
	for (size_t i = 2; i <= std::thread::hardware_concurrency(); i++) {
		buf0.thread = i;

		std::cout << buf0.thread << "�X���b�h�ł̌v�Z:";
		double time2 = static_cast<double>(calc());
		std::cout << time2 << "�~���b������܂����B->";

		std::cout << (1 - time2 / time1) * 100 << "%�������ł��܂����B" << std::endl;
	}
	system("pause");
}