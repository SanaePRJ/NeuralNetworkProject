#include <iostream>
#include <random>
#include <time.h>

#include "Matrix-for-Cpp-4.0.3/Matrix/Matrix/Matrix.hpp"
#include "NeuralNetwork.hpp"


void OR() {
	uint32_t seed = 2024;
	size_t   n    = 5000;

	Sanae::NN<double> OR(2, 3, 2, 0.03, seed);
	srand(seed);

	for (size_t i = 0; i < n; i++) {
		Sanae::Matrix<double> Q = { {static_cast<double>(rand() % 2),static_cast<double>(rand() % 2)} };
		Sanae::Matrix<double> A = { {static_cast<double>((bool)(Q[0][0] + Q[0][1])),static_cast<double>(!(bool)(Q[0][0] + Q[0][1]))} };

		OR.Learn(Q, A);
	}

	double accuracy = 0;
	for (size_t i = 0; i < n; i++) {
		Sanae::Matrix<double> Q = { {static_cast<double>(rand() % 2),static_cast<double>(rand() % 2)} };
		Sanae::Matrix<double> A = OR.Predict(Q);

		if (static_cast<bool>(Q[0][0] + Q[0][1]) && A[0][0] > A[0][1]) {
			accuracy += 1;
		}
		else if (!static_cast<bool>(Q[0][0] + Q[0][1]) && A[0][0] < A[0][1]) {
			accuracy += 1;
		}
	}

	std::cout << "OR正答率:" << accuracy / static_cast<double>(n) * 100 << "%";
}
void AND() {
	uint32_t seed = 2024;
	size_t   n    = 5000;

	Sanae::NN<double> AND(2, 4, 2, 0.03, seed);
	srand(seed);

	for (size_t i = 0; i < n; i++) {
		Sanae::Matrix<double> Q = { {static_cast<double>(rand() % 2),static_cast<double>(rand() % 2)} };
		Sanae::Matrix<double> A = { {static_cast<double>((bool)(Q[0][0] == Q[0][1])),static_cast<double>((bool)(Q[0][0] != Q[0][1]))} };

		AND.Learn(Q, A);
	}

	double accuracy = 0;
	for (size_t i = 0; i < n; i++) {
		Sanae::Matrix<double> Q = { {static_cast<double>(rand() % 2),static_cast<double>(rand() % 2)} };
		Sanae::Matrix<double> A = AND.Predict(Q);

		if (static_cast<bool>(Q[0][0] == Q[0][1]) && A[0][0] > A[0][1]) {
			accuracy += 1;
		}
		else if (static_cast<bool>(Q[0][0] != Q[0][1]) && A[0][0] < A[0][1]) {
			accuracy += 1;
		}
	}

	std::cout << "AND正答率:" << accuracy / static_cast<double>(n) * 100 << "%";
}

int main() {
	AND();
	OR();
}