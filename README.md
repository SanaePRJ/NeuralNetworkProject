# NeuralNetwork-SanaeProject
���̃��|�W�g���ł�NeuralNetwork�������N���X���쐬���Ă���B  
"�[��������DeepLearning"���Q�l��Tanh���C���[�⏑����Ă��Ȃ��������a�덷�̃~�j�o�b�`�����Ή����C���[�Ȃǂ�ǉ����Ă���B  
  
�s��̉��Z�ł͓�Project�ŊJ�����Ă���Matrix�^���g�p���Ă���B  
mnist�f�[�^�͊e���Ń_�E�����[�h���Ă��������B

# �T���v���R�[�h
## mnist�f�[�^�ɂ�鐔���̎���
mnist�f�[�^�Z�b�g���g�p���摜�̐��������ʂ���B

	void mnist() {
		srand((uint32_t)time(NULL));

		Sanae::ReadCSV    mnist("mnist_train.csv");
		//���̓m�[�h��:784,���ԑw�m�[�h��:50,�o�͑w�m�[�h��:10,�w�K��:30%,���ԑw:2�w
		Sanae::NN<double> Machine(784, 200, 10, 0.3, rand(),2);


		//mnist�f�[�^��28*28
		const uint64_t MnistSize = 784;
		//batch�T�C�Y
		uint64_t       batch     = 50;

		//�w�K��
		uint64_t       LearnCount   = 500;
		//�\����
		uint64_t       PredictCount = 500;


		//Mnist�f�[�^�̓����Ɖ摜�f�[�^��CSV�t�@�C������ǂݎ��B
		auto get_mnist = [&mnist, &Machine, &MnistSize]() -> std::pair<double, std::vector<double>> {
			//����
			double Answer = std::stoi(mnist.ReadData());

			//�摜�f�[�^��ǂݍ���
			std::vector<double> Question(MnistSize, 0);
			for (uint64_t i = 0; i < MnistSize; i++)
				Question[i] = (std::stod(mnist.ReadData()) / 255 * 0.99 + 0.0001);   //���K��

			//����,�摜�f�[�^��pair�ő���B
			return std::pair<double, std::vector<double>>{Answer, Question};
			};

		//�w�K������B
		auto Learn = [&get_mnist, &mnist, &Machine, &batch, &MnistSize]() {
			Sanae::Matrix<double> A = std::pair<size_t, size_t>{ batch,0 };   //���
			Sanae::Matrix<double> Q = std::pair<size_t, size_t>{ batch,0 };   //����

			//batch�T�C�Y�������ǂݎ��B
			for (size_t i = 0; i < batch; i++) {
				std::pair<double, std::vector<double>> MnistData = get_mnist();   //Mnist�f�[�^�ǂݎ��

				//������z��ɂ���B3->{0,0,0,1,0,0,0,0,0,0}
				std::vector<double> buf_A(10, 0);
				buf_A[static_cast<uint64_t>(MnistData.first)] = 1;

				A[i] = { buf_A };
				Q[i] = { MnistData.second };
			}

			//�w�K->�Ԃ�l�Ƃ��đ����֐��̕Ԃ�l��Ԃ��B
			return Machine.Learn(Q, A);
			};

		//�\��
		auto predict = [&mnist, &Machine, &get_mnist, &MnistSize]() {
			//mnist�f�[�^�ǂݍ���
			std::pair<double, std::vector<double>> MnistData = get_mnist();

			//�摜�f�[�^���i�[
			Sanae::Matrix<double> Q = std::pair<size_t, size_t>{ 1,0 };
			Q[0] = MnistData.second;

			//�\�����ʂ�A�Ɋi�[
			Sanae::Matrix<double> A = Machine.Predict(Q);

			//�ő�l�������ł��邽�ߍő�l�̂���index��\�����ʂƂ���B
			std::vector<double>::iterator iter = std::max_element(A[0].begin(), A[0].end());
			size_t index = std::distance(A[0].begin(), iter);

			//�����ł������ꍇ1,�s�����ł������ꍇ��0��Ԃ��B
			if (index != MnistData.first)
				return 0;
			else
				return 1;
			};

		//�w�K������
		for (size_t i = 0; i < LearnCount; i++) {
			if (i % (LearnCount / 50) == 0)
				std::cout << i / static_cast<double>(LearnCount) * 100 << "%...�w�K����" << std::endl;

			Learn();
		}

		//test�f�[�^���J��
		mnist.close();
		mnist.open("mnist_test.csv");

		//�\��
		double accuracy = 0;
		for (size_t i = 0; i < PredictCount; i++)
			accuracy += predict();

		std::cout << "������:" << accuracy / static_cast<double>(PredictCount) * 100 << "%";
	}
## �����f�[�^�̐���
mnist����w�K�����摜���琔���̉摜�𐶐�����B

	void makemnist() {
		srand((uint32_t)time(NULL));

		Sanae::ReadCSV    mnist("mnist_train.csv");
		Sanae::NN<double, Sanae::Layer_ReLU<double>, Sanae::Layer_IdentityWithLoss<double>> Machine(10, 200, 784, 0.3, rand(), 4);

		//mnist�f�[�^��28*28
		const uint64_t MnistSize = 784;
		//batch�T�C�Y
		uint64_t       batch = 50;

		//�w�K��
		uint64_t       LearnCount = 500;
		//�\����
		uint64_t       PredictCount = 500;


		//Mnist�f�[�^�̓����Ɖ摜�f�[�^��CSV�t�@�C������ǂݎ��B
		auto get_mnist = [&mnist, &Machine, &MnistSize]() -> std::pair<double, std::vector<double>> {
			//����
			double Answer = std::stoi(mnist.ReadData());

			//�摜�f�[�^��ǂݍ���
			std::vector<double> Question(MnistSize, 0);
			for (uint64_t i = 0; i < MnistSize; i++)
				Question[i] = (std::stod(mnist.ReadData()) / 255 * 0.99 + 0.0001);   //���K��

			//����,�摜�f�[�^��pair�ő���B
			return std::pair<double, std::vector<double>>{Answer, Question};
			};

		//�w�K������B
		auto Learn = [&get_mnist, &mnist, &Machine, &batch, &MnistSize]() {
			Sanae::Matrix<double> Q = std::pair<size_t, size_t>{ batch,0 };   //���
			Sanae::Matrix<double> A = std::pair<size_t, size_t>{ batch,0 };   //����

			//batch�T�C�Y�������ǂݎ��B
			for (size_t i = 0; i < batch; i++) {
				std::pair<double, std::vector<double>> MnistData = get_mnist();   //Mnist�f�[�^�ǂݎ��

				//������z��ɂ���B3->{0,0,0,1,0,0,0,0,0,0}
				std::vector<double> buf_A(10, 0);
				buf_A[static_cast<uint64_t>(MnistData.first)] = 1;

				Q[i] = { buf_A };
				A[i] = { MnistData.second };
			}

			//�w�K->�Ԃ�l�Ƃ��đ����֐��̕Ԃ�l��Ԃ��B
			return Machine.Learn(Q, A);
			};

		//�\��
		auto predict = [&mnist, &Machine, &get_mnist, &MnistSize]() {
			Sanae::Matrix<double> Q = std::pair<size_t, size_t>{ 1,0 };   //���

			std::pair<double, std::vector<double>> MnistData = get_mnist();   //Mnist�f�[�^�ǂݎ��

			//����z��ɂ���B3->{0,0,0,1,0,0,0,0,0,0}
			std::vector<double> buf_Q(10, 0);
			buf_Q[static_cast<uint64_t>(MnistData.first)] = 1;
			Q[0] = { buf_Q };

			system("pause");
			std::cout << MnistData.first << std::endl;

			Sanae::Matrix<double> plot = Machine.Predict(Q);
			for (size_t i = 0; i < plot[0].size(); i++)
				plot[0][i] = (plot[0][i] - 0.0001) / 0.99 * 255;

			Sanae::PLOT << plot[0];

			system("pause");
			};

		//�w�K������
		for (size_t i = 0; i < LearnCount; i++) {
			if (i % (LearnCount / 50) == 0)
				std::cout << i / static_cast<double>(LearnCount) * 100 << "%...�w�K����" << std::endl;

			Learn();
		}

		//test�f�[�^���J��
		mnist.close();
		mnist.open("mnist_test.csv");

		//�\��
		for (size_t i = 0; i < PredictCount; i++)
			predict();
	}