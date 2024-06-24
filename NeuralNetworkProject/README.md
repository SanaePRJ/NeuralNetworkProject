# Matrix�^�̏Љ�
Matrix�^�́A�s���\�����邽�߂̃e���v���[�g�N���X�ł���A�s��̍쐬�A����A�v�Z���s�����߂̋@�\��񋟂��܂��B�ȉ���Matrix�^�̎�ȓ����Ƌ@�\��������܂��B  

WebSite:	http://sanae.starfree.jp  
Author:		SanaeProject

- ���̐�������`ChatGpt(Gpt3.5)`�ɂ�萶�����܂����B

## ����
- **�e���v���[�g�N���X**: �v�f�̌^(`ty`)�Ɉˑ����Ē�`����邽�߁A�C�ӂ̃f�[�^�^�ōs��������܂��B
- **���I�T�C�Y**: �s���Ɨ񐔂����I�Ɍ��肳��A�T�C�Y�̕ύX�⏉�������\�ł��B
- **�s�񉉎Z�̃T�|�[�g**: ���Z�A���Z�A�X�J���[�{�A�s��ςȂǂ̉��Z����������Ă��܂��B
- **���x�ȍs��v�Z**: �t�s��̌v�Z��LU�����Ɋ�Â��s�񎮂̌v�Z�ȂǁA���x�Ȑ��w�I���삪�T�|�[�g����Ă��܂��B
- **�G���[�n���h�����O**: �s���ȍs�񑀍�ɑ΂����O�������g�ݍ��܂�Ă��܂��B

## ��ȃt�@�C���\��
- **Matrix.h**: Matrix�^�̒�`���܂܂�Ă���A�N���X�����o�[�A�e���v���[�g�A�s�񉉎Z�A���x�ȍs��v�Z�Ȃǂ̎������L�q����Ă��܂��B
- **MatrixUtil.hpp**: �s�񑀍�Ɋւ��郆�[�e�B���e�B�֐����錾����A�s��̊�{�I�ȑ�����T�|�[�g���܂��B
- **MatrixCalc.hpp**: �s�񉉎Z�Ɋւ���֐����錾����A���Z�A���Z�A�X�J���[�{�A�A�_�}�[���ρA�s��ςȂǂ̉��Z���`���܂��B
- **MatrixAdvCalc.hpp**: ���x�ȍs��v�Z�Ɋւ���֐����錾����A�t�s��̌v�Z�ALU�����A�s�񎮂̌v�Z�Ȃǂ��������܂��B

## ���p���@
Matrix�^�͈ȉ��̂悤�Ɏg�p�ł��܂�:
- �s��̍쐬�Ə������A�T�C�Y�̕ύX
- �s�񉉎Z�̎��s�i���Z�A���Z�A�s��ςȂǁj
- ���x�ȍs��v�Z�i�t�s��̌v�Z�A�s�񎮂̎擾�Ȃǁj

## ��O����
Matrix�^�ł͕s���ȍs�񑀍�ɑ΂����O�Ƃ���`InvalidMatrix`�N���X���g�p����A�G���[���b�Z�[�W��񋟂��܂��B

## �g�p��
```cpp
#include "Matrix.h"

using namespace Sanae;

int main() {
    try {
        // �R���X�g���N�^�ɂ�鏉����
        Sanae::Matrix<double> mat1{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
        std::cout << "�s�� mat1:" << std::endl << mat1 << std::endl;

        Sanae::Matrix<double> mat2(std::make_pair(2, 3));
        std::cout << "�s�� mat2 (2x3):" << std::endl << mat2 << std::endl;

        Sanae::Matrix<double> mat3(mat1);
        std::cout << "�s�� mat3 (mat1�̃R�s�[):" << std::endl << mat3 << std::endl;

        // ���Z�q�ɂ�鑀��
        mat1 += mat1;
        std::cout << "�s�� mat1 ���Z�� (mat1 += mat1):" << std::endl << mat1 << std::endl;

        mat1 -= mat3;
        std::cout << "�s�� mat1 ���Z�� (mat1 -= mat3):" << std::endl << mat1 << std::endl;

        mat1 *= 2;
        std::cout << "�s�� mat1 �X�J���[�{�� (mat1 *= 2):" << std::endl << mat1 << std::endl;

        Sanae::Matrix<double> mat4 = mat1 * mat3;
        std::cout << "�s�� mat4 (mat1 * mat3):" << std::endl << mat4 << std::endl;

        // �s��̓]�u
        Sanae::Matrix<double> mat5 = mat1.Transpose();
        std::cout << "�s�� mat5 (mat1�̓]�u):" << std::endl << mat5 << std::endl;

        // �s��̃T�C�Y�ύX
        mat2.ReSize(std::make_pair(3, 3));
        std::cout << "�s�� mat2 �T�C�Y�ύX�� (3x3):" << std::endl << mat2 << std::endl;

        // �P�ʍs��ƃ[���s��
        Sanae::Matrix<double> mat6 = Sanae::Matrix<double>::Identity(3);
        std::cout << "�P�ʍs�� mat6 (3x3):" << std::endl << mat6 << std::endl;

        Sanae::Matrix<double> mat7 = Sanae::Matrix<double>::Zero(3);
        std::cout << "�[���s�� mat7 (3x3):" << std::endl << mat7 << std::endl;

        // �s�񎮂Ƌt�s��
        double det = mat1.Det();
        std::cout << "�s�� mat1 �̍s��: " << det << std::endl;

        Sanae::Matrix<double> mat8 = mat1.Inverse();
        std::cout << "�s�� mat8 (mat1�̋t�s��):" << std::endl << mat8 << std::endl;

        // �L���X�g
        Sanae::Matrix<float> matFloat = mat1;
        std::cout << "�s�� matFloat (mat1����L���X�g):" << std::endl << matFloat << std::endl;

        // �A�N�Z�X
        std::cout << "�v�f mat1[1][1]: " << mat1[1][1] << std::endl;

        // ���n
        Sanae::Matrix<double> mat9{ {10, 11, 12}, {13, 14, 15}, {16, 17, 18} };
        std::cout << "�s�� mat9 (���n�O):" << std::endl << mat9 << std::endl;

        mat9 << mat1;
        std::cout << "�s�� mat9 (mat9 << mat1):" << std::endl << mat9 << std::endl;

        // ��r
        bool isEqual = (mat1 == mat3);
        std::cout << "��r (mat1 == mat3): " << (isEqual ? "true" : "false") << std::endl;

        bool isNotEqual = (mat1 != mat3);
        std::cout << "��r (mat1 != mat3): " << (isNotEqual ? "true" : "false") << std::endl;

    }
    catch (InvalidMatrix& e) {
        std::cerr << "�s��G���[: " << e.what() << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "�G���[: " << e.what() << std::endl;
    }

    return 0;
}