#ifndef __SPARSE_H__
#define __SPARSE_H__

struct ccsmatrix
{
	double *value; //������ �������� ������� �� ��������
	int *row; //������ ������� �����
	int *colindex; //������ �������� ������ ��������
	int number_of_nonzero_elem; //����� ��������� ��������� � �������
	int size; //������ �������
};

// ��������� 2 ���������� ������� � ������� C�S (3 �������, ���������� � ����)
// ���������� C = A * B, C - � ������� C�S (3 �������, ���������� � ����)
ccsmatrix ccs_multiplicate_matrix(ccsmatrix A, ccsmatrix B);
#endif // __SPARSE_H__