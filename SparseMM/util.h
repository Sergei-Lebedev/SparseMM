#ifndef __UTIL_H__
#define __UTIL_H__

#include "sparse.h"

// ������� ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// �������� ������ ��� ���� value, row � colindex
// ���������� ����� �������� mtx
void InitializeMatrix(int N, int NZ, ccsmatrix &mtx);

// ����������� ������ ��� ����� mtx
void FreeMatrix(ccsmatrix &mtx);

// ������� ����� imtx � omtx, ������� ������ ��� ���� value, row � colindex
void CopyMatrix(ccsmatrix imtx, ccsmatrix &omtx);

// ��������� 2 ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// ���������� max|Cij|, ��� C = A - B
// ���������� ������� ���������� ��������: 0 - ��, 1 - �� ��������� ������� (N)
int CompareMatrix(ccsmatrix mtx1, ccsmatrix mtx2, double &diff);

// ��������� 2 ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// ���������� max|Cij|, ��� C = A - B
// ��� ��������� ���������� ������� �� MKL
// ���������� ������� ���������� ��������: 0 - ��, 1 - �� ��������� ������� (N)
int SparseDiff(ccsmatrix A, ccsmatrix B, double &diff);

// ��������� 2 ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// ���������� C = A * B, C - � ������� CCS (3 �������, ���������� � ����)
//                       ������ ��� C � ������ ��������� �� ����������
// ���������� ������� ���������� ��������: 0 - ��, 1 - �� ��������� ������� (N)
// ���������� ����� ������
int SparseMKLMult(ccsmatrix A, ccsmatrix B, ccsmatrix &C, double &time);

// ���������� ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// � ������ ������� cntInRow ��������� ���������
void GenerateRegularCCS(int seed, int N, int cntInRow, ccsmatrix& mtx);

// ���������� ���������� ������� � ������� C�S (3 �������, ���������� � ����)
// ����� ��������� ��������� � ������� ������ �� 1 �� cntInRow
// ����� ����� - ���������� ��������
void GenerateSpecialCCS(int seed, int N, int cntInRow, ccsmatrix& mtx);

#endif // __UTIL_H__