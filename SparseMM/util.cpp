#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include "mkl_spblas.h"
#include "util.h"

using namespace std;

const double MAX_VAL = 100.0;

// ���� - ��� �� ��������������� ��������� ��������� �����
bool isSrandCalled = false;

// ������� ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// �������� ������ ��� ���� Value, Col � RowIndex
// ���������� ����� �������� mtx
void InitializeMatrix(int N, int NZ, ccsmatrix &mtx)
{
	mtx.size = N;
	mtx.number_of_nonzero_elem = NZ;
  mtx.value = new double[NZ];
  mtx.row      = new int[NZ];
  mtx.colindex = new int[N + 1];
}

// ����������� ������, ���������� ��� ���� mtx
void FreeMatrix(ccsmatrix &mtx)
{
  delete [] mtx.value;
  delete [] mtx.colindex;
  delete [] mtx.row;
}

// ������� ����� imtx � omtx, ������� ������ ��� ���� Value, Col � RowIndex
void CopyMatrix(ccsmatrix imtx, ccsmatrix &omtx)
{
  // ������������� �������������� �������
	int N  = imtx.size;
	int NZ = imtx.number_of_nonzero_elem;
  InitializeMatrix(N, NZ, omtx);
  // �����������
  memcpy(omtx.value   , imtx.value   , NZ * sizeof(double));
  memcpy(omtx.colindex     , imtx.colindex     , (N + 1) * sizeof(int));
  memcpy(omtx.row, imtx.row, (NZ) * sizeof(int));
}

// ��������� 2 ���������� ������� � ������� CCS (3 �������, ���������� � ����)
// ���������� max|Cij|, ��� C = A - B
// ���������� ������� ���������� ��������: 0 - ��, 1 - �� ��������� ������� (N)
int CompareMatrix(ccsmatrix mtx1, ccsmatrix mtx2, double &diff)
{
	int N = mtx1.size;
  int i, j;
  int i1, i2;
  double **p;

  // ��������� �� ������?
  if (mtx1.size != mtx2.size)
    return 1; // �� ������ ������

  N = mtx1.size;
  // �������� ������� �������
  p = new double*[N];
  for (i = 0; i < N; i++) 
  {
    p[i] = new double[N];  
    for (j = 0; j < N; j++)
      p[i][j] = 0.0;
  }

  // ����������� ������ ������� � �������
  for (i = 0; i < N; i++)
  {
	  i1 = mtx1.colindex[i];
    i2 = mtx1.colindex[i + 1] - 1;
    for (j = i1; j <= i2; j++)
		p[mtx1.row[j]][i] = mtx1.value[j];
  }

  // ��������� ������ ������� �� �������
  for (i = 0; i < N; i++)
  {
	  i1 = mtx2.colindex[i];
    i2 = mtx2.colindex[i + 1] - 1;
    for (j = i1; j <= i2; j++)
		p[mtx2.row[j]][i] -= mtx2.value[j];
  }

  // ���������� ������������� ����������
  double max = 0.0;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (fabs(p[i][j]) > max)
        max = fabs(p[i][j]);
  diff = max;

  // �������� ������� �������
  for (i = 0; i < N; i++)
    delete[] p[i];
  delete[] p;

  return 0; // ������ ������
}

// ��������� 2 ���������� ������� � ������� CRS (3 �������, ���������� � ����)
// ���������� max|Cij|, ��� C = A - B
// ��� ��������� ���������� ������� �� MKL
// ���������� ������� ���������� ��������: 0 - ��, 1 - �� ��������� ������� (N)
int SparseDiff(ccsmatrix A, ccsmatrix B, double &diff)
{
	if (A.size != B.size)
    return 1;

  int n = A.size;

  // ����� ��������� C = A - B, ��������� MKL
  // ��������� ������ � ����� MKL
  double *c = 0; // ��������
  int *jc = 0;   // ������ �������� (��������� � �������)
  int *ic;   // ������� ������ ��������� ����� (��������� � �������)
  
  // �������� ��������� ��� ������ ������� MKL
  // ��������������� ������� A � B � �������
  int i, j;
  for (i = 0; i < A.number_of_nonzero_elem; i++)
	  A.row[i]++;
  for (i = 0; i < B.number_of_nonzero_elem; i++)
	  B.row[i]++;
  for (j = 0; j <= A.size; j++)
  {
	  A.colindex[j]++;
	  B.colindex[j]++;
  }

  // ������������ �������, ����������� C = A + beta*op(B)
  char trans = 'N'; // ������� � ���, op(B) = B - �� ����� ��������������� B

// ������ ��������, �������� �� ��, ��� ����� ���������� ������
// request = 0: ������ ��� �������������� ������� �.�. �������� �������
// ���� �� �� �����, ������� ������ ���������� ��� �������� ����������,
// ����������:
// 1) �������� ������ ��� ������� �������� ����� ic: "���-�� �����+1" ���������;
// 2) ������� ������� � ���������� request = 1 - � ������� ic ����� �������� 
//                                                         ��������� �������
// 3) �������� ������ ��� �������� c � jc 
//    (���-�� ��������� = ic[���-�� �����]-1)
// 4) ������� ������� � ���������� request = 2
  int request;

// ��� ���� ������������� ������: ���� ����������� ���������, ����� �� 
// ������������� ������� A, B � C. � ��� ��������������, ��� ��� �������
// �����������, �������������, �������� ������� "No-No-Yes", �������
// ������������� ������ ��������, ����� ����� ����� �� 1 �� 7 ������������
  int sort = 8;

// beta = -1 -> C = A + (-1.0) * B;
  double beta = -1.0;

// ���������� ��������� ���������.
// ������������ ������ ���� request = 0
  int nzmax = -1;

// ��������� ����������
  int info;

// ������� ������ ��� ������� � ������� C
  ic = new int[n + 1];
// ��������� ���������� ��������� ��������� � ������� C
  request = 1;
  mkl_dcsradd(&trans, &request, &sort, &n, &n, A.value, A.row, A.colindex, 
	  &beta, B.value, B.row, B.colindex, c, jc, ic, &nzmax, &info);
  int nzc = ic[n] - 1;
  c = new double[nzc];
  jc = new int[nzc];
// ��������� C = A - B
  request = 2;
  mkl_dcsradd(&trans, &request, &sort, &n, &n, A.value, A.row, A.colindex, 
	  &beta, B.value, B.row, B.colindex, c, jc, ic, &nzmax, &info);
// ��������� max|Cij|
  diff = 0.0;
  for (i = 0; i < nzc; i++)
  {
    double var = fabs(c[i]);
    if (var > diff)
      diff = var;
  }
// �������� � ����������� ���� ������� A � B
  for (i = 0; i < A.number_of_nonzero_elem; i++)
	  A.row[i]--;
  for (i = 0; i < B.number_of_nonzero_elem; i++)
	  B.row[i]--;
  for (j = 0; j <= n; j++)
  {
	  A.colindex[j]--;
	  B.colindex[j]--;
  }
// ��������� ������
  delete [] c;
  delete [] ic;
  delete [] jc;

  return 0;
}

// ��������� 2 ���������� ������� � ������� CRS (3 �������, ���������� � ����)
// ���������� C = A * B, C - � ������� CRS (3 �������, ���������� � ����)
//                       ������ ��� C � ������ ��������� �� ����������
// ���������� ������� ���������� ��������: 0 - ��, 1 - �� ��������� ������� (N)
// ���������� ����� ������
int SparseMKLMult(ccsmatrix A, ccsmatrix B, ccsmatrix &C, double &time)
{
	if (A.size != B.size)
    return 1;

  int n = A.size;

  // �������� ��������� ��� ������ ������� MKL
  // ��������������� ������� A � B � �������
  int i, j;
  for (i = 0; i < A.number_of_nonzero_elem; i++)
	  A.row[i]++;
  for (i = 0; i < B.number_of_nonzero_elem; i++)
    B.row[i]++;
  for (j = 0; j <= n; j++)
  {
	  A.colindex[j]++;
	  B.colindex[j]++;
  }

  // ������������ �������, ����������� C = op(A) * B
  char trans = 'N'; // ������� � ���, op(A) = A - �� ����� ��������������� A

// ������ ��������, �������� �� ��, ��� ����� ���������� ������
// request = 0: ������ ��� �������������� ������� �.�. �������� �������
// ���� �� �� �����, ������� ������ ���������� ��� �������� ����������,
// ����������:
// 1) �������� ������ ��� ������� �������� ����� ic: "���-�� �����+1" ���������;
// 2) ������� ������� � ���������� request = 1 - � ������� ic ����� �������� 
//                                                         ��������� �������
// 3) �������� ������ ��� �������� c � jc 
//    (���-�� ��������� = ic[���-�� �����]-1)
// 4) ������� ������� � ���������� request = 2
  int request;

// ��� ���� ������������� ������: ���� ����������� ���������, ����� �� 
// ������������� ������� A, B � C. � ��� ��������������, ��� ��� �������
// �����������, �������������, �������� ������� "No-No-Yes", �������
// ������������� ������ ��������, ����� ����� ����� �� 1 �� 7 ������������
  int sort = 8;

// ���������� ��������� ���������.
// ������������ ������ ���� request = 0
  int nzmax = -1;

// ��������� ����������
  int info;

  clock_t start = clock();

// ������� ������ ��� ������� � ������� C
  C.colindex = new int[n + 1];
// ��������� ���������� ��������� ��������� � ������� C
  request = 1;
  C.value = 0;
  C.row = 0;
  mkl_dcsrmultcsr(&trans, &request, &sort, &n, &n, &n, A.value, A.row, 
	  A.colindex, B.value, B.row, B.colindex, C.value, C.row,
	  C.colindex, &nzmax, &info);

  int nzc = C.colindex[n] - 1;
  C.value = new double[nzc];
  C.row = new int[nzc];
// ��������� C = A * B
  request = 2;

  mkl_dcsrmultcsr(&trans, &request, &sort, &n, &n, &n, A.value, A.row, 
	  A.colindex, B.value, B.row, B.colindex, C.value, C.row,
	  C.colindex, &nzmax, &info);
  C.size = n;
  C.number_of_nonzero_elem = nzc;

  clock_t finish = clock();

  // �������� � ����������� ���� ������� A, B � �
  for (i = 0; i < A.number_of_nonzero_elem; i++)
    A.row[i]--;
  for (i = 0; i < B.number_of_nonzero_elem; i++)
    B.row[i]--;
  for (i = 0; i < C.number_of_nonzero_elem; i++)
    C.row[i]--;
  for (j = 0; j <= n; j++)
  {
	  A.colindex[j]--;
	  B.colindex[j]--;
	  C.colindex[j]--;
  }

  time = double(finish - start) / double(CLOCKS_PER_SEC);

  return 0;
}

double next()
{
  return ((double)rand() / (double)RAND_MAX);
}

// ���������� ���������� ������� � ������� CRS (3 �������, ���������� � ����)
// � ������ ������ cntInRow ��������� ���������
void GenerateRegularCCS(int seed, int N, int cntInRow, ccsmatrix& mtx)
{
  int i, j, k, f, tmp, notNull, c;

  if (!isSrandCalled)
  {
    srand(seed);
    isSrandCalled = true;
  }

  notNull = cntInRow * N;
  InitializeMatrix(N, notNull, mtx);

  for(i = 0; i < N; i++)
  {
    // ��������� ������ ����� � ������� i
    for(j = 0; j < cntInRow; j++)
    {
      do
      {
        mtx.row[i * cntInRow + j] = rand() % N;
        f = 0;
        for (k = 0; k < j; k++)
          if (mtx.row[i * cntInRow + j] == mtx.row[i * cntInRow + k])
            f = 1;
      } while (f == 1);
    }
    // ��������� ������ ������� � ������ i
    for (j = 0; j < cntInRow - 1; j++)
      for (k = 0; k < cntInRow - 1; k++)
        if (mtx.row[i * cntInRow + k] > mtx.row[i * cntInRow + k + 1])
        {
          tmp = mtx.row[i * cntInRow + k];
          mtx.row[i * cntInRow + k] = mtx.row[i * cntInRow + k + 1];
          mtx.row[i * cntInRow + k + 1] = tmp;
        }
  }

  // ��������� ������ ��������
  for (i = 0; i < cntInRow * N; i++)
    mtx.value[i] = next() * MAX_VAL;

  // ��������� ������ �������� �����
  c = 0;
  for (i = 0; i <= N; i++)
  {
	  mtx.colindex[i] = c;
    c += cntInRow;
  }
}

// ���������� ���������� ������� � ������� CRS (3 �������, ���������� � ����)
// ����� ��������� ��������� � ������� ������ �� 1 �� cntInRow
// ����� ����� - ���������� ��������
void GenerateSpecialCCS(int seed, int N, int cntInRow, ccsmatrix& mtx)
{
  if (!isSrandCalled)
  {
    srand(seed);
    isSrandCalled = true;
  }

  double end = pow((double)cntInRow, 1.0 / 3.0);
  double step = end / N;

  vector<int>* columns = new vector<int>[N];
  int NZ = 0;

  for (int i = 0; i < N; i++)
  {
    int rowNZ = int(pow((double(i + 1) * step), 3) + 1);
    NZ += rowNZ;
    int num1 = (rowNZ - 1) / 2;
    int num2 = rowNZ - 1 - num1;

    if (rowNZ != 0)
    {
      if (i < num1)
      {
        num2 += num1 - i;
        num1 = i;
        for(int j = 0; j < i; j++)
          columns[i].push_back(j);
        columns[i].push_back(i);
        for(int j = 0; j < num2; j++)
          columns[i].push_back(i + 1 + j);

      }
      else
      {
        if (N - i - 1 < num2)
        {
          num1 += num2 - (N - 1 - i);
          num2 = N - i - 1;
        }
        for (int j = 0; j < num1; j++)
          columns[i].push_back(i - num1 + j);
        columns[i].push_back(i);
        for (int j = 0; j < num2; j++)
          columns[i].push_back(i + j + 1);
      }
    }
  }

  InitializeMatrix(N, NZ, mtx);

  int count = 0;
  int sum = 0;
  for (int i = 0; i < N; i++)
  {
	  mtx.colindex[i] = sum;
    sum += columns[i].size();
    for (unsigned int j = 0; j < columns[i].size(); j++)
    {
      mtx.row[count] = columns[i][j];
      mtx.value[count] = next();
      count++;
    }
  }
  mtx.colindex[N] = sum;

  delete [] columns;
}
