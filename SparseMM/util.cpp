#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include "mkl_spblas.h"
#include "util.h"

using namespace std;

const double MAX_VAL = 100.0;

// Флаг - был ли инициализирован генератор случайных чисел
bool isSrandCalled = false;

// Создает квадратную матрицу в формате CCS (3 массива, индексация с нуля)
// Выделяет память под поля Value, Col и RowIndex
// Возвращает через параметр mtx
void InitializeMatrix(int N, int NZ, ccsmatrix &mtx)
{
	mtx.size = N;
	mtx.number_of_nonzero_elem = NZ;
  mtx.value = new double[NZ];
  mtx.row      = new int[NZ];
  mtx.colindex = new int[N + 1];
}

// Освобождает память, выделенную под поля mtx
void FreeMatrix(ccsmatrix &mtx)
{
  delete [] mtx.value;
  delete [] mtx.colindex;
  delete [] mtx.row;
}

// Создает копию imtx в omtx, выделяя память под поля Value, Col и RowIndex
void CopyMatrix(ccsmatrix imtx, ccsmatrix &omtx)
{
  // Инициализация результирующей матрицы
	int N  = imtx.size;
	int NZ = imtx.number_of_nonzero_elem;
  InitializeMatrix(N, NZ, omtx);
  // Копирование
  memcpy(omtx.value   , imtx.value   , NZ * sizeof(double));
  memcpy(omtx.colindex     , imtx.colindex     , (N + 1) * sizeof(int));
  memcpy(omtx.row, imtx.row, (NZ) * sizeof(int));
}

// Принимает 2 квадратных матрицы в формате CCS (3 массива, индексация с нуля)
// Возвращает max|Cij|, где C = A - B
// Возвращает признак успешности операции: 0 - ОК, 1 - не совпадают размеры (N)
int CompareMatrix(ccsmatrix mtx1, ccsmatrix mtx2, double &diff)
{
	int N = mtx1.size;
  int i, j;
  int i1, i2;
  double **p;

  // Совпадает ли размер?
  if (mtx1.size != mtx2.size)
    return 1; // Не совпал размер

  N = mtx1.size;
  // Создание плотной матрицы
  p = new double*[N];
  for (i = 0; i < N; i++) 
  {
    p[i] = new double[N];  
    for (j = 0; j < N; j++)
      p[i][j] = 0.0;
  }

  // Копирование первой матрицы в плотную
  for (i = 0; i < N; i++)
  {
	  i1 = mtx1.colindex[i];
    i2 = mtx1.colindex[i + 1] - 1;
    for (j = i1; j <= i2; j++)
		p[mtx1.row[j]][i] = mtx1.value[j];
  }

  // Вычитание второй матрицы из плотной
  for (i = 0; i < N; i++)
  {
	  i1 = mtx2.colindex[i];
    i2 = mtx2.colindex[i + 1] - 1;
    for (j = i1; j <= i2; j++)
		p[mtx2.row[j]][i] -= mtx2.value[j];
  }

  // Вычисление максимального отклонения
  double max = 0.0;
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      if (fabs(p[i][j]) > max)
        max = fabs(p[i][j]);
  diff = max;

  // Удаление плотной матрицы
  for (i = 0; i < N; i++)
    delete[] p[i];
  delete[] p;

  return 0; // Совпал размер
}

// Принимает 2 квадратных матрицы в формате CRS (3 массива, индексация с нуля)
// Возвращает max|Cij|, где C = A - B
// Для сравнения использует функцию из MKL
// Возвращает признак успешности операции: 0 - ОК, 1 - не совпадают размеры (N)
int SparseDiff(ccsmatrix A, ccsmatrix B, double &diff)
{
	if (A.size != B.size)
    return 1;

  int n = A.size;

  // Будем вычислять C = A - B, используя MKL
  // Структуры данных в стиле MKL
  double *c = 0; // Значения
  int *jc = 0;   // Номера столбцов (нумерация с единицы)
  int *ic;   // Индексы первых элементов строк (нумерация с единицы)
  
  // Настроим параметры для вызова функции MKL
  // Переиндексируем матрицы A и B с единицы
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

  // Используется функция, вычисляющая C = A + beta*op(B)
  char trans = 'N'; // говорит о том, op(B) = B - не нужно транспонировать B

// Хитрый параметр, влияющий на то, как будет выделяться память
// request = 0: память для результирующей матрицы д.б. выделена заранее
// Если мы не знаем, сколько памяти необходимо для зранения результата,
// необходимо:
// 1) выделить память для массива индексов строк ic: "Кол-во строк+1" элементов;
// 2) вызвать функцию с параметром request = 1 - в массиве ic будет заполнен 
//                                                         последний элемент
// 3) выделить память для массивов c и jc 
//    (кол-во элементов = ic[Кол-во строк]-1)
// 4) вызвать функцию с параметром request = 2
  int request;

// Еще один нетривиальный момент: есть возможность настроить, нужно ли 
// упорядочивать матрицы A, B и C. У нас предполагается, что все матрицы
// упорядочены, следовательно, выбираем вариант "No-No-Yes", который
// соответствует любому значению, кроме целых чисел от 1 до 7 включительно
  int sort = 8;

// beta = -1 -> C = A + (-1.0) * B;
  double beta = -1.0;

// Количество ненулевых элементов.
// Используется только если request = 0
  int nzmax = -1;

// Служебная информация
  int info;

// Выделим память для индекса в матрице C
  ic = new int[n + 1];
// Сосчитаем количество ненулевых элементов в матрице C
  request = 1;
  mkl_dcsradd(&trans, &request, &sort, &n, &n, A.value, A.row, A.colindex, 
	  &beta, B.value, B.row, B.colindex, c, jc, ic, &nzmax, &info);
  int nzc = ic[n] - 1;
  c = new double[nzc];
  jc = new int[nzc];
// Сосчитаем C = A - B
  request = 2;
  mkl_dcsradd(&trans, &request, &sort, &n, &n, A.value, A.row, A.colindex, 
	  &beta, B.value, B.row, B.colindex, c, jc, ic, &nzmax, &info);
// Сосчитаем max|Cij|
  diff = 0.0;
  for (i = 0; i < nzc; i++)
  {
    double var = fabs(c[i]);
    if (var > diff)
      diff = var;
  }
// Приведем к нормальному виду матрицы A и B
  for (i = 0; i < A.number_of_nonzero_elem; i++)
	  A.row[i]--;
  for (i = 0; i < B.number_of_nonzero_elem; i++)
	  B.row[i]--;
  for (j = 0; j <= n; j++)
  {
	  A.colindex[j]--;
	  B.colindex[j]--;
  }
// Освободим память
  delete [] c;
  delete [] ic;
  delete [] jc;

  return 0;
}

// Принимает 2 квадратных матрицы в формате CRS (3 массива, индексация с нуля)
// Возвращает C = A * B, C - в формате CRS (3 массива, индексация с нуля)
//                       Память для C в начале считается не выделенной
// Возвращает признак успешности операции: 0 - ОК, 1 - не совпадают размеры (N)
// Возвращает время работы
int SparseMKLMult(ccsmatrix A, ccsmatrix B, ccsmatrix &C, double &time)
{
	if (A.size != B.size)
    return 1;

  int n = A.size;

  // Настроим параметры для вызова функции MKL
  // Переиндексируем матрицы A и B с единицы
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

  // Используется функция, вычисляющая C = op(A) * B
  char trans = 'N'; // говорит о том, op(A) = A - не нужно транспонировать A

// Хитрый параметр, влияющий на то, как будет выделяться память
// request = 0: память для результирующей матрицы д.б. выделена заранее
// Если мы не знаем, сколько памяти необходимо для хранения результата,
// необходимо:
// 1) выделить память для массива индексов строк ic: "Кол-во строк+1" элементов;
// 2) вызвать функцию с параметром request = 1 - в массиве ic будет заполнен 
//                                                         последний элемент
// 3) выделить память для массивов c и jc 
//    (кол-во элементов = ic[Кол-во строк]-1)
// 4) вызвать функцию с параметром request = 2
  int request;

// Еще один нетривиальный момент: есть возможность настроить, нужно ли 
// упорядочивать матрицы A, B и C. У нас предполагается, что все матрицы
// упорядочены, следовательно, выбираем вариант "No-No-Yes", который
// соответствует любому значению, кроме целых чисел от 1 до 7 включительно
  int sort = 8;

// Количество ненулевых элементов.
// Используется только если request = 0
  int nzmax = -1;

// Служебная информация
  int info;

  clock_t start = clock();

// Выделим память для индекса в матрице C
  C.colindex = new int[n + 1];
// Сосчитаем количество ненулевых элементов в матрице C
  request = 1;
  C.value = 0;
  C.row = 0;
  mkl_dcsrmultcsr(&trans, &request, &sort, &n, &n, &n, A.value, A.row, 
	  A.colindex, B.value, B.row, B.colindex, C.value, C.row,
	  C.colindex, &nzmax, &info);

  int nzc = C.colindex[n] - 1;
  C.value = new double[nzc];
  C.row = new int[nzc];
// Сосчитаем C = A * B
  request = 2;

  mkl_dcsrmultcsr(&trans, &request, &sort, &n, &n, &n, A.value, A.row, 
	  A.colindex, B.value, B.row, B.colindex, C.value, C.row,
	  C.colindex, &nzmax, &info);
  C.size = n;
  C.number_of_nonzero_elem = nzc;

  clock_t finish = clock();

  // Приведем к нормальному виду матрицы A, B и С
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

// Генерирует квадратную матрицу в формате CRS (3 массива, индексация с нуля)
// В каждой строке cntInRow ненулевых элементов
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
    // Формируем номера строк в столбце i
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
    // Сортируем номера столцов в строке i
    for (j = 0; j < cntInRow - 1; j++)
      for (k = 0; k < cntInRow - 1; k++)
        if (mtx.row[i * cntInRow + k] > mtx.row[i * cntInRow + k + 1])
        {
          tmp = mtx.row[i * cntInRow + k];
          mtx.row[i * cntInRow + k] = mtx.row[i * cntInRow + k + 1];
          mtx.row[i * cntInRow + k + 1] = tmp;
        }
  }

  // Заполняем массив значений
  for (i = 0; i < cntInRow * N; i++)
    mtx.value[i] = next() * MAX_VAL;

  // Заполняем массив индексов строк
  c = 0;
  for (i = 0; i <= N; i++)
  {
	  mtx.colindex[i] = c;
    c += cntInRow;
  }
}

// Генерирует квадратную матрицу в формате CRS (3 массива, индексация с нуля)
// Число ненулевых элементов в строках растет от 1 до cntInRow
// Закон роста - кубическая парабола
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
