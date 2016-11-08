#ifndef __UTIL_H__
#define __UTIL_H__

#include "sparse.h"

// Создает квадратную матрицу в формате CCS (3 массива, индексация с нуля)
// Выделяет память под поля value, row и colindex
// Возвращает через параметр mtx
void InitializeMatrix(int N, int NZ, ccsmatrix &mtx);

// Освобождает память для полей mtx
void FreeMatrix(ccsmatrix &mtx);

// Создает копию imtx в omtx, выделяя память под поля value, row и colindex
void CopyMatrix(ccsmatrix imtx, ccsmatrix &omtx);

// Принимает 2 квадратных матрицы в формате CCS (3 массива, индексация с нуля)
// Возвращает max|Cij|, где C = A - B
// Возвращает признак успешности операции: 0 - ОК, 1 - не совпадают размеры (N)
int CompareMatrix(ccsmatrix mtx1, ccsmatrix mtx2, double &diff);

// Принимает 2 квадратных матрицы в формате CCS (3 массива, индексация с нуля)
// Возвращает max|Cij|, где C = A - B
// Для сравнения использует функцию из MKL
// Возвращает признак успешности операции: 0 - ОК, 1 - не совпадают размеры (N)
int SparseDiff(ccsmatrix A, ccsmatrix B, double &diff);

// Принимает 2 квадратных матрицы в формате CCS (3 массива, индексация с нуля)
// Возвращает C = A * B, C - в формате CCS (3 массива, индексация с нуля)
//                       Память для C в начале считается не выделенной
// Возвращает признак успешности операции: 0 - ОК, 1 - не совпадают размеры (N)
// Возвращает время работы
int SparseMKLMult(ccsmatrix A, ccsmatrix B, ccsmatrix &C, double &time);

// Генерирует квадратную матрицу в формате CCS (3 массива, индексация с нуля)
// В каждом столбце cntInRow ненулевых элементов
void GenerateRegularCCS(int seed, int N, int cntInRow, ccsmatrix& mtx);

// Генерирует квадратную матрицу в формате CСS (3 массива, индексация с нуля)
// Число ненулевых элементов в строках растет от 1 до cntInRow
// Закон роста - кубическая парабола
void GenerateSpecialCCS(int seed, int N, int cntInRow, ccsmatrix& mtx);

#endif // __UTIL_H__