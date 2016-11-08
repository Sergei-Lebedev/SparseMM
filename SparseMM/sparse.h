#ifndef __SPARSE_H__
#define __SPARSE_H__

struct ccsmatrix
{
	double *value; //массив значений матрицы по столбцам
	int *row; //массив номеров строк
	int *colindex; //массив индексов начала столбцов
	int number_of_nonzero_elem; //число ненулевых элементов в матрице
	int size; //размер матрицы
};

// Принимает 2 квадратных матрицы в формате CСS (3 массива, индексация с нуля)
// Возвращает C = A * B, C - в формате CСS (3 массива, индексация с нуля)
ccsmatrix ccs_multiplicate_matrix(ccsmatrix A, ccsmatrix B);
#endif // __SPARSE_H__