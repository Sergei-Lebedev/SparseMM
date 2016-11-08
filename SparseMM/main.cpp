// ”множение разреженных матриц в формате CRS
// ћатрица ј: случайным образом выбираютс€ позиции 50 ненулевых элементов 
//   в каждой строке
// ћатрица B: кол-во ненулевых элементов от строки к строке растет так, чтобы
//    последн€€ строка содержала максимальное кол-во (50) ненулевых элементов
//    –ост происходит по кубическому закону

// ***** ќптимизаци€ 3 - учет "регул€рности" матриц *****

#include <stdio.h>
#include <stdlib.h>
#include "util.h"
#include "sparse.h"

const double EPSILON = 0.000001;

// argv[1] - пор€док матрицы
// argv[2] - количество ненулей в строках матрицы с регул€рной структурой
int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    printf("Invalid input parameters\n");
    return 1;
  }
  
  int N = atoi(argv[1]);
  int NZ = atoi(argv[2]);

  if ((NZ > N) || (N <= 0) || (NZ <= 0))
  {
    printf("Incorrect arguments of main\n");
    return 1;
  }

  ccsmatrix A, B, BT, C;

  GenerateRegularCCS(1, N, NZ, A);
  GenerateSpecialCCS(2, N, NZ, B);

  double timeM, timeM1;
  C = ccs_multiplicate_matrix(A, B);

  ccsmatrix CM;
  double diff;

  SparseMKLMult(B, A, CM, timeM1);
  int error = SparseDiff(C, CM, diff);
  if (diff < EPSILON && !error)
    printf("OK\n");
  else
    printf("not OK\n");
  printf("A.size = %d A.nz=%d B.nz=%d C.nz=%d\n", A.size, A.number_of_nonzero_elem, B.number_of_nonzero_elem, C.number_of_nonzero_elem);

  FreeMatrix(A);
  FreeMatrix(B);
  FreeMatrix(C);
  FreeMatrix(CM);

  return 0;
}
