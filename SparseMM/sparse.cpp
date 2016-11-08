#include <time.h>
#include <vector>
#include <stdlib.h>
#include <math.h>

#include "util.h"
#include "sparse.h"

using namespace std;

void default_init(ccsmatrix &matrix)
{
    matrix.size = 0;
    matrix.number_of_nonzero_elem = 0;
    matrix.row = 0;
    matrix.value = 0;
    matrix.colindex = 0;
}

// ��������� 2 ���������� ������� � ������� C�S (3 �������, ���������� � ����)
// ���������� C = A * B, C - � ������� C�S (3 �������, ���������� � ����)
ccsmatrix ccs_multiplicate_matrix(ccsmatrix A, ccsmatrix B)
{
	ccsmatrix C;
	default_init(C);
  return C;
}
