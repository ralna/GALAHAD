/* fdct.c */
/* Full test for the FDC C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_fdc.h"

int main(void) {

    // Derived types
    void *data;
    struct fdc_control_type control;
    struct fdc_inform_type inform;

    // Set problem data
    int m = 3; // number of rows
    int n = 4; // number of columns
    int A_ne = 10; // number of nonzeros
    int A_col[] = {0, 1, 2, 3, 0, 1, 2, 3, 1, 3}; // column indices
    int A_ptr[] = {0, 4, 8, 10}; // row pointers
    double A_val[] = {1.0, 2.0, 3.0, 4.0, 2.0, -4.0, 6.0, -8.0, 5.0, 10.0};
    double b[] = {5.0, 10.0, 0.0};

    // Set output storage
    int depen[m]; // dependencies, if any
    int n_depen;
    int status;

    printf(" C sparse matrix indexing\n");

    // Initialize FDC
    fdc_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing

    // Start from 0

    fdc_find_dependent_rows( &control, &data, &inform, &status, m, n, A_ne,
                             A_col, A_ptr, A_val, b, &n_depen, depen );

    if(status == 0){
      if(n_depen == 0){
        printf("FDC_find_dependent - no dependent rows, status = %1i\n",
               status);
      }else{
        printf("FDC_find_dependent - dependent rows(s):" );
        for( int i = 0; i < n_depen; i++) printf(" %i", depen[i]);
        printf(", status = %i\n", status);
      }
    }else{
        printf("FDC_find_dependent - exit status = %1i\n", status);
    }

    // Delete internal workspace
    fdc_terminate( &data, &control, &inform );
}

