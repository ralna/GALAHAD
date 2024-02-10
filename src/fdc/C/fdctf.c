/* fdctf.c */
/* Full test for the FDC C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_fdc.h"

int main(void) {

    // Derived types
    void *data;
    struct fdc_control_type control;
    struct fdc_inform_type inform;

    // Set problem data
    ipc_ m = 3; // number of rows
    ipc_ n = 4; // number of columns
    ipc_ A_ne = 10; // number of nonzeros
    ipc_ A_col[] = {1, 2, 3, 4, 1, 2, 3, 4, 2, 4}; // column indices
    ipc_ A_ptr[] = {1, 5, 9, 11}; // row pointers
    rpc_ A_val[] = {1.0, 2.0, 3.0, 4.0, 2.0, -4.0, 6.0, -8.0, 5.0, 10.0};
    rpc_ b[] = {5.0, 10.0, 0.0};

    // Set output storage
    ipc_ depen[m]; // dependencies, if any
    ipc_ n_depen;
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n");

    // Initialize FDC
    fdc_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = true; // Fortran sparse matrix indexing

    // Start from 0

    fdc_find_dependent_rows( &control, &data, &inform, &status, m, n, A_ne,
                             A_col, A_ptr, A_val, b, &n_depen, depen );

    if(status == 0){
      if(n_depen == 0){
        printf("FDC_find_dependent - no dependent rows, status = %" i_ipc_ "\n",
               status);
      }else{
        printf("FDC_find_dependent - dependent rows(s):" );
        for( ipc_ i = 0; i < n_depen; i++) printf(" %" i_ipc_ "", depen[i]);
        printf(", status = %" i_ipc_ "\n", status);
      }
    }else{
        printf("FDC_find_dependent - exit status = %1" i_ipc_ "\n", status);
    }

    // Delete internal workspace
    fdc_terminate( &data, &control, &inform );
}

