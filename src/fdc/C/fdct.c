/* fdct.c */
/* Full test for the FDC C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
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
    ipc_ A_col[] = {0, 1, 2, 3, 0, 1, 2, 3, 1, 3}; // column indices
    ipc_ A_ptr[] = {0, 4, 8, 10}; // row pointers
    rpc_ A_val[] = {1.0, 2.0, 3.0, 4.0, 2.0, -4.0, 6.0, -8.0, 5.0, 10.0};
    rpc_ b[] = {5.0, 10.0, 0.0};

    // Set output storage
    ipc_ depen[m]; // dependencies, if any
    ipc_ n_depen;
    ipc_ status;

    printf(" C sparse matrix indexing\n");

    // Initialize FDC
    fdc_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    control.use_sls = true;
    strcpy(control.symmetric_linear_solver, "sytr ");

    // Start from 0

    fdc_find_dependent_rows( &control, &data, &inform, &status, m, n, A_ne,
                             A_col, A_ptr, A_val, b, &n_depen, depen );

    if(status == 0){
      if(n_depen == 0){
        printf("FDC_find_dependent - no dependent rows, status = %1" i_ipc_ "\n",
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

