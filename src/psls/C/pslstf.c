/* pslstf.c */
/* Full test for the PSLS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_psls.h"

int main(void) {

    // Derived types
    void *data;
    struct psls_control_type control;
    struct psls_inform_type inform;

    // Set problem data
    ipc_ n = 5; // dimension of A
    ipc_ ne = 7; // number of elements of A
    ipc_ dense_ne = n * ( n + 1 ) / 2; // number of elements of dense A

    ipc_ row[] = {1, 2, 2, 3, 3, 4, 5}; // A indices & values, NB lower triangle
    ipc_ col[] = {1, 1, 5, 2, 3, 3, 5};
    ipc_ ptr[] = {1, 2, 4, 6, 7, 8};
    rpc_ val[] = {2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0};
    rpc_ dense[] = {2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                      0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0};
    char st = ' ';
    ipc_ status;
    ipc_ status_apply;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( ipc_ d=1; d <= 3; d++){

        // Initialize PSLS
        psls_initialize( &data, &control, &status );
        control.preconditioner = 2; // band preconditioner
        control.semi_bandwidth = 1; // semibandwidth
        strcpy( control.definite_linear_solver, "sils" );

        // Set user-defined control options
        control.f_indexing = true; // fortran sparse matrix indexing

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                psls_import( &control, &data, &status, n,
                             "coordinate", ne, row, col, NULL );
                psls_form_preconditioner( &data, &status, ne, val );
                break;
            printf(" case %1" i_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                psls_import( &control, &data, &status, n,
                             "sparse_by_rows", ne, NULL, col, ptr );
                psls_form_preconditioner( &data, &status, ne, val );
                break;
            case 3: // dense
                st = 'D';
                psls_import( &control, &data, &status, n,
                             "dense", ne, NULL, NULL, NULL );
                psls_form_preconditioner( &data, &status, dense_ne, dense );
                break;
            }

        // Set right-hand side b in x
        rpc_ x[] = {8.0, 45.0, 31.0, 15.0, 17.0};   // values

        if(status == 0){
          psls_information( &data, &inform, &status );
          psls_apply_preconditioner( &data, &status_apply, n, x );
        }else{
          status_apply = - 1;
        }

        printf("%c storage: status from form & factorize = %" i_ipc_ 
               " apply = %" i_ipc_ "\n", st, status, status_apply );

        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);

        // Delete internal workspace
        psls_terminate( &data, &control, &inform );
    }
}

