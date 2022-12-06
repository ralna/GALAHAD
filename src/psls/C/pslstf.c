/* pslstf.c */
/* Full test for the PSLS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_psls.h"

int main(void) {

    // Derived types
    void *data;
    struct psls_control_type control;
    struct psls_inform_type inform;

    // Set problem data
    int n = 5; // dimension of A
    int ne = 7; // number of elements of A
    int dense_ne = n * ( n + 1 ) / 2; // number of elements of dense A

    int row[] = {1, 2, 2, 3, 3, 4, 5}; // A indices & values, NB lower triangle
    int col[] = {1, 1, 5, 2, 3, 3, 5};
    int ptr[] = {1, 2, 4, 6, 7, 8};
    double val[] = {2.0, 3.0, 6.0, 4.0, 1.0, 5.0, 1.0};
    double dense[] = {2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0,
                      0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0};
    char st;
    int status;
    int status_apply;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( int d=1; d <= 3; d++){

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
            printf(" case %1i break\n",d);
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
        double x[] = {8.0, 45.0, 31.0, 15.0, 17.0};   // values

        if(status == 0){
          psls_information( &data, &inform, &status );
          psls_apply_preconditioner( &data, &status_apply, n, x );
        }else{
          status_apply = - 1;
        }

        printf("%c storage: status from form & factorize = %i apply = %i\n",
                   st, status, status_apply );

        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);

        // Delete internal workspace
        psls_terminate( &data, &control, &inform );
    }
}

