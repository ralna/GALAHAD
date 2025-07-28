/* sslst.c */
/* Full test for the SSLS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_ssls.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct ssls_control_type control;
    struct ssls_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension of H
    ipc_ m = 2; // dimension of C
    ipc_ H_ne = 4; // number of elements of H
    ipc_ A_ne = 3; // number of elements of A
    ipc_ C_ne = 3; // number of elements of C
    ipc_ H_dense_ne = 6; // number of elements of H
    ipc_ A_dense_ne = 6; // number of elements of A
    ipc_ C_dense_ne = 3; // number of elements of C
    ipc_ H_row[] = {1, 2, 3, 3}; // row indices, NB lower triangle
    ipc_ H_col[] = {1, 2, 3, 1};
    ipc_ H_ptr[] = {1, 2, 3, 5};
    ipc_ A_row[] = {1, 1, 2};
    ipc_ A_col[] = {1, 2, 3};
    ipc_ A_ptr[] = {1, 3, 4};
    ipc_ C_row[] = {1, 2, 2}; // row indices, NB lower triangle
    ipc_ C_col[] = {1, 1, 2};
    ipc_ C_ptr[] = {1, 2, 4};
    rpc_ H_val[] = {1.0, 2.0, 3.0, 1.0};
    rpc_ A_val[] = {2.0, 1.0, 1.0};
    rpc_ C_val[] = {4.0, 1.0, 2.0};
    rpc_ H_dense[] = {1.0, 0.0, 2.0, 1.0, 0.0, 3.0};
    rpc_ A_dense[] = {2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    rpc_ C_dense[] = {4.0, 1.0, 2.0};
    rpc_ H_diag[] = {1.0, 1.0, 2.0};
    rpc_ C_diag[] = {4.0, 2.0};
    rpc_ H_scid[] = {2.0};
    rpc_ C_scid[] = {2.0};
    rpc_ sol[n+m];

    char st = ' ';
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( ipc_ d=1; d <= 7; d++){

        // Initialize SSLS
        ssls_initialize( &data, &control, &status );
        //control.print_level = 1;
        //control.sls_control.print_level = 1;
        strcpy(control.symmetric_linear_solver, "sytr ") ;

        // Set user-defined control options
        control.f_indexing = true; // fortran sparse matrix indexing

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                ssls_import( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL,
                           "coordinate", A_ne, A_row, A_col, NULL,
                           "coordinate", C_ne, C_row, C_col, NULL );
                ssls_factorize_matrix( &data, &status,
                                       H_ne, H_val,
                                       A_ne, A_val,
                                       C_ne, C_val );
                break;
            printf(" case %1" i_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                ssls_import( &control, &data, &status, n, m,
                            "sparse_by_rows", H_ne, NULL, H_col, H_ptr,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr,
                            "sparse_by_rows", C_ne, NULL, C_col, C_ptr );
                ssls_factorize_matrix( &data, &status,
                                       H_ne, H_val,
                                       A_ne, A_val,
                                       C_ne, C_val );
                break;
            case 3: // dense
                st = 'D';
                ssls_import( &control, &data, &status, n, m,
                            "dense", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "dense", C_ne, NULL, NULL, NULL );
                ssls_factorize_matrix( &data, &status,
                                       H_dense_ne, H_dense,
                                       A_dense_ne, A_dense,
                                       C_dense_ne, C_dense );
                break;
            case 4: // diagonal
                st = 'L';
                ssls_import( &control, &data, &status, n, m,
                            "diagonal", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "diagonal", C_ne, NULL, NULL, NULL );
                ssls_factorize_matrix( &data, &status,
                                       n, H_diag,
                                       A_dense_ne, A_dense,
                                       m, C_diag );
                break;

            case 5: // scaled identity
                st = 'S';
                ssls_import( &control, &data, &status, n, m,
                            "scaled_identity", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "scaled_identity", C_ne, NULL, NULL, NULL );
                ssls_factorize_matrix( &data, &status,
                                       1, H_scid,
                                       A_dense_ne, A_dense,
                                       1, C_scid );
                break;
            case 6: // identity
                st = 'I';
                ssls_import( &control, &data, &status, n, m,
                            "identity", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "identity", C_ne, NULL, NULL, NULL );
                ssls_factorize_matrix( &data, &status,
                                       0, H_val,
                                       A_dense_ne, A_dense,
                                       0, C_val );
                break;
            case 7: // zero
                st = 'Z';
                ssls_import( &control, &data, &status, n, m,
                            "identity", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "zero", C_ne, NULL, NULL, NULL );
                ssls_factorize_matrix( &data, &status,
                                       0, H_val,
                                       A_dense_ne, A_dense,
                                       0, NULL );
                break;
            }

        // check that the factorization succeeded
        if(status != 0){
            ssls_information( &data, &inform, &status );
            printf("%c: SSLS_solve factorization exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
            continue;
        } 

        // Set right-hand side ( a, b ) so that solution is vec(1.0)
        switch(d){
            case 4: // diagonal
                sol[0] = 3.0;
                sol[1] = 2.0;
                sol[2] = 3.0;
                sol[3] = -1.0;
                sol[4] = -1.0;
                break;
            case 5: // scaled identity
                sol[0] = 4.0;
                sol[1] = 3.0;
                sol[2] = 3.0;
                sol[3] = 1.0;
                sol[4] = -1.0;
                break;
            case 6: // identity
                sol[0] = 3.0;
                sol[1] = 2.0;
                sol[2] = 2.0;
                sol[3] = 2.0;
                sol[4] = 0.0;
                break;
            case 7: // zero
                sol[0] = 3.0;
                sol[1] = 2.0;
                sol[2] = 2.0;
                sol[3] = 3.0;
                sol[4] = 1.0;
                break;
            default: 
                sol[0] = 4.0;
                sol[1] = 3.0;
                sol[2] = 5.0;
                sol[3] = -2.0;
                sol[4] = -2.0;
                break;
            }

        ssls_solve_system( &data, &status, n, m, sol );

        // check that the factorization succeeded
        if(status != 0){
            ssls_information( &data, &inform, &status );
            printf("%c: SSLS_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
            continue;
        } 

        ssls_information( &data, &inform, &status );
        if(inform.status == 0){
            printf("%c: status = %1" i_ipc_ "", st, inform.status);
          printf(" sol = ");
          for(int loop = 0; loop < n+m; loop++)
#ifdef REAL_128
             printf("%f ", (double)sol[loop]);
#else
             printf("%f ", sol[loop]);
#endif
           printf("\n");
        }else{
            printf("%c: SSLS_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }

        // Delete internal workspace
        ssls_terminate( &data, &control, &inform );
    }
}

