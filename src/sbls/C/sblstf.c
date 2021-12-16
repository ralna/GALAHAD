/* sblstf.c */
/* Full test for the SBLS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "sbls.h"

int main(void) {

    // Derived types
    void *data;
    struct sbls_control_type control;
    struct sbls_inform_type inform;

    // Set problem data
    int n = 3; // dimension of H
    int m = 2; // dimension of C
    int H_ne = 4; // number of elements of H
    int A_ne = 3; // number of elements of A
    int C_ne = 3; // number of elements of C
    int H_dense_ne = 6; // number of elements of H
    int A_dense_ne = 6; // number of elements of A
    int C_dense_ne = 3; // number of elements of C
    int H_row[] = {1, 2, 3, 3}; // row indices, NB lower triangle
    int H_col[] = {1, 2, 3, 1};
    int H_ptr[] = {1, 2, 3, 5}; 
    int A_row[] = {1, 1, 2};
    int A_col[] = {1, 2, 3};
    int A_ptr[] = {1, 3, 4};
    int C_row[] = {1, 2, 2}; // row indices, NB lower triangle
    int C_col[] = {1, 1, 2};
    int C_ptr[] = {1, 2, 4};
    double H_val[] = {1.0, 2.0, 3.0, 1.0};
    double A_val[] = {2.0, 1.0, 1.0};
    double C_val[] = {4.0, 1.0, 2.0};
    double H_dense[] = {1.0, 0.0, 2.0, 1.0, 0.0, 3.0};
    double A_dense[] = {2.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double C_dense[] = {4.0, 1.0, 2.0};
    double H_diag[] = {1.0, 1.0, 2.0};
    double C_diag[] = {4.0, 2.0};
    double H_scid[] = {2.0};
    double C_scid[] = {2.0};

    char st;
    int status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( int d=1; d <= 7; d++){

        // Initialize SBLS
        sbls_initialize( &data, &control, &status );
        control.preconditioner = 2;
        control.factorization = 2;
        control.get_norm_residual = true;

        // Set user-defined control options
        control.f_indexing = true; // fortran sparse matrix indexing

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                sbls_import( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL,
                           "coordinate", A_ne, A_row, A_col, NULL,
                           "coordinate", C_ne, C_row, C_col, NULL );
                sbls_factorize_matrix( &data, &status, n, 
                                       H_ne, H_val, 
                                       A_ne, A_val, 
                                       C_ne, C_val, NULL );
                break;
            printf(" case %1i break\n",d);
            case 2: // sparse by rows
                st = 'R';
                sbls_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", H_ne, NULL, H_col, H_ptr,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr,
                            "sparse_by_rows", C_ne, NULL, C_col, C_ptr );
                sbls_factorize_matrix( &data, &status, n, 
                                       H_ne, H_val, 
                                       A_ne, A_val, 
                                       C_ne, C_val, NULL );
                break;
            case 3: // dense
                st = 'D';
                sbls_import( &control, &data, &status, n, m,
                            "dense", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "dense", C_ne, NULL, NULL, NULL );
                sbls_factorize_matrix( &data, &status, n, 
                                       H_dense_ne, H_dense, 
                                       A_dense_ne, A_dense, 
                                       C_dense_ne, C_dense,
                                       NULL );
                break;
            case 4: // diagonal
                st = 'L';
                sbls_import( &control, &data, &status, n, m,
                            "diagonal", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "diagonal", C_ne, NULL, NULL, NULL );
                sbls_factorize_matrix( &data, &status, n, 
                                       n, H_diag, 
                                       A_dense_ne, A_dense, 
                                       m, C_diag, 
                                       NULL );
                break;

            case 5: // scaled identity
                st = 'S';
                sbls_import( &control, &data, &status, n, m, 
                            "scaled_identity", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "scaled_identity", C_ne, NULL, NULL, NULL );
                sbls_factorize_matrix( &data, &status, n,
                                       1, H_scid, 
                                       A_dense_ne, A_dense, 
                                       1, C_scid, 
                                       NULL );
                break;
            case 6: // identity
                st = 'I';
                sbls_import( &control, &data, &status, n, m, 
                            "identity", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "identity", C_ne, NULL, NULL, NULL );
                sbls_factorize_matrix( &data, &status, n, 
                                       0, H_val, 
                                       A_dense_ne, A_dense, 
                                       0, C_val, NULL );
                break;
            case 7: // zero
                st = 'Z';
                sbls_import( &control, &data, &status, n, m, 
                            "identity", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL,
                            "zero", C_ne, NULL, NULL, NULL );
                sbls_factorize_matrix( &data, &status, n, 
                                       0, H_val, 
                                       A_dense_ne, A_dense, 
                                       0, NULL, NULL );
                break;
            }

        // Set right-hand side ( a, b )
        double sol[] = {3.0, 2.0, 4.0, 2.0, 0.0};   // values

        sbls_solve_system( &data, &status, n, m, sol );

        sbls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c: residual = %9.1e status = %1i\n",
                   st, inform.norm_residual, inform.status);
        }else{
            printf("%c: SBLS_solve exit status = %1i\n", st, inform.status);
        }
        //printf("sol: ");
        //for( int i = 0; i < n+m; i++) printf("%f ", x[i]);

        // Delete internal workspace
        sbls_terminate( &data, &control, &inform );
    }
}

