/* cllst.c */
/* Full test for the CLLS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_clls.h"

int main(void) {

    // Derived types
    void *data;
    struct clls_control_type control;
    struct clls_inform_type inform;

    // Set problem data
    int n = 3; // dimension
    int m = 2; // number of general constraints
    int H_ne = 3; // Hesssian elements
    int H_row[] = {0, 1, 2 };   // row indices, NB lower triangle
    int H_col[] = {0, 1, 2};    // column indices, NB lower triangle
    int H_ptr[] = {0, 1, 2, 3}; // row pointers
    double H_val[] = {1.0, 1.0, 1.0 };   // values
    double g[] = {0.0, 2.0, 0.0};   // linear term in the objective
    double f = 1.0;  // constant term in the objective
    int L_ne = 4; // Jacobian elements
    int L_row[] = {0, 0, 1, 1}; // row indices
    int L_col[] = {0, 1, 1, 2}; // column indices
    int L_ptr[] = {0, 2, 4}; // row pointers
    double L_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
    double c_l[] = {1.0, 2.0};   // constraint lower bound
    double c_u[] = {2.0, 2.0};   // constraint upper bound
    double x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    double x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound

    // Set output storage
    double c[m]; // constraint values
    int x_stat[n]; // variable status
    int c_stat[m]; // constraint status
    char st;
    int status;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of qp storage formats\n\n");

    for( int d=1; d <= 7; d++){

        // Initialize CLLS
        clls_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        // Start from 0
        double x[] = {0.0,0.0,0.0};
        double y[] = {0.0,0.0};
        double z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                clls_import( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL,
                           "coordinate", L_ne, L_row, L_col, NULL );
                clls_solve_clls( &data, &status, n, m, H_ne, H_val, g, f, 
                              L_ne, L_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;
            printf(" case %1i break\n",d);
            case 2: // sparse by rows
                st = 'R';
                clls_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", H_ne, NULL, H_col, H_ptr,
                            "sparse_by_rows", L_ne, NULL, L_col, L_ptr );
                clls_solve_clls( &data, &status, n, m, H_ne, H_val, g, f, 
                              L_ne, L_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;
            case 3: // dense
                st = 'D';
                int H_dense_ne = 6; // number of elements of H
                int L_dense_ne = 6; // number of elements of A
                double H_dense[] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
                double L_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                clls_import( &control, &data, &status, n, m,
                            "dense", H_ne, NULL, NULL, NULL,
                            "dense", L_ne, NULL, NULL, NULL );
                clls_solve_clls( &data, &status, n, m, H_dense_ne, H_dense, g, 
                              f, L_dense_ne, L_dense, c_l, c_u, x_l, x_u, 
                              x, c, y, z, x_stat, c_stat );
                break;
            case 4: // diagonal
                st = 'L';
                clls_import( &control, &data, &status, n, m,
                            "diagonal", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", L_ne, NULL, L_col, L_ptr );
                clls_solve_clls( &data, &status, n, m, H_ne, H_val, g, f, 
                                L_ne, L_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                                x_stat, c_stat );
                break;

            case 5: // scaled identity
                st = 'S';
                clls_import( &control, &data, &status, n, m, 
                            "scaled_identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", L_ne, NULL, L_col, L_ptr );
                clls_solve_clls( &data, &status, n, m, H_ne, H_val, g, f, 
                                L_ne, L_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                                x_stat, c_stat );
                break;
            case 6: // identity
                st = 'I';
                clls_import( &control, &data, &status, n, m, 
                            "identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", L_ne, NULL, L_col, L_ptr );
                clls_solve_clls( &data, &status, n, m, H_ne, H_val, g, f, 
                                L_ne, L_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                                x_stat, c_stat );
                break;
            case 7: // zero
                st = 'Z';
                clls_import( &control, &data, &status, n, m, 
                            "zero", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", L_ne, NULL, L_col, L_ptr );
                clls_solve_clls( &data, &status, n, m, H_ne, H_val, g, f, 
                                L_ne, L_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                                x_stat, c_stat );
                break;



            }
        clls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: CLLS_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        clls_terminate( &data, &control, &inform );
    }

}

