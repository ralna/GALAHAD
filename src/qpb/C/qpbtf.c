/* qpbtf.c */
/* Full test for the QPB C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_qpb.h"

int main(void) {

    // Derived types
    void *data;
    struct qpb_control_type control;
    struct qpb_inform_type inform;

    // Set problem data
    int n = 3; // dimension
    int m = 2; // number of general constraints
    int H_ne = 3; // Hesssian elements
    int H_row[] = {1, 2, 3 };   // row indices, NB lower triangle
    int H_col[] = {1, 2, 3};    // column indices, NB lower triangle
    int H_ptr[] = {1, 2, 3, 4}; // row pointers
    double H_val[] = {1.0, 1.0, 1.0 };   // values
    double g[] = {0.0, 2.0, 0.0};   // linear term in the objective
    double f = 1.0;  // constant term in the objective
    int A_ne = 4; // Jacobian elements
    int A_row[] = {1, 1, 2, 2}; // row indices
    int A_col[] = {1, 2, 2, 3}; // column indices
    int A_ptr[] = {1, 3, 5}; // row pointers
    double A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
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

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of qp storage formats\n\n");

    for( int d=1; d <= 7; d++){

        // Initialize QPB
        qpb_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        // Start from 0
        double x[] = {0.0,0.0,0.0};
        double y[] = {0.0,0.0};
        double z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                qpb_import( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL,
                           "coordinate", A_ne, A_row, A_col, NULL );
                qpb_solve_qp( &data, &status, n, m, H_ne, H_val, g, f, 
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;
            printf(" case %1i break\n",d);
            case 2: // sparse by rows
                st = 'R';
                qpb_import( &control, &data, &status, n, m, 
                            "sparse_by_rows", H_ne, NULL, H_col, H_ptr,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpb_solve_qp( &data, &status, n, m, H_ne, H_val, g, f, 
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;
            case 3: // dense
                st = 'D';
                int H_dense_ne = 6; // number of elements of H
                int A_dense_ne = 6; // number of elements of A
                double H_dense[] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
                double A_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                qpb_import( &control, &data, &status, n, m,
                            "dense", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL );
                qpb_solve_qp( &data, &status, n, m, H_dense_ne, H_dense, g, f, 
                              A_dense_ne, A_dense, c_l, c_u, x_l, x_u, 
                              x, c, y, z, x_stat, c_stat );
                break;
            case 4: // diagonal
                st = 'L';
                qpb_import( &control, &data, &status, n, m,
                            "diagonal", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpb_solve_qp( &data, &status, n, m, H_ne, H_val, g, f, 
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;

            case 5: // scaled identity
                st = 'S';
                qpb_import( &control, &data, &status, n, m, 
                            "scaled_identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpb_solve_qp( &data, &status, n, m, H_ne, H_val, g, f, 
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;
            case 6: // identity
                st = 'I';
                qpb_import( &control, &data, &status, n, m, 
                            "identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpb_solve_qp( &data, &status, n, m, H_ne, H_val, g, f, 
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;
            case 7: // zero
                st = 'Z';
                qpb_import( &control, &data, &status, n, m, 
                            "zero", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpb_solve_qp( &data, &status, n, m, H_ne, H_val, g, f, 
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z, 
                              x_stat, c_stat );
                break;



            }
        qpb_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: QPB_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        qpb_terminate( &data, &control, &inform );
    }
}

