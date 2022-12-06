/* bqpbt.c */
/* Full test for the BQPB C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_bqpb.h"

int main(void) {

    // Derived types
    void *data;
    struct bqpb_control_type control;
    struct bqpb_inform_type inform;

    // Set problem data
    int n = 3; // dimension
    int H_ne = 3; // Hesssian elements
    int H_row[] = {0, 1, 2 };   // row indices, NB lower triangle
    int H_col[] = {0, 1, 2};    // column indices, NB lower triangle
    int H_ptr[] = {0, 1, 2, 3}; // row pointers
    double H_val[] = {1.0, 1.0, 1.0 };   // values
    double g[] = {2.0, 0.0, 0.0};   // linear term in the objective
    double f = 1.0;  // constant term in the objective
    double x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    double x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound

    // Set output storage
    int x_stat[n]; // variable status
    char st;
    int status;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of qp storage formats\n\n");

    for( int d=1; d <= 7; d++){

        // Initialize BQPB
        bqpb_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        // Start from 0
        double x[] = {0.0,0.0,0.0};
        double z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                bqpb_import( &control, &data, &status, n,
                            "coordinate", H_ne, H_row, H_col, NULL );
                bqpb_solve_qp( &data, &status, n, H_ne, H_val, g, f,
                               x_l, x_u, x, z, x_stat );
                break;
            printf(" case %1i break\n",d);
            case 2: // sparse by rows
                st = 'R';
                bqpb_import( &control, &data, &status, n,
                             "sparse_by_rows", H_ne, NULL, H_col, H_ptr );
                bqpb_solve_qp( &data, &status, n, H_ne, H_val, g, f,
                               x_l, x_u, x, z, x_stat );
                break;
            case 3: // dense
                st = 'D';
                int H_dense_ne = 6; // number of elements of H
                double H_dense[] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
                bqpb_import( &control, &data, &status, n,
                             "dense", H_ne, NULL, NULL, NULL );
                bqpb_solve_qp( &data, &status, n, H_dense_ne, H_dense, g, f,
                               x_l, x_u, x, z, x_stat );
                break;
            case 4: // diagonal
                st = 'L';
                bqpb_import( &control, &data, &status, n,
                             "diagonal", H_ne, NULL, NULL, NULL );
                bqpb_solve_qp( &data, &status, n, H_ne, H_val, g, f,
                               x_l, x_u, x, z, x_stat );
                break;

            case 5: // scaled identity
                st = 'S';
                bqpb_import( &control, &data, &status, n,
                             "scaled_identity", H_ne, NULL, NULL, NULL );
                bqpb_solve_qp( &data, &status, n, H_ne, H_val, g, f,
                               x_l, x_u, x, z, x_stat );
                break;
            case 6: // identity
                st = 'I';
                bqpb_import( &control, &data, &status, n,
                             "identity", H_ne, NULL, NULL, NULL );
                bqpb_solve_qp( &data, &status, n, H_ne, H_val, g, f,
                               x_l, x_u, x, z, x_stat );
                break;
            case 7: // zero
                st = 'Z';
                bqpb_import( &control, &data, &status, n,
                             "zero", H_ne, NULL, NULL, NULL );
                bqpb_solve_qp( &data, &status, n, H_ne, H_val, g, f,
                               x_l, x_u, x, z, x_stat );
                break;



            }
        bqpb_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: BQPB_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        bqpb_terminate( &data, &control, &inform );
    }

    // test shifted least-distance interface
    for( int d=1; d <= 1; d++){

        // Initialize BQPB
        bqpb_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        // Start from 0
        double x[] = {0.0,0.0,0.0};
        double z[] = {0.0,0.0,0.0};

        // Set shifted least-distance data

        double w[] = {1.0,1.0,1.0};
        double x_0[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'W';
                bqpb_import( &control, &data, &status, n,
                           "shifted_least_distance", H_ne, NULL, NULL, NULL );
                bqpb_solve_sldqp( &data, &status, n, w, x_0, g, f,
                                 x_l, x_u, x, z, x_stat );
                break;

            }
        bqpb_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: BQPB_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        bqpb_terminate( &data, &control, &inform );
    }

}

