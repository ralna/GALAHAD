/* wcptf.c */
/* Full test for the WCP C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_wcp.h"

int main(void) {

    // Derived types
    void *data;
    struct wcp_control_type control;
    struct wcp_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ m = 2; // number of general constraints
    real_wp_ g[] = {0.0, 2.0, 0.0};   // linear term in the objective
    ipc_ A_ne = 4; // Jacobian elements
    ipc_ A_row[] = {1, 1, 2, 2}; // row indices
    ipc_ A_col[] = {1, 2, 2, 3}; // column indices
    ipc_ A_ptr[] = {1, 3, 5}; // row pointers
    real_wp_ A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
    real_wp_ c_l[] = {1.0, 2.0};   // constraint lower bound
    real_wp_ c_u[] = {2.0, 2.0};   // constraint upper bound
    real_wp_ x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    real_wp_ x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound

    // Set output storage
    real_wp_ c[m]; // constraint values
    ipc_ x_stat[n]; // variable status
    ipc_ c_stat[m]; // constraint status
    char st;
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of wcp storage formats\n\n");

    for( ipc_ d=1; d <= 3; d++){

        // Initialize WCP
        wcp_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        // Start from 0
        real_wp_ x[] = {0.0,0.0,0.0};
        real_wp_ y_l[] = {0.0,0.0};
        real_wp_ y_u[] = {0.0,0.0};
        real_wp_ z_l[] = {0.0,0.0,0.0};
        real_wp_ z_u[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                wcp_import( &control, &data, &status, n, m,
                            "coordinate", A_ne, A_row, A_col, NULL );
                wcp_find_wcp( &data, &status, n, m, g, A_ne, A_val,
                              c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u,
                              x_stat, c_stat );
                break;
            printf(" case %1i break\n",d);
            case 2: // sparse by rows
                st = 'R';
                wcp_import( &control, &data, &status, n, m,
                             "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                wcp_find_wcp( &data, &status, n, m, g, A_ne, A_val,
                              c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u,
                              x_stat, c_stat );
                break;
            case 3: // dense
                st = 'D';
                ipc_ A_dense_ne = 6; // number of elements of A
                real_wp_ A_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                wcp_import( &control, &data, &status, n, m,
                             "dense", A_dense_ne, NULL, NULL, NULL );
                wcp_find_wcp( &data, &status, n, m, g, A_dense_ne, A_dense,
                              c_l, c_u, x_l, x_u, x, c, y_l, y_u, z_l, z_u,
                              x_stat, c_stat );
                break;
            }
        wcp_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: WCP_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        wcp_terminate( &data, &control, &inform );
    }
}

