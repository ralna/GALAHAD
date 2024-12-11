/* lpbt.c */
/* Full test for the LPB C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_lpb.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct lpb_control_type control;
    struct lpb_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ m = 2; // number of general constraints
    rpc_ g[] = {0.0, 2.0, 0.0};   // linear term in the objective
    rpc_ f = 1.0;  // constant term in the objective
    ipc_ A_ne = 4; // Jacobian elements
    ipc_ A_row[] = {0, 0, 1, 1}; // row indices
    ipc_ A_col[] = {0, 1, 1, 2}; // column indices
    ipc_ A_ptr[] = {0, 2, 4}; // row pointers
    rpc_ A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
    rpc_ c_l[] = {1.0, 2.0};   // constraint lower bound
    rpc_ c_u[] = {2.0, 2.0};   // constraint upper bound
    rpc_ x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    rpc_ x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound

    // Set output storage
    rpc_ c[m]; // constraint values
    ipc_ x_stat[n]; // variable status
    ipc_ c_stat[m]; // constraint status
    char st = ' ';
    ipc_ status;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of lp storage formats\n\n");

    for( ipc_ d=1; d <= 3; d++){

        // Initialize LPB
        lpb_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        // Start from 0
        rpc_ x[] = {0.0,0.0,0.0};
        rpc_ y[] = {0.0,0.0};
        rpc_ z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                lpb_import( &control, &data, &status, n, m,
                           "coordinate", A_ne, A_row, A_col, NULL );
                lpb_solve_lp( &data, &status, n, m, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;
            printf(" case %1" i_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                lpb_import( &control, &data, &status, n, m,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                lpb_solve_lp( &data, &status, n, m, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;
            case 3: // dense
                st = 'D';
                ipc_ A_dense_ne = 6; // number of elements of A
                rpc_ A_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                lpb_import( &control, &data, &status, n, m,
                            "dense", A_ne, NULL, NULL, NULL );
                lpb_solve_lp( &data, &status, n, m, g, f,
                              A_dense_ne, A_dense, c_l, c_u, x_l, x_u,
                              x, c, y, z, x_stat, c_stat );
                break;
            }
        lpb_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_f.h
#include "galahad_pquad_f.h"
#else
            printf("%c:%6" i_ipc_ " iterations. Optimal objective " 
                   "value = %.2f status = %1" i_ipc_ "\n",
                   st, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf("%c: LPB_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        lpb_terminate( &data, &control, &inform );
    }
}
