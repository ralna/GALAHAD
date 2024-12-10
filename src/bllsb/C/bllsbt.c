/* bllsbt.c */
/* Full test for the BLLSB C interface using C sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_bllsb.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct bllsb_control_type control;
    struct bllsb_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ o = 4; // number of observations
    rpc_ sigma = 1.0; // regularization weight
    rpc_ b[] = {2.0, 2.0, 3.0, 1.0};   // observations
    rpc_ x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    rpc_ x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound
    rpc_ w[] = {1.0, 1.0, 1.0, 2.0}; // weights

    // Set output storage
    rpc_ r[o]; // residual values
    ipc_ x_stat[n]; // variable status
    char st[3];
    ipc_ status;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of bllsb storage formats\n\n");

    for( ipc_ d=1; d <= 5; d++){

        // Initialize BLLSB
        bllsb_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        strcpy(control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.fdc_control.symmetric_linear_solver, "sytr ") ;
        control.fdc_control.use_sls = true;

        // Start from 0
        rpc_ x[] = {0.0,0.0,0.0};
        rpc_ z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                strcpy(st, "CO");
                {
                ipc_ Ao_ne = 7; // objective Jacobian elements
                ipc_ Ao_row[] = {0, 0, 1, 1, 2, 2, 3};   // row indices
                ipc_ Ao_col[] = {0, 1, 1, 2, 0, 2, 1};    // column indices
                rpc_ Ao_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // vals
                bllsb_import( &control, &data, &status, n, o,
                              "coordinate", Ao_ne, Ao_row, Ao_col, 0, NULL );
                bllsb_solve_blls( &data, &status, n, o, Ao_ne, Ao_val, b,
                                  sigma, x_l, x_u, x, r, z, x_stat, w );
                }
                break;
            case 2: // sparse by rows
                strcpy(st, "SR");
                {
                ipc_ Ao_ne = 7; // objective Jacobian elements
                ipc_ Ao_col[] = {0, 1, 1, 2, 0, 2, 1};    // column indices
                ipc_ Ao_ptr_ne = o + 1; // number of row pointers
                ipc_ Ao_ptr[] = {0, 2, 4, 6, 7}; // row pointers
                rpc_ Ao_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // vals
                bllsb_import( &control, &data, &status, n, o,
                              "sparse_by_rows", Ao_ne, NULL, Ao_col,
                              Ao_ptr_ne, Ao_ptr );
                bllsb_solve_blls( &data, &status, n, o, Ao_ne, Ao_val, b,
                                  sigma, x_l, x_u, x, r, z, x_stat, w );
                }
                break;
            case 3: // sparse by columns
                strcpy(st, "SC");
                {
                ipc_ Ao_ne = 7; // objective Jacobian elements
                ipc_ Ao_row[] = {0, 2, 0, 1, 3, 1, 2};   // row indices
                ipc_ Ao_ptr_ne = n + 1; // number of column pointers
                ipc_ Ao_ptr[] = {0, 2, 5, 7}; // column pointers
                rpc_ Ao_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // vals
                bllsb_import( &control, &data, &status, n, o,
                              "sparse_by_columns", Ao_ne, Ao_row, NULL,
                              Ao_ptr_ne, Ao_ptr );
                bllsb_solve_blls( &data, &status, n, o, Ao_ne, Ao_val, b,
                                  sigma, x_l, x_u, x, r, z, x_stat, w );
                }
                break;
            case 4: // dense by rows
                strcpy(st, "DR");
                {
                ipc_ Ao_ne = 12; // objective Jacobian elements
                rpc_ Ao_dense[] = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                       1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
                bllsb_import( &control, &data, &status, n, o,
                             "dense", Ao_ne, NULL, NULL, 0, NULL );
                bllsb_solve_blls( &data, &status, n, o, Ao_ne, Ao_dense, b,
                                  sigma, x_l, x_u, x, r, z, x_stat, w );
                }
                break;
            case 5: // dense by cols
                strcpy(st, "DC");
                {
                ipc_ Ao_ne = 12; // objective Jacobian elements
                rpc_ Ao_dense[] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                                       0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
                bllsb_import( &control, &data, &status, n, o,
                             "dense_by_columns", Ao_ne, NULL, NULL, 0, NULL );
                bllsb_solve_blls( &data, &status, n, o, Ao_ne, Ao_dense, b,
                                  sigma, x_l, x_u, x, r, z, x_stat, w );
                }
                break;
            }
        bllsb_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_sf.h
#include "galahad_pquad_sf.h"
#else
            printf("%s:%6" i_ipc_ " iterations. Optimal objective " 
                   "value = %.2f status = %1" i_ipc_ "\n",
                   st, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf("%s: BLLSB_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        bllsb_terminate( &data, &control, &inform );
    }
}
