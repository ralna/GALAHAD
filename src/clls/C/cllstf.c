/* cllstf.c */
/* Full test for the CLLS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_clls.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct clls_control_type control;
    struct clls_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ o = 4; // number of observations
    ipc_ m = 2; // number of general constraints
    rpc_ sigma = 1.0; // regularization weight
    rpc_ b[] = {2.0, 2.0, 3.0, 1.0};   // observations
    rpc_ c_l[] = {1.0, 2.0};   // constraint lower bound
    rpc_ c_u[] = {2.0, 2.0};   // constraint upper bound
    rpc_ x_l[] = {-1.0, - INFINITY, - INFINITY}; // variable lower bound
    rpc_ x_u[] = {1.0, INFINITY, 2.0}; // variable upper bound
    rpc_ w[] = {1.0, 1.0, 1.0, 2.0}; // weights

    // Set output storage
    rpc_ r[o]; // residual values
    rpc_ c[m]; // constraint values
    ipc_ x_stat[n]; // variable status
    ipc_ c_stat[m]; // constraint status
    char st[3];
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of clls storage formats\n\n");

    for( ipc_ d=1; d <= 5; d++){

        // Initialize CLLS
        clls_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        strcpy(control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.fdc_control.symmetric_linear_solver, "sytr ") ;
        control.fdc_control.use_sls = true;

        // Start from 0
        rpc_ x[] = {0.0,0.0,0.0};
        rpc_ y[] = {0.0,0.0};
        rpc_ z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                strcpy(st, "CO");
                {
                ipc_ Ao_ne = 7; // objective Jacobian elements
                ipc_ Ao_row[] = {1, 1, 2, 2, 3, 3, 4};   // row indices
                ipc_ Ao_col[] = {1, 2, 2, 3, 1, 3, 2};    // column indices
                rpc_ Ao_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // vals
                ipc_ A_ne = 4; // constraint Jacobian elements
                ipc_ A_row[] = {1, 1, 2, 2}; // row indices
                ipc_ A_col[] = {1, 2, 2, 3}; // column indices
                rpc_ A_val[] = {2.0, 1.0, 1.0, 1.0}; // values
                clls_import( &control, &data, &status, n, o, m,
                            "coordinate", Ao_ne, Ao_row, Ao_col, 0, NULL,
                            "coordinate", A_ne, A_row, A_col, 0, NULL );
                clls_solve_clls( &data, &status, n, o, m, Ao_ne, Ao_val, b,
                                 sigma, A_ne, A_val, c_l, c_u, x_l, x_u,
                                 x, r, c, y, z, x_stat, c_stat, w );
                }
                break;
            case 2: // sparse by rows
                strcpy(st, "SR");
                {
                ipc_ Ao_ne = 7; // objective Jacobian elements
                ipc_ Ao_col[] = {1, 2, 2, 3, 1, 3, 2};    // column indices
                ipc_ Ao_ptr_ne = o + 1; // number of row pointers
                ipc_ Ao_ptr[] = {1, 3, 5, 7, 8}; // row pointers
                rpc_ Ao_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // vals
                ipc_ A_ne = 4; // constraint Jacobian elements
                ipc_ A_col[] = {1, 2, 2, 3}; // column indices
                ipc_ A_ptr_ne = m + 1; // number of row pointers
                ipc_ A_ptr[] = {1, 3, 5}; // row pointers
                rpc_ A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
                clls_import( &control, &data, &status, n, o, m,
                             "sparse_by_rows", Ao_ne, NULL, Ao_col,
                             Ao_ptr_ne, Ao_ptr,
                             "sparse_by_rows", A_ne, NULL, A_col,
                             A_ptr_ne, A_ptr );
                clls_solve_clls( &data, &status, n, o, m, Ao_ne, Ao_val, b,
                                 sigma, A_ne, A_val, c_l, c_u, x_l, x_u,
                                 x, r, c, y, z, x_stat, c_stat, w );
                }
                break;
            case 3: // sparse by columns
                strcpy(st, "SC");
                {
                ipc_ Ao_ne = 7; // objective Jacobian elements
                ipc_ Ao_row[] = {1, 3, 1, 2, 4, 2, 3};   // row indices
                ipc_ Ao_ptr_ne = n + 1; // number of column pointers
                ipc_ Ao_ptr[] = {1, 3, 6, 8}; // column pointers
                rpc_ Ao_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // vals
                ipc_ A_ne = 4; // constraint Jacobian elements
                ipc_ A_row[] = {1, 1, 2, 2}; // row indices
                ipc_ A_ptr_ne = n + 1; // number of column pointers
                ipc_ A_ptr[] = {1, 2, 4, 5}; // column pointers
                rpc_ A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
                clls_import( &control, &data, &status, n, o, m,
                             "sparse_by_columns", Ao_ne, Ao_row, NULL,
                             Ao_ptr_ne, Ao_ptr,
                             "sparse_by_columns", A_ne, A_row, NULL,
                             A_ptr_ne, A_ptr );
                clls_solve_clls( &data, &status, n, o, m, Ao_ne, Ao_val, b,
                                 sigma, A_ne, A_val, c_l, c_u, x_l, x_u,
                                 x, r, c, y, z, x_stat, c_stat, w );
                }
                break;
            case 4: // dense by rows
                strcpy(st, "DR");
                {
                ipc_ Ao_ne = 12; // objective Jacobian elements
                rpc_ Ao_dense[] = {1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                       1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
                ipc_ A_ne = 6; // constraint Jacobian elements
                rpc_ A_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                clls_import( &control, &data, &status, n, o, m,
                             "dense", Ao_ne, NULL, NULL, 0, NULL,
                             "dense", A_ne, NULL, NULL, 0, NULL );
                clls_solve_clls( &data, &status, n, o, m, Ao_ne, Ao_dense, b,
                                 sigma, A_ne, A_dense, c_l, c_u, x_l, x_u,
                                 x, r, c, y, z, x_stat, c_stat, w );
                }
                break;
            case 5: // dense by cols
                strcpy(st, "DC");
                {
                ipc_ Ao_ne = 12; // objective Jacobian elements
                rpc_ Ao_dense[] = {1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                                       0.0, 1.0, 0.0, 1.0, 1.0, 0.0};
                ipc_ A_ne = 6; // constraint Jacobian elements
                rpc_ A_dense[] = {2.0, 0.0, 1.0, 1.0, 0.0, 1.0};
                clls_import( &control, &data, &status, n, o, m,
                             "dense_by_columns", Ao_ne, NULL, NULL, 0, NULL,
                             "dense_by_columns", A_ne, NULL, NULL, 0, NULL );
                clls_solve_clls( &data, &status, n, o, m, Ao_ne, Ao_dense, b,
                                 sigma, A_ne, A_dense, c_l, c_u, x_l, x_u,
                                 x, r, c, y, z, x_stat, c_stat, w );
                }
                break;
            }
        clls_information( &data, &inform, &status );

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
            printf("%s: CLLS_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        clls_terminate( &data, &control, &inform );
    }
}
