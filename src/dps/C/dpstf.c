/* dpstf.c */
/* Full test for the DPS C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_dps.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct dps_control_type control;
    struct dps_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension of H
    ipc_ H_ne = 4; // number of elements of H
    ipc_ H_dense_ne = 6; // number of elements of H
    ipc_ H_row[] = {1, 2, 3, 3}; // row indices, NB lower triangle
    ipc_ H_col[] = {1, 2, 3, 1};
    ipc_ H_ptr[] = {1, 2, 3, 5};
    rpc_ H_val[] = {1.0, 2.0, 3.0, 4.0};
    rpc_ H_dense[] = {1.0, 0.0, 2.0, 4.0, 0.0, 3.0};
    rpc_ f = 0.96;
    rpc_ radius = 1.0;
    rpc_ half_radius = 0.5;
    rpc_ c[] = {0.0, 2.0, 0.0};

    char st = ' ';
    ipc_ status;
    rpc_ x[n];

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    for( ipc_ storage_type=1; storage_type <= 3; storage_type++){

    // Initialize DPS
      dps_initialize( &data, &control, &status );

    // Set user-defined control options
      control.f_indexing = true; // fortran sparse matrix indexing
      strcpy(control.symmetric_linear_solver,"sytr ") ;

      switch(storage_type){
        case 1: // sparse co-ordinate storage
            st = 'C';
            // import the control parameters and structural data
            dps_import( &control, &data, &status, n,
                       "coordinate", H_ne, H_row, H_col, NULL );
            // solve the problem
            dps_solve_tr_problem( &data, &status, n, H_ne, H_val,
                                  c, f, radius, x );
            break;
        case 2: // sparse by rows
            st = 'R';
            // import the control parameters and structural data
            dps_import( &control, &data, &status, n,
                        "sparse_by_rows", H_ne, NULL, H_col, H_ptr );
            dps_solve_tr_problem( &data, &status, n, H_ne, H_val,
                                  c, f, radius, x );
            break;
        case 3: // dense
            st = 'D';
            // import the control parameters and structural data
            dps_import( &control, &data, &status, n,
                        "dense", H_ne, NULL, NULL, NULL );
            dps_solve_tr_problem( &data, &status, n, H_dense_ne, H_dense,
                                  c, f, radius, x );
            break;
        }

      dps_information( &data, &inform, &status );
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_dps.h
#include "galahad_pquad_dps.h"
#else
      printf("format %c: DPS_solve_problem exit status   = %1" i_ipc_ 
             ", f = %.2f\n", st, inform.status, inform.obj );
#endif

      switch(storage_type){
        case 1: // sparse co-ordinate storage
            st = 'C';
            // solve the problem
            dps_resolve_tr_problem( &data, &status, n,
                                    c, f, half_radius, x );
            break;
        case 2: // sparse by rows
            st = 'R';
            dps_resolve_tr_problem( &data, &status, n,
                                    c, f, half_radius, x );
            break;
        case 3: // dense
            st = 'D';
            dps_resolve_tr_problem( &data, &status, n,
                                    c, f, half_radius, x );
            break;
        }

      dps_information( &data, &inform, &status );
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_dpsr.h
#include "galahad_pquad_dpsr.h"
#else
      printf("format %c: DPS_resolve_problem exit status = %1" i_ipc_ 
             ", f = %.2f\n", st, inform.status, inform.obj );
#endif
      //printf("x: ");
      //for( ipc_ i = 0; i < n+m; i++) printf("%f ", x[i]);

      // Delete internal workspace
      dps_terminate( &data, &control, &inform );
   }
}

