/* eqptf.c */
/* Full test for the EQP C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_eqp.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct eqp_control_type control;
    struct eqp_inform_type inform;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ m = 2; // number of general constraints
    ipc_ H_ne = 3; // Hesssian elements
    ipc_ H_row[] = {1, 2, 3 };   // row indices, NB lower triangle
    ipc_ H_col[] = {1, 2, 3};    // column indices, NB lower triangle
    ipc_ H_ptr[] = {1, 2, 3, 4}; // row pointers
    rpc_ H_val[] = {1.0, 1.0, 1.0 };   // values
    rpc_ g[] = {0.0, 2.0, 0.0};   // linear term in the objective
    rpc_ f = 1.0;  // constant term in the objective
    ipc_ A_ne = 4; // Jacobian elements
    ipc_ A_row[] = {1, 1, 2, 2}; // row indices
    ipc_ A_col[] = {1, 2, 2, 3}; // column indices
    ipc_ A_ptr[] = {1, 3, 5}; // row pointers
    rpc_ A_val[] = {2.0, 1.0, 1.0, 1.0 }; // values
    rpc_ c[] = {3.0, 0.0};   // rhs of the constraints

    // Set output storage
    char st = ' ';
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of qp storage formats\n\n");

    for( ipc_ d=1; d <= 6; d++){

        // Initialize EQP
        eqp_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        control.fdc_control.use_sls = true ;
        strcpy(control.fdc_control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.sbls_control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.sbls_control.definite_linear_solver, "sytr ") ;

        // Start from 0
        rpc_ x[] = {0.0,0.0,0.0};
        rpc_ y[] = {0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                eqp_import( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL,
                           "coordinate", A_ne, A_row, A_col, NULL );
                eqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c, x, y );
                break;
            case 2: // sparse by rows
                st = 'R';
                eqp_import( &control, &data, &status, n, m,
                            "sparse_by_rows", H_ne, NULL, H_col, H_ptr,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                eqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c, x, y );
                break;
            case 3: // dense
                st = 'D';
                ipc_ H_dense_ne = 6; // number of elements of H
                ipc_ A_dense_ne = 6; // number of elements of A
                rpc_ H_dense[] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
                rpc_ A_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                eqp_import( &control, &data, &status, n, m,
                            "dense", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL );
                eqp_solve_qp( &data, &status, n, m, H_dense_ne, H_dense, g, f,
                              A_dense_ne, A_dense, c, x, y );
                break;
            case 4: // diagonal
                st = 'L';
                eqp_import( &control, &data, &status, n, m,
                            "diagonal", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                eqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c, x, y );
                break;

            case 5: // scaled identity
                st = 'S';
                eqp_import( &control, &data, &status, n, m,
                            "scaled_identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                eqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c, x, y );
                break;
            case 6: // identity
                st = 'I';
                eqp_import( &control, &data, &status, n, m,
                            "identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                eqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c, x, y );
                break;
            case 7: // zero
                st = 'Z';
                eqp_import( &control, &data, &status, n, m,
                            "zero", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                eqp_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c, x, y );
                break;
            }
        eqp_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
#include "galahad_pquad_if.h"
#else
            printf("%c:%6" i_ipc_ " cg iterations. Optimal objective " 
                   "value = %.2f status = %1" i_ipc_ "\n",
                   st, inform.cg_iter, inform.obj, inform.status);
#endif
        }else{
            printf("%c: EQP_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        eqp_terminate( &data, &control, &inform );
    }

    // test shifted least-distance interface
    for( ipc_ d=1; d <= 1; d++){

        // Initialize EQP
        eqp_initialize( &data, &control, &status );
        control.fdc_control.use_sls = true ;
        strcpy(control.fdc_control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.sbls_control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.sbls_control.definite_linear_solver, "sytr ") ;

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        // Start from 0
        rpc_ x[] = {0.0,0.0,0.0};
        rpc_ y[] = {0.0,0.0};

        // Set shifted least-distance data

        rpc_ w[] = {1.0,1.0,1.0};
        rpc_ x_0[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'W';
                eqp_import( &control, &data, &status, n, m,
                           "shifted_least_distance", H_ne, NULL, NULL, NULL,
                           "coordinate", A_ne, A_row, A_col, NULL );
                eqp_solve_sldqp( &data, &status, n, m, w, x_0, g, f,
                                 A_ne, A_val, c, x, y );
                break;

            }
        eqp_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
#include "galahad_pquad_if.h"
#else
            printf("%c:%6" i_ipc_ " cg iterations. Optimal objective " 
                   "value = %.2f status = %1" i_ipc_ "\n",
                   st, inform.cg_iter, inform.obj, inform.status);
#endif
        }else{
            printf("%c: EQP_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        eqp_terminate( &data, &control, &inform );
    }

}
