/* qpatf.c */
/* Full test for the QPA C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_qpa.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct qpa_control_type control;
    struct qpa_inform_type inform;

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
    rpc_ rho_g = 0.1;  // penalty paramter for general constraints
    rpc_ rho_b = 0.1;  // penalty paramter for simple bound constraints
    ipc_ A_ne = 4; // Jacobian elements
    ipc_ A_row[] = {1, 1, 2, 2}; // row indices
    ipc_ A_col[] = {1, 2, 2, 3}; // column indices
    ipc_ A_ptr[] = {1, 3, 5}; // row pointers
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

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of qp storage formats\n\n");

    for( ipc_ d=1; d <= 7; d++){

        // Initialize QPA
        qpa_initialize( &data, &control, &status );
        strcpy(control.symmetric_linear_solver, "sytr ") ;

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        // Start from 0
        rpc_ x[] = {0.0,0.0,0.0};
        rpc_ y[] = {0.0,0.0};
        rpc_ z[] = {0.0,0.0,0.0};

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                qpa_import( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL,
                           "coordinate", A_ne, A_row, A_col, NULL );
                qpa_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;
            printf(" case %1" i_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                qpa_import( &control, &data, &status, n, m,
                            "sparse_by_rows", H_ne, NULL, H_col, H_ptr,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpa_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;
            case 3: // dense
                st = 'D';
                ipc_ H_dense_ne = 6; // number of elements of H
                ipc_ A_dense_ne = 6; // number of elements of A
                rpc_ H_dense[] = {1.0, 0.0, 1.0, 0.0, 0.0, 1.0};
                rpc_ A_dense[] = {2.0, 1.0, 0.0, 0.0, 1.0, 1.0};
                qpa_import( &control, &data, &status, n, m,
                            "dense", H_ne, NULL, NULL, NULL,
                            "dense", A_ne, NULL, NULL, NULL );
                qpa_solve_qp( &data, &status, n, m, H_dense_ne, H_dense, g, f,
                              A_dense_ne, A_dense, c_l, c_u, x_l, x_u,
                              x, c, y, z, x_stat, c_stat );
                break;
            case 4: // diagonal
                st = 'L';
                qpa_import( &control, &data, &status, n, m,
                            "diagonal", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpa_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;

            case 5: // scaled identity
                st = 'S';
                qpa_import( &control, &data, &status, n, m,
                            "scaled_identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpa_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;
            case 6: // identity
                st = 'I';
                qpa_import( &control, &data, &status, n, m,
                            "identity", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpa_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;
            case 7: // zero
                st = 'Z';
                qpa_import( &control, &data, &status, n, m,
                            "zero", H_ne, NULL, NULL, NULL,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                qpa_solve_qp( &data, &status, n, m, H_ne, H_val, g, f,
                              A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                              x_stat, c_stat );
                break;



            }
        qpa_information( &data, &inform, &status );

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
            printf("%c: QPA_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        qpa_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of l_1 qp storage formats\n\n");

    qpa_initialize( &data, &control, &status );
    strcpy(control.symmetric_linear_solver, "sytr ") ;

    // Set user-defined control options
    control.f_indexing = true; // Fortran sparse matrix indexing

    // Start from 0
    rpc_ x[] = {0.0,0.0,0.0};
    rpc_ y[] = {0.0,0.0};
    rpc_ z[] = {0.0,0.0,0.0};

    // solve the l_1qp problem
    qpa_import( &control, &data, &status, n, m,
               "coordinate", H_ne, H_row, H_col, NULL,
               "coordinate", A_ne, A_row, A_col, NULL );
    qpa_solve_l1qp( &data, &status, n, m, H_ne, H_val, g, f, rho_g, rho_b,
                    A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                    x_stat, c_stat );

    qpa_information( &data, &inform, &status );

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
        printf("%c: QPA_solve exit status = %1" i_ipc_ "\n", st, inform.status);
    }
    // Start from 0
    for( ipc_ i=0; i <= n-1; i++) x[i] = 0.0;
    for( ipc_ i=0; i <= m-1; i++) y[i] = 0.0;
    for( ipc_ i=0; i <= n-1; i++) z[i] = 0.0;

    // solve the bound constrained l_1qp problem
    qpa_import( &control, &data, &status, n, m,
               "coordinate", H_ne, H_row, H_col, NULL,
               "coordinate", A_ne, A_row, A_col, NULL );
    qpa_solve_bcl1qp( &data, &status, n, m, H_ne, H_val, g, f, rho_g,
                      A_ne, A_val, c_l, c_u, x_l, x_u, x, c, y, z,
                      x_stat, c_stat );

    qpa_information( &data, &inform, &status );

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
        printf("%c: QPA_solve exit status = %1" i_ipc_ "\n", st, inform.status);
    }

    // Delete internal workspace
    qpa_terminate( &data, &control, &inform );
}

