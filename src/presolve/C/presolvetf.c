/* presolvetf.c */
/* Full test for the PRESOLVE C interface using Fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_presolve.h"

int main(void) {

    // Derived types
    void *data;
    struct presolve_control_type control;
    struct presolve_inform_type inform;

    // Set problem data
    ipc_ n = 6; // dimension
    ipc_ m = 5; // number of general constraints
    ipc_ H_ne = 1; // Hesssian elements
    ipc_ H_row[] = {1};   // row indices, NB lower triangle
    ipc_ H_col[] = {1};    // column indices, NB lower triangle
    ipc_ H_ptr[] = {1, 2, 2, 2, 2, 2, 2}; // row pointers
    rpc_ H_val[] = {1.0};   // values
    rpc_ g[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // linear term in the objective
    rpc_ f = 1.0;  // constant term in the objective
    ipc_ A_ne = 8; // Jacobian elements
    ipc_ A_row[] = {3, 3, 3, 4, 4, 5, 5, 5}; // row indices
    ipc_ A_col[] = {3, 4, 5, 3, 6, 4, 5, 6}; // column indices
    ipc_ A_ptr[] = {1, 1, 1, 4, 6, 9}; // row pointers
    rpc_ A_val[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // values
    rpc_ c_l[] = { 0.0, 0.0, 2.0, 1.0, 3.0};   // constraint lower bound
    rpc_ c_u[] = {1.0, 1.0, 3.0, 3.0, 3.0};   // constraint upper bound
    rpc_ x_l[] = {-3.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // variable lower bound
    rpc_ x_u[] = {3.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // variable upper bound

    // Set output storage
    char st = ' ';
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" basic tests of qp storage formats\n\n");

    for( ipc_ d=1; d <= 7; d++){

      ipc_ n_trans, m_trans, H_ne_trans, A_ne_trans;

        // Initialize PRESOLVE
        presolve_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                presolve_import_problem( &control, &data, &status, n, m,
                           "coordinate", H_ne, H_row, H_col, NULL, H_val, g, f,
                           "coordinate", A_ne, A_row, A_col, NULL, A_val,
                           c_l, c_u, x_l, x_u,
                           &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;
            printf(" case %1" i_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                presolve_import_problem( &control, &data, &status, n, m,
                       "sparse_by_rows", H_ne, NULL, H_col, H_ptr, H_val, g, f,
                       "sparse_by_rows", A_ne, NULL, A_col, A_ptr, A_val,
                       c_l, c_u, x_l, x_u,
                       &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;
            case 3: // dense
                st = 'D';
                ipc_ H_dense_ne = n*(n+1)/2; // number of elements of H
                ipc_ A_dense_ne = m*n; // number of elements of A
                rpc_ H_dense[] = {1.0,
                                    0.0, 0.0,
                                    0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                rpc_ A_dense[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                                    0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
                                    0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
                presolve_import_problem( &control, &data, &status, n, m,
                            "dense", H_dense_ne, NULL, NULL, NULL, H_dense, g,
                            f, "dense", A_dense_ne, NULL, NULL, NULL, A_dense,
                            c_l, c_u, x_l, x_u,
                            &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;
            case 4: // diagonal
                st = 'L';
                presolve_import_problem( &control, &data, &status, n, m,
                            "diagonal", n, NULL, NULL, NULL, H_val, g, f,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr, A_val,
                            c_l, c_u, x_l, x_u,
                            &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;

            case 5: // scaled identity
                st = 'S';
                presolve_import_problem( &control, &data, &status, n, m,
                        "scaled_identity", 1, NULL, NULL, NULL, H_val, g, f,
                        "sparse_by_rows", A_ne, NULL, A_col, A_ptr, A_val,
                        c_l, c_u, x_l, x_u,
                        &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;
            case 6: // identity
                st = 'I';
                presolve_import_problem( &control, &data, &status, n, m,
                            "identity", 0, NULL, NULL, NULL, NULL, g, f,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr, A_val,
                            c_l, c_u, x_l, x_u,
                            &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;
            case 7: // zero
                st = 'Z';
                presolve_import_problem( &control, &data, &status, n, m,
                            "zero", 0, NULL, NULL, NULL, NULL, g, f,
                            "sparse_by_rows", A_ne, NULL, A_col, A_ptr, A_val,
                            c_l, c_u, x_l, x_u,
                            &n_trans, &m_trans, &H_ne_trans, &A_ne_trans );
                break;
            }

        //printf("%c: n, m, h_ne, a_ne = %2" i_ipc_ ", %2" i_ipc_ ", %2" i_ipc_ ", %2" i_ipc_ "\n",
        //           st, n_trans, m_trans, H_ne_trans, A_ne_trans);
        rpc_ f_trans;  // transformed constant term in the objective
        ipc_ H_ptr_trans[n_trans+1]; // transformed Hessian row pointers
        ipc_ H_col_trans[H_ne_trans]; // transformed Hessian column indices
        rpc_ H_val_trans[H_ne_trans]; // transformed Hessian values
        rpc_ g_trans[n_trans]; // transformed gradient
        ipc_ A_ptr_trans[m_trans+1]; // transformed Jacobian row pointers
        ipc_ A_col_trans[A_ne_trans]; // transformed Jacobian column indices
        rpc_ A_val_trans[A_ne_trans]; // transformed Jacobian values
        rpc_ x_l_trans[n_trans]; // transformed lower variable bounds
        rpc_ x_u_trans[n_trans]; // transformed upper variable bounds
        rpc_ c_l_trans[m_trans]; // transformed lower constraint bounds
        rpc_ c_u_trans[m_trans]; // transformed upper constraint bounds
        rpc_ y_l_trans[m_trans]; // transformed lower multiplier bounds
        rpc_ y_u_trans[m_trans]; // transformed upper multiplier bounds
        rpc_ z_l_trans[n_trans]; // transformed lower dual variable bounds
        rpc_ z_u_trans[n_trans]; // transformed upper dual variable bounds

        presolve_transform_problem( &data, &status, n_trans, m_trans,
                               H_ne_trans, H_col_trans, H_ptr_trans,
                               H_val_trans, g_trans, &f_trans, A_ne_trans,
                               A_col_trans, A_ptr_trans, A_val_trans,
                               c_l_trans, c_u_trans, x_l_trans, x_u_trans,
                               y_l_trans, y_u_trans, z_l_trans, z_u_trans );

        rpc_ x_trans[n_trans]; // transformed variables
        for( ipc_ i = 0; i < n_trans; i++) x_trans[i] = 0.0;
        rpc_ c_trans[m_trans]; // transformed constraints
        for( ipc_ i = 0; i < m_trans; i++) c_trans[i] = 0.0;
        rpc_ y_trans[m_trans]; // transformed Lagrange multipliers
        for( ipc_ i = 0; i < m_trans; i++) y_trans[i] = 0.0;
        rpc_ z_trans[n_trans]; // transformed dual variables
        for( ipc_ i = 0; i < n_trans; i++) z_trans[i] = 0.0;

        rpc_ x[n]; // primal variables
        rpc_ c[m]; // constraint values
        rpc_ y[m]; // Lagrange multipliers
        rpc_ z[n]; // dual variables

        //printf("%c: n_trans, m_trans, n, m = %2" i_ipc_ ", %2" i_ipc_ ", %2" i_ipc_ ", %2" i_ipc_ "\n",
        //           st, n_trans, m_trans, n, m );
        presolve_restore_solution( &data, &status, n_trans, m_trans,
                  x_trans, c_trans, y_trans, z_trans, n, m, x, c, y, z );

        presolve_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c:%6" i_ipc_ " transformations, n, m = %2" i_ipc_ ", %2" i_ipc_ ", status = %1" i_ipc_ "\n",
                   st, inform.nbr_transforms, n_trans, m_trans, inform.status);
        }else{
            printf("%c: PRESOLVE_solve exit status = %1" i_ipc_ "\n", st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        presolve_terminate( &data, &control, &inform );
    }
}
