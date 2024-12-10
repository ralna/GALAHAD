/* nlstf.c */
/* Full test for the NLS interface using Fortran sparse matrix indexing */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2021 */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_nls.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

// Define imax
ipc_ imax(ipc_ a, ipc_ b) {
    return (a > b) ? a : b;
};

// Custom userdata struct
struct userdata_type {
   rpc_ p;
};

// Function prototypes

ipc_ res( ipc_ n, ipc_ m, const rpc_ x[], rpc_ c[], const void * );
ipc_ jac( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
         const void * );
ipc_ hess( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
          rpc_ hval[], const void * );
ipc_ jacprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose,
             rpc_ u[], const rpc_ v[], bool got_j, const void * );
ipc_ hessprod( ipc_ n, ipc_ m, const rpc_ x[], const rpc_ y[],
              rpc_ u[], const rpc_ v[], bool got_h, const void * );
ipc_ rhessprods( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[], const rpc_ v[],
                rpc_ pval[], bool got_h, const void * );
ipc_ scale( ipc_ n, ipc_ m, const rpc_ x[], rpc_ u[],
           const rpc_ v[], const void * );
ipc_ jac_dense( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
               const void * );
ipc_ hess_dense( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
                rpc_ hval[], const void * );
ipc_ rhessprods_dense( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[],
                      const rpc_ v[], rpc_ pval[], bool got_h,
                      const void * );

int main(void) {

    // Derived types
    void *data;
    struct nls_control_type control;
    struct nls_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 1.0;

    // Set problem data
    ipc_ n = 2; // # variables
    ipc_ m = 3; // # residuals
    ipc_ j_ne = 5; // Jacobian elements
    ipc_ h_ne = 2; // Hesssian elements
    ipc_ p_ne = 2; // residual-Hessians-vector products elements
    ipc_ J_row[] = {1, 2, 2, 3, 3}; // Jacobian J
    ipc_ J_col[] = {1, 1, 2, 1, 2}; //
    ipc_ J_ptr[] = {1, 2, 4, 6};    // row pointers
    ipc_ H_row[] = {1, 2};          // Hessian H
    ipc_ H_col[] = {1, 2};          // NB lower triangle
    ipc_ H_ptr[] = {1, 2, 3};       // row pointers
    ipc_ P_row[] = {1, 2};          // residual-Hessians-vector product matrix
    ipc_ P_ptr[] = {1, 2, 3, 3};    // column pointers

    // Set storage
    rpc_ g[n]; // gradient
    rpc_ c[m]; // residual
    rpc_ y[m]; // multipliers
    char st = ' ';
    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    printf(" tests options for all-in-one storage format\n\n");

    for( ipc_ d=1; d <= 5; d++){
//  for( ipc_ d=5; d <= 5; d++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        // control.print_level = 1;
        control.jacobian_available = 2;
        control.hessian_available = 2;
        control.model = 6;
        rpc_ x[] = {1.5,1.5}; // starting point
        rpc_ W[] = {1.0, 1.0, 1.0}; // weights

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                nls_import( &control, &data, &status, n, m,
                            "coordinate", j_ne, J_row, J_col, NULL,
                            "coordinate", h_ne, H_row, H_col, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac,
                                    h_ne, hess, p_ne, rhessprods );
                break;
            case 2: // sparse by rows
                st = 'R';
                nls_import( &control, &data, &status, n, m,
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac,
                                    h_ne, hess, p_ne, rhessprods );
                break;
            case 3: // dense
                st = 'D';
                nls_import( &control, &data, &status, n, m,
                            "dense", j_ne, NULL, NULL, NULL,
                            "dense", h_ne, NULL, NULL, NULL,
                            "dense", p_ne, NULL, NULL, NULL, W );
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac_dense,
                                    h_ne, hess_dense, p_ne, rhessprods_dense );
                break;
            case 4: // diagonal
                st = 'I';
                nls_import( &control, &data, &status, n, m,
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "diagonal", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                nls_solve_with_mat( &data, &userdata, &status,
                                    n, m, x, c, g, res, j_ne, jac,
                                    h_ne, hess, p_ne, rhessprods );
                break;
            case 5: // access by products
                st = 'P';
                nls_import( &control, &data, &status, n, m,
                            "absent", j_ne, NULL, NULL, NULL,
                            "absent", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                nls_solve_without_mat( &data, &userdata, &status,
                                       n, m, x, c, g, res, jacprod,
                                       hessprod, p_ne, rhessprods );
                break;
        }

        nls_information( &data, &inform, &status );

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
            printf("%c: NLS_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    ipc_ eval_status;
    rpc_ u[imax(m,n)], v[imax(m,n)];
    rpc_ J_val[j_ne], J_dense[m*n];
    rpc_ H_val[h_ne], H_dense[n*(n+1)/2], H_diag[n];
    rpc_ P_val[p_ne], P_dense[m*n];
    bool transpose;
    bool got_j = false;
    bool got_h = false;

    for( ipc_ d=1; d <= 5; d++){
//  for( ipc_ d=1; d <= 4; d++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2;
        control.hessian_available = 2;
        control.model = 6;
        rpc_ x[] = {1.5,1.5}; // starting point
        rpc_ W[] = {1.0, 1.0, 1.0}; // weights

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                nls_import( &control, &data, &status, n, m,
                            "coordinate", j_ne, J_row, J_col, NULL,
                            "coordinate", h_ne, H_row, H_col, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_val, y,
                                              h_ne, H_val, v, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac( n, m, j_ne, x, J_val, &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess( n, m, h_ne, x, y, H_val, &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val,
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1" i_ipc_ " of status should not occur\n",
                        status);
                      break;
                  }
                }
                break;
            case 2: // sparse by rows
                st = 'R';
                nls_import( &control, &data, &status, n, m,
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_val, y,
                                              h_ne, H_val, v, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac( n, m, j_ne, x, J_val, &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess( n, m, h_ne, x, y, H_val, &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val,
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1" i_ipc_ " of status should not occur\n",
                        status);
                      break;
                  }
                }
                break;
            case 3: // dense
                st = 'D';
                nls_import( &control, &data, &status, n, m,
                            "dense", j_ne, NULL, NULL, NULL,
                            "dense", h_ne, NULL, NULL, NULL,
                            "dense", p_ne, NULL, NULL, NULL, W );
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, m*n, J_dense, y,
                                              n*(n+1)/2, H_dense, v, m*n,
                                              P_dense );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac_dense( n, m, j_ne, x, J_dense,
                                               &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess_dense( n, m, h_ne, x, y, H_dense,
                                                &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods_dense( n, m, p_ne, x, v, P_dense,
                                                      got_h, &userdata );
                  }else{
                      printf(" the value %1" i_ipc_ " of status should not occur\n",
                        status);
                      break;
                  }
                }
                break;
            case 4: // diagonal
                st = 'I';
                nls_import( &control, &data, &status, n, m,
                            "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                            "diagonal", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                while(true){ // reverse-communication loop
                  nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                              n, m, x, c, g, j_ne, J_val, y,
                                              n, H_diag, v, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 3){ // evaluate J
                      eval_status = jac( n, m, j_ne, x, J_val, &userdata );
                  }else if(status == 4){ // evaluate H
                      eval_status = hess( n, m, h_ne, x, y, H_diag, &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val,
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1" i_ipc_ " of status should not occur\n",
                        status);
                      break;
                  }
                }
                break;
            case 5: // access by products
                st = 'P';
//              control.print_level = 1;
                nls_import( &control, &data, &status, n, m,
                            "absent", j_ne, NULL, NULL, NULL,
                            "absent", h_ne, NULL, NULL, NULL,
                            "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
                while(true){ // reverse-communication loop
                  nls_solve_reverse_without_mat( &data, &status, &eval_status,
                                                 n, m, x, c, g, &transpose,
                                                 u, v, y, p_ne, P_val );
                  if(status == 0){ // successful termination
                        break;
                  }else if(status < 0){ // error exit
                      break;
                  }else if(status == 2){ // evaluate c
                      eval_status = res( n, m, x, c, &userdata );
                  }else if(status == 5){ // evaluate u + J v or u + J'v
                      eval_status = jacprod( n, m, x, transpose, u, v, got_j,
                                             &userdata );
                  }else if(status == 6){ // evaluate u + H v
                      eval_status = hessprod( n, m, x, y, u, v, got_h,
                                              &userdata );
                  }else if(status == 7){ // evaluate P
                      eval_status = rhessprods( n, m, p_ne, x, v, P_val,
                                                got_h, &userdata );
                  }else{
                      printf(" the value %1" i_ipc_ " of status should not occur\n",
                        status);
                      break;
                  }
                }
                break;
        }

        nls_information( &data, &inform, &status );

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
            printf("%c: NLS_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, direct access\n\n");

    for( ipc_ model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2;
        control.hessian_available = 2;
        control.model = model;
        rpc_ x[] = {1.5,1.5}; // starting point
        rpc_ W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m,
                    "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
        nls_solve_with_mat( &data, &userdata, &status,
                            n, m, x, c, g, res, j_ne, jac,
                            h_ne, hess, p_ne, rhessprods );

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_nf.h
#include "galahad_pquad_nf.h"
#else
            printf(" %1" i_ipc_ ":%6" i_ipc_ 
                   " iterations. Optimal objective value = %.2f"
                   " status = %1" i_ipc_ "\n",
                   model, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf(" %" i_ipc_ ": NLS_solve exit status = %1" i_ipc_ 
                   "\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, access by products\n\n");

    for( ipc_ model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2;
        control.hessian_available = 2;
        control.model = model;
        rpc_ x[] = {1.5,1.5}; // starting point
        rpc_ W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m,
                    "absent", j_ne, NULL, NULL, NULL,
                    "absent", h_ne, NULL, NULL, NULL,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
        nls_solve_without_mat( &data, &userdata, &status,
                               n, m, x, c, g, res, jacprod,
                               hessprod, p_ne, rhessprods );
        nls_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_pf.h
#include "galahad_pquad_pf.h"
#else
            printf("P%1" i_ipc_ ":%6" i_ipc_ " iterations. Optimal objective "
                   "value = %.2f status = %1" i_ipc_ "\n",
                   model, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf("P%" i_ipc_ ": NLS_solve exit status = %1" i_ipc_ 
                   "\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, reverse access\n\n");

    for( ipc_ model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2;
        control.hessian_available = 2;
        control.model = model;
        rpc_ x[] = {1.5,1.5}; // starting point
        rpc_ W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m,
                    "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                    "sparse_by_rows", h_ne, NULL, H_col, H_ptr,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
        while(true){ // reverse-communication loop
          nls_solve_reverse_with_mat( &data, &status, &eval_status,
                                      n, m, x, c, g, j_ne, J_val, y,
                                      h_ne, H_val, v, p_ne, P_val );
          if(status == 0){ // successful termination
                break;
          }else if(status < 0){ // error exit
              break;
          }else if(status == 2){ // evaluate c
              eval_status = res( n, m, x, c, &userdata );
          }else if(status == 3){ // evaluate J
              eval_status = jac( n, m, j_ne, x, J_val, &userdata );
          }else if(status == 4){ // evaluate H
              eval_status = hess( n, m, h_ne, x, y, H_val, &userdata );
          }else if(status == 7){ // evaluate P
              eval_status = rhessprods( n, m, p_ne, x, v, P_val,
                                        got_h, &userdata );
          }else{
              printf(" the value %1" i_ipc_ " of status should not occur\n",
                status);
              break;
          }
        }

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_pf.h
#include "galahad_pquad_pf.h"
#else
            printf("P%1" i_ipc_ ":%6" i_ipc_ " iterations. Optimal objective "
                   "value = %.2f status = %1" i_ipc_ "\n",
                   model, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf(" %" i_ipc_ ": NLS_solve exit status = %1" i_ipc_ 
                   "\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }

    printf("\n basic tests of models used, reverse access by products\n\n");

    for( ipc_ model=3; model <= 8; model++){

        // Initialize NLS
        nls_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = true; // Fortran sparse matrix indexing
        //control.print_level = 1;
        control.jacobian_available = 2;
        control.hessian_available = 2;
        control.model = model;
        rpc_ x[] = {1.5,1.5}; // starting point
        rpc_ W[] = {1.0, 1.0, 1.0}; // weights

        nls_import( &control, &data, &status, n, m,
                    "absent", j_ne, NULL, NULL, NULL,
                    "absent", h_ne, NULL, NULL, NULL,
                    "sparse_by_columns", p_ne, P_row, NULL, P_ptr, W );
        while(true){ // reverse-communication loop
          nls_solve_reverse_without_mat( &data, &status, &eval_status,
                                         n, m, x, c, g, &transpose,
                                         u, v, y, p_ne, P_val );
          if(status == 0){ // successful termination
                break;
          }else if(status < 0){ // error exit
              break;
          }else if(status == 2){ // evaluate c
              eval_status = res( n, m, x, c, &userdata );
          }else if(status == 5){ // evaluate u + J v or u + J'v
              eval_status = jacprod( n, m, x, transpose, u, v, got_j,
                                     &userdata );
          }else if(status == 6){ // evaluate u + H v
              eval_status = hessprod( n, m, x, y, u, v, got_h,
                                      &userdata );
          }else if(status == 7){ // evaluate P
              eval_status = rhessprods( n, m, p_ne, x, v, P_val,
                                        got_h, &userdata );
          }else{
              printf(" the value %1" i_ipc_ " of status should not occur\n",
                status);
              break;
          }
        }

        nls_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_pf.h
#include "galahad_pquad_pf.h"
#else
            printf("P%1" i_ipc_ ":%6" i_ipc_ " iterations. Optimal objective "
                   "value = %.2f status = %1" i_ipc_ "\n",
                   model, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf("P%" i_ipc_ ": NLS_solve exit status = %1" i_ipc_ 
                   "\n", model, inform.status);
        }
        // Delete internal workspace
        nls_terminate( &data, &control, &inform );
    }
}

// compute the residuals
ipc_ res( ipc_ n, ipc_ m, const rpc_ x[], rpc_ c[], const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    rpc_ p = myuserdata->p;
    c[0] = pow(x[0],2.0) + p;
    c[1] = x[0] + pow(x[1],2.0);
    c[2] = x[0] - x[1];
    return 0;
}

// compute the Jacobian
ipc_ jac( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
         const void *userdata ){
    jval[0] = 2.0 * x[0];
    jval[1] = 1.0;
    jval[2] = 2.0 * x[1];
    jval[3] = 1.0;
    jval[4] = - 1.0;
    return 0;
}

// compute the Hessian
ipc_ hess( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
           rpc_ hval[], const void *userdata ){
    hval[0] = 2.0 * y[0];
    hval[1] = 2.0 * y[1];
    return 0;
}

// compute Jacobian-vector products
ipc_ jacprod( ipc_ n, ipc_ m, const rpc_ x[], const bool transpose, rpc_ u[],
             const rpc_ v[], bool got_j, const void *userdata ){
    if (transpose) {
      u[0] = u[0] + 2.0 * x[0] * v[0] + v[1] + v[2];
      u[1] = u[1] + 2.0 * x[1] * v[1] - v[2];
    }else{
      u[0] = u[0] + 2.0 * x[0] * v[0];
      u[1] = u[1] + v[0]  + 2.0 * x[1] * v[1];
      u[2] = u[2] + v[0] - v[1];
    }
    return 0;
}

// compute Hessian-vector products
ipc_ hessprod( ipc_ n, ipc_ m, const rpc_ x[], const rpc_ y[], rpc_ u[],
              const rpc_ v[], bool got_h, const void *userdata ){
    u[0] = u[0] + 2.0 * y[0] * v[0];
    u[1] = u[1] + 2.0 * y[1] * v[1];
    return 0;
}

// compute residual-Hessians-vector products
ipc_ rhessprods( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[], const rpc_ v[],
                rpc_ pval[], bool got_h, const void *userdata ){
    pval[0] = 2.0 * v[0];
    pval[1] = 2.0 * v[1];
    return 0;
}

// scale v
ipc_ scale( ipc_ n, ipc_ m, const rpc_ x[], rpc_ u[],
           const rpc_ v[], const void *userdata ){
    u[0] = v[0];
    u[1] = v[1];
    return 0;
}

// compute the dense Jacobian
ipc_ jac_dense( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ jval[],
               const void *userdata ){
    jval[0] = 2.0 * x[0];
    jval[1] = 0.0;
    jval[2] = 1.0;
    jval[3] = 2.0 * x[1];
    jval[4] = 1.0;
    jval[5] = - 1.0;
    return 0;
}

// compute the dense Hessian
ipc_ hess_dense( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
                rpc_ hval[], const void *userdata ){
    hval[0] = 2.0 * y[0];
    hval[1] = 0.0;
    hval[2] = 2.0 * y[1];
    return 0;
}

// compute dense residual-Hessians-vector products
ipc_ rhessprods_dense( ipc_ n, ipc_ m, ipc_ pne, const rpc_ x[],
                      const rpc_ v[], rpc_ pval[], bool got_h,
                      const void *userdata ){
    pval[0] = 2.0 * v[0];
    pval[1] = 0.0;
    pval[2] = 0.0;
    pval[3] = 2.0 * v[1];
    pval[4] = 0.0;
    pval[5] = 0.0;
    return 0;
}
