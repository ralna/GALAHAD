/* bqptf.c */
/* Full test for the BQP C interface using fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_bqp.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct bqp_control_type control;
    struct bqp_inform_type inform;

    // Set problem data
    ipc_ n = 10; // dimension
    ipc_ H_ne = 2 * n - 1; // Hesssian elements, NB lower triangle
    ipc_ H_dense_ne = n * ( n + 1 ) / 2; // dense Hessian elements
    ipc_ H_row[H_ne]; // row indices,
    ipc_ H_col[H_ne]; // column indices
    ipc_ H_ptr[n+1];  // row pointers
    rpc_ H_val[H_ne]; // values
    rpc_ H_dense[H_dense_ne]; // dense values
    rpc_ H_diag[n];   // diagonal values
    rpc_ g[n];  // linear term in the objective
    rpc_ f = 1.0;  // constant term in the objective
    rpc_ x_l[n]; // variable lower bound
    rpc_ x_u[n]; // variable upper bound
    rpc_ x[n]; // variables
    rpc_ z[n]; // dual variables

    // Set output storage
    ipc_ x_stat[n]; // variable status
    char st = ' ';
    ipc_ i, l, status;

    g[0] = 2.0;
    for( ipc_ i = 1; i < n; i++) g[i] = 0.0;
    x_l[0] = -1.0;
    for( ipc_ i = 1; i < n; i++) x_l[i] = - INFINITY;
    x_u[0] = 1.0;
    x_u[1] = INFINITY;
    for( ipc_ i = 2; i < n; i++) x_u[i] = 2.0;

    // H = tridiag(2,1), H_dense = diag(2)

    l = 0 ;
    H_ptr[0] = l + 1;
    H_row[l] = 1; H_col[l] = 1; H_val[l] = 2.0;
    for( ipc_ i = 1; i < n; i++)
    {
      l = l + 1;
      H_ptr[i] = l + 1;
      H_row[l] = i + 1; H_col[l] = i; H_val[l] = 1.0;
      l = l + 1;
      H_row[l] = i + 1; H_col[l] = i + 1; H_val[l] = 2.0;
    }
    H_ptr[n] = l + 2;

    l = - 1;
    for( ipc_ i = 0; i < n; i++)
    {
      H_diag[i] = 2.0;
      for( ipc_ j = 0; j <= i; j++)
      {
        l = l + 1;
        if ( j < i - 1 ) {
          H_dense[l] = 0.0;
        }
        else if ( j == i - 1 ) {
          H_dense[l] = 1.0;
        }
        else {
          H_dense[l] = 2.0;
        }
      }
    }

    printf(" fortran sparse matrix indexing\n\n");

    printf(" basic tests of bqp storage formats\n\n");

    for( ipc_ d=1; d <= 4; d++){

        // Initialize BQP
        bqp_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // fortran sparse matrix indexing

        // Start from 0
        for( ipc_ i = 0; i < n; i++) x[i] = 0.0;
        for( ipc_ i = 0; i < n; i++) z[i] = 0.0;

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                bqp_import( &control, &data, &status, n,
                            "coordinate", H_ne, H_row, H_col, NULL );
                bqp_solve_given_h( &data, &status, n, H_ne, H_val, g, f,
                                   x_l, x_u, x, z, x_stat );
                break;
            printf(" case %1" i_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                bqp_import( &control, &data, &status, n,
                             "sparse_by_rows", H_ne, NULL, H_col, H_ptr );
                bqp_solve_given_h( &data, &status, n, H_ne, H_val, g, f,
                                   x_l, x_u, x, z, x_stat );
                break;
            case 3: // dense
                st = 'D';
                bqp_import( &control, &data, &status, n,
                             "dense", H_dense_ne, NULL, NULL, NULL );
                bqp_solve_given_h( &data, &status, n, H_dense_ne, H_dense,
                                   g, f, x_l, x_u, x, z, x_stat );
                break;
            case 4: // diagonal
                st = 'L';
                bqp_import( &control, &data, &status, n,
                             "diagonal", H_ne, NULL, NULL, NULL );
                bqp_solve_given_h( &data, &status, n, n, H_diag, g, f,
                                   x_l, x_u, x, z, x_stat );
                break;
            }
        bqp_information( &data, &inform, &status );

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
            printf("%c: BQP_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        //printf("x: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        bqp_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    ipc_ nz_v_start, nz_v_end, nz_prod_end;
    ipc_ nz_v[n], nz_prod[n], mask[n];
    rpc_ v[n], prod[n];

    nz_prod_end = 0;

    // Initialize BQP
    bqp_initialize( &data, &control, &status );
   // control.print_level = 1;

   // Set user-defined control options
    control.f_indexing = true; // fortran sparse matrix indexing

    // Start from 0
    for( ipc_ i = 0; i < n; i++) x[i] = 0.0;
    for( ipc_ i = 0; i < n; i++) z[i] = 0.0;

    st = 'I';
    for( ipc_ i = 0; i < n; i++) mask[i] = 0;
    bqp_import_without_h( &control, &data, &status, n ) ;
    while(true){ // reverse-communication loop
        bqp_solve_reverse_h_prod( &data, &status, n, g, f, x_l, x_u,
                                  x, z, x_stat, v, prod,
                                  nz_v, &nz_v_start, &nz_v_end,
                                  nz_prod, nz_prod_end );
        if(status == 0){ // successful termination
            break;
        }else if(status < 0){ // error exit
            break;
        }else if(status == 2){ // evaluate Hv
          prod[0] = 2.0 * v[0] + v[1];
          for( ipc_ i = 1; i < n-1; i++) prod[i] = 2.0 * v[i] + v[i-1] + v[i+1];
          prod[n-1] = 2.0 * v[n-1] + v[n-2];
        }else if(status == 3){ // evaluate Hv for sparse v
          for( ipc_ i = 0; i < n; i++) prod[i] = 0.0;
          for( ipc_ l = nz_v_start - 1; l < nz_v_end; l++){
             i = nz_v[l]-1;
             if (i > 0) prod[i-1] = prod[i-1] + v[i];
             prod[i] = prod[i] + 2.0 * v[i];
             if (i < n-1) prod[i+1] = prod[i+1] + v[i];
           }
        }else if(status == 4){ // evaluate sarse Hv for sparse v
          nz_prod_end = 0;
          for( ipc_ l = nz_v_start - 1; l < nz_v_end; l++){
             i = nz_v[l]-1;
             if (i > 0){
               if (mask[i-1] == 0){
                 mask[i-1] = 1;
                 nz_prod[nz_prod_end] = i - 1;
                 nz_prod_end = nz_prod_end + 1;
                 prod[i-1] = v[i];
               }else{
                 prod[i-1] = prod[i-1] + v[i];
               }
             }
             if (mask[i] == 0){
               mask[i] = 1;
               nz_prod[nz_prod_end] = i;
               nz_prod_end = nz_prod_end + 1;
               prod[i] = 2.0 * v[i];
             }else{
               prod[i] = prod[i] + 2.0 * v[i];
             }
             if (i < n-1){
               if (mask[i+1] == 0){
                 mask[i+1] = 1;
                 nz_prod[nz_prod_end] = i + 1;
                 nz_prod_end = nz_prod_end + 1;
               }
               prod[i+1] = prod[i+1] + v[i];
             }
          }
          for( ipc_ l = 0; l < nz_prod_end; l++) mask[nz_prod[l]] = 0;
        }else{
            printf(" the value %1" i_ipc_ " of status should not occur\n", 
                   status);
            break;
        }
    }

    // Record solution information
    bqp_information( &data, &inform, &status );

    // Print solution details
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
        printf("%c: BQP_solve exit status = %1" i_ipc_ "\n", 
                st, inform.status);
    }
    //printf("x: ");
    //for( ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
    //printf("\n");
    //printf("gradient: ");
    //for( ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
    //printf("\n");

    // Delete internal workspace
    bqp_terminate( &data, &control, &inform );
}
