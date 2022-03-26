/* bqpt.c */
/* Full test for the BQP C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_bqp.h"

int main(void) {

    // Derived types
    void *data;
    struct bqp_control_type control;
    struct bqp_inform_type inform;

    // Set problem data
    int n = 10; // dimension
    int H_ne = 2 * n - 1; // Hesssian elements, NB lower triangle
    int H_dense_ne = n * ( n + 1 ) / 2; // dense Hessian elements
    int H_row[H_ne]; // row indices, 
    int H_col[H_ne]; // column indices
    int H_ptr[n+1];  // row pointers
    double H_val[H_ne]; // values
    double H_dense[H_dense_ne]; // dense values
    double H_diag[n];   // diagonal values
    double g[n];  // linear term in the objective
    double f = 1.0;  // constant term in the objective
    double x_l[n]; // variable lower bound
    double x_u[n]; // variable upper bound
    double x[n]; // variables
    double z[n]; // dual variables

    // Set output storage
    int x_stat[n]; // variable status
    char st;
    int i, l, status;

    g[0] = 2.0;
    for( int i = 1; i < n; i++) g[i] = 0.0;
    x_l[0] = -1.0;
    for( int i = 1; i < n; i++) x_l[i] = - INFINITY;
    x_u[0] = 1.0;
    x_u[1] = INFINITY;
    for( int i = 2; i < n; i++) x_u[i] = 2.0;

    // H = tridiag(2,1), H_dense = diag(2)

    l = 0 ; 
    H_ptr[0] = l;
    H_row[l] = 0; H_col[l] = 0; H_val[l] = 2.0;
    for( int i = 1; i < n; i++)
    {
      l = l + 1; 
      H_ptr[i] = l;
      H_row[l] = i; H_col[l] = i - 1; H_val[l] = 1.0;
      l = l + 1;
      H_row[l] = i; H_col[l] = i; H_val[l] = 2.0;
    }
    H_ptr[n] = l + 1;

    l = - 1 ; 
    for( int i = 0; i < n; i++)
    {
      H_diag[i] = 2.0;
      for( int j = 0; j <= i; j++)
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

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of bqp storage formats\n\n");

    for( int d=1; d <= 4; d++){

        // Initialize BQP
        bqp_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        // Start from 0
        for( int i = 0; i < n; i++) x[i] = 0.0;
        for( int i = 0; i < n; i++) z[i] = 0.0;

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                bqp_import( &control, &data, &status, n,
                            "coordinate", H_ne, H_row, H_col, NULL );
                bqp_solve_given_h( &data, &status, n, H_ne, H_val, g, f, 
                                   x_l, x_u, x, z, x_stat );
                break;
            printf(" case %1i break\n",d);
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
            printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%c: BQP_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        bqp_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    int nz_v_start, nz_v_end, nz_prod_end;
    int nz_v[n], nz_prod[n], mask[n];
    double v[n], prod[n];

    nz_prod_end = 0;

    // Initialize BQP
    bqp_initialize( &data, &control, &status );
   // control.print_level = 1;

   // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing

    // Start from 0
    for( int i = 0; i < n; i++) x[i] = 0.0;
    for( int i = 0; i < n; i++) z[i] = 0.0;

    st = 'I';
    for( int i = 0; i < n; i++) mask[i] = 0;
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
          for( int i = 1; i < n-1; i++) prod[i] = 2.0 * v[i] + v[i-1] + v[i+1];
          prod[n-1] = 2.0 * v[n-1] + v[n-2];
        }else if(status == 3){ // evaluate Hv for sparse v
          for( int i = 0; i < n; i++) prod[i] = 0.0;
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
             i = nz_v[l];
             if (i > 0) prod[i-1] = prod[i-1] + v[i];
             prod[i] = prod[i] + 2.0 * v[i];
             if (i < n-1) prod[i+1] = prod[i+1] + v[i];
           }
        }else if(status == 4){ // evaluate sarse Hv for sparse v
          nz_prod_end = 0;
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
             i = nz_v[l];
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
                 prod[i+1] = prod[i+1] + v[i];
               }else{
                 prod[i+1] = prod[i+1] + v[i];
               }
             }
          }
          for( int l = 0; l < nz_prod_end; l++) mask[nz_prod[l]] = 0;
        }else{
            printf(" the value %1i of status should not occur\n", status);
            break;
        }
    }

    // Record solution information
    bqp_information( &data, &inform, &status );

    // Print solution details
    if(inform.status == 0){
        printf("%c:%6i iterations. Optimal objective value = %5.2f status = %1i\n", 
               st, inform.iter, inform.obj, inform.status);
    }else{
        printf("%c: BQP_solve exit status = %1i\n", st, inform.status);
    }
    //printf("x: ");
    //for( int i = 0; i < n; i++) printf("%f ", x[i]);
    //printf("\n");
    //printf("gradient: ");
    //for( int i = 0; i < n; i++) printf("%f ", g[i]);
    //printf("\n");

    // Delete internal workspace
    bqp_terminate( &data, &control, &inform );
}
