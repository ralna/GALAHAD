/* bllst.c */
/* Full test for the BLLS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_blls.h"

// Define imax
int imax(int a, int b) {
    return (a > b) ? a : b;
};

// Custom userdata struct
struct userdata_type {
   real_wp_ scale;
};

// Function prototypes
int prec( int n, const real_wp_ v[], real_wp_ p[], const void * );

int main(void) {

    // Derived types
    void *data;
    struct blls_control_type control;
    struct blls_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.scale = 1.0;

    // Set problem data
    int n = 10; // dimension
    int o = n + 1; // number of residuals
    int Ao_ne = 2 * n; // sparse Jacobian elements
    int Ao_dense_ne = o * n; // dense Jacobian elements
    // row-wise storage
    int Ao_row[Ao_ne]; // row indices,
    int Ao_col[Ao_ne]; // column indices
    int Ao_ptr_ne = o+1; // number of row pointers
    int Ao_ptr[ Ao_ptr_ne];  // row pointers
    real_wp_ Ao_val[Ao_ne]; // values
    real_wp_ Ao_dense[Ao_dense_ne]; // dense values
    // column-wise storage
    int Ao_by_col_row[Ao_ne]; // row indices,
    int Ao_by_col_ptr_ne = n+1; // number of column pointers
    int Ao_by_col_ptr[Ao_by_col_ptr_ne];  // column pointers
    real_wp_ Ao_by_col_val[Ao_ne]; // values
    real_wp_ Ao_by_col_dense[Ao_dense_ne]; // dense values
    real_wp_ b[o];  // linear term in the objective
    real_wp_ x_l[n]; // variable lower bound
    real_wp_ x_u[n]; // variable upper bound
    real_wp_ x[n]; // variables
    real_wp_ z[n]; // dual variables
    real_wp_ r[o]; // residual
    real_wp_ g[n]; // gradient
    real_wp_ w[o]; // weights

    // Set output storage
    int x_stat[n]; // variable status
    char st[3];
    int i, l, status;

    x_l[0] = -1.0;
    for( int i = 1; i < n; i++) x_l[i] = - INFINITY;
    x_u[0] = 1.0;
    x_u[1] = INFINITY;
    for( int i = 2; i < n; i++) x_u[i] = 2.0;

    //   A = (  I  )  and b = ( i * e )
    //       ( e^T )          ( n + 1 )

    for( int i = 0; i < n; i++) b[i] = i + 1;
    b[n] = n+1;

    //w[0] = 2.0;
    w[0] = 1.0;
    for( int i = 1; i < o; i++) w[i] = 1.0;

    // A by rows

    for( int i = 0; i < n; i++)
    {
      Ao_ptr[i] = i;
      Ao_row[i] = i; Ao_col[i] = i; Ao_val[i] = 1.0;
    }
    Ao_ptr[n] = n;
    for( int i = 0; i < n; i++)
    {
      Ao_row[n+i] = n; Ao_col[n+i] = i; Ao_val[n+i] = 1.0;
    }
    Ao_ptr[o] = Ao_ne;
    l = - 1;
    for( int i = 0; i < n; i++)
    {
      for( int j = 0; j < n; j++)
      {
        l = l + 1;
        if ( i == j ) {
          Ao_dense[l] = 1.0;
        }
        else {
          Ao_dense[l] = 0.0;
        }
      }
    }
    for( int j = 0; j < n; j++)
    {
      l = l + 1;
      Ao_dense[l] = 1.0;
    }

    // A by columns

    l = - 1;
    for( int j = 0; j < n; j++)
    {
      l = l + 1;  Ao_by_col_ptr[j] = l ;
      Ao_by_col_row[l] = j ; Ao_by_col_val[l] = 1.0;
      l = l + 1;
      Ao_by_col_row[l] = n ; Ao_by_col_val[l] = 1.0;
    }
    Ao_by_col_ptr[n] = Ao_ne;
    l = - 1;
    for( int j = 0; j < n; j++)
    {
      for( int i = 0; i < n; i++)
      {
        l = l + 1;
        if ( i == j ) {
          Ao_by_col_dense[l] = 1.0;
        }
        else {
          Ao_by_col_dense[l] = 0.0;
        }
      }
      l = l + 1;
      Ao_by_col_dense[l] = 1.0;
    }

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of blls storage formats\n\n");

    for( int d=1; d <= 5; d++){

        // Initialize BLLS
        blls_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;

        // Start from 0
        for( int i = 0; i < n; i++) x[i] = 0.0;
        for( int i = 0; i < n; i++) z[i] = 0.0;

        switch(d){
            case 1: // sparse co-ordinate storage
                strcpy( st, "CO" );
                blls_import( &control, &data, &status, n, o,
                            "coordinate", Ao_ne, Ao_row, Ao_col, 0, NULL );
                blls_solve_given_a( &data, &userdata, &status, n, o,
                                    Ao_ne, Ao_val, b, x_l, x_u,
                                    x, z, r, g, x_stat, w, prec );
                break;
            case 2: // sparse by rows
                strcpy( st, "SR" );
                blls_import( &control, &data, &status, n, o,
                             "sparse_by_rows", Ao_ne, NULL, Ao_col,
                             Ao_ptr_ne, Ao_ptr );
                blls_solve_given_a( &data, &userdata, &status, n, o,
                                    Ao_ne, Ao_val, b, x_l, x_u,
                                    x, z, r, g, x_stat, w, prec );
                break;
            case 3: // dense by rows
                strcpy( st, "DR" );
                blls_import( &control, &data, &status, n, o,
                             "dense_by_rows", Ao_dense_ne,
                              NULL, NULL,0,  NULL );
                blls_solve_given_a( &data, &userdata, &status, n, o,
                                    Ao_dense_ne, Ao_dense, b, x_l, x_u,
                                    x, z, r, g, x_stat, w, prec );
                break;
            case 4: // sparse by columns
                strcpy( st, "SC" );
                blls_import( &control, &data, &status, n, o,
                             "sparse_by_columns", Ao_ne, Ao_by_col_row,
                             NULL, Ao_by_col_ptr_ne, Ao_by_col_ptr );
                blls_solve_given_a( &data, &userdata, &status, n, o,
                                    Ao_ne, Ao_by_col_val, b, x_l, x_u,
                                    x, z, r, g, x_stat, w, prec );
                break;
            case 5: // dense by columns
                strcpy( st, "DC" );
                blls_import( &control, &data, &status, n, o,
                             "dense_by_columns", Ao_dense_ne,
                             NULL, NULL, 0, NULL);
                blls_solve_given_a( &data, &userdata, &status, n, o,
                                    Ao_dense_ne, Ao_by_col_dense, b, x_l, x_u,
                                    x, z, r, g, x_stat, w, prec );
                break;
            }
        blls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%s:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%s: BLLS_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        blls_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    int on;
    on = imax( o, n );
    int eval_status, nz_v_start, nz_v_end, nz_p_end;
    int nz_v[on], nz_p[o], mask[o];
    real_wp_ v[on], p[on];

    nz_p_end = 0;

    // Initialize BLLS
    blls_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing
    // control.print_level = 1;

    // Start from 0
    for( int i = 0; i < n; i++) x[i] = 0.0;
    for( int i = 0; i < n; i++) z[i] = 0.0;

    strcpy( st, "RC" );
    for( int i = 0; i < o; i++) mask[i] = 0;
    blls_import_without_a( &control, &data, &status, n, o ) ;
    while(true){ // reverse-communication loop
        blls_solve_reverse_a_prod( &data, &status, &eval_status, n, o, b,
                                   x_l, x_u, x, z, r, g, x_stat, v, p,
                                   nz_v, &nz_v_start, &nz_v_end,
                                   nz_p, nz_p_end, w );
        if(status == 0){ // successful termination
            break;
        }else if(status < 0){ // error exit
            break;
        }else if(status == 2){ // evaluate p = Av
          p[n]=0.0;
          for( int i = 0; i < n; i++){
            p[i] = v[i];
            p[n] = p[n] + v[i];
          }
        }else if(status == 3){ // evaluate p = A^Tv
          for( int i = 0; i < n; i++) p[i] = v[i] + v[n];
        }else if(status == 4){ // evaluate p = Av for sparse v
          p[n]=0.0;
          for( int i = 0; i < n; i++) p[i] = 0.0;
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
            i = nz_v[l];
            p[i] = v[i];
            p[n] = p[n] + v[i];
          }
        }else if(status == 5){ // evaluate p = sparse Av for sparse v
          nz_p_end = 0;
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
            i = nz_v[l];
            if (mask[i] == 0){
              mask[i] = 1;
              nz_p[nz_p_end] = i;
              nz_p_end = nz_p_end + 1;
              p[i] = v[i];
            }
            if (mask[n] == 0){
              mask[n] = 1;
              nz_p[nz_p_end] = n;
              nz_p_end = nz_p_end + 1;
              p[n] = v[i];
            }else{
              p[n] = p[n] + v[i];
            }
          }
          for( int l = 0; l < nz_p_end; l++) mask[nz_p[l]] = 0;
        }else if(status == 6){ // evaluate p = sparse A^Tv
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
            i = nz_v[l];
            p[i] = v[i] + v[n];
          }
        }else if(status == 7){ // evaluate p = P^{-}v
          for( int i = 0; i < n; i++) p[i] = userdata.scale * v[i];
        }else{
            printf(" the value %1i of status should not occur\n", status);
            break;
        }
        eval_status = 0;
    }

    // Record solution information
    blls_information( &data, &inform, &status );

    // Print solution details
    if(inform.status == 0){
        printf("%s:%6i iterations. Optimal objective value = %5.2f"
               " status = %1i\n",
               st, inform.iter, inform.obj, inform.status);
    }else{
        printf("%s: BLLS_solve exit status = %1i\n", st, inform.status);
    }
    //printf("x: ");
    //for( int i = 0; i < n; i++) printf("%f ", x[i]);
    //printf("\n");
    //printf("gradient: ");
    //for( int i = 0; i < n; i++) printf("%f ", g[i]);
    //printf("\n");

    // Delete internal workspace
    blls_terminate( &data, &control, &inform );
}

// Apply preconditioner
int prec( int n, const real_wp_ v[], real_wp_ p[], const void *userdata ){
  struct userdata_type *myuserdata = (struct userdata_type *) userdata;
  real_wp_ scale = myuserdata->scale;
  for( int i = 0; i < n; i++) p[i] = scale * v[i];
   return 0;
}
