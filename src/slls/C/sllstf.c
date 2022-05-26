/* sllstf.c */
/* Full test for the SLLS C interface using fortran sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_slls.h"

// define max

#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

// Custom userdata struct
struct userdata_type {
   double scale;
};

// Function prototypes
int prec( int n, const double v[], double p[], const void * );

int main(void) {

    // Derived types
    void *data;
    struct slls_control_type control;
    struct slls_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.scale = 1.0;

    // Set problem data
    int n = 10; // dimension
    int m = n + 1; // number of residuals
    int A_ne = 2 * n; // sparse Jacobian elements
    int A_dense_ne = m * n; // dense Jacobian elements
    // row-wise storage
    int A_row[A_ne]; // row indices,
    int A_col[A_ne]; // column indices
    int A_ptr[m+1];  // row pointers
    double A_val[A_ne]; // values
    double A_dense[A_dense_ne]; // dense values
    // column-wise storage
    int A_by_col_row[A_ne]; // row indices,
    int A_by_col_ptr[n+1];  // column pointers
    double A_by_col_val[A_ne]; // values
    double A_by_col_dense[A_dense_ne]; // dense values
    double b[m];  // linear term in the objective
    double x[n]; // variables
    double z[n]; // dual variables
    double c[m]; // residual
    double g[n]; // gradient

    // Set output storage
    int x_stat[n]; // variable status
    char st[3];
    int i, l, status;

    //   A = (  I  )  and b = ( i * e )
    //       ( e^T )          ( n + 1 )

    for( int i = 0; i < n; i++) b[i] = i + 1;
    b[n] = n+1;

    // A by rows

    for( int i = 0; i < n; i++)
    {
      A_ptr[i] = i + 1;
      A_row[i] = i + 1; A_col[i] = i + 1; A_val[i] = 1.0;
    }
    A_ptr[n] = n + 1;
    for( int i = 0; i < n; i++)
    {
      A_row[n+i] = m; A_col[n+i] = i + 1; A_val[n+i] = 1.0;
    }
    A_ptr[m] = A_ne + 1;
    l = - 1;
    for( int i = 0; i < n; i++)
    {
      for( int j = 0; j < n; j++)
      {
        l = l + 1;
        if ( i == j ) {
          A_dense[l] = 1.0;
        }
        else {
          A_dense[l] = 0.0;
        }
      }
    }
    for( int j = 0; j < n; j++)
    {
      l = l + 1;
      A_dense[l] = 1.0;
    }

    // A by columns

    l = - 1;
    for( int j = 0; j < n; j++)
    {
      l = l + 1;  A_by_col_ptr[j] = l + 1;
      A_by_col_row[l] = j + 1; A_by_col_val[l] = 1.0;
      l = l + 1;
      A_by_col_row[l] = m; A_by_col_val[l] = 1.0;
    }
    A_by_col_ptr[n] = A_ne;
    l = - 1;
    for( int j = 0; j < n; j++)
    {
      for( int i = 0; i < n; i++)
      {
        l = l + 1;
        if ( i == j ) {
          A_by_col_dense[l] = 1.0;
        }
        else {
          A_by_col_dense[l] = 0.0;
        }
      }
      l = l + 1;
      A_by_col_dense[l] = 1.0;
    }

    printf(" fortran sparse matrix indexing\n\n");

    printf(" basic tests of slls storage formats\n\n");

    for( int d=1; d <= 5; d++){

        // Initialize SLLS
        slls_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = true; // fortran sparse matrix indexing

        // Start from 0
        for( int i = 0; i < n; i++) x[i] = 0.0;
        for( int i = 0; i < n; i++) z[i] = 0.0;

        switch(d){
            case 1: // sparse co-ordinate storage
                strcpy( st, "CO" );
                slls_import( &control, &data, &status, n, m,
                            "coordinate", A_ne, A_row, A_col, NULL );
                slls_solve_given_a( &data, &userdata, &status, n, m,
                                    A_ne, A_val, b,
                                    x, z, c, g, x_stat, prec );
                break;
            case 2: // sparse by rows
                strcpy( st, "SR" );
                slls_import( &control, &data, &status, n, m,
                             "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                slls_solve_given_a( &data, &userdata, &status, n, m,
                                    A_ne, A_val, b,
                                    x, z, c, g, x_stat, prec );
                break;
            case 3: // dense by rows
                strcpy( st, "DR" );
                slls_import( &control, &data, &status, n, m,
                             "dense_by_rows", A_dense_ne, NULL, NULL, NULL );
                slls_solve_given_a( &data, &userdata, &status, n, m,
                                    A_dense_ne, A_dense, b,
                                    x, z, c, g, x_stat, prec );
                break;
            case 4: // sparse by columns
                strcpy( st, "SC" );
                slls_import( &control, &data, &status, n, m,
                             "sparse_by_columns", A_ne, A_by_col_row,
                             NULL, A_by_col_ptr );
                slls_solve_given_a( &data, &userdata, &status, n, m,
                                    A_ne, A_by_col_val, b,
                                    x, z, c, g, x_stat, prec );
                break;
            case 5: // dense by columns
                strcpy( st, "DC" );
                slls_import( &control, &data, &status, n, m,
                             "dense_by_columns", A_dense_ne, NULL, NULL, NULL );
                slls_solve_given_a( &data, &userdata, &status, n, m,
                                    A_dense_ne, A_by_col_dense, b,
                                    x, z, c, g, x_stat, prec );
                break;
            }
        slls_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%s:%6i iterations. Optimal objective value = %5.2f"
                   " status = %1i\n",
                   st, inform.iter, inform.obj, inform.status);
        }else{
            printf("%s: SLLS_solve exit status = %1i\n", st, inform.status);
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");
        //printf("gradient: ");
        //for( int i = 0; i < n; i++) printf("%f ", g[i]);
        //printf("\n");

        // Delete internal workspace
        slls_terminate( &data, &control, &inform );
    }

    printf("\n tests reverse-communication options\n\n");

    // reverse-communication input/output
    int nm;
    nm = max( n, m );
    int eval_status, nz_v_start, nz_v_end, nz_p_end;
    int nz_v[nm], nz_p[m], mask[m];
    double v[nm], p[nm];

    nz_p_end = 0;

    // Initialize SLLS
    slls_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = true; // fortran sparse matrix indexing

    // Start from 0
    for( int i = 0; i < n; i++) x[i] = 0.0;
    for( int i = 0; i < n; i++) z[i] = 0.0;

    strcpy( st, "RC" );
    for( int i = 0; i < m; i++) mask[i] = 0;
    slls_import_without_a( &control, &data, &status, n, m ) ;
    while(true){ // reverse-communication loop
        slls_solve_reverse_a_prod( &data, &status, &eval_status, n, m, b,
                                   x, z, c, g, x_stat, v, p,
                                   nz_v, &nz_v_start, &nz_v_end,
                                   nz_p, nz_p_end );
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
            i = nz_v[l]-1;
            p[i] = v[i];
            p[n] = p[n] + v[i];
          }
        }else if(status == 5){ // evaluate p = sparse Av for sparse v
          nz_p_end = 0;
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
            i = nz_v[l]-1;
            if (mask[i] == 0){
              mask[i] = 1;
              nz_p[nz_p_end] = i+1;
              nz_p_end = nz_p_end + 1;
              p[i] = v[i];
            }
            if (mask[n] == 0){
              mask[n] = 1;
              nz_p[nz_p_end] = m;
              nz_p_end = nz_p_end + 1;
              p[n] = v[i];
            }else{
              p[n] = p[n] + v[i];
            }
          }
          for( int l = 0; l < nz_p_end; l++) mask[nz_p[l]] = 0;
        }else if(status == 6){ // evaluate p = sparse A^Tv
          for( int l = nz_v_start - 1; l < nz_v_end; l++){
            i = nz_v[l]-1;
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
    slls_information( &data, &inform, &status );

    // Print solution details
    if(inform.status == 0){
        printf("%s:%6i iterations. Optimal objective value = %5.2f"
               " status = %1i\n",
               st, inform.iter, inform.obj, inform.status);
    }else{
        printf("%s: SLLS_solve exit status = %1i\n", st, inform.status);
    }
    //printf("x: ");
    //for( int i = 0; i < n; i++) printf("%f ", x[i]);
    //printf("\n");
    //printf("gradient: ");
    //for( int i = 0; i < n; i++) printf("%f ", g[i]);
    //printf("\n");

    // Delete internal workspace
    slls_terminate( &data, &control, &inform );
}

// Apply preconditioner
int prec( int n, const double v[], double p[], const void *userdata ){
  struct userdata_type *myuserdata = (struct userdata_type *) userdata;
  double scale = myuserdata->scale;
  for( int i = 0; i < n; i++) p[i] = scale * v[i];
   return 0;
}
