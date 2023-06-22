/* llsrtf.c */
/* Full test for the LLSR C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_llsr.h"

int main(void) {

    // Derived types
    void *data;
    struct llsr_control_type control;
    struct llsr_inform_type inform;
    int i, l;

    // Set problem data
    // set dimensions
    int m = 100;
    int n = 2*m+1;
    // A = ( I : Diag(1:n) : e )
    int A_ne = 3*m;
    int A_row[A_ne];
    int A_col[A_ne];
    int A_ptr[m+1];
    real_wp_ A_val[A_ne];

    // store A in sparse formats
    l=0;
    for( i=0; i < m; i++){
     A_ptr[i] = l;
     A_row[l] = i;
     A_col[l] = i;
     A_val[l] = 1.0;
     l++;
     A_row[l] = i;
     A_col[l] = m+i;
     A_val[l] = i+1;
     l++;
     A_row[l] = i;
     A_col[l] = n-1;
     A_val[l] = 1.0;
     l++;
    }
    A_ptr[m] = l;

    // store A in dense format
    int A_dense_ne = m * n;
    real_wp_ A_dense_val[A_dense_ne];
    for( i=0; i < A_dense_ne; i++) A_dense_val[i] = 0.0;
    l=-1;
    for( i=1; i <= m; i++){
     A_dense_val[l+i] = 1.0;
     A_dense_val[l+m+i] = i;
     A_dense_val[l+n] = 1.0;
     l=l+n;
    }

    // S = diag(1:n)**2
    int S_ne = n;
    int S_row[S_ne];
    int S_col[S_ne];
    int S_ptr[n+1];
    real_wp_ S_val[S_ne];

    // store S in sparse formats
    for( i=0; i < n; i++){
     S_row[i] = i;
     S_col[i] = i;
     S_ptr[i] = i;
     S_val[i] = (i+1)*(i+1);
    }
    S_ptr[n] = n;

    // store S in dense format
    int S_dense_ne = n*(n+1)/2;
    real_wp_ S_dense_val[S_dense_ne];
    for( i=0; i < S_dense_ne; i++) S_dense_val[i] = 0.0;
    l=-1;
    for( i=1; i <= n; i++){
      S_dense_val[l+i] = i*i;
      l=l+i;
    }

   // b is a vector of ones
    real_wp_ b[m]; // observations
    for( i=0; i < m; i++){
      b[i] = 1.0;
    }

   // cubic regularization, weight is one
    real_wp_ power = 3.0;
    real_wp_ weight = 1.0;

   // Set output storage
    real_wp_ x[n]; // solution
    char st;
    int status;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of problem storage formats\n\n");

    // loop over storage formats
    for( int d=1; d<=4; d++){

        // Initialize LLSR
        llsr_initialize( &data, &control, &status );
        strcpy(control.definite_linear_solver, "potr ") ;
        strcpy(control.sbls_control.symmetric_linear_solver, "sytr ") ;
        strcpy(control.sbls_control.definite_linear_solver, "potr ") ;
        // control.print_level = 1;

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        // use s or not (1 or 0)
        for( int use_s=0; use_s<=1; use_s++){
           switch(d){
               case 1: // sparse co-ordinate storage
                   st = 'C';
                   llsr_import( &control, &data, &status, m, n,
                               "coordinate", A_ne, A_row, A_col, NULL );
                   if(use_s == 0){
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_ne, A_val, b, x, 0, NULL );
                   }else{
                      llsr_import_scaling( &control, &data, &status, n,
                                           "coordinate", S_ne, S_row,
                                           S_col, NULL );
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_ne, A_val, b, x, S_ne, S_val );
                   }
                   break;
               case 2: // sparse by rows
                   st = 'R';
                   llsr_import( &control, &data, &status, m, n,
                                "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                   if(use_s == 0){
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_ne, A_val, b, x, 0, NULL );
                   }else{
                      llsr_import_scaling( &control, &data, &status, n,
                                           "sparse_by_rows", S_ne, NULL,
                                           S_col, S_ptr );
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_ne, A_val, b, x, S_ne, S_val );
                   }
                   break;
               case 3: // dense
                   st = 'D';
                   llsr_import( &control, &data, &status, m, n,
                                "dense", A_dense_ne, NULL, NULL, NULL );
                   if(use_s == 0){
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_dense_ne, A_dense_val, b, x,
                                          0, NULL );
                   }else{
                      llsr_import_scaling( &control, &data, &status, n,
                                           "dense", S_dense_ne,
                                           NULL, NULL, NULL );
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_dense_ne, A_dense_val, b, x,
                                          S_dense_ne, S_dense_val );
                   }
                   break;
               case 4: // diagonal
                   st = 'I';
                   llsr_import( &control, &data, &status, m, n,
                                "coordinate", A_ne, A_row, A_col, NULL );
                   if(use_s == 0){
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_ne, A_val, b, x, 0, NULL );
                   }else{
                      llsr_import_scaling( &control, &data, &status, n,
                                           "diagonal", S_ne, NULL, NULL, NULL );
                      llsr_solve_problem( &data, &status, m, n, power, weight,
                                          A_ne, A_val, b, x, S_ne, S_val );
                   }
                   break;
               }
           llsr_information( &data, &inform, &status );

           if(inform.status == 0){
               printf("storage type %c%1i:  status = %1i, ||r|| = %5.2f\n",
                      st, use_s, inform.status, inform.r_norm );
           }else{
               printf("storage type %c%1i: LLSR_solve exit status = %1i\n",
                      st, use_s, inform.status);
           }
        }
        //printf("x: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);
        //printf("\n");

        // Delete internal workspace
        llsr_terminate( &data, &control, &inform );
    }
}