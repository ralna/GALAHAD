/* slst.c */
/* Full test for the SLS C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "sls.h"

int maxabsarray(double a[],int n, double *maxabs);

int main(void) {

    // Derived types
    void *data;
    struct sls_control_type control;
    struct sls_inform_type inform;

    // Set problem data
    int n = 5; // dimension of A
    int ne = 7; // number of entries of A
    int dense_ne = 15; // number of elements of A as a dense matrix
    int row[] = {0, 1, 1, 2, 2, 3, 4}; // row indices, NB lower triangle
    int col[] = {0, 0, 4, 1, 2, 2, 4}; // column indices
    int ptr[] = {0, 1, 3, 5, 6, 7}; // pointers to indices
    double val[] = {2.0, 3.0, 6.0, 4.0,  1.0, 5.0, 1.0}; // values
    double dense[] = {2.0, 3.0, 0.0, 0.0, 4.0, 1.0, 0.0, 
                      0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 1.0};
    double rhs[] = {8.0, 45.0, 31.0, 15.0, 17.0};
    double sol[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int i, status;
    double x[n];
    double error[n];

    double norm_residual;
    double good_x = pow( DBL_EPSILON, 0.3333 );

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of storage formats\n\n");

    printf(" storage          RHS   refine  partial\n");
    for( int d=1; d <= 3; d++){

        // Initialize SLS - use the sils solver
        sls_initialize( "sils", &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        switch(d){ // import matrix data and factorize
            case 1: // sparse co-ordinate storage
                printf(" coordinate     ");
                sls_analyse_matrix( &control, &data, &status, n,
                                    "coordinate", ne, row, col, NULL );
                sls_factorize_matrix( &data, &status, ne, val );
                break;
            case 2: // sparse by rows
                printf(" sparse by rows ");
                sls_analyse_matrix( &control, &data, &status, n, 
                                    "sparse_by_rows", ne, NULL, col, ptr );
                sls_factorize_matrix( &data, &status, ne, val );
                break;
            case 3: // dense
                printf(" dense          ");
                sls_analyse_matrix( &control, &data, &status, n,
                                    "dense", ne, NULL, NULL, NULL );
                sls_factorize_matrix( &data, &status, dense_ne, dense );
                break;
            }

        // Set right-hand side and solve the system
        for(i=0; i<n; i++) x[i] = rhs[i];
        sls_solve_system( &data, &status, n, x );
        sls_information( &data, &inform, &status );

        if(inform.status == 0){
          for(i=0; i<n; i++) error[i] = x[i]-sol[i];
          status = maxabsarray( error, n, &norm_residual );
          if(norm_residual < good_x){
            printf("   ok  ");
          }else{
            printf("  fail ");
          }
        }else{
            printf(" SLS_solve exit status = %1i\n", inform.status);
        }
        //printf("sol: ");
        //for( int i = 0; i < n; i++) printf("%f ", x[i]);

        // resolve, this time using iterative refinement
        control.max_iterative_refinements = 1;
        sls_reset_control( &control, &data, &status );
        for(i=0; i<n; i++) x[i] = rhs[i];
        sls_solve_system( &data, &status, n, x );
        sls_information( &data, &inform, &status );

        if(inform.status == 0){
          for(i=0; i<n; i++) error[i] = x[i]-sol[i];
          status = maxabsarray( error, n, &norm_residual );
          if(norm_residual < good_x){
            printf("    ok  ");
          }else{
            printf("   fail ");
          }
        }else{
            printf(" SLS_solve exit status = %1i\n", inform.status);
        }

        // obtain the solution by part solves
        for(i=0; i<n; i++) x[i] = rhs[i];
        sls_partial_solve_system( "L", &data, &status, n, x );
        sls_partial_solve_system( "D", &data, &status, n, x );
        sls_partial_solve_system( "U", &data, &status, n, x );
        sls_information( &data, &inform, &status );

        if(inform.status == 0){
          for(i=0; i<n; i++) error[i] = x[i]-sol[i];
          status = maxabsarray( error, n, &norm_residual );
          if(norm_residual < good_x){
            printf("    ok  ");
          }else{
            printf("   fail ");
          }
        }else{
            printf(" SLS_solve exit status = %1i\n", inform.status);
        }

        // Delete internal workspace
        sls_terminate( &data, &control, &inform );
        printf("\n");
    }
}

int maxabsarray(double a[],int n, double *maxabs)
 {
    int i;
    double b,max;
    max=abs(a[0]);
    for(i=1; i<n; i++)
    {
        b = abs(a[i]);
	if(max<b)
          max=b;       
    }
    *maxabs=max;
 }
