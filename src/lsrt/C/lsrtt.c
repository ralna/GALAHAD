/* lsrtt.c */
/* Full test for the LSRT C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_lsrt.h"

int main(void) {

    // Derived types
    void *data;
    struct lsrt_control_type control;
    struct lsrt_inform_type inform;

    // Set problem data
    int n = 50; // dimensions
    int m = 2 * n;

    int status;
    double power = 3.0;
    double weight = 1.0;
    double x[n];
    double u[m];
    double v[n];

    // Initialize lsrt
    lsrt_initialize( &data, &control, &status );

    status = 1;
    control.print_level = 0;
    lsrt_import_control( &control, &data, &status );

    for( int i = 0; i < m; i++) u[i] = 1.0; // b = 1

    // iteration loop to find the minimizer with A^T = (I:diag(1:n))
    while(true){ // reverse-communication loop
      lsrt_solve_problem( &data, &status, m, n, power, weight, x, u, v );
      if ( status == 0 ) { // successful termination
          break;
      } else if ( status < 0 ) { // error exit
          break;
      } else if ( status == 2 ) { // form u <- u + A * v
        for( int i = 0; i < n; i++) {
          u[i] = u[i] + v[i];
          u[n+i] = u[n+i] + (i+1)*v[i];
        }
      } else if ( status == 3 ) { // form v <- v + A^T * u
        for( int i = 0; i < n; i++) v[i] = v[i] + u[i] + (i+1) * u[n+i];
      } else if ( status == 4 ) { // restart
        for( int i = 0; i < m; i++) u[i] = 1.0;
      }else{
          printf(" the value %1i of status should not occur\n", 
            status);
          break;
      }
    }
    lsrt_information( &data, &inform, &status );
    printf("lsrt_solve_problem exit status = %i,"
           " f = %.2f\n", inform.status, inform.obj );
    // Delete internal workspace
    lsrt_terminate( &data, &control, &inform );
}
