/* l2rtt.c */
/* Full test for the L2RT C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_l2rt.h"

int main(void) {

    // Derived types
    void *data;
    struct l2rt_control_type control;
    struct l2rt_inform_type inform;

    // Set problem data
    int n = 50; // dimensions
    int m = 2 * n;

    int status;
    real_wp_ power = 3.0;
    real_wp_ weight = 1.0;
    real_wp_ shift = 1.0;
    real_wp_ x[n];
    real_wp_ u[m];
    real_wp_ v[n];

    // Initialize l2rt
    l2rt_initialize( &data, &control, &status );

    status = 1;
    control.print_level = 0;
    l2rt_import_control( &control, &data, &status );

    for( int i = 0; i < m; i++) u[i] = 1.0; // b = 1

    // iteration loop to find the minimizer with A^T = (I:diag(1:n))
    while(true){ // reverse-communication loop
      l2rt_solve_problem( &data, &status, m, n, power, weight, shift, x, u, v );
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
    l2rt_information( &data, &inform, &status );
    printf("l2rt_solve_problem exit status = %i,"
           " f = %.2f\n", inform.status, inform.obj );
    // Delete internal workspace
    l2rt_terminate( &data, &control, &inform );
}
