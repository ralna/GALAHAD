/* l2rtt.c */
/* Full test for the L2RT C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_l2rt.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct l2rt_control_type control;
    struct l2rt_inform_type inform;

    // Set problem data
    ipc_ n = 50; // dimensions
    ipc_ m = 2 * n;

    ipc_ status;
    rpc_ power = 3.0;
    rpc_ weight = 1.0;
    rpc_ shift = 1.0;
    rpc_ x[n];
    rpc_ u[m];
    rpc_ v[n];

    // Initialize l2rt
    l2rt_initialize( &data, &control, &status );

    status = 1;
    control.print_level = 0;
    l2rt_import_control( &control, &data, &status );

    for( ipc_ i = 0; i < m; i++) u[i] = 1.0; // b = 1

    // iteration loop to find the minimizer with A^T = (I:diag(1:n))
    while(true){ // reverse-communication loop
      l2rt_solve_problem( &data, &status, m, n, power, weight, shift, x, u, v );
      if ( status == 0 ) { // successful termination
          break;
      } else if ( status < 0 ) { // error exit
          break;
      } else if ( status == 2 ) { // form u <- u + A * v
        for( ipc_ i = 0; i < n; i++) {
          u[i] = u[i] + v[i];
          u[n+i] = u[n+i] + (i+1)*v[i];
        }
      } else if ( status == 3 ) { // form v <- v + A^T * u
        for( ipc_ i = 0; i < n; i++) v[i] = v[i] + u[i] + (i+1) * u[n+i];
      } else if ( status == 4 ) { // restart
        for( ipc_ i = 0; i < m; i++) u[i] = 1.0;
      }else{
          printf(" the value %1" i_ipc_ " of status should not occur\n",
            status);
          break;
      }
    }
    l2rt_information( &data, &inform, &status );
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_l2rt.h
#include "galahad_pquad_l2rt.h"
#else
    printf("l2rt_solve_problem exit status = %" i_ipc_
           ", f = %.2f\n", inform.status, inform.obj );
#endif
    // Delete internal workspace
    l2rt_terminate( &data, &control, &inform );
}
