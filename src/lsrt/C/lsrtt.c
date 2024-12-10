/* lsrtt.c */
/* Full test for the LSRT C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_lsrt.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct lsrt_control_type control;
    struct lsrt_inform_type inform;

    // Set problem data
    ipc_ n = 50; // dimensions
    ipc_ m = 2 * n;

    ipc_ status;
    rpc_ power = 3.0;
    rpc_ weight = 1.0;
    rpc_ x[n];
    rpc_ u[m];
    rpc_ v[n];

    // Initialize lsrt
    lsrt_initialize( &data, &control, &status );

    status = 1;
    control.print_level = 0;
    lsrt_import_control( &control, &data, &status );

    for( ipc_ i = 0; i < m; i++) u[i] = 1.0; // b = 1

    // iteration loop to find the minimizer with A^T = (I:diag(1:n))
    while(true){ // reverse-communication loop
      lsrt_solve_problem( &data, &status, m, n, power, weight, x, u, v );
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
    lsrt_information( &data, &inform, &status );
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_lsrt.h
#include "galahad_pquad_lsrt.h"
#else
    printf("lsrt_solve_problem exit status = %" i_ipc_
           ", f = %.2f\n", inform.status, inform.obj );
    // Delete internal workspace
    lsrt_terminate( &data, &control, &inform );
#endif
}
