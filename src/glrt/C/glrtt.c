/* glrtt.c */
/* Full test for the GLRT C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_glrt.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {

    // Derived types
    void *data;
    struct glrt_control_type control;
    struct glrt_inform_type inform;

    // Set problem data
    ipc_ n = 100; // dimension

    ipc_ status;
    rpc_ weight;
    rpc_ power = 3.0;
    rpc_ x[n];
    rpc_ r[n];
    rpc_ vector[n];
    rpc_ h_vector[n];

    // Initialize glrt
    glrt_initialize( &data, &control, &status );

    // use a unit M ?
    for( ipc_ unit_m=0; unit_m <= 1; unit_m++){
      if ( unit_m == 0 ){
        control.unitm = false;
      } else {
        control.unitm = true;
      }
      // control.print_level = 1;
      glrt_import_control( &control, &data, &status );
      // resolve with a larger weight ?
      for( ipc_ new_weight=0; new_weight <= 1; new_weight++){
        if ( new_weight == 0 ){
           weight = 1.0;
           status = 1;
        } else {
           weight = 10.0;
           status = 6;
        }
        for( ipc_ i = 0; i < n; i++) r[i] = 1.0;

        // iteration loop to find the minimizer
        while(true){ // reverse-communication loop
          glrt_solve_problem( &data, &status, n, power, weight, x, r, vector );
          if ( status == 0 ) { // successful termination
              break;
          } else if ( status < 0 ) { // error exit
              break;
          } else if ( status == 2 ) { // form the preconditioned vector
            for( ipc_ i = 0; i < n; i++) vector[i] = vector[i] / 2.0;
          } else if ( status == 3 ) { // form the Hessian-vector product
            h_vector[0] =  2.0 * vector[0] + vector[1];
            for( ipc_ i = 1; i < n-1; i++){
              h_vector[i] = vector[i-1] + 2.0 * vector[i] + vector[i+1];
            }
            h_vector[n-1] = vector[n-2] + 2.0 * vector[n-1];
            for( ipc_ i = 0; i < n; i++) vector[i] = h_vector[i];
          } else if ( status == 4 ) { // restart
            for( ipc_ i = 0; i < n; i++) r[i] = 1.0;
          }else{
              printf(" the value %1" i_ipc_ " of status should not occur\n",
                status);
              break;
          }
        }
        glrt_information( &data, &inform, &status );
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_glrt.h
#include "galahad_pquad_glrt.h"
//        printf("MR = %1" i_ipc_ "%1" i_ipc_ 
//               " glrt_solve_problem exit status = %" i_ipc_ ", f = %.2e\n", 
//               unit_m, new_weight, inform.status, inform.obj_regularized );
#else
        printf("MR = %1" i_ipc_ "%1" i_ipc_ 
               " glrt_solve_problem exit status = %" i_ipc_ ", f = %.2f\n", 
               unit_m, new_weight, inform.status, inform.obj_regularized );
#endif
      }
    }
   // Delete internal workspace
   glrt_terminate( &data, &control, &inform );
}
