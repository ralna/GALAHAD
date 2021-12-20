/* gltrt.c */
/* Full test for the GLTR C interface */

#include <stdio.h>
#include <math.h>
#include "gltr.h"

int main(void) {

    // Derived types
    void *data;
    struct gltr_control_type control;
    struct gltr_inform_type inform;

    // Set problem data
    int n = 100; // dimension

    int status;
    double radius;
    double x[n];
    double r[n];
    double vector[n];
    double h_vector[n];

    // Initialize gltr
    gltr_initialize( &data, &control, &status );

    // use a unit M ?
    for( int unit_m=0; unit_m <= 1; unit_m++){
      if ( unit_m == 0 ){
        control.unitm = false;
      } else {
        control.unitm = true;
      }
      gltr_import_control( &control, &data, &status );

      // resolve with a smaller radius ?
      for( int new_radius=0; new_radius <= 1; new_radius++){
        if ( new_radius == 0 ){
           radius = 1.0; 
           status = 1;
        } else {
           radius = 0.1; 
           status = 4;
        }
        for( int i = 0; i < n; i++) r[i] = 1.0;

        // iteration loop to find the minimizer
        while(true){ // reverse-communication loop
          gltr_solve_problem( &data, &status, n, radius, x, r, vector );
          if ( status == 0 ) { // successful termination
              break;
          } else if ( status < 0 ) { // error exit
              break;
          } else if ( status == 2 ) { // form the preconditioned vector
            for( int i = 0; i < n; i++) vector[i] = vector[i] / 2.0;
          } else if ( status == 3 ) { // form the Hessian-vector product
            h_vector[0] =  2.0 * vector[0] + vector[1];
            for( int i = 1; i < n-1; i++){
              h_vector[i] = vector[i-1] + 2.0 * vector[i] + vector[i+1];
            }
            h_vector[n-1] = vector[n-2] + 2.0 * vector[n-1];
            for( int i = 0; i < n; i++) vector[i] = h_vector[i];
          } else if ( status == 5 ) { // restart
            for( int i = 0; i < n; i++) r[i] = 1.0;
          }else{
              printf(" the value %1i of status should not occur\n", 
                status);
              break;
          }
        }
        gltr_information( &data, &inform, &status );
        printf("MR = %1i%1i gltr_solve_problem exit status = %i,"
             " f = %.2f\n", unit_m, new_radius, inform.status, inform.obj );
      }
    }
   // Delete internal workspace
   gltr_terminate( &data, &control, &inform );
}
