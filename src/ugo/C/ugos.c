/* ugos.c */
/* Spec test for the UGO C interface */

#include <stdio.h>
#include <math.h>
#include "ugo.h"

struct userdata_type {
   double a;
};

// Evaluate test problem objective, first and second derivatives 
int fgh(double x, double *f, double *g, double *h, const void *userdata){
   struct userdata_type *myuserdata = (struct userdata_type *) userdata;
   double a = myuserdata->a;

   *f = x * x * cos( a*x );
   *g = - a * x * x * sin( a*x ) + 2.0 * x * cos( a*x );
   *h = - a * a* x * x * cos( a*x ) - 4.0 * a * x * sin( a*x ) 
        + 2.0 * cos( a*x );
   return 0;
}

int main(void) {

    // Derived types
    void *data;
    struct ugo_control_type control;
    struct ugo_inform_type inform;

    // Initialize UGO

    ugo_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.print_level = 1;
    control.maxit = 100;
    control.lipschitz_estimate_used = 3;

    // User data
    struct userdata_type userdata;
    userdata.a = 10.0;

    // Test problem bounds
    double x_l = -1.0; 
    double x_u = 2.0;     

    // Test problem objective, gradient, Hessian values
    double x, f, g, h;
    int status;

    // import problem data
    ugo_import( &control, &data, &status, &x_l, &x_u );

    // Set for initial entry
    status = 1; 
    
    // Call UGO_solve
    ugo_solve_direct( &data, &userdata, &status, &x, &f, &g, &h, fgh );    

    // Record solution information
    ugo_information( &data, &inform, &status );
    if(inform.status == 0){ // successful return
        printf("UGO successful solve\n");
        printf("iter: %d \n", inform.iter);
        printf("x: %f \n", x);
        printf("f: %f \n", f);
        printf("g: %f \n", g);
        printf("h: %f \n", h);
        printf("f_eval: %d \n", inform.f_eval);
        printf("time: %f \n", inform.time.clock_total);
        printf("status: %d \n", inform.status);
    }else{ // error returns
        printf("UGO error in solve\n");
        printf("status: %d \n", inform.status);
    }

    // Delete internal workspace
    ugo_terminate( &data, &control, &inform );

    return 0;
}
