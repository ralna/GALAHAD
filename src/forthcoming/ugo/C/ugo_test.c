/* ugo_test.c */
/* Simple code to test the UGO reverse communication C interface */

#include <stdio.h>
#include <math.h>
#include "ugo.h"

// Test problem objective
double objf(double x){
    double a = 10.0;
    return x * x * cos( a*x );
}

// Test problem first derivative
double gradf(double x){
    double a = 10.0;
    return - a * x * x * sin( a*x ) + 2.0 * x * cos( a*x );
}

// Test problem second derivative
double hessf(double x){
    double a = 10.0;
    return - a * a* x * x * cos( a*x ) - 4.0 * a * x * sin( a*x ) + 2.0 * cos( a*x );
}

int main(void) {

    // Derived types
    void *data;
    struct ugo_control_type control;
    struct ugo_inform_type inform;

    // Initialize UGO
    ugo_initialize(&data, &control, &inform);

    // Set user-defined control options
    //control.print_level = 1;
    //control.maxit = 100;
    //control.lipschitz_estimate_used = 3;

    // Read options from specfile
    char specfile[] = "UGO.SPC";
    ugo_read_specfile(&control, specfile);

    // Test problem bounds
    double x_l = -1.0; 
    double x_u = 2.0;

    // Test problem objective, gradient, Hessian values
    double x, f, g, h;

    // Set for initial entry
    inform.status = 1; 

    // Solve the problem: min f(x), x_l <= x <= x_u
    while(true){

        // Call UGO_solve
        ugo_solve(x_l, x_u, &x, &f, &g, &h, &control, &inform, &data, NULL, NULL);

        // Evaluate f(x) and its derivatives as required
        if(inform.status >= 2){ // need objective
            f = objf(x);
            if(inform.status >= 3){ // need first derivative
                g = gradf(x);
                if(inform.status >= 4){ // need second derivative
                    h = hessf(x);
                }
            }
        } else { // the solution has been found (or an error has occured)
            break;
        }
    }

    // Print solution details
    printf("iter: %d \n", inform.iter);
    printf("x: %f \n", x);
    printf("f: %f \n", f);
    printf("g: %f \n", g);
    printf("h: %f \n", h);
    printf("f_eval: %d \n", inform.f_eval);
    printf("time: %f \n", inform.time.clock_total);
    printf("status: %d \n", inform.status);

    // Delete internal workspace
    ugo_terminate(&data, &control, &inform);

    return 0;
}
