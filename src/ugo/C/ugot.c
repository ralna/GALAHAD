/* ugo_test.c */
/* Simple code to test the UGO reverse communication C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_ugo.h"
#include <string.h>

// Test problem objective
real_wp_ objf(real_wp_ x){
    real_wp_ a = 10.0;
    return x * x * cos( a*x );
}

// Test problem first derivative
real_wp_ gradf(real_wp_ x){
    real_wp_ a = 10.0;
    return - a * x * x * sin( a*x ) + 2.0 * x * cos( a*x );
}

// Test problem second derivative
real_wp_ hessf(real_wp_ x){
    real_wp_ a = 10.0;
    return - a * a* x * x * cos( a*x ) - 4.0 * a * x * sin( a*x )
            + 2.0 * cos( a*x );
}

int main(void) {

    // Derived types
    void *data;
    struct ugo_control_type control;
    struct ugo_inform_type inform;

    // Initialize UGO
    ipc_ status, eval_status;
    ugo_initialize( &data, &control, &status );

    // Set user-defined control options
    control.print_level = 1;
    //control.maxit = 100;
    //control.lipschitz_estimate_used = 3;
    strcpy(control.prefix, "'ugo: '");

    // Read options from specfile
    char specfile[] = "UGO.SPC";
    ugo_read_specfile(&control, specfile);

    // Test problem bounds
    real_wp_ x_l = -1.0;
    real_wp_ x_u = 2.0;

    // Test problem objective, gradient, Hessian values
    real_wp_ x, f, g, h;

    // import problem data
    ugo_import( &control, &data, &status, &x_l, &x_u );

    // Set for initial entry
    status = 1;

    // Solve the problem: min f(x), x_l <= x <= x_u
    while(true){

        // Call UGO_solve
        ugo_solve_reverse(&data, &status, &eval_status, &x, &f, &g, &h );

        // Evaluate f(x) and its derivatives as required
        if(status >= 2){ // need objective
            f = objf(x);
            if(status >= 3){ // need first derivative
                g = gradf(x);
                if(status >= 4){ // need second derivative
                    h = hessf(x);
                }
            }
        } else { // the solution has been found (or an error has occured)
            break;
        }
    }
    // Record solution information
    ugo_information( &data, &inform, &status );

    if(inform.status == 0){
        printf("%i evaluations. Optimal objective value = %5.2f"
          " status = %1i\n", inform.f_eval, f, inform.status);
    }else{
        printf("BGO_solve exit status = %1i\n", inform.status);
    }

    // Delete internal workspace
    ugo_terminate( &data, &control, &inform );

    return 0;
}
