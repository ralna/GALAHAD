/* trus4.c */
/* Spec test for the TRU C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_tru.h"

int main(void) {

    // Derived types
    void *data;
    struct tru_control_type control;
    struct tru_inform_type inform;

    // Initialize TRU
    int status;
    tru_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    //control.print_level = 1;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    real_wp_ x[] = {1.,1.,1.}; // start from one
    real_wp_ infty = 1e20; // infinity
    char H_type[] = "absent"; // specify Hessian-vector products
    
    // Reverse-communication input/output
    int eval_status;
    real_wp_ f;
    real_wp_ g[n];
    real_wp_ u[n], v[n];

    //control.maxit=2;
    //control.print_level=5;
    // Set Hessian storage format, structure and problem bounds
    tru_import( &control, &data, &status, n, H_type, ne, NULL, NULL, NULL );

    // Solve the problem
    while(true){ // reverse-communication loop

        // Call TRU_solve
        tru_solve_reverse_without_mat( &data, &status, &eval_status, 
                                       n, x, f, g, u, v );

        // Evaluate f(x) and its derivatives as required
        if(status == 0){ // successful termination
            break; // break while loop
        }else if(status == 2){ // obtain the objective function
            f = pow(x[0] + x[2] + 4.0, 2) + pow(x[1] + x[2], 2) + cos(x[0]);
            eval_status = 0; // record successful evaluation
        }else if(status == 3){ // obtain the gradient
            g[0] = 2.0 * ( x[0] + x[2] + 4.0 ) - sin(x[0]);
            g[1] = 2.0 * ( x[1] + x[2] );
            g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] );
            eval_status = 0; // record successful evaluation
        }else if(status == 5){ // obtain Hessian-vector product
            u[0] = u[0] + 2.0 * ( v[0] + v[2] ) - cos( x[0] ) * v[0];
            u[1] = u[1] + 2.0 * ( v[1] + v[2] );
            u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
            eval_status = 0; // record successful evaluation
        }else if(status == 6){ // apply the preconditioner
            u[0] = 0.5 * v[0];
            u[1] = 0.5 * v[1];
            u[2] = 0.25 * v[2];
            eval_status = 0; // record successful evaluation
        }else{ // an error has occured
            break; // break while loop
        }
    }

    // Record solution information
    tru_information( &data, &inform, &status );

    // Print solution details
    printf("iter: %d \n", inform.iter);
    printf("x: ");
    for(int i = 0; i < n; i++) printf("%f ", x[i]);
    printf("\n");
    printf("objective: %f \n", inform.obj);
    printf("gradient: ");
    for(int i = 0; i < n; i++) printf("%f ", g[i]);
    printf("\n");
    printf("f_eval: %d \n", inform.f_eval);
    printf("time: %f \n", inform.time.clock_total);
    printf("status: %d \n", inform.status);

    // Delete internal workspace
    tru_terminate( &data, &control, &inform );

    return 0;
}
