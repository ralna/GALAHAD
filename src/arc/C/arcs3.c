/* arcs3.c */
/* Spec test for the ARC C interface */

#include <stdio.h>
#include <math.h>
#include "arc.h"

int main(void) {

    // Derived types
    void *data;
    struct arc_control_type control;
    struct arc_inform_type inform;

    // Initialize ARC
    arc_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    control.print_level = 1;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    double x[] = {1.,1.,1.}; // start from one
    double infty = 1e20; // infinity
    char H_type[] = "coordinate"; // specify co-ordinate storage
    int H_row[] = {0, 2, 1, 2, 2}; // Hessian H
    int H_col[] = {0, 0, 1, 1, 2}; // NB lower triangle
    
    // Reverse-communication input/output
    int status, eval_status;
    double f;
    double g[n];
    double u[n], v[n];
    double H_val[ne]; 

    // Set Hessian storage format, structure and problem bounds
    arc_import( &control, &data, &status, n, H_type, ne, H_row, H_col, NULL );

    // Solve the problem
    while(true){ // reverse-communication loop

        // Call ARC_solve
        arc_solve_reverse_with_mat( &data, &status, &eval_status, 
                                    n, x, f, g, ne, H_val, u, v );

        // Evaluate f(x) and its derivatives as required
        if(status == 0){ // successful termination
            printf("ARC successful solve\n");
            break; // break while loop
        }else if(status == 2){ // obtain the objective function
            f = pow(x[0] + x[2] + 4.0, 2) + pow(x[1] + x[2], 2) + cos(x[0]);
            eval_status = 0; // record successful evaluation
        }else if(status == 3){ // obtain the gradient
            g[0] = 2.0 * ( x[0] + x[2] + 4.0 ) - sin(x[0]);
            g[1] = 2.0 * ( x[1] + x[2] );
            g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] );
            eval_status = 0; // record successful evaluation
        }else if(status == 4){ // obtain Hessian evaluation
            H_val[0] = 2.0 - cos(x[0]);
            H_val[1] = 2.0;
            H_val[2] = 2.0;
            H_val[3] = 2.0;
            H_val[4] = 4.0;
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
    arc_information( &data, &inform, &status );

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
    arc_terminate( &data, &control, &inform );

    return 0;
}
