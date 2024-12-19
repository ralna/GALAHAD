/* trus3.c */
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
    ipc_ status;
    tru_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    //control.print_level = 1;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ ne = 5; // Hesssian elements
    rpc_ x[] = {1.,1.,1.}; // start from one
    char H_type[] = "coordinate"; // specify co-ordinate storage
    ipc_ H_row[] = {0, 2, 1, 2, 2}; // Hessian H
    ipc_ H_col[] = {0, 0, 1, 1, 2}; // NB lower triangle

    // Reverse-communication input/output
    ipc_ eval_status;
    rpc_ f = 0.0;
    rpc_ g[n];
    rpc_ u[n], v[n];
    rpc_ H_val[ne];

    // Set Hessian storage format, structure and problem bounds
    tru_import( &control, &data, &status, n, H_type, ne, H_row, H_col, NULL );

    // Solve the problem
    while(true){ // reverse-communication loop

        // Call TRU_solve
        tru_solve_reverse_with_mat( &data, &status, &eval_status,
                                    n, x, f, g, ne, H_val, u, v );

        // Evaluate f(x) and its derivatives as required
        if(status == 0){ // successful termination
            printf("TRU successful solve\n");
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
    tru_information( &data, &inform, &status );

    // Print solution details
    printf("iter: %" d_ipc_ " \n", inform.iter);
    printf("x: ");
#ifdef REAL_128
    for(ipc_ i = 0; i < n; i++) printf("%f ", (double)x[i]);
    printf("\nobjective: %f \ngradient: ", (double)inform.obj);
    for(ipc_ i = 0; i < n; i++) printf("%f ", (double)g[i]);
    printf("\nf_eval: %" d_ipc_ " \n", inform.f_eval);
    printf("time: %f \n", (double)inform.time.clock_total);
#else
    for(ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
    printf("\nobjective: %f \ngradient: ", inform.obj);
    for(ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
    printf("\nf_eval: %" d_ipc_ " \n", inform.f_eval);
    printf("time: %f \n", inform.time.clock_total);
#endif
    printf("status: %" d_ipc_ " \n", inform.status);

    // Delete internal workspace
    tru_terminate( &data, &control, &inform );

    return 0;
}
