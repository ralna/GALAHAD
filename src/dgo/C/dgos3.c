/* dgos3.c */
/* Spec test for the DGO C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_dgo.h"

int main(void) {

    // Derived types
    void *data;
    struct dgo_control_type control;
    struct dgo_inform_type inform;

    // Initialize DGO
    ipc_ status;
    dgo_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    control.maxit = 20000;
    // control.print_level = 1;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ ne = 5; // Hesssian elements
    rpc_ x[] = {1.,1.,1.}; // start from one
    rpc_ x_l[] = {-10.0,-10.0,-10.0};
    rpc_ x_u[] = {1.0,1.0,1.0};
    char H_type[] = "coordinate"; // specify co-ordinate storage
    ipc_ H_row[] = {0, 2, 1, 2, 2}; // Hessian H
    ipc_ H_col[] = {0, 0, 1, 1, 2}; // NB lower triangle

    // Reverse-communication input/output
    ipc_ eval_status;
    rpc_ f = 0.0;
    rpc_ g[n];
    rpc_ u[n], v[n];
    rpc_ H_val[ne];
    rpc_ freq = 10.0;
    rpc_ mag = 1000.0;

    // Set Hessian storage format, structure and problem bounds
    dgo_import( &control, &data, &status, n, x_l, x_u,
                H_type, ne, H_row, H_col, NULL );

    // Solve the problem
    while(true){ // reverse-communication loop

        // Call DGO_solve
        dgo_solve_reverse_with_mat( &data, &status, &eval_status, n, x, f, g,
                                    ne, H_val, u, v );

        // Evaluate f(x) and its derivatives as required
        if(status == 0){ // successful termination
            printf("DGO successful solve\n");
            break; // break while loop
        }else if(status == 2){ // obtain the objective function
            f = pow(x[0] + x[2] + 4.0, 2) + pow(x[1] + x[2], 2)
                + mag*cos(freq*x[0]) + x[0] + x[1] + x[2];
            eval_status = 0; // record successful evaluation
        }else if(status == 3){ // obtain the gradient
            g[0] = 2.0 * ( x[0] + x[2] + 4.0 )
                   - mag*freq*sin(freq*x[0]) + 1.0;;
            g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
            g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] ) + 1.0;
            eval_status = 0; // record successful evaluation
        }else if(status == 4){ // obtain Hessian evaluation
            H_val[0] = 2.0 - mag*freq*freq*cos(freq*x[0]);
            H_val[1] = 2.0;
            H_val[2] = 2.0;
            H_val[3] = 2.0;
            H_val[4] = 4.0;
            eval_status = 0; // record successful evaluation
        }else if(status == 5){ // obtain Hessian-vector product
            u[0] = u[0] + 2.0 * ( v[0] + v[2] )
                    - mag*freq*freq*cos(freq*x[0]) * v[0];
            u[1] = u[1] + 2.0 * ( v[1] + v[2] );
            u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
            eval_status = 0; // record successful evaluation
        }else if(status == 6){ // apply the preconditioner
            u[0] = 0.5 * v[0];
            u[1] = 0.5 * v[1];
            u[2] = 0.25 * v[2];
            eval_status = 0; // record successful evaluation
        }else if(status == 23){ // obtain objective and gradient
            f = pow(x[0] + x[2] + 4.0, 2) + pow(x[1] + x[2], 2)
                + mag*cos(freq*x[0]);
            g[0] = 2.0 * ( x[0] + x[2] + 4.0 )
                   - mag*freq* sin(freq*x[0]) + 1.0;
            g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
            g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] ) + 1.0;
            eval_status = 0; // record successful evaluation
        }else if(status == 25){ // obtain objective and Hessian-vector product
            f = pow(x[0] + x[2] + 4.0, 2) + pow(x[1] + x[2], 2)
                   + mag*cos(freq*x[0]);
            u[0] = u[0] + 2.0 * ( v[0] + v[2] )
                   - mag*freq*freq*cos(freq*x[0]) * v[0];
            u[1] = u[1] + 2.0 * ( v[1] + v[2] );
            u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
            eval_status = 0; // record successful evaluation
        }else if(status == 35){ // obtain gradient and Hessian-vector product
            g[0] = 2.0 * ( x[0] + x[2] + 4.0 ) - mag*freq*sin(freq*x[0]) + 1.0;
            g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
            g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] ) + 1.0;
            u[0] = u[0] + 2.0 * ( v[0] + v[2] )
                   - mag*freq*freq*cos( freq*x[0] ) * v[0];
            u[1] = u[1] + 2.0 * ( v[1] + v[2] );
            u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
            eval_status = 0; // record successful evaluation
        }else if(status == 235){ // obtain obj, grad and Hess-vector product
            f = pow(x[0] + x[2] + 4.0, 2) + pow(x[1] + x[2], 2)
                + mag*cos(freq*x[0]);
            g[0] = 2.0 * ( x[0] + x[2] + 4.0 ) - mag*freq*sin(freq*x[0]) + 1.0;
            g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
            g[2] = 2.0 * ( x[0] + x[2] + 4.0 ) + 2.0 * ( x[1] + x[2] ) + 1.0;
            u[0] = u[0] + 2.0 * ( v[0] + v[2] )
                   - mag*freq*freq*cos(freq*x[0]) * v[0];
            u[1] = u[1] + 2.0 * ( v[1] + v[2] );
            u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
            eval_status = 0; // record successful evaluation
        }else{ // an error has occured
            break; // break while loop
        }
    }

    // Record solution information
    dgo_information( &data, &inform, &status );

    // Print solution details
    printf("iter: %" d_ipc_ " \n", inform.trb_inform.iter);
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
    dgo_terminate( &data, &control, &inform );

    return 0;
}
