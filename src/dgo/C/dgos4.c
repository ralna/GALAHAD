/* dgos4.c */
/* Spec test for the DGO C interface */

#include <stdio.h>
#include <math.h>
#include "dgo.h"

int main(void) {

    // Derived types
    void *data;
    struct dgo_control_type control;
    struct dgo_inform_type inform;

    // Initialize DGO
    dgo_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    control.maxit = 20000;
    // control.print_level = 1;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    double x[] = {1.,1.,1.}; // start from one
    double x_l[] = {-10.0,-10.0,-10.0}; 
    double x_u[] = {1.0,1.0,1.0};
    char H_type[] = "absent"; // specify Hessian-vector producrs
    
    // Reverse-communication input/output
    int eval_status, nnz_u, nnz_v;
    double f;
    double g[n];
    double u[n], v[n];
    int index_nz_u[n], index_nz_v[n];
    double freq = 10.0;
    double mag = 1000.0;

    // Set Hessian storage format, structure and problem bounds
    int status;
    dgo_import( &control, &data, &status, n, x_l, x_u, 
                H_type, ne, NULL, NULL, NULL );

    // Solve the problem
    while(true){ // reverse-communication loop

        // Call DGO_solve
        dgo_solve_reverse_without_mat( &data, &status, &eval_status, 
                                       n, x, f, g, u, v,index_nz_v, &nnz_v, 
                                       index_nz_u, nnz_u );

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
         }else if(status == 7){ // obtain sparse Hessian-vector product
            double tmp[] = {0., 0., 0.};
            bool used[] = {false, false, false};
            for(int i = 0; i < nnz_v; i++){
                int j = index_nz_v[i];
                switch(j){
                    case 0:
                        tmp[0] = tmp[0] + 2.0 * v[0] 
                                 - mag*freq*freq*cos(freq*x[0]) * v[0];
                        used[0] = true;
                        tmp[2] = tmp[2] + 2.0 * v[0];
                        used[2] = true;
                        break;
                    case 1:
                        tmp[1] = tmp[1] + 2.0 * v[1];
                        used[1] = true;
                        tmp[2] = tmp[2] + 2.0 * v[1];
                        used[2] = true;
                        break;
                    case 2:
                        tmp[0] = tmp[0] + 2.0 * v[2];
                        used[0] = true;
                        tmp[1] = tmp[1] + 2.0 * v[2];
                        used[1] = true;
                        tmp[2] = tmp[2] + 4.0 * v[2];
                        used[2] = true;
                        break;
                }
            }
            nnz_u = 0;
            for(int j = 0; j < 3; j++){
                if(used[j]){
                u[j] = tmp[j];
                nnz_u = nnz_u + 1;
                index_nz_u[nnz_u-1] = j;
                }
            }
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
    if(inform.status == 0 || inform.status == -18){
        if(inform.status == 0){ // successful return
          printf("DGO successful solve\n");
        }else{
          printf("DGO solve budget limit reached\n");
        }
        printf("TR iter: %d \n", inform.trb_inform.iter);
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
    }else{ // error returns
        printf("DGO error in solve\n");
        printf("status: %d \n", inform.status);
    }

    // Delete internal workspace
    dgo_terminate( &data, &control, &inform );

    return 0;
}
