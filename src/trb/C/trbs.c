/* trbs.c */
/* Spec test for the TRB C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_trb.h"

// Custom userdata struct
struct userdata_type {
   real_wp_ p;
};

// Function prototypes
ipc_ fun( ipc_ n, const real_wp_ x[], real_wp_ *f, const void * );
ipc_ grad( ipc_ n, const real_wp_ x[], real_wp_ g[], const void * );
ipc_ hess( ipc_ n, ipc_ ne, const real_wp_ x[], real_wp_ hval[], const void * );

ipc_ main(void) {

    // Derived types
    void *data;
    struct trb_control_type control;
    struct trb_inform_type inform;

    // Initialize TRB
    ipc_ status;
    trb_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    //control.print_level = 1;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ ne = 5; // Hesssian elements
    real_wp_ x[] = {1,1,1}; // start from one
    real_wp_ infty = 1e20; // infinity
    real_wp_ x_l[] = {-infty,-infty, 0.};
    real_wp_ x_u[] = {1.1,1.1,1.1};
    char H_type[] = "coordinate"; // specify co-ordinate storage
    ipc_ H_row[] = {0, 2, 1, 2, 2}; // Hessian H
    ipc_ H_col[] = {0, 0, 1, 1, 2}; // NB lower triangle

    // Set storage
    real_wp_ g[n]; // gradient

    // Set Hessian storage format, structure and problem bounds
    trb_import( &control, &data, &status, n, x_l, x_u,
                H_type, ne, H_row, H_col, NULL );

    // Call TRB_solve
    trb_solve_with_mat( &data, &userdata, &status,
                        n, x, g, ne, fun, grad, hess, NULL );

    // Record solution information
    trb_information( &data, &inform, &status );

    // Print solution details
    if(inform.status == 0){ // successful return
        printf("TRB successful solve\n");
        printf("iter: %d \n", inform.iter);
        printf("x: ");
        for(ipc_ i = 0; i < n; i++) printf("%f ", x[i]);
        printf("\n");
        printf("objective: %f \n", inform.obj);
        printf("gradient: ");
        for(ipc_ i = 0; i < n; i++) printf("%f ", g[i]);
        printf("\n");
        printf("f_eval: %d \n", inform.f_eval);
        printf("time: %f \n", inform.time.clock_total);
        printf("status: %d \n", inform.status);
    }else{ // error returns
        printf("TRB error in solve\n");
        printf("status: %d \n", inform.status);
    }

    // Delete internal workspace
    trb_terminate( &data, &control, &inform );

    return 0;
}

// Objective function
ipc_ fun(ipc_ n, const real_wp_ x[], real_wp_ *f, const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + cos(x[0]);
    return 0;
}

// Gradient of the objective
ipc_ grad(ipc_ n, const real_wp_ x[], real_wp_ g[], const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - sin(x[0]);
    g[1] = 2.0 * ( x[1] + x[2] );
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] );
    return 0;
}

// Hessian of the objective
ipc_ hess(ipc_ n, ipc_ ne, const real_wp_ x[], real_wp_ hval[],
         const void *userdata){
    hval[0] = 2.0 - cos(x[0]);
    hval[1] = 2.0;
    hval[2] = 2.0;
    hval[3] = 2.0;
    hval[4] = 4.0;
    return 0;
}
