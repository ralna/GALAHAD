/* arcs2.c */
/* Spec test for the ARC C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_arc.h"

// Custom userdata struct
struct userdata_type {
   rpc_ p;
};

// Function prototypes
ipc_ fun(ipc_ n, const rpc_ x[], rpc_ *f, const void *);
ipc_ grad(ipc_ n, const rpc_ x[], rpc_ g[], const void *);
ipc_ hessprod(ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[],
             bool got_h, const void *);

int main(void) {

    // Derived types
    void *data;
    struct arc_control_type control;
    struct arc_inform_type inform;

    // Initialize ARC
    ipc_ status;
    arc_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    //control.print_level = 1;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ ne = 5; // Hesssian elements
    rpc_ x[] = {1.,1.,1.}; // start from one
    char H_type[] = "absent"; // specify Hessian-vector products

    // Set storage
    rpc_ g[n]; // gradient

    // Set Hessian storage format, structure and problem bounds
    arc_import( &control, &data, &status, n, H_type, ne, NULL, NULL, NULL );

    // Call ARC_solve
    arc_solve_without_mat( &data, &userdata, &status,
                           n, x, g, fun, grad, hessprod, NULL );

    // Record solution information
    arc_information( &data, &inform, &status );

    if(inform.status == 0){ // successful return
        printf("ARC successful solve\n");
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
    }else{ // error returns
        printf("ARC error in solve\n");
        printf("status: %" d_ipc_ " \n", inform.status);
    }

    // Delete internal workspace
    arc_terminate( &data, &control, &inform );

    return 0;
}

// Objective function
ipc_ fun(ipc_ n, const rpc_ x[], rpc_ *f, const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    rpc_ p = myuserdata->p;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + cos(x[0]);
    return 0;
}

// Gradient of the objective
ipc_ grad(ipc_ n, const rpc_ x[], rpc_ g[], const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    rpc_ p = myuserdata->p;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - sin(x[0]);
    g[1] = 2.0 * ( x[1] + x[2] );
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] );
    return 0;
}

// Hessian-vector product
ipc_ hessprod(ipc_ n, const rpc_ x[], rpc_ u[], const rpc_ v[],
             bool got_h, const void *userdata){
    u[0] = u[0] + 2.0 * ( v[0] + v[2] ) - cos( x[0] ) * v[0];
    u[1] = u[1] + 2.0 * ( v[1] + v[2] );
    u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
    return 0;
}
