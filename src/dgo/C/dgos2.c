/* dgos2.c */
/* Spec test for the DGO C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_dgo.h"

// Custom userdata struct
struct userdata_type {
   real_wp_ p;
};

// Function prototypes
ipc_ fun( ipc_ n, const real_wp_ x[], real_wp_ *f, const void * );
ipc_ grad( ipc_ n, const real_wp_ x[], real_wp_ g[], const void * );
ipc_ hessprod( ipc_ n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[],
              bool got_h, const void * );
ipc_ shessprod( ipc_ n, const real_wp_ x[], ipc_ nnz_v, const ipc_ index_nz_v[],
               const real_wp_ v[], int *nnz_u, ipc_ index_nz_u[], real_wp_ u[],
               bool got_h, const void * );

ipc_ main(void) {

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
    // control.trb_control.maxit = 10;
    // control.print_level = 1;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    ipc_ n = 3; // dimension
    ipc_ ne = 5; // Hesssian elements
    real_wp_ x[] = {1,1,1}; // start from one
    real_wp_ x_l[] = {-10.0,-10.0,-10.0};
    real_wp_ x_u[] = {1.0,1.0,1.0};
    char H_type[] = "absent"; // specify co-ordinate storage

    // Set storage
    real_wp_ g[n]; // gradient

    // Set Hessian storage format, structure and problem bounds
    dgo_import( &control, &data, &status, n, x_l, x_u,
                H_type, ne, NULL, NULL, NULL );

    // Call DGO_solve
    dgo_solve_without_mat( &data, &userdata, &status, n, x, g,
                           fun, grad, hessprod, shessprod, NULL );

    // Record solution information
    dgo_information( &data, &inform, &status );

    if(inform.status == 0 || inform.status == -18){
        if(inform.status == 0){ // successful return
          printf("DGO successful solve\n");
        }else{
          printf("DGO solve budget limit reached\n");
        }
        printf("TR iter: %d \n", inform.trb_inform.iter);
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
        printf("DGO error in solve\n");
        printf("status: %d \n", inform.status);
    }

    // Delete internal workspace
    dgo_terminate( &data, &control, &inform );

    return 0;
}

// Objective function
ipc_ fun( ipc_ n, const real_wp_ x[], real_wp_ *f, const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;
    real_wp_ freq = 10.0;
    real_wp_ mag = 1000.0;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + mag*cos(freq*x[0])
        + x[0] + x[1] + x[2];
    return 0;
}

// Gradient of the objective
ipc_ grad( ipc_ n, const real_wp_ x[], real_wp_ g[], const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;
    real_wp_ freq = 10.0;
    real_wp_ mag = 1000.0;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - mag*freq*sin(freq*x[0]) + 1.0;
    g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] ) + 1.0;
    return 0;
}

// Hessian-vector product
ipc_ hessprod( ipc_ n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[],
              bool got_h, const void *userdata){
    real_wp_ freq = 10.0;
    real_wp_ mag = 1000.0;
    u[0] = u[0] + 2.0 * ( v[0] + v[2] ) - mag*freq*freq*cos(freq*x[0]) * v[0];
    u[1] = u[1] + 2.0 * ( v[1] + v[2] );
    u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
    return 0;
}

// Sparse Hessian-vector product
ipc_ shessprod(ipc_ n, const real_wp_ x[], ipc_ nnz_v, const ipc_ index_nz_v[],
              const real_wp_ v[], int *nnz_u, ipc_ index_nz_u[], real_wp_ u[],
              bool got_h, const void *userdata){
    real_wp_ freq = 10.0;
    real_wp_ mag = 1000.0;
    real_wp_ p[] = {0., 0., 0.};
    bool used[] = {false, false, false};
    for(ipc_ i = 0; i < nnz_v; i++){
        ipc_ j = index_nz_v[i];
        switch(j){
            case 0:
                p[0] = p[0] + 2.0 * v[0] - mag*freq*freq*cos(freq*x[0]) * v[0];
                used[0] = true;
                p[2] = p[2] + 2.0 * v[0];
                used[2] = true;
                break;
            case 1:
                p[1] = p[1] + 2.0 * v[1];
                used[1] = true;
                p[2] = p[2] + 2.0 * v[1];
                used[2] = true;
                break;
            case 2:
                p[0] = p[0] + 2.0 * v[2];
                used[0] = true;
                p[1] = p[1] + 2.0 * v[2];
                used[1] = true;
                p[2] = p[2] + 4.0 * v[2];
                used[2] = true;
                break;
        }
    }
    *nnz_u = 0;
    for(ipc_ j = 0; j < 3; j++){
        if(used[j]){
        u[j] = p[j];
        *nnz_u = *nnz_u + 1;
        index_nz_u[*nnz_u-1] = j;
        }
    }
    return 0;
}
