/* bgos.c */
/* Spec test for the BGO C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_bgo.h"

// Custom userdata struct
struct userdata_type {
   real_wp_ p, freq, mag;
};

// Function prototypes
int fun( int n, const real_wp_ x[], real_wp_ *f, const void * );
int grad( int n, const real_wp_ x[], real_wp_ g[], const void * );
int hess( int n, int ne, const real_wp_ x[], real_wp_ hval[], const void * );
int hessprod( int n, const real_wp_ x[], real_wp_ u[], const real_wp_ v[],
              bool got_h, const void * );

int main(void) {

    // Derived types
    void *data;
    struct bgo_control_type control;
    struct bgo_inform_type inform;

    // Initialize BGO
    int status;
    bgo_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    control.attempts_max = 1000;
    control.max_evals = 1000;
    control.trb_control.maxit = 10;
    control.print_level = 0;
    control.trb_control.subproblem_direct = false;
    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;
    userdata.freq = 10.0;
    userdata.mag = 1000.0;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    real_wp_ x[] = {0,0,0}; // start from zero
    real_wp_ x_l[] = {-10.0,-10.0,-10.0};
    real_wp_ x_u[] = {0.5,0.5,0.5};
    char H_type[] = "coordinate"; // specify co-ordinate storage
    int H_row[] = {0, 1, 2, 2, 2}; // Hessian H
    int H_col[] = {0, 1, 0, 1, 2}; // NB lower triangle

    // Set storage
    real_wp_ g[n]; // gradient

    // Set Hessian storage format, structure and problem bounds
    bgo_import( &control, &data, &status, n, x_l, x_u,
                H_type, ne, H_row, H_col, NULL );

    // Call BGO_solve
    //bgo_solve_with_mat( &data, &userdata, &status,
    //                  n, x, g, ne, fun, grad, hess, hessprod, NULL );
    bgo_solve_with_mat( &data, &userdata, &status,
                      n, x, g, ne, fun, grad, hess, NULL, NULL );

    // Record solution information
    bgo_information( &data, &inform, &status );

    if(inform.status == 0){ // successful return
        printf("BGO successful solve\n");
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
        printf("BGO error in solve\n");
        printf("status: %d \n", inform.status);
    }

    // Delete internal workspace
    bgo_terminate( &data, &control, &inform );

    return 0;
}

// Objective function
int fun( int n, const real_wp_ x[], real_wp_ *f, const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;
    real_wp_ freq = myuserdata->freq;
    real_wp_ mag = myuserdata->mag;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) 
           + mag * cos( freq * x[0]) + x[0] + x[1] + x[2];
    return 0;
}

// Gradient of the objective
int grad( int n, const real_wp_ x[], real_wp_ g[], const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ p = myuserdata->p;
    real_wp_ freq = myuserdata->freq;
    real_wp_ mag = myuserdata->mag;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - mag * freq * sin(freq * x[0]) + 1.0;
    g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] ) + 1.0;
    return 0;
}

// Hessian of the objective
int hess( int n, int ne, const real_wp_ x[], real_wp_ hval[], 
          const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    real_wp_ freq = myuserdata->freq;
    real_wp_ mag = myuserdata->mag;
    hval[0] = 2.0 - mag * freq * freq * cos(freq * x[0]);
    hval[1] = 2.0;
    hval[2] = 2.0;
    hval[3] = 2.0;
    hval[4] = 4.0;
    return 0;
}

