/* trus.c */
/* Spec test for the TRU C interface */

#include <stdio.h>
#include <math.h>
#include "tru.h"

// Custom userdata struct
struct userdata_type {
   double p;
};

// Function prototypes
int fun(int n, const double x[], double *f, const void *);
int grad(int n, const double x[], double g[], const void *);
int hess(int n, int ne, const double x[], double hval[], const void *);

int main(void) {

    // Derived types
    void *data;
    struct tru_control_type control;
    struct tru_inform_type inform;   

    // Initialize TRU
    tru_initialize( &data, &control, &inform );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing (default)
    // control.print_level = 1;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    double x[] = {1,1,1}; // start from one
    double infty = 1e20; // infinity
    char H_type[] = "coordinate"; // specify co-ordinate storage
    int H_row[] = {0, 2, 1, 2, 2}; // Hessian H
    int H_col[] = {0, 0, 1, 1, 2}; // NB lower triangle

    // Set storage
    double g[n]; // gradient
    
    // Set Hessian storage format, structure and problem bounds
    int status;
    tru_import( &control, &data, &status, n, H_type, ne, H_row, H_col, NULL );

    // Call TRU_solve
    tru_solve_with_mat( &data, &userdata, &status,
                        n, x, g, ne, fun, grad, hess, NULL );

    // Record solution information
    tru_information( &data, &inform, &status );
    
    if(inform.status == 0){ // successful return
        printf("TRU successful solve\n");
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
    }else{ // error returns
        printf("TRU error in solve\n");
        printf("status: %d \n", inform.status);
    }

    // Delete internal workspace
    tru_terminate( &data, &control, &inform );

    return 0;
}

// Objective function 
int fun(int n, const double x[], double *f, const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    double p = myuserdata->p;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + cos(x[0]);
    return 0;
}

// Gradient of the objective
int grad(int n, const double x[], double g[], const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    double p = myuserdata->p;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - sin(x[0]);
    g[1] = 2.0 * ( x[1] + x[2] );
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] );
    return 0;
}

// Hessian of the objective
int hess(int n, int ne, const double x[], double hval[], const void *userdata){
    hval[0] = 2.0 - cos(x[0]);
    hval[1] = 2.0;
    hval[2] = 2.0;
    hval[3] = 2.0;
    hval[4] = 4.0;
    return 0;
}
