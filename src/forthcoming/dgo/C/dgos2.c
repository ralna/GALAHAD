/* dgos2.c */
/* Spec test for the DGO C interface */

#include <stdio.h>
#include <math.h>
#include "dgo.h"

// Custom userdata struct
struct userdata_type {
   double p;
};

// Function prototypes
int fun( int n, const double x[], double *f, const void * );
int grad( int n, const double x[], double g[], const void * );
int hessprod( int n, const double x[], double u[], const double v[], 
              bool got_h, const void * );
int shessprod( int n, const double x[], int nnz_v, const int index_nz_v[], 
               const double v[], int *nnz_u, int index_nz_u[], double u[], 
               bool got_h, const void * );

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
    // control.trb_control.maxit = 10;
    // control.print_level = 1;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 4.0;

    // Set problem data
    int n = 3; // dimension
    int ne = 5; // Hesssian elements
    double x[] = {1,1,1}; // start from one
    double x_l[] = {-10.0,-10.0,-10.0}; 
    double x_u[] = {1.0,1.0,1.0};
    char H_type[] = "absent"; // specify co-ordinate storage

    // Set storage
    double g[n]; // gradient
    
    // Set Hessian storage format, structure and problem bounds
    int status;
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

// Objective function 
int fun( int n, const double x[], double *f, const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    double p = myuserdata->p;
    double freq = 10.0;
    double mag = 1000.0;

    *f = pow(x[0] + x[2] + p, 2) + pow(x[1] + x[2], 2) + mag*cos(freq*x[0])
        + x[0] + x[1] + x[2];
    return 0;
}

// Gradient of the objective
int grad( int n, const double x[], double g[], const void *userdata){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    double p = myuserdata->p;
    double freq = 10.0;
    double mag = 1000.0;

    g[0] = 2.0 * ( x[0] + x[2] + p ) - mag*freq*sin(freq*x[0]) + 1.0;
    g[1] = 2.0 * ( x[1] + x[2] ) + 1.0;
    g[2] = 2.0 * ( x[0] + x[2] + p ) + 2.0 * ( x[1] + x[2] ) + 1.0;
    return 0;
}

// Hessian-vector product
int hessprod( int n, const double x[], double u[], const double v[], 
              bool got_h, const void *userdata){
    double freq = 10.0;
    double mag = 1000.0;
    u[0] = u[0] + 2.0 * ( v[0] + v[2] ) - mag*freq*freq*cos(freq*x[0]) * v[0];
    u[1] = u[1] + 2.0 * ( v[1] + v[2] );
    u[2] = u[2] + 2.0 * ( v[0] + v[1] + 2.0 * v[2] );
    return 0;
}

// Sparse Hessian-vector product
int shessprod(int n, const double x[], int nnz_v, const int index_nz_v[], 
              const double v[], int *nnz_u, int index_nz_u[], double u[], 
              bool got_h, const void *userdata){
    double freq = 10.0;
    double mag = 1000.0;
    double p[] = {0., 0., 0.};
    bool used[] = {false, false, false};
    for(int i = 0; i < nnz_v; i++){
        int j = index_nz_v[i];
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
    for(int j = 0; j < 3; j++){
        if(used[j]){
        u[j] = p[j];
        *nnz_u = *nnz_u + 1;
        index_nz_u[*nnz_u-1] = j;
        }
    }
    return 0;
}
