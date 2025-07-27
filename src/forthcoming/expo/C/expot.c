/* expot.c */
/* Full test for the EXPO C interface using C sparse matrix indexing */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2025 */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_expo.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

// Define imax
ipc_ imax(ipc_ a, ipc_ b) {
    return (a > b) ? a : b;
};

// Custom userdata struct
struct userdata_type {
   rpc_ p;
};

// Function prototypes
ipc_ fc( ipc_ n, ipc_ m, const rpc_ x[], rpc_ *f, rpc_ c[], const void * );
ipc_ gj( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ g[], 
         rpc_ jval[], const void * );
ipc_ hl( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
         rpc_ hval[], const void * );
ipc_ gj_dense( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ g[], 
               rpc_ jval[],  const void * );
ipc_ hl_dense( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
               rpc_ hval[], const void * );

int main(void) {

    // Derived types
    void *data;
    struct expo_control_type control;
    struct expo_inform_type inform;

    // Set user data
    struct userdata_type userdata;
    userdata.p = 9.0;

    // Set problem data
    ipc_ n = 2; // # variables
    ipc_ m = 5; // # constraints
    ipc_ j_ne = 10; // Jacobian elements
    ipc_ h_ne = 2; // Hesssian elements
    // 0-based indices
    ipc_ J_row[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4}; // Jacobian J
    ipc_ J_col[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}; //
    ipc_ J_ptr[] = {0, 2, 4, 6, 8, 10 };           // row pointers
    ipc_ H_row[] = {0, 1};     // Hessian H
    ipc_ H_col[] = {0, 1};     // NB lower triangle
    ipc_ H_ptr[] = {0, 1, 2};  // row pointers

    // Set storage
    rpc_ y[m]; // multipliers
    rpc_ z[n]; // dual variables
    rpc_ c[m]; // constraints
    rpc_ gl[n]; // gradients
    rpc_ xl[] = {-50.0, -50.0}; // lower variable bounds
    rpc_ xu[] = {50.0, 50.0}; // upper variable bounds
    rpc_ cl[] = {0.0, 0.0, 0.0, 0.0, 0.0}; // lower constraint bounds
    rpc_ cu[] = {INFINITY, INFINITY, INFINITY, 
                 INFINITY, INFINITY}; // upper constraint bounds
    char st = ' ';
    ipc_ status;

    printf(" C sparse matrix indexing\n\n");

    printf(" tests options for all-in-one storage format\n\n");

//  for( ipc_ d=1; d <= 3; d++){
    for( ipc_ d=1; d <= 4; d++){

        // Initialize EXPO
        expo_initialize( &data, &control, &inform );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing
        //control.print_level = 1;
        control.max_it = 20;
        control.max_eval = 100;
        control.stop_abs_p = 0.00001;
        control.stop_abs_d = 0.00001;
        control.stop_abs_c = 0.00001;

        rpc_ x[] = {3.0,1.0}; // starting point

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                expo_import( &control, &data, &status, n, m, 
                             "coordinate", j_ne, J_row, J_col, NULL,
                             "coordinate", h_ne, H_row, H_col, NULL );
                expo_solve_hessian_direct( &data, &userdata, &status,
                                           n, m, j_ne, h_ne,
                                           cl, cu, xl, xu, x, y, z, c, gl, 
                                           fc, gj, hl );
                break;
            case 2: // sparse by rows
                st = 'R';
                expo_import( &control, &data, &status, n, m,
                             "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                             "sparse_by_rows", h_ne, NULL, H_col, H_ptr );
                expo_solve_hessian_direct( &data, &userdata, &status,
                                           n, m, j_ne, h_ne,
                                           cl, cu, xl, xu, x, y, z, c, gl,
                                           fc, gj, hl );
                break;
            case 3: // dense
                st = 'D';
                expo_import( &control, &data, &status, n, m,
                             "dense", j_ne, NULL, NULL, NULL,
                             "dense", h_ne, NULL, NULL, NULL );
                expo_solve_hessian_direct( &data, &userdata, &status,
                                           n, m, j_ne, h_ne,
                                           cl, cu, xl, xu, x, y, z, c, gl,
                                           fc, gj_dense, hl_dense );
                break;
            case 4: // diagonal
                st = 'I';
                expo_import( &control, &data, &status, n, m, 
                             "sparse_by_rows", j_ne, NULL, J_col, J_ptr,
                             "diagonal", h_ne, NULL, NULL, NULL );
                expo_solve_hessian_direct( &data, &userdata, &status, 
                                           n, m, j_ne, h_ne,
                                           cl, cu, xl, xu, x, y, z, c, gl, 
                                           fc, gj, hl );
                break;
        }

        expo_information( &data, &inform, &status );

        if(inform.status == 0){
#ifdef REAL_128
// interim replacement for quad output: $GALAHAD/include/galahad_pquad_f.h
#include "galahad_pquad_f.h"
#else
            printf("%c:%6" i_ipc_ " iterations. Optimal objective " 
                   "value = %.2f status = %1" i_ipc_ "\n",
                   st, inform.iter, inform.obj, inform.status);
#endif
        }else{
            printf("%c: EXPO_solve exit status = %1" i_ipc_ "\n", 
                   st, inform.status);
        }
        // Delete internal workspace
        expo_terminate( &data, &control, &inform );
    }
}

// compute the function and constraint values
ipc_ fc( ipc_ n, ipc_ m, const rpc_ x[], rpc_ *f, rpc_ c[], 
         const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    rpc_ p = myuserdata->p;
    *f = pow(x[0],2.0) + pow(x[1],2.0);
    c[0] = x[0] + x[1] - 1.0;
    c[1] = pow(x[0],2.0) + pow(x[1],2.0) - 1.0;
    c[2] = p * pow(x[0],2.0) + pow(x[1],2.0) - p;
    c[3] = pow(x[0],2.0) - x[1];
    c[4] = pow(x[1],2.0) - x[0];
    return 0;
}

// compute the gradient and Jacobian
ipc_ gj( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ g[], rpc_ jval[],
         const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    rpc_ p = myuserdata->p;
    g[0] = 2.0 * x[0];
    g[1] = 2.0 * x[1];
    jval[0] = 1.0;
    jval[1] = 1.0;
    jval[2] = 2.0 * x[0];
    jval[3] = 2.0 * x[1];
    jval[4] = 2.0 * p * x[0];
    jval[5] = 2.0 * x[1];
    jval[6] = 2.0 * x[0];
    jval[7] = - 1.0;
    jval[8] = - 1.0;
    jval[9] = 2.0 * x[1];
    return 0;
}

// compute the Hessian of the Lagrangian
ipc_ hl( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
         rpc_ hval[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    rpc_ p = myuserdata->p;
    hval[0] = 2.0 - 2.0 * (y[1] + p * y[2] + y[3]);
    hval[1] = 2.0 - 2.0 * (y[1] + y[2] + y[4]);
    return 0;
}

// compute the gradient and dense Jacobian
ipc_ gj_dense( ipc_ n, ipc_ m, ipc_ jne, const rpc_ x[], rpc_ g[], 
               rpc_ jval[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    rpc_ p = myuserdata->p;
    g[0] = 2.0 * x[0];
    g[1] = 2.0 * x[1];
    jval[0] = 1.0;
    jval[1] = 1.0;
    jval[2] = 2.0 * x[0];
    jval[3] = 2.0 * x[1];
    jval[4] = 2.0 * p * x[0];
    jval[5] = 2.0 * x[1];
    jval[6] = 2.0 * x[0];
    jval[7] = - 1.0;
    jval[8] = - 1.0;
    jval[9] = 2.0 * x[1];
    return 0;
}

// compute the dense Hessian of the Lagrangian
ipc_ hl_dense( ipc_ n, ipc_ m, ipc_ hne, const rpc_ x[], const rpc_ y[],
               rpc_ hval[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    rpc_ p = myuserdata->p;
    hval[0] = 2.0 - 2.0 * (y[1] + p * y[2] +  y[3]);
    hval[1] = 0.0;
    hval[2] = 2.0 - 2.0 * (y[1] + y[2] + y[4]);
    return 0;
}




















