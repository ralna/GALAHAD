/* expotm.c */
/* Test for the EXPO C interface using C sparse matrix indexing */
/* and simulataneous use of different precisions and integer types */
/* Jari Fowkes & Nick Gould, STFC-Rutherford Appleton Laboratory, 2025 */

#include <stdio.h>
#include <math.h>
#include "galahad_c.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

// Custom userdata struct
struct userdata_type {
   double p;
};
struct userdata_type_s_64 {
   float p;
};

// Function prototypes
int fc( int32_t n, int32_t m, const double x[], double *f, double c[], 
        const void * );
int gj( int32_t n, int32_t m, int32_t jne, const double x[], double g[], 
         double jval[], const void * );
int hl( int32_t n, int32_t m, int32_t hne, const double x[], const double y[],
         double hval[], const void * );
int fc_s_64( int64_t n, int64_t m, const float x[], float *f, float c[], 
            const void * );
int gj_s_64( int64_t n, int64_t m, int64_t jne, const float x[], float g[], 
            float jval[], const void * );
int hl_s_64( int64_t n, int64_t m, int64_t hne, const float x[], 
             const float y[], float hval[], const void * );

int main(void) {

    // double precision reals and 32-bit integers 
    // Derived types
    void *data;
    struct expo_control_type control;
    struct expo_inform_type inform;

    printf(" expo test with double precision reals and 32-bit integers\n\n");

    // Set user data
    struct userdata_type userdata;
    userdata.p = 9.0;

    // Set problem data
    int32_t n = 2; // # variables
    int32_t m = 5; // # constraints
    int32_t j_ne = 10; // Jacobian elements
    int32_t h_ne = 2; // Hesssian (lower triangle) elements
    // 0-based indices
    int32_t J_row[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4}; // Jacobian J
    int32_t J_col[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}; 
    int32_t H_row[] = {0, 1}; // Hessian H, NB lower triangle
    int32_t H_col[] = {0, 1};

    // Set storage
    double y[m]; // multipliers
    double z[n]; // dual variables
    double c[m]; // constraints
    double gl[n]; // gradients
    double xl[] = {-50.0, -50.0}; // lower variable bounds
    double xu[] = {50.0, 50.0}; // upper variable bounds
    double cl[] = {0.0, 0.0, 0.0, 0.0, 0.0}; // lower constraint bounds
    double cu[] = {INFINITY, INFINITY, INFINITY, 
                   INFINITY, INFINITY}; // upper constraint bounds
    double x[] = {3.0,1.0}; // starting point
    int32_t status;

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

    // Solve the problem
    expo_import( &control, &data, &status, n, m, 
                 "coordinate", j_ne, J_row, J_col, NULL,
                 "coordinate", h_ne, H_row, H_col, NULL );
    expo_solve_hessian_direct( &data, &userdata, &status,
                               n, m, j_ne, h_ne,
                               cl, cu, xl, xu, x, y, z, c, gl, 
                                           fc, gj, hl );
    expo_information( &data, &inform, &status );

    if(inform.status == 0){
      printf("%6i iterations. Optimal objective value" 
            " = %.2f status = %1i\n",  inform.iter, inform.obj, inform.status);
    }else{
      printf("EXPO_solve exit status = %1i\n", inform.status);
    }
    // Delete internal workspace
    expo_terminate( &data, &control, &inform );

    // double precision reals and 32-bit integers 
    // Derived types
    void *data_s_64;
    struct expo_control_type_s_64 control_s_64;
    struct expo_inform_type_s_64 inform_s_64;

    printf("\n expo test with single precision reals and 64-bit integers\n\n");

    // Set user data
    struct userdata_type_s_64 userdata_s_64;
    userdata_s_64.p = 9.0;

    // Set problem data
    int64_t n_64 = 2; // # variables
    int64_t m_64 = 5; // # constraints
    int64_t j_ne_64 = 10; // Jacobian elements
    int64_t h_ne_64 = 2; // Hesssian elements
    // 0-based indices
    int64_t J_row_64[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4}; // Jacobian J
    int64_t J_col_64[] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    int64_t H_row_64[] = {0, 1}; // Hessian H, NB lower triangle
    int64_t H_col_64[] = {0, 1}; 

    // Set storage
    float y_s[m]; // multipliers
    float z_s[n]; // dual variables
    float c_s[m]; // constraints
    float gl_s[n]; // gradients
    float xl_s[] = {-50.0, -50.0}; // lower variable bounds
    float xu_s[] = {50.0, 50.0}; // upper variable bounds
    float cl_s[] = {0.0, 0.0, 0.0, 0.0, 0.0}; // lower constraint bounds
    float cu_s[] = {INFINITY, INFINITY, INFINITY, 
                   INFINITY, INFINITY}; // upper constraint bounds
    float x_s[] = {3.0,1.0}; // starting point
    int64_t status_64;

    // Initialize EXPO
    expo_initialize_s_64( &data_s_64, &control_s_64, &inform_s_64 );

    // Set user-defined control options
    control_s_64.f_indexing = false; // C sparse matrix indexing
    //control_s_64.print_level = 1;
    control_s_64.max_it = 20;
    control_s_64.max_eval = 100;
    control_s_64.stop_abs_p = 0.001;
    control_s_64.stop_abs_d = 0.001;
    control_s_64.stop_abs_c = 0.001;
    control_s_64.tru_control.error = 0;

    // Solve the problem
    expo_import_s_64( &control_s_64, &data_s_64, &status_64, n_64, m_64, 
                 "coordinate", j_ne_64, J_row_64, J_col_64, NULL,
                 "coordinate", h_ne_64, H_row_64, H_col_64, NULL );
    expo_solve_hessian_direct_s_64( &data_s_64, &userdata_s_64, &status_64,
                                    n_64, m_64, j_ne_64, h_ne_64,
                                    cl_s, cu_s, xl_s, xu_s, 
                                    x_s, y_s, z_s, c_s, gl_s, 
                                    fc_s_64, gj_s_64, hl_s_64 );
    expo_information_s_64( &data_s_64, &inform_s_64, &status_64 );

    if(inform_s_64.status == 0){
      printf("%6li iterations. Optimal objective value = %.2f status = %1li\n",
          inform_s_64.iter, inform_s_64.obj, inform_s_64.status);
    }else{
      printf("EXPO_solve exit status = %1li\n", inform_s_64.status);
    }
    // Delete internal workspace
    expo_terminate_s_64( &data_s_64, &control_s_64, &inform_s_64 );

}

// compute the function and constraint values
int fc( int32_t n, int32_t m, const double x[], double *f, double c[], 
         const void *userdata ){
    struct userdata_type *myuserdata = ( struct userdata_type * ) userdata;
    double p = myuserdata->p;
    *f = pow(x[0],2.0) + pow(x[1],2.0);
    c[0] = x[0] + x[1] - 1.0;
    c[1] = pow(x[0],2.0) + pow(x[1],2.0) - 1.0;
    c[2] = p * pow(x[0],2.0) + pow(x[1],2.0) - p;
    c[3] = pow(x[0],2.0) - x[1];
    c[4] = pow(x[1],2.0) - x[0];
    return 0;
}

// compute the gradient and Jacobian
int gj( int32_t n, int32_t m, int32_t jne, const double x[], double g[], 
        double jval[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    double p = myuserdata->p;
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
int hl( int32_t n, int32_t m, int32_t hne, const double x[], const double y[],
         double hval[], const void *userdata ){
    struct userdata_type *myuserdata = (struct userdata_type *) userdata;
    double p = myuserdata->p;
    hval[0] = 2.0 - 2.0 * (y[1] + p * y[2] + y[3]);
    hval[1] = 2.0 - 2.0 * (y[1] + y[2] + y[4]);
    return 0;
}

// compute the function and constraint values
int fc_s_64( int64_t n, int64_t m, const float x[], float *f, float c[], 
            const void *userdata_s_64 ){
    struct userdata_type_s_64 *myuserdata 
      = (struct userdata_type_s_64 *) userdata_s_64;
    float p = myuserdata->p;
    *f = pow(x[0],2.0) + pow(x[1],2.0);
    c[0] = x[0] + x[1] - 1.0;
    c[1] = pow(x[0],2.0) + pow(x[1],2.0) - 1.0;
    c[2] = p * pow(x[0],2.0) + pow(x[1],2.0) - p;
    c[3] = pow(x[0],2.0) - x[1];
    c[4] = pow(x[1],2.0) - x[0];
    return 0;
}

// compute the gradient and Jacobian
int gj_s_64( int64_t n, int64_t m, int64_t jne, const float x[], 
            float g[], float jval[], const void *userdata_s_64 ){
    struct userdata_type_s_64 *myuserdata 
      = (struct userdata_type_s_64 *) userdata_s_64;
    float p = myuserdata->p;
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
int hl_s_64( int64_t n, int64_t m, int64_t hne, const float x[], 
            const float y[], float hval[], const void *userdata_s_64 ){
    struct userdata_type_s_64 *myuserdata 
      = (struct userdata_type_s_64 *) userdata_s_64;
    float p = myuserdata->p;
    hval[0] = 2.0 - 2.0 * (y[1] + p * y[2] + y[3]);
    hval[1] = 2.0 - 2.0 * (y[1] + y[2] + y[4]);
    return 0;
}





















