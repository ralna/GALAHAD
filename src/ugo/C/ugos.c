/* ugos.c */
/* Spec test for the UGO C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_ugo.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

struct userdata_type {
   rpc_ a;
};

// Evaluate test problem objective, first and second derivatives
ipc_ fgh(rpc_ x, rpc_ *f, rpc_ *g, rpc_ *h, const void *userdata){
   struct userdata_type *myuserdata = (struct userdata_type *) userdata;
   rpc_ a = myuserdata->a;

   *f = x * x * cos( a*x );
   *g = - a * x * x * sin( a*x ) + 2.0 * x * cos( a*x );
   *h = - a * a* x * x * cos( a*x ) - 4.0 * a * x * sin( a*x )
        + 2.0 * cos( a*x );
   return 0;
}

int main(void) {

    // Derived types
    void *data;
    struct ugo_control_type control;
    struct ugo_inform_type inform;

    // Initialize UGO

    ipc_ status;
    ugo_initialize( &data, &control, &status );

    // Set user-defined control options
    // control.print_level = 1;
    // control.maxit = 100;
    // control.lipschitz_estimate_used = 3;

    // User data
    struct userdata_type userdata;
    userdata.a = 10.0;

    // Test problem bounds
    rpc_ x_l = -1.0;
    rpc_ x_u = 2.0;

    // Test problem objective, gradient, Hessian values
    rpc_ x, f, g, h;

    // import problem data
    ugo_import( &control, &data, &status, &x_l, &x_u );

    // Set for initial entry
    status = 1;

    // Call UGO_solve
    ugo_solve_direct( &data, &userdata, &status, &x, &f, &g, &h, fgh );

    // Record solution information
    ugo_information( &data, &inform, &status );

    if(inform.status == 0){
#ifdef REAL_128
        printf("%" i_ipc_ " evaluations. Optimal objective value = %5.2f"
               " at x = %5.2f, status = %1" i_ipc_ "\n", 
               inform.f_eval, (double)f, (double)x, inform.status);
#else
        printf("%" i_ipc_ " evaluations. Optimal objective value = %5.2f"
               " at x = %5.2f, status = %1" i_ipc_ "\n", 
               inform.f_eval, f, x, inform.status);
#endif
    }else{
        printf("UGO_solve exit status = %1" i_ipc_ "\n", inform.status);
    }

    // Delete internal workspace
    ugo_terminate( &data, &control, &inform );
    return 0;
}
