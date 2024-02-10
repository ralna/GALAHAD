/* crotf.c */
/* Full test for the CRO C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_cro.h"

int main(void) {

    // Derived types
    void *data;
    struct cro_control_type control;
    struct cro_inform_type inform;

    // Set problem dimensions
    ipc_ n = 11; // dimension
    ipc_ m = 3; // number of general constraints
    ipc_ m_equal = 1; // number of equality constraints

    //  describe the objective function

    ipc_ H_ne = 21;
    rpc_ H_val[] = {1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,
                        1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0};
    ipc_ H_col[] = {1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11};
    ipc_ H_ptr[] = {1,2,4,6,8,10,12,14,16,18,20,22};
    rpc_ g[] = {0.5,-0.5,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-0.5};

    //  describe constraints

    ipc_ A_ne = 30;
    rpc_ A_val[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    ipc_ A_col[] = {1,2,3,4,5,6,7,8,9,10,11,3,4,5,6,7,8,9,10,11,
                   2,3,4,5,6,7,8,9,10,11 };
    ipc_ A_ptr[] = {1,12,21,31};
    rpc_ c_l[] = {10.0,9.0,-INFINITY};
    rpc_ c_u[] = {10.0,INFINITY,10.0};
    rpc_ x_l[] = {0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    rpc_ x_u[] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,
                      INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};

    // provide optimal variables, Lagrange multipliers and dual variables
    rpc_ x[] = {0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0};
    rpc_ c[] = {10.0,9.0,10.0};
    rpc_ y[] = { -1.0,1.5,-2.0};
    rpc_ z[] = {2.0,4.0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5};
    // provide interior-point constraint and variable status
    ipc_ c_stat[] = {-1,-1,1};
    ipc_ x_stat[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

    // Set output storage

    ipc_ status;

    printf(" Fortran sparse matrix indexing\n\n");

    // Initialize CRO
    cro_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = true; // Fortran sparse matrix indexing

    // crossover the solution
    cro_crossover_solution( &data, &control, &inform,
                            n, m, m_equal,
                            H_ne, H_val, H_col, H_ptr,
                            A_ne, A_val, A_col, A_ptr,
                            g, c_l, c_u, x_l, x_u, x, c, y, z,
                            x_stat, c_stat );

    printf(" CRO_crossover exit status = %1" i_ipc_ "\n", inform.status);

    // Delete internal workspace
    cro_terminate( &data, &control, &inform );

}
