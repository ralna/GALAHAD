/* crot.c */
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
    int n = 11; // dimension
    int m = 3; // number of general constraints
    int m_equal = 1; // number of equality constraints

    //  describe the objective function

    int H_ne = 21;
    real_wp_ H_val[] = {1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,
                        1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0,0.5,1.0};
    int H_col[] = {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10};
    int H_ptr[] = {0,1,3,5,7,9,11,13,15,17,19,21};
    real_wp_ g[] = {0.5,-0.5,-1.0,-1.0,-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-0.5};

    //  describe constraints

    int A_ne = 30;
    real_wp_ A_val[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                        1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    int A_col[] = {0,1,2,3,4,5,6,7,8,9,10,2,3,4,5,6,7,8,9,10,
                   1,2,3,4,5,6,7,8,9,10};
    int A_ptr[] = {0,11,20,30};
    real_wp_ c_l[] = {10.0,9.0,-INFINITY};
    real_wp_ c_u[] = {10.0,INFINITY,10.0};
    real_wp_ x_l[] = {0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    real_wp_ x_u[]  = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,
                       INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};

    // provide optimal variables, Lagrange multipliers and dual variables
    real_wp_ x[] = {0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0, 1.0,1.0,1.0};
    real_wp_ c[] = {10.0,9.0,10.0};
    real_wp_ y[] = { -1.0,1.5,-2.0};
    real_wp_ z[] = {2.0,4.0,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5};
    // provide interior-point constraint and variable status
    int c_stat[] = {-1,-1,1};
    int x_stat[] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

    // Set output storage

    char st;
    int status;

    printf(" C sparse matrix indexing\n\n");

    // Initialize CRO
    cro_initialize( &data, &control, &status );

    // Set user-defined control options
    control.f_indexing = false; // C sparse matrix indexing

    // crossover the solution
    cro_crossover_solution( &data, &control, &inform, 
                            n, m, m_equal, 
                            H_ne, H_val, H_col, H_ptr, 
                            A_ne, A_val, A_col, A_ptr, 
                            g, c_l, c_u, x_l, x_u, x, c, y, z, 
                            x_stat, c_stat );

    printf(" CRO_crossover exit status = %1i\n", inform.status);

    // Delete internal workspace
    cro_terminate( &data, &control, &inform );

}
