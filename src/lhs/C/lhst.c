/* lhss.c */
/* Spec test for the LHS C interface */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_lhs.h"

int main(void) {

    // Derived types
    void *data;
    struct lhs_control_type control;
    struct lhs_inform_type inform;

    // Initialize LHS
    lhs_initialize(&data, &control, &inform);

    // Parameters
    ipc_ n_dimen = 7; // dimension
    ipc_ n_points = 2; // points required
    ipc_ X[n_points][n_dimen]; // points
    ipc_ seed;

    // Set a random seed
    lhs_get_seed(&seed);
    seed = 1;
    // Generate points
    lhs_ihs(n_dimen, n_points, &seed, (ipc_*)X, &control, &inform, &data);
    if(inform.status == 0){ // successful return
        printf("LHS successful\n");
        for(ipc_ j = 0; j < n_points; j++){
            printf("Point %" d_ipc_ " = ", j);
            for(ipc_ i = 0; i < n_dimen; i++){
                printf("%" d_ipc_ " ", X[j][i]);
            }
            printf("\n");
        }
    }else{ // error returns
        printf("LHS exit status = %" d_ipc_ " \n", inform.status);
    }

    // Delete internal workspace
    lhs_terminate(&data, &control, &inform);

    return 0;
}
