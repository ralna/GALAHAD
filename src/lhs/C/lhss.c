/* lhss.c */
/* Spec test for the LHS C interface */

#include <stdio.h>
#include <math.h>
#include "lhs.h"

int main(void) {

    // Derived types
    void *data;
    struct lhs_control_type control;
    struct lhs_inform_type inform;

    // Initialize LHS
    lhs_initialize(&data, &control, &inform);

    // Parameters
    int n_dimen = 7; // dimension
    int n_points = 2; // points required
    int X[n_dimen][n_points]; // points
    int seed;
    
    // Set a random seed
    lhs_get_seed(&seed);                             

    // Generate points
    lhs_ihs(n_dimen, n_points, &seed, X, &control, &inform, &data);
    if(inform.status == 0){ // successful return
        printf("LHS successful\n");
        for(int j = 0; j < n_points; j++){
            printf("Point %d = ", j);
            for(int i = 0; i < n_dimen; i++){
                printf("%d ", X[i][j]);
            }
            printf("\n");
        }
    }else{ // error returns
        printf("LHS exit status = %d \n", inform.status);
    }

    // Delete internal workspace
    lhs_terminate(&data, &control, &inform);

    return 0;
}
