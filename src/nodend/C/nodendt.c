/* nodendt.c */
/* Full test for the NODEND C interface using C sparse matrix indexing */

#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_nodend.h"

int main(void) {

    // Derived types
    void *data;
    struct nodend_control_type control;
    struct nodend_inform_type inform;

    // Set problem data
    ipc_ n = 10; // dimension
    ipc_ A_ne = 2 * n - 1; // Hesssian elements, NB lower triangle
    ipc_ A_dense_ne = n * ( n + 1 ) / 2; // dense Hessian elements
    ipc_ A_row[A_ne]; // row indices,
    ipc_ A_col[A_ne]; // column indices
    ipc_ A_ptr[n+1];  // row pointers
    ipc_ perm[n]; // permutation,

    // Set output storage
    char st = ' ';
    ipc_ i, l, status;

    // A = tridiag(2,1)

    l = 0 ;
    A_ptr[0] = l;
    A_row[l] = 0; A_col[l] = 0;
    for( ipc_ i = 1; i < n; i++)
    {
      l = l + 1;
      A_ptr[i] = l;
      A_row[l] = i; A_col[l] = i - 1;
      l = l + 1;
      A_row[l] = i; A_col[l] = i;
    }
    A_ptr[n] = l + 1;

    printf(" C sparse matrix indexing\n\n");

    printf(" basic tests of nodend storage formats\n\n");

    for( ipc_ d=1; d <= 3; d++){

        // Initialize NODEND
        nodend_initialize( &data, &control, &status );

        // Set user-defined control options
        control.f_indexing = false; // C sparse matrix indexing

        switch(d){
            case 1: // sparse co-ordinate storage
                st = 'C';
                nodend_order( &control, &data, &status, n, perm,
                              "coordinate", A_ne, A_row, A_col, NULL );
                break;
            printf(" case %1" d_ipc_ " break\n",d);
            case 2: // sparse by rows
                st = 'R';
                nodend_order( &control, &data, &status, n, perm,
                              "sparse_by_rows", A_ne, NULL, A_col, A_ptr );
                break;
            case 3: // dense
                st = 'D';
                nodend_order( &control, &data, &status, n, perm,
                              "dense", A_dense_ne, NULL, NULL, NULL );
                break;
            }
        nodend_information( &data, &inform, &status );

        if(inform.status == 0){
            printf("%c: NODEND_order success, perm: ", st);
            for( i = 0; i < n; i++) printf("%1" d_ipc_ " ", perm[i]);
            printf("\n");
        }else{
            printf("%c: NODEND_order exit status = %1" d_ipc_ "\n", 
                   st, inform.status);
        }

        // Terminate NODEND
        nodend_terminate( &data );

    }
}
