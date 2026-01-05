/* ssidstf.c - Example code for GALAHAD_SSIDS package, GALified by */
/* Jari Fowkes, Nick Gould, STFC-Rutherford Appleton Laboratory, */
/* and Alexis Montoison, Argone National Laboratory, 2025-08-19 */
#include <stdlib.h>
/*#include <stdint.h>*/
#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "galahad_ssids.h"
#ifdef REAL_128
#endif

int main(void) {
   /* Derived types */
   void *akeep, *fkeep;
   struct ssids_control_type control;
   struct ssids_inform_type inform;

   // Initialize derived types
   akeep = NULL; fkeep = NULL; // Important that these are NULL to start with
   ssids_default_control(&control);
   control.array_base = 1; // Fortran sparse matrix indexing
   control.nodend_control.print_level = 0;

    printf(" Fortran sparse matrix indexing\n\n");

   /* Data for matrix:
    * ( 2  1         )
    * ( 1  4  1    1 )
    * (    1  3  2   )
    * (       2 -1   )
    * (    1       2 ) */
   bool posdef = false;
   ipc_ n = 5;
   longc_ ptr[] = {1, 3, 6, 8, 9, 10};
   ipc_ row[] = {1, 2, 2, 3, 5, 3, 4, 4, 5};
   rpc_ val[] = {2.0, 1.0, 4.0, 1.0, 1.0, 3.0, 2.0, -1.0, 2.0};

   // The right-hand side with solution (1.0, 2.0, 3.0, 4.0, 5.0)
   rpc_ x[] = {4.0, 17.0, 19.0, 2.0, 12.0};

   // perform analysis and factorization with data checking
   bool check = true;
   ssids_analyse(check, n, NULL, ptr, row, NULL, &akeep, &control, &inform);
   if(inform.flag<0) {
      ssids_free(&akeep, &fkeep);
      exit(1);
   }
   ssids_factor(posdef, NULL, NULL, val, NULL, akeep, &fkeep, &control,
                &inform);
   if(inform.flag<0) {
      ssids_free( &akeep, &fkeep );
      exit(1);
   }

   // solve
   ssids_solve1(0, x, akeep, fkeep, &control, &inform);
   if(inform.flag<0) {
      ssids_free(&akeep, &fkeep);
      exit(1);
   }
   printf("The computed solution is:");
   for(int i=0; i<n; i++) printf(" %9.2e", (double) x[i]);
   printf("\n");

   /* Determine and print the pivot order */
   ipc_ piv_order[5];
   ssids_enquire_indef(akeep, fkeep, &control, &inform, piv_order, NULL);
   printf("Pivot order:");
   for(int i=0; i<n; i++) printf(" %3d", (int) piv_order[i]);
   printf("\n");

   ipc_ cuda_error = ssids_free(&akeep, &fkeep);
   if(cuda_error!=0) exit(1);

   return 0;
}
