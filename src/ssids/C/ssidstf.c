/* ssidstf.c - Example code for SPRAL_SSIDS package, GALified by */
/* Jari Fowkes, Nick Gould, STFC-Rutherford Appleton Laboratory, */
/* and Alexis Montoison, Argone National Laboratory, 2025 */
#include <stdlib.h>
/*#include <stdint.h>*/
#include <stdio.h>
#include <math.h>
#include "galahad_precision.h"
#include "galahad_cfunctions.h"
#include "spral_ssids.h"
#ifdef REAL_128
#include <quadmath.h>
#endif

int main(void) {
   /* Derived types */
   void *akeep, *fkeep;
   struct spral_ssids_options options;
   struct spral_ssids_inform inform;

   // Initialize derived types
   akeep = NULL; fkeep = NULL; // Important that these are NULL to start with
   spral_ssids_default_options(&options);
   options.array_base = 1; // Fortran sparse matrix indexing
   options.nodend_options.print_level = 0;

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

   // perform analyis and factorization with data checking
   bool check = true;
   spral_ssids_analyse(check, n, NULL, ptr, row, NULL, &akeep, &options,
                       &inform);
   if(inform.flag<0) {
      spral_ssids_free(&akeep, &fkeep);
      exit(1);
   }
   spral_ssids_factor(posdef, NULL, NULL, val, NULL, akeep, &fkeep, &options,
                      &inform);
   if(inform.flag<0) {
      spral_ssids_free( &akeep, &fkeep );
      exit(1);
   }

   // solve
   spral_ssids_solve1(0, x, akeep, fkeep, &options, &inform);
   if(inform.flag<0) {
      spral_ssids_free(&akeep, &fkeep);
      exit(1);
   }
   printf("The computed solution is:");
   for(int i=0; i<n; i++) printf(" %9.2e", x[i]);
   printf("\n");

   /* Determine and print the pivot order */
   ipc_ piv_order[5];
   spral_ssids_enquire_indef(akeep, fkeep, &options, &inform, piv_order, NULL);
   printf("Pivot order:");
   for(int i=0; i<n; i++) printf(" %3d", piv_order[i]);
   printf("\n");

   ipc_ cuda_error = spral_ssids_free(&akeep, &fkeep);
   if(cuda_error!=0) exit(1);

   return 0;
}
