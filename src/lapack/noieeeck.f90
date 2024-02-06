#include "galahad_lapack.h"

       INTEGER(ip_) FUNCTION IEEECK(ispec, zero, one)
         USE GALAHAD_KINDS
         INTEGER(ip_) :: ispec
         REAL :: one, zero
         REAL :: nan1, nan2, nan3, nan4, nan5, nan6, neginf, negzro,      &
           newzro, posinf
         IEEECK = 0
         RETURN
       END FUNCTION
