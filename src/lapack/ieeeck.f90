#include "galahad_lapack.h"

       INTEGER(ip_) FUNCTION IEEECK(ispec, zero, one)
         USE GALAHAD_KINDS
         INTEGER(ip_) :: ispec
         REAL :: one, zero
         REAL :: nan1, nan2, nan3, nan4, nan5, nan6, neginf, negzro,      &
           newzro, posinf
         IEEECK = 1
         posinf = one/zero
         IF (posinf<=one) THEN
           IEEECK = 0
           RETURN
         END IF
         neginf = -one/zero
         IF (neginf>=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         negzro = one/(neginf+one)
         IF (negzro/=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         neginf = one/negzro
         IF (neginf>=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         newzro = negzro + zero
         IF (newzro/=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         posinf = one/newzro
         IF (posinf<=one) THEN
           IEEECK = 0
           RETURN
         END IF
         neginf = neginf*posinf
         IF (neginf>=zero) THEN
           IEEECK = 0
           RETURN
         END IF
         posinf = posinf*posinf
         IF (posinf<=one) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (ispec==0) RETURN
         nan1 = posinf + neginf
         nan2 = posinf/neginf
         nan3 = posinf/posinf
         nan4 = posinf*zero
         nan5 = neginf*negzro
         nan6 = nan5*zero
         IF (nan1==nan1) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan2==nan2) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan3==nan3) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan4==nan4) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan5==nan5) THEN
           IEEECK = 0
           RETURN
         END IF
         IF (nan6==nan6) THEN
           IEEECK = 0
           RETURN
         END IF
         RETURN
       END FUNCTION
