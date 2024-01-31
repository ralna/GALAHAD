! THIS VERSION: GALAHAD 4.1 - 2022-12-08 AT 07:05 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*  G A L A H A D _ S M T   M O D U L E  *-*-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. December 1st 1997
!   update released with GALAHAD Version 2.0. February 16th 2005
!   replaced by the current use of the functionally-equivalent
!    HSL package ZD11 in GALAHAD 2.4. June 18th, 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SMT_precision

     USE hsl_zd11_precision, SMT_type => ZD11_type, SMT_put => ZD11_put,       &
                             SMT_get => ZD11_get

!  ==========================
!  sparse matrix derived type
!  ==========================

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SMT_type, SMT_put, SMT_get

!  End of module GALAHAD_SMT

   END MODULE GALAHAD_SMT_precision
