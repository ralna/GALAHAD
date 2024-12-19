! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 16:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-  L A N C E L O T  -B-  H S L _ r o u t i n e s   M O D U L E  -*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 5th 1995

   MODULE LANCELOT_HSL_inter_precision

!  Define generic interfaces to HSL routines

     USE GALAHAD_HSL_inter_precision, ONLY: MA27_initialize => MA27I,          &
                                            MA27_analyse => MA27A,             &
                                            MA27_factorize => MA27B,           &
                                            MA27_solve => MA27C,               &
                                            MA61_initialize => MA61I,          &
                                            MA61_compress => MA61D

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: MA27_initialize, MA27_analyse, MA27_factorize, MA27_solve,      &
               MA61_initialize, MA61_compress

!  End of module LANCELOT_HSL_inter

   END MODULE LANCELOT_HSL_inter_precision
