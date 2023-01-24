! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D   R U N L A N C E L O T _ S I F  *-*-*-*-*-*-*-*-

!  Nick Gould, Dominique Orban and Philippe Toint, for GALAHAD productions
!  Copyright reserved
!  March 14th 2003

   PROGRAM RUNLANCELOT_SIF_precision
   USE GALAHAD_KINDS_precision
   USE GALAHAD_USELANCELOT_precision

!  Main program for the SIF interface to LANCELOT B, an augmented Lagrangian
!  algorithm for nonlinear programming

!  Problem input characteristics

   INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
   CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Open the data input file

   OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD'  )
   REWIND input

!  Call the SIF interface

   CALL USE_LANCELOT( input )

!  Close the data input file

   CLOSE( input  )
   STOP

!  End of RUNLANCELOT_SIF_precision

   END PROGRAM RUNLANCELOT_SIF_precision
