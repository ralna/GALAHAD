! THIS VERSION: GALAHAD 3.3 - 08/07/2021 AT 09:15 GMT

!-*-*-*-*-*-*-*-*-  G A L A H A D _ D G O   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Nick Gould

!  History -
!   originally released GALAHAD Version 3.3. July 7th 2021

!  Return a description of a GALAHAD status error code

   PROGRAM GALAHAD_error
   USE GALAHAD_SYMBOLS
   INTEGER, PARAMETER :: in = 5
   INTEGER, PARAMETER :: out = 6
   INTEGER :: status
   READ( in, * ) status
   CALL  SYMBOLS_status( status, out, '', 'GALAHAD' )
   STOP
   END PROGRAM GALAHAD_error
