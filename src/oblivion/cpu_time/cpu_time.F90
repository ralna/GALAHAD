! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D   C P U _ T I M E   M O D U L E  *-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 27th 1995

    MODULE GALAHAD_CPU_time

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CPU_TIME

    CONTAINS

!-*-*-*-*-*-  G A L A H A D   C P U _ T I M E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE CPU_TIME( time )

!  Return the cpu time (to be replaced in 1995 by a Fortran intrinsic)

!NAG  USE F90_UNIX

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      REAL ( KIND = KIND( 1.0E0 ) ), INTENT( OUT ) :: time

! For all machines using the NAG f90 compiler

!NAG  TYPE( TMS ) BUFFER
!NAG  INTEGER :: i
!NAG  i = TIMES( BUFFER )
!NAG  time = REAL( BUFFER%UTIME ) / REAL( CLOCK_TICKS_PER_SECOND( ) )

! For CRAY (UNICOS)

!CRAY CALL SECOND( time )

! For general Unix etime (Linux, DEC/Compaq OSF/Ultrix/Tru64, HP, SUN, SGI ...)

!GEN  REAL ( KIND = KIND( 1.0E0 ) ) :: ETIME, DUMMY( 2 )
!GEN  EXTERNAL ETIME
!GEN  time = ETIME( DUMMY )

! For DEC (VAX D arithmetic)

!VAXD INCLUDE '($JPIDEF)'
!VAXD INTEGER  i
!VAXD CALL LIB$GETJPI( JPI$_CPUTIM, , , i )
!VAXD time = I / 100.

! For DEC (VAX G arithmetic)

!VAXG INCLUDE '($JPIDEF)'
!VAXG INTEGER          i
!VAXG CALL LIB$GETJPI( JPI$_CPUTIM, , , i )
!VAXG time = I / 100.

! For IBM (AIX)

!RS6  INTEGER, EXTERNAL :: MCLOCK
!RS6  time = REAL( MCLOCK( ) ) / 100.0

! For IBM (CMS)

!IBM  INTEGER :: i
!IBM  REAL ( KIND = KIND = KIND( 1.0D0 ) ) :: dum8
!IBM  CALL CPUTIME( dum8, i )
!IBM  time = dum8 / 1000000.0

! For PC (Waterloo Fortran Compiler)

!WFC  INCLUDE 'FSUBLIB.FI'
!WFC  INTEGER * 2  hrs, mins, secs, hsecs
!WFC  CALL GETTIM( hrs, mins, secs, hsecs )
!WFC  time = 3600. * hrs + 60. * mins + secs + hsecs / 100.

! For PC (Salford Fortran Compiler)

!SALF CALL CLOCK@(DUM)
!SALF time = DUM

      RETURN

!  End of subroutine CPU_TIME

      END SUBROUTINE CPU_TIME

!  End of module GALAHAD_CPU_time

    END MODULE GALAHAD_CPU_time
