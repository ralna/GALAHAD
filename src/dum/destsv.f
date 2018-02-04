! THIS VERSION: 21/01/2009 AT 09:00 GMT.

!-*-*-*-*-  G A L A H A D   d u m m y  D E S T S V  S U B R O U T I N E  -*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 21st 2009

!  ---------------------------------------------------------------
!  This subroutine definition should be replaced by its actual
!  MINPACK2 counterpart if available

      subroutine destsv(n,r,ldr,svmin,z)
      integer ldr, n
      double precision svmin
      double precision r(ldr,n), z(n)

!  Dummy subroutine DESTSV available with GALAHAD

      WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

 2000 FORMAT( /, 
     *    ' We regret that the solution options that you have', /, 
     *    ' chosen are not all freely available with GALAHAD.', /,
     *    ' This code is part of the GQT package (itself part', /, 
     *    " of MINPACK2), and may be obtained from Jorge More'",/, 
     *    ' (more@mcs.anl.gov)by ftp froe:', /, 
     *    '   ftp://ftp.mcs.anl.gov/pub/MINPACK-2/gqt/', /, 
     *    ' If you have the GQT package, this option may be ', /,  
     *    ' enabled by replacing this dummy subroutine',/,   
     *    ' with its MINPACK2 namesake and recompiling it.', /,
     *    ' See $GALAHAD/src/makedefs/packages for details.', //,
     *    ' *** EXECUTION TERMINATING *** ', / )

!  End of dummy subroutine DESTSV

      end
