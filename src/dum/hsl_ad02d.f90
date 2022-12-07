! THIS VERSION: 25/06/2002 AT 14:00:00 PM.
! Updated 25/06/2002: additional warning information added

!-*-*-*-*-*  L A N C E L O T  -B-  DUMMY AD02_FORWARD  M O D U L E S *-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  June 28th 1996

MODULE HSL_AD02_FORWARD_DOUBLE

      IMPLICIT NONE
      PRIVATE
      INTEGER, PARAMETER :: WP = KIND(1D0)
      PUBLIC :: AD02_INITIALIZE
    
!  Dummy HSL_AD02_FORWARD_DOUBLE module

      TYPE, PUBLIC :: AD02_REAL
        PRIVATE
        INTEGER :: P
      END TYPE AD02_REAL

      TYPE, PUBLIC :: AD02_DATA
        PRIVATE
        INTEGER :: LEVEL
      END TYPE AD02_DATA

CONTAINS

      SUBROUTINE AD02_INITIALIZE(DEGREE,A,VALUE,DATA,FULL_THRESHOLD)
        INTEGER, INTENT (IN) :: DEGREE
!       TYPE (AD02_REAL), INTENT (OUT) :: A
        TYPE (AD02_REAL) :: A
        INTEGER, OPTIONAL, INTENT (IN) :: FULL_THRESHOLD
        TYPE (AD02_DATA), POINTER :: DATA
        REAL (WP), INTENT (IN) :: VALUE
    
!  Dummy subroutine available with LANCELOT

        WRITE ( 6, 2000 )
        STOP

!  Non-executable statements

 2000    FORMAT( /, ' We regret that the solution options that you have ', /, &
                    ' chosen are not all freely available with LANCELOT.', //,&
                    ' If you have HSL (formerly the Harwell Subroutine',      &
                    ' Library), this ', /,                                    &
                    ' option may be enabled by replacing the dummy ', /,      &
                    ' module HSL_AD02_FORWARD_DOUBLE with its H...', /,       &
                    ' namesake and dependencies. See', /,                     &
                    '   $GALAHAD/src/makedefs/packages for details.', //,     &
                    ' *** EXECUTION TERMINATING *** ', / )

      END SUBROUTINE AD02_INITIALIZE

END MODULE HSL_AD02_FORWARD_DOUBLE

!  THIS VERSION: 28/06/1996 AT 09:00:00 AM

!-*-*-*-*-*  L A N C E L O T  -B-  DUMMY AD02_BACKWARD  M O D U L E S *-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  June 28th 1996

MODULE HSL_AD02_BACKWARD_DOUBLE

      IMPLICIT NONE
      PRIVATE
      INTEGER, PARAMETER :: WP = KIND(1D0)
      PUBLIC :: AD02_INITIALIZE
    
!  Dummy HSL_AD02_BACKWARD_DOUBLE module

      TYPE, PUBLIC :: AD02_REAL
        PRIVATE
        INTEGER :: P
      END TYPE AD02_REAL

      TYPE, PUBLIC :: AD02_DATA
        PRIVATE
        INTEGER :: LEVEL
      END TYPE AD02_DATA

CONTAINS

      SUBROUTINE AD02_INITIALIZE(DEGREE,A,VALUE,DATA,FULL_THRESHOLD)
        INTEGER, INTENT (IN) :: DEGREE
!       TYPE (AD02_REAL), INTENT (OUT) :: A
        TYPE (AD02_REAL) :: A
        INTEGER, OPTIONAL, INTENT (IN) :: FULL_THRESHOLD
        TYPE (AD02_DATA), POINTER :: DATA
        REAL (WP), INTENT (IN) :: VALUE
    
!  Dummy subroutine available with LANCELOT

        WRITE ( 6, 2000 )
        STOP

!  Non-executable statements

 2000    FORMAT( /, ' We regret that the solution options that you have ', /, &
                    ' chosen are not all freely available with LANCELOT.', //,&
                    ' If you have HSL (formerly the Harwell Subroutine',      &
                    ' Library), this ', /,                                    &
                    ' option may be enabled by replacing the dummy ', /,      &
                    ' module HSL_AD02_BACKWARD_DOUBLE with its HSL ',/,       &
                    ' namesake and dependencies. See', /,                     &
                    '   $GALAHAD/src/makedefs/packages for details.', //,     &
                    ' *** EXECUTION TERMINATING *** ', / )

      END SUBROUTINE AD02_INITIALIZE

END MODULE HSL_AD02_BACKWARD_DOUBLE
