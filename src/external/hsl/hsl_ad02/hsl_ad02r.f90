! THIS VERSION: GALAHAD 5.0 - 2024-03-17 AT 11:25 GMT.

#include "hsl_subset.h"

!-*-*-*-*-*  L A N C E L O T  -B-  DUMMY AD02_FORWARD  M O D U L E S *-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  June 28th 1996

  MODULE hsl_ad02_forward_real
      USE HSL_KINDS_real, ONLY: ip_, rp_
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: AD02_INITIALIZE

!  Dummy HSL_AD02_FORWARD_REAL module

      TYPE, PUBLIC :: AD02_REAL
        PRIVATE
        INTEGER ( KIND = ip_ ) :: p
      END TYPE AD02_REAL

      TYPE, PUBLIC :: AD02_DATA
        PRIVATE
        INTEGER ( KIND = ip_ ) :: level
      END TYPE AD02_DATA

  CONTAINS

      SUBROUTINE AD02_INITIALIZE(DEGREE,A,VALUE,DATA,FULL_THRESHOLD)
      INTEGER ( KIND = ip_ ), INTENT (IN) :: DEGREE
!     TYPE (AD02_REAL), INTENT (OUT) :: A
      TYPE (AD02_REAL) :: A
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT (IN) :: FULL_THRESHOLD
      TYPE (AD02_DATA), POINTER :: DATA
      REAL ( KIND = rp_ ), INTENT (IN) :: VALUE

!  Dummy subroutine available with LANCELOT

      WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

 2000  FORMAT( /, ' We regret that the solution options that you have ', /, &
                  ' chosen are not all freely available with LANCELOT.', //,&
                  ' If you have HSL (formerly the Harwell Subroutine',      &
                  ' Library), this ', /,                                    &
                  ' option may be enabled by replacing the dummy ', /,      &
                  ' module HSL_AD02_FORWARD_REAL with its H...', /,         &
                  ' namesake and dependencies. See', /,                     &
                  '   $GALAHAD/src/makedefs/packages for details.', //,     &
                  ' *** EXECUTION TERMINATING *** ', / )

      END SUBROUTINE AD02_INITIALIZE

END MODULE hsl_ad02_forward_real

!-*-*-*-*-*  L A N C E L O T  -B-  DUMMY AD02_BACKWARD  M O D U L E S *-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  June 28th 1996

MODULE hsl_ad02_backward_real
      USE HSL_KINDS_real, ONLY: ip_, rp_
      IMPLICIT NONE
      PRIVATE
      PUBLIC :: AD02_INITIALIZE

!  Dummy HSL_AD02_BACKWARD_REAL module

      TYPE, PUBLIC :: AD02_REAL
        PRIVATE
        INTEGER ( KIND = ip_ ) :: p
      END TYPE AD02_REAL

      TYPE, PUBLIC :: AD02_DATA
        PRIVATE
        INTEGER ( KIND = ip_ ) :: level
      END TYPE AD02_DATA

CONTAINS

      SUBROUTINE AD02_INITIALIZE(DEGREE,A,VALUE,DATA,FULL_THRESHOLD)
      INTEGER ( KIND = ip_ ), INTENT (IN) :: DEGREE
!     TYPE (AD02_REAL), INTENT (OUT) :: A
      TYPE (AD02_REAL) :: A
      INTEGER ( KIND = ip_ ), OPTIONAL, INTENT (IN) :: FULL_THRESHOLD
      TYPE (AD02_DATA), POINTER :: DATA
      REAL ( KIND = rp_ ), INTENT (IN) :: VALUE

!  Dummy subroutine available with LANCELOT

      WRITE ( 6, 2000 )
      STOP

!  Non-executable statements

 2000 FORMAT( /, ' We regret that the solution options that you have ', /,  &
                  ' chosen are not all freely available with LANCELOT.', //,&
                  ' If you have HSL (formerly the Harwell Subroutine',      &
                  ' Library), this ', /,                                    &
                  ' option may be enabled by replacing the dummy ', /,      &
                  ' module HSL_AD02_BACKWARD_REAL with its HSL ',/,         &
                  ' namesake and dependencies. See', /,                     &
                  '   $GALAHAD/src/makedefs/packages for details.', //,     &
                  ' *** EXECUTION TERMINATING *** ', / )

      END SUBROUTINE AD02_INITIALIZE

  END MODULE hsl_ad02_backward_real
