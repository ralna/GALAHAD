! THIS VERSION: GALAHAD 4.3 - 2024-01-04 AT 14:30 GMT

#ifdef GALAHAD_64BIT_INTEGER
  INTEGER, PARAMETER :: ip_ = INT64
#else
  INTEGER, PARAMETER :: ip_ = INT32
#endif

!-*-*-*-*-*-*-  L A N C E L O T  -B-  KB07  S U B R O U T I N E S *-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 13th 1995

      SUBROUTINE KB07AI( COUNT, n, INDEX )

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: n
      INTEGER ( KIND = ip_ ), DIMENSION( : ) :: COUNT, INDEX

      RETURN
      END SUBROUTINE KB07AI

