! THIS VERSION: GALAHAD 4.1 - 2022-11-02 AT 15:50 GMT.

!-*-*-*-  G A L A H A D  -  D U M M Y   M U M P S   S U B R O U T I N E  -*-*-*-

      SUBROUTINE DMUMPS( mumps_par )
      USE GALAHAD_MUMPS_TYPES_double
      IMPLICIT NONE
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )
      TYPE ( DMUMPS_STRUC( long = long ) ) :: mumps_par
      mumps_par%INFOG( 1 ) = - 999  ! error code
      RETURN
      END SUBROUTINE DMUMPS
