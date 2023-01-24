#include "galahad_modules.h"

   PROGRAM SILS_EXAMPLE  !  GALAHAD 3.3 - 05/05/2021 AT 16:30 GMT.
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SMT_precision
   USE GALAHAD_SILS_precision
   IMPLICIT NONE
   TYPE( SMT_type ) :: matrix
   TYPE( SILS_control ) :: control
   TYPE( SILS_ainfo ) :: ainfo
   TYPE( SILS_finfo ) :: finfo
   TYPE( SILS_sinfo ) :: sinfo
   TYPE( SILS_factors ) :: factors
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 5
   INTEGER ( KIND = ip_ ), PARAMETER :: ne = 7
   REAL ( KIND = rp_ ) :: B( n ), X( n )
   INTEGER ( KIND = ip_ ) :: i
! Read matrix order and number of entries
   matrix%n = n
   matrix%ne = ne
! Allocate and set matrix
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 1, 2, 2, 3, 3, 5 /)
   matrix%col( : ne ) = (/ 1, 2, 3, 5, 3, 4, 5 /)
   matrix%val( : ne ) = (/ 2.0_rp_, 3.0_rp_, 4.0_rp_, 6.0_rp_, 1.0_rp_,        &
                           5.0_rp_, 1.0_rp_ /)
   CALL SMT_put( matrix%type, 'COORDINATE', i )     ! Specify co-ordinate
! Set right-hand side
   B( : n ) = (/ 8.0_rp_,  45.0_rp_,  31.0_rp_,  15.0_rp_,  17.0_rp_ /)
! Initialize the structures
   CALL SILS_INITIALIZE( factors, control )
! Analyse
   CALL SILS_ANALYSE( matrix, factors, control, ainfo )
   IF ( ainfo%FLAG < 0 ) THEN
    WRITE(6,"( ' Failure of SILS_ANALYSE with AINFO%FLAG=',I0)" ) ainfo%FLAG
    STOP
   END IF
! Factorize
   CALL SILS_FACTORIZE( matrix, factors, control, finfo )
   IF ( finfo%FLAG < 0 ) THEN
     WRITE(6,"(' Failure of SILS_FACTORIZE with FINFO%FLAG=', I0)") finfo%FLAG
     STOP
   END IF
! Solve without refinement
   X = B
   CALL SILS_SOLVE( matrix, factors, X, control, sinfo )
   IF ( sinfo%FLAG == 0 ) WRITE(6,                                             &
     "(' Solution without refinement is',/,(5F12.4))") X
! Perform one refinement
   CALL SILS_SOLVE( matrix, factors, X, control, sinfo, B )
   IF( sinfo%FLAG == 0 ) WRITE(6,                                              &
     "(' Solution after one refinement is',/,(5F12.4))") X
! Clean up
   DEALLOCATE( matrix%type, matrix%val, matrix%row, matrix%col )
   STOP
END PROGRAM SILS_EXAMPLE
