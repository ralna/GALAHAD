! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_LHS_TEST  !! far from complete
   USE GALAHAD_KINDS_precision
   USE GALAHAD_LHS_precision
   IMPLICIT NONE
   TYPE ( LHS_control_type ) :: control
   TYPE ( LHS_inform_type ) :: inform
   TYPE ( LHS_data_type ) :: data
   INTEGER ( KIND = ip_ ), PARAMETER :: n_dimen = 7    ! dimension
   INTEGER ( KIND = ip_ ), PARAMETER :: n_points = 2   ! # points required
   INTEGER ( KIND = ip_ ) :: X( n_dimen, n_points )    ! points
   INTEGER ( KIND = ip_ ) :: j, seed
   CALL LHS_initialize( data, control, inform )             ! set controls
   CALL LHS_get_seed( seed )                                ! set a random seed
   CALL LHS_ihs( n_dimen, n_points, seed, X,                                   &
                 control, inform, data )                    ! generate points
   IF ( inform%status == 0 ) THEN                           ! Successful return
    DO j = 1, n_points
      WRITE( 6, "( ' point ', I0, ' =', 7I3 )" ) j, X( : , j )
    END DO
   ELSE                                                      ! Error returns
     WRITE( 6, "( ' LHS_ihs exit status = ', I6 ) " ) inform%status
   END IF
   CALL LHS_terminate( data, control, inform )    ! deallocate workspace arrays
   END PROGRAM GALAHAD_LHS_TEST
