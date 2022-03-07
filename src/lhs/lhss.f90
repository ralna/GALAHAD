   PROGRAM GALAHAD_LHS_EXAMPLE  !  GALAHAD 2.8 - 29/06/2016 AT 15:40 GMT
   USE GALAHAD_LHS_double       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                ! set precision
   TYPE ( LHS_control_type ) :: control
   TYPE ( LHS_inform_type ) :: inform
   TYPE ( LHS_data_type ) :: data
   INTEGER, PARAMETER :: n_dimen = 7                        ! dimension
   INTEGER, PARAMETER :: n_points = 2                       ! # points required
   INTEGER :: X( n_dimen, n_points )                        ! points
   INTEGER :: j, seed
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
   END PROGRAM GALAHAD_LHS_EXAMPLE
