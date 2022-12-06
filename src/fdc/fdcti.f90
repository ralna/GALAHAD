! THIS VERSION: GALAHAD 4.0 - 20/01/2022 AT 09:30 GMT.
   PROGRAM GALAHAD_FDC_test    !! to be expanded
   USE GALAHAD_FDC_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )   ! set precision
   INTEGER, PARAMETER :: n = 4, m = 3, a_ne = 10
   INTEGER :: n_depen, status
   INTEGER :: A_ptr( m + 1 ), A_col( a_ne )
   REAL ( KIND = wp ) :: A_val( a_ne ), B( m )
   INTEGER :: DEPEN( m )
   TYPE ( FDC_full_data_type ) :: data
   TYPE ( FDC_control_type ) :: control
   TYPE ( FDC_inform_type ) :: inform
   A_val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 2.0_wp, -4.0_wp, 6.0_wp,         &
             -8.0_wp, 5.0_wp, 10.0_wp /)
   A_col = (/ 1, 2, 3, 4, 1, 2, 3, 4, 2, 4 /)
   A_ptr = (/ 1, 5, 9, 11 /)
   B = (/ 5.0_wp, 10.0_wp, 0.0_wp /)
   CALL FDC_initialize( data, control, inform )  ! Initialize control parameters
   CALL FDC_find_dependent_rows( control, data, inform, status, m, n,          &
                                 A_col, A_ptr, A_val, B, n_depen, DEPEN )

! Check for dependencies
   IF ( status == 0 ) THEN              ! Successful return
     IF ( n_depen == 0 ) THEN
       WRITE( 6, "( ' FDC_find_dependent - no dependencies ' )" )
     ELSE
       WRITE( 6, "( ' FDC_find_dependent - dependent constraint(s):', 3I3 )")  &
         DEPEN( : n_depen )
     END IF
   ELSE                                        !  Error returns
     WRITE( 6, "( ' FDC_find_dependent exit status = ', I6 ) " ) inform%status
   END IF
   CALL FDC_terminate( data, control, inform ) ! Delete workspace
   END PROGRAM GALAHAD_FDC_test

