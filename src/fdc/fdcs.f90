! THIS VERSION: GALAHAD 2.4 - 26/04/2010 AT 13:30 GMT.
   PROGRAM GALAHAD_FDC_example
   USE GALAHAD_FDC_double                      ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )   ! set precision
   INTEGER, PARAMETER :: n = 4, m = 3, a_ne = 10
   INTEGER :: A_ptr( m + 1 ), A_col( a_ne )
   REAL ( KIND = wp ) :: A_val( a_ne ), C( m )
   INTEGER, ALLOCATABLE :: C_depen( : ) ! Remeber to nullify C_depen
   INTEGER :: n_depen
   TYPE ( FDC_data_type ) :: data
   TYPE ( FDC_control_type ) :: control        
   TYPE ( FDC_inform_type ) :: inform
   A_val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 2.0_wp, -4.0_wp, 6.0_wp,         &
             -8.0_wp, 5.0_wp, 10.0_wp /)
   A_col = (/ 1, 2, 3, 4, 1, 2, 3, 4, 2, 4 /)
   A_ptr = (/ 1, 5, 9, 11 /)
   C = (/ 5.0_wp, 10.0_wp, 0.0_wp /)
   CALL FDC_initialize( data, control, inform )  ! Initialize control parameters
   CALL FDC_find_dependent( n, m, A_val, A_col, A_ptr, C, n_depen, C_depen,    &
     data, control, inform )                  ! Check for dependencies
   IF ( inform%status == 0 ) THEN              ! Successful return
     IF ( n_depen == 0 ) THEN
       WRITE( 6, "( ' FDC_find_dependent - no dependencies ' )" )
     ELSE
       WRITE( 6, "( ' FDC_find_dependent - dependent constraint(s):', 3I3 )")  &
         C_depen
     END IF
   ELSE                                        !  Error returns
     WRITE( 6, "( ' FDC_find_dependent exit status = ', I6 ) " ) inform%status
   END IF
   CALL FDC_terminate( data, control, inform, C_depen ) ! Delete workspace
   END PROGRAM GALAHAD_FDC_example

