! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_FDC_test    !! to be expanded
   USE GALAHAD_KINDS_precision
   USE GALAHAD_FDC_precision
   IMPLICIT NONE
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 4, m = 3, a_ne = 10
   INTEGER ( KIND = ip_ ) :: A_ptr( m + 1 ), A_col( a_ne )
   REAL ( KIND = rp_ ) :: A_val( a_ne ), B( m )
   INTEGER ( KIND = ip_ ), ALLOCATABLE :: DEPEN( : )
   INTEGER ( KIND = ip_ ) :: n_depen
   TYPE ( FDC_data_type ) :: data
   TYPE ( FDC_control_type ) :: control
   TYPE ( FDC_inform_type ) :: inform
   A_val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 2.0_rp_, -4.0_rp_, 6.0_rp_,  &
             -8.0_rp_, 5.0_rp_, 10.0_rp_ /)
   A_col = (/ 1, 2, 3, 4, 1, 2, 3, 4, 2, 4 /)
   A_ptr = (/ 1, 5, 9, 11 /)
   B = (/ 5.0_rp_, 10.0_rp_, 0.0_rp_ /)
   CALL FDC_initialize( data, control, inform )  ! Initialize control parameters
!  control%print_level = 1
   control%use_sls = .TRUE.
   control%symmetric_linear_solver = 'sytr'
   CALL FDC_find_dependent( n, m, A_val, A_col, A_ptr, B, n_depen, DEPEN,      &
                            data, control, inform ) ! Check for dependencies
   WRITE( 6, "( ' linear solver used: ', A )" ) inform%SLS_inform%solver
   IF ( inform%status == 0 ) THEN              ! Successful return
     IF ( n_depen == 0 ) THEN
       WRITE( 6, "( ' FDC_find_dependent - no dependencies ' )" )
     ELSE
       WRITE( 6, "( ' FDC_find_dependent - dependent constraint(s):', 3I3 )")  &
         DEPEN( : n_depen )
     END IF
   ELSE                                        !  Error returns
     WRITE( 6, "( ' FDC_find_dependent exit status = ', I6 ) " ) inform%status
   END IF
   CALL FDC_terminate( data, control, inform, DEPEN ) ! Delete workspace
   END PROGRAM GALAHAD_FDC_test

