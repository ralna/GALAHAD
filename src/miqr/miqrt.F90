! THIS VERSION: GALAHAD 4.2 - 2023-08-10 AT 07:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_MIQR_TEST
   USE GALAHAD_KINDS_precision
   USE GALAHAD_MIQR_precision
   IMPLICIT NONE
   TYPE ( SMT_type ) :: mat
   TYPE ( MIQR_data_type ) :: data
   TYPE ( MIQR_control_type ) :: control
   TYPE ( MIQR_inform_type ) :: inform
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: SOL
   INTEGER ( KIND = ip_ ) :: s
   INTEGER ( KIND = ip_ ), PARAMETER :: prob_number = 1
! set problem data
   SELECT CASE ( prob_number )
   CASE ( 2 )
     mat%m = 3 ; mat%n = 3 ; mat%ne = 4
   CASE ( 3 )
     mat%m = 3 ; mat%n = 3 ; mat%ne = 4
   CASE DEFAULT
     mat%m = 4 ; mat%n = 3 ; mat%ne = 5
   END SELECT
   ALLOCATE( mat%ptr( mat%m + 1 ), mat%col( mat%ne ), mat%val( mat%ne ) )
   SELECT CASE ( prob_number )
   CASE ( 2 )
     mat%ptr = (/ 1, 3, 4, 5 /)
     mat%col = (/ 1, 2, 1, 3 /)
     mat%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_ /)
   CASE ( 3 )
     mat%ptr = (/ 1, 3, 4, 5 /)
     mat%col = (/ 1, 2, 2, 3 /)
     mat%val = (/ 2.0_rp_, 1.0_rp_, 3.0_rp_, 4.0_rp_ /)
   CASE DEFAULT
     mat%ptr = (/ 1, 3, 4, 5, 6 /)
     mat%col = (/ 1, 2, 1, 2, 3 /)
     mat%val = (/ 1.0_rp_, 2.0_rp_, 3.0_rp_, 4.0_rp_, 5.0_rp_ /)
   END SELECT
   CALL SMT_put( mat%type, 'SPARSE_BY_ROWS', s )
! problem data complete
   CALL MIQR_initialize( data, control, inform ) ! Initialize control parameters
   control%print_level = 1
!  control%multi_level = .FALSE.
!  data%control%deallocate_after_factorization = .TRUE.
   CALL MIQR_form( mat, data, control, inform ) ! form factors
   ALLOCATE( SOL( mat%n ) )
   SELECT CASE ( prob_number )
   CASE ( 2 )
     SOL = (/ 14.0, 10.0, 48.0 /)
   CASE ( 3 )
     SOL = (/ 8.0, 22.0, 48.0 /)
   CASE DEFAULT
     SOL = (/ 14.0, 42.0, 75.0 /)
   END SELECT
   CALL MIQR_apply( SOL, .TRUE., data, inform )
   WRITE( 6, "( ' sol(transpose) ', /, ( 5ES12.4 ) )" ) SOL
   CALL MIQR_apply( SOL, .FALSE., data, inform )
   WRITE( 6, "( ' sol ', /, ( 5ES12.4 ) )" ) SOL
   CALL MIQR_terminate( data, control, inform )  !  delete internal workspace
   DEALLOCATE( SOL )
   END PROGRAM GALAHAD_MIQR_TEST

