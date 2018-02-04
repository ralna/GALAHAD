! THIS VERSION: GALAHAD 2.6 - 13/05/2014 AT 15:00 GMT.
   PROGRAM GALAHAD_MIQR_EXAMPLE
   USE GALAHAD_MIQR_double                   ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( SMT_type ) :: A   
   TYPE ( MIQR_data_type ) :: data
   TYPE ( MIQR_control_type ) :: control        
   TYPE ( MIQR_inform_type ) :: inform
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
   INTEGER :: s
! set problem data
   A%m = 4 ; A%n = 3 ; A%ne = 5
! sparse row-wise storage format
   CALL SMT_put( A%type, 'SPARSE_BY_ROWS', s ) ! storage for A
   ALLOCATE( A%ptr( A%m + 1 ), A%col( A%ne ), A%val( A%ne ) ) 
   A%val = (/ 1.0_wp, 2.0_wp, 3.0_wp, 4.0_wp, 5.0_wp /) ! matrix A
   A%col = (/ 1, 2, 1, 2, 3 /)
   A%ptr = (/ 1, 3, 4, 5, 6 /)                   ! set row pointers
! problem data complete
   CALL MIQR_initialize( data, control, inform ) ! Initialize control parameters
   CALL MIQR_form( A, data, control, inform )    ! form factors
   ALLOCATE( SOL( A%n ) )
   SOL = (/ 14.0, 42.0, 75.0 /)                  ! set b 
   CALL MIQR_apply( SOL, .TRUE., data, inform )  ! solve R^T z = b
   WRITE( 6, "( ' z ', /, ( 5ES12.4 ) )" ) SOL
   CALL MIQR_apply( SOL, .FALSE., data, inform ) ! solve R x = z
   WRITE( 6, "( ' x ', /, ( 5ES12.4 ) )" ) SOL
   CALL MIQR_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_MIQR_EXAMPLE
