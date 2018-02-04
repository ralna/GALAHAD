   PROGRAM GALAHAD_IR_EXAMPLE  !  GALAHAD 2.3 - 16/10/2008 AT 11:30 GMT.
   USE GALAHAD_IR_double                           ! double precision version
   USE GALAHAD_SMT_double
   USE GALAHAD_SILS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )      ! set precision
   TYPE ( SMT_type ) :: A
   TYPE ( SILS_control ) :: CNTL
   TYPE ( SILS_ainfo ) :: AINFO
   TYPE ( SILS_finfo ) :: FINFO
   TYPE ( SILS_sinfo ) :: SINFO
   TYPE ( SILS_factors) :: FACTORS
   TYPE ( IR_data_type ) :: data
   TYPE ( IR_control_type ) :: control
   TYPE ( IR_inform_type ) :: inform
   INTEGER :: i
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION ( : ) :: B, X
   READ( 5, * ) A%n,  A%ne                        ! read dimensions from file
   ALLOCATE( B( A%n ), X( A%n ) )                 ! Allocate space for A, b and x
   ALLOCATE( A%VAL( A%ne ), A%ROW( A%ne ), A%COL( A%ne ) )
   READ( 5, * ) ( A%ROW( i ), A%COL( i ), A%VAL( i ), i = 1, A%ne )
   READ( 5, * ) B                                 ! read A and b from file
   CALL SILS_INITIALIZE( FACTORS, CNTL )          ! initialize SILS structures
   CALL SILS_ANALYSE( A, FACTORS, CNTL, AINFO )   ! order the rows of A
   IF ( AINFO%FLAG < 0 ) THEN                     ! check for errors
    WRITE( 6,'( A, I0 )' ) ' Failure of SILS_ANALYSE with flag = ', AINFO%FLAG
    STOP
   END IF
   CALL SILS_FACTORIZE( A, FACTORS, CNTL, FINFO ) ! factorize A
   IF( FINFO%FLAG < 0 ) THEN                      ! check for errors
    WRITE( 6,'( A, I0 )' ) ' Failure of SILS_FACTORIZE with flag = ', FINFO%FLAG
    STOP
   END IF
   CALL IR_initialize( data, control )            ! initialize IR structures
   control%itref_max = 2                          ! perform 2 iterations
   control%acceptable_residual_relative = 0.1 * EPSILON( 1.0D0 ) ! high accuracy
   X = B
   CALL IR_SOLVE( A, FACTORS, X, data, control, CNTL, inform ) ! find x
   IF ( inform%status == 0 ) THEN                 ! check for errors
     WRITE( 6, '( A, /, ( 3F20.16 ) )' ) ' Solution after refinement is', X
   ELSE
    WRITE( 6,'( A, I2 )' ) ' Failure of IR_solve with status = ', inform%status
   END IF
   CALL IR_terminate( data, control, inform )     ! delete internal workspace
   CALL SILS_finalize( FACTORS, CNTL, i )
   END PROGRAM GALAHAD_IR_EXAMPLE
