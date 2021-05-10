   PROGRAM SILS_EXAMPLE  !  GALAHAD 3.3 - 05/05/2021 AT 16:30 GMT.
   USE GALAHAD_SMT_double
   USE GALAHAD_SILS_double
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE( SMT_type ) :: matrix
   TYPE( SILS_control ) :: control
   TYPE( SILS_ainfo ) :: ainfo
   TYPE( SILS_finfo ) :: finfo
   TYPE( SILS_sinfo ) :: sinfo
   TYPE( SILS_factors ) :: factors
   INTEGER, PARAMETER :: n = 5
   INTEGER, PARAMETER :: ne = 7
   REAL ( KIND = wp ) :: B( n ), X( n )
   INTEGER :: i
! Read matrix order and number of entries
   matrix%n = n
   matrix%ne = ne
! Allocate and set matrix
   ALLOCATE( matrix%val( ne ), matrix%row( ne ), matrix%col( ne ) )
   matrix%row( : ne ) = (/ 1, 1, 2, 2, 3, 3, 5 /)
   matrix%col( : ne ) = (/ 1, 2, 3, 5, 3, 4, 5 /)
   matrix%val( : ne ) = (/ 2.0_wp, 3.0_wp, 4.0_wp, 6.0_wp, 1.0_wp,             &
                           5.0_wp, 1.0_wp /)
   CALL SMT_put( matrix%type, 'COORDINATE', i )     ! Specify co-ordinate
! Set right-hand side
   B( : n ) = (/ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp /)
! Initialize the structures
   CALL SILS_INITIALIZE( factors, control )
! Analyse
   CALL SILS_ANALYSE( matrix, factors, control, ainfo )
   IF ( ainfo%FLAG < 0 ) THEN
    WRITE(6,"( ' Failure of SILS_ANALYSE with AINFO%FLAG=',I2)" ) ainfo%FLAG
    STOP
   END IF
! Factorize
   CALL SILS_FACTORIZE( matrix, factors, control, finfo )
   IF ( finfo%FLAG < 0 ) THEN
     WRITE(6,"(' Failure of SILS_FACTORIZE with FINFO%FLAG=', I2)") finfo%FLAG
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
