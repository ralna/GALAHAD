  PROGRAM SSIDS_EXAMPLE   !  GALAHAD 2.8 - 06/09/2016 AT 12:00 GMT.
   USE SPRAL_ssids
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   TYPE( ssids_inform ) :: inform
   TYPE( ssids_akeep ) :: akeep
   TYPE( ssids_fkeep ) :: fkeep
   TYPE( ssids_options ) :: options
   INTEGER :: i, n, ne, cuda_error
!  INTEGER :: j, l
   INTEGER, ALLOCATABLE :: ROW( : ), PTR( : ), ORDER( : )
   REAL ( KIND = wp ), ALLOCATABLE :: B( : ), X( : ), VAL( : )
! Read matrix order and number of entries
   READ( 5, * ) n
! Allocate arrays of appropriate sizes
   ALLOCATE( PTR( n + 1 ), B( n ), X( n ) )
! Read matrix and right-hand side
   READ( 5, * ) ( PTR( i ), i = 1, n + 1 )
!write(6,*) ' ptr ', PTR
   ne = ptr( n + 1 ) - 1
   ALLOCATE( VAL( ne ), ROW( ne ) )
   READ( 5, * ) ( ROW( i ), i = 1, ne )
!write(6,*) ' row ', ROW
   READ( 5, * ) ( VAL( i ),  i = 1, ne )
!write(6,*) ' val ', VAL
   READ( 5, * ) B
!write(6,*) ' b ', B
! Analyse
   IF ( .TRUE. ) THEN
     options%ordering = 0
     ALLOCATE( ORDER( n ) )
     DO i = 1, n ; ORDER( i ) = i ; END DO
     CALL ssids_analyse( .FALSE., n, PTR, ROW, akeep, options, inform,         &
                         val = VAL, order = ORDER )
     DEALLOCATE( ORDER )
   ELSE
     CALL ssids_analyse( .FALSE., n, PTR, ROW, akeep, options, inform,         &
                         val = VAL )
   END IF
   IF ( inform%flag < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SSIDS_analyse with flag = ', inform%flag
     STOP
   END IF
! Factorize
   CALL ssids_factor( .FALSE., VAL, akeep, fkeep, options, inform,             &
                      ptr = PTR, row = ROW )
   IF ( inform%flag < 0 ) THEN
     WRITE( 6, '( A, I0 )' )                                                   &
          ' Failure of SSIDS_factor with flag = ', inform%flag
     STOP
   END IF
! Solve using iterative refinement and ask for high relative accuracy
   X = B
   CALL ssids_solve( X, akeep, fkeep, options, inform )
   IF ( inform%flag == 0 ) WRITE( 6, '( A, /, ( 3F20.16 ) )' )                 &
     ' Solution is', X
!   DO j = 1, n
!     DO l = PTR( j ), PTR( j + 1 ) - 1
!       i = ROW( l )
!       IF ( i == j ) THEN
!         B( i ) = B( i ) - VAL( l ) * X( i )
!       ELSE
!         B( i ) = B( i ) - VAL( l ) * X( j )
!         B( j ) = B( j ) - VAL( l ) * X( i )
!       END IF
!     END DO
!   END DO
!write(6,*) 'b ', B
! Clean up
   CALL ssids_free(akeep, fkeep, cuda_error )
   DEALLOCATE( VAL, ROW, PTR, X, B )
   STOP
   END PROGRAM SSIDS_EXAMPLE
