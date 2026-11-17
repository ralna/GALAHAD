    PROGRAM SLBLT_EXAMPLE   !  GALAHAD 5.6 - 2026-11-16 AT 11:50 GMT.
    USE GALAHAD_SLBLT_double
    IMPLICIT NONE
    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
    TYPE( SLBLT_inform_type ) :: inform
    TYPE( SLBLT_akeep_type ) :: akeep
    TYPE( SLBLT_fkeep_type ) :: fkeep
    TYPE( SLBLT_control_type ) :: control
    INTEGER :: i, n, ne, flag
    INTEGER, ALLOCATABLE :: ROW( : ), PTR( : ), ORDER( : )
    REAL ( KIND = wp ), ALLOCATABLE :: B( : ), X( : ), VAL( : )
!  set dimension
    n = 5
!  allocate arrays of appropriate sizes
    ALLOCATE( PTR( n + 1 ), B( n ), X( n ) )
!  set matrix and right-hand side
    PTR = [ 1, 3, 5, 7, 7, 8 ]
    ne = PTR( n + 1 ) - 1
    ALLOCATE( VAL( ne ), ROW( ne ) )
    ROW = [ 1, 2, 3, 5, 3, 4, 5 ]
    VAL = [ 2.0_wp, 3.0_wp, 4.0_wp, 6.0_wp, 1.0_wp, 5.0_wp, 1.0_wp ]
    B = [ 8.0_wp, 45.0_wp, 31.0_wp, 15.0_wp, 17.0_wp ]
!  analyse
    control%ordering = 0
    ALLOCATE( ORDER( n ) )
    DO i = 1, n ; ORDER( i ) = i ; END DO
    CALL SLBLT_analyse( .FALSE., n, PTR, ROW, akeep, control, inform,          &
                        val = VAL, order = ORDER )
    DEALLOCATE( ORDER )
    IF ( inform%flag < 0 ) THEN
      WRITE( 6, "( ' Failure of SLBLT_analyse with flag = ', I0 )" )           &
        inform%flag
      STOP
    END IF
!  factorize
    CALL SLBLT_factor( .FALSE., VAL, akeep, fkeep, control, inform,            &
                       ptr = PTR, row = ROW )
    IF ( inform%flag < 0 ) THEN
      WRITE( 6, '( A, I0 )' )                                                  &
           ' Failure of SLBLT_factor with flag = ', inform%flag
      STOP
    END IF
!  solve using iterative refinement and ask for high relative accuracy
    X = B
    CALL SLBLT_solve( X, akeep, fkeep, control, inform )
    IF ( inform%flag == 0 ) WRITE( 6, "( ' Solution is', /, ( 5F12.8 ) )" ) X
    CALL SLBLT_free(akeep, fkeep, flag )
    DEALLOCATE( VAL, ROW, PTR, X, B )
    STOP
    END PROGRAM SLBLT_EXAMPLE
