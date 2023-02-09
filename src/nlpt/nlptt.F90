! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
PROGRAM GALAHAD_NLPT_EXAMPLE
  USE GALAHAD_NLPT_double      ! the problem type
  USE GALAHAD_SYMBOLS
  IMPLICIT NONE
  INTEGER, PARAMETER                :: wp = KIND( 1.0D+0 )
  INTEGER,           PARAMETER      :: iout = 6        ! stdout and stderr
  REAL( KIND = wp ), PARAMETER      :: INFINITY = (10.0_wp)**19
  TYPE( NLPT_problem_type     )     :: problem
  INTEGER                           :: test
  INTEGER, DIMENSION( 4 )           :: perm, col, row
  INTEGER, DIMENSION( 3 )           :: ptr

! Set the problem up.
  problem%pname    = 'NLPT-TEST'
  problem%n        = 2
  ALLOCATE( problem%vnames( problem%n ), problem%x( problem%n )  ,             &
            problem%x_l( problem%n )   , problem%x_u( problem%n ),             &
            problem%g( problem%n )     , problem%z( problem%n )  )
  problem%m        = 2
  ALLOCATE( problem%equation( problem%m ), problem%linear( problem%m ),        &
            problem%c( problem%m ) , problem%c_l( problem%m ),                 &
            problem%c_u( problem%m), problem%y( problem%m ),                   &
            problem%cnames( problem%m ) ) 
  problem%J_ne     = 4
  ALLOCATE( problem%J_val( problem%J_ne ), problem%J_row( problem%J_ne ),      &
            problem%J_col( problem%J_ne ) )
  problem%H_ne     = 3
  ALLOCATE( problem%H_val( problem%H_ne ), problem%H_row( problem%H_ne ),      &
            problem%H_col( problem%H_ne ) )
  problem%H_type   = GALAHAD_COORDINATE
  problem%J_type   = GALAHAD_COORDINATE
  problem%vnames   = (/    'X1'  ,    'X2'   /)
  problem%x        = (/   0.0D0  ,   1.0D0   /)
  problem%x_l      = (/   0.0D0  , -INFINITY /)
  problem%x_u      = (/  INFINITY,  INFINITY /)
  problem%cnames   = (/    'C1'  ,    'C2'   /)
  problem%c        = (/   1.0D0  ,   1.0D0   /)
  problem%c_l      = (/ -INFINITY,   0.0D0   /)
  problem%c_u      = (/   1.0D0  ,  INFINITY /)
  problem%y        = (/  -1.0D0  ,   0.0D0   /)
  problem%equation = (/  .FALSE. ,  .FALSE.  /)
  problem%linear   = (/  .FALSE. ,   .TRUE.  /)
  problem%z        = (/   1.0D0  ,   0.0D0   /)
  problem%f        = -2.0_wp
  problem%g        = (/   1.0D0  ,  -1.0D0   /)
  problem%J_row    = (/     1    ,     1     ,     2     ,     2     /)
  problem%J_col    = (/     1    ,     2     ,     1     ,     2     /)
  problem%J_val    = (/   0.0D0  ,   2.0D0   ,  -1.0D0   ,   1.0D0   /)
  problem%H_row    = (/     1    ,     2     ,     2     /)
  problem%H_col    = (/     1    ,     1     ,     2     /)
  problem%H_val    = (/   2.0D0  ,   1.0D0   ,   2.0D0   /)
  problem%infinity = INFINITY
! Select the next test case.
  test = 0
  DO
     test = test + 1

     WRITE( iout, * ) ' '
     WRITE( iout, * )                                                          &
     ' ========================================================================'
     WRITE( iout, * ) ' '

     SELECT CASE( test )
     CASE ( 1 )
        WRITE( iout, * ) ' ', test, ')         Normal run'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        CALL NLPT_write_problem( problem, iout, GALAHAD_DETAILS )
     CASE( 2 )
        WRITE( iout, * ) ' ', test,                                            &
             ')         No names, no matrices additional bounds'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        DEALLOCATE( problem%vnames, problem%cnames )
        problem%x_u      = (/   2.0D0  ,   3.0D0   /)
        problem%c_l      = (/ -10.0D0  ,   0.0D0   /)
        problem%c_u      = (/   1.0D0  , INFINITY  /)
        CALL NLPT_write_problem( problem, iout, GALAHAD_ACTION )
        ALLOCATE( problem%vnames( problem%n ), problem%cnames( problem%m ) )
        problem%vnames   = (/    'X1'  ,    'X2'   /)
        problem%cnames   = (/    'C1'  ,    'C2'   /)
     CASE ( 3 )
        WRITE( iout, * ) ' ', test, ')        No bound, no matrices'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        DEALLOCATE( problem%x_l, problem%x_u )
        CALL NLPT_write_problem( problem, iout, GALAHAD_ACTION )
     CASE ( 4 )
        WRITE( iout, * ) ' ', test,                                            &
             ')        The same, without names, an equality'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        DEALLOCATE( problem%vnames, problem%cnames )
        problem%c_l      = (/   1.0D0  ,   0.0D0   /)
        problem%equation = (/  .TRUE.  ,  .FALSE.  /) 
        CALL NLPT_write_problem( problem, iout, GALAHAD_ACTION )
        ALLOCATE( problem%x_l( problem%n ), problem%x_u( problem%n ))
        problem%x_l      = (/   0.0D0  , -INFINITY /)
        problem%x_u      = (/  INFINITY,  INFINITY /)
        ALLOCATE( problem%vnames( problem%n ), problem%cnames( problem%m ) )
        problem%vnames   = (/    'X1'  ,    'X2'   /)
        problem%cnames   = (/    'C1'  ,    'C2'   /)
     CASE ( 5 )
        WRITE( iout, * ) ' ', test, ')        Lower print level'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        CALL NLPT_write_problem( problem, iout, GALAHAD_TRACE )
     CASE ( 6 )
        WRITE( iout, * ) ' ', test,                                            &
                         ')        Transform Jacobian to sparse by rows'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        CALL NLPT_J_perm_from_C_to_Srow( problem, perm, col, ptr )
        WRITE( iout,  * ) ' perm =', perm
        WRITE( iout,  * ) ' col  =', col
        WRITE( iout,  * ) ' ptr  =', ptr
     CASE ( 7 )
        WRITE( iout, * ) ' ', test,                                            &
                         ')        Transform Jacobian to sparse by columns'
        WRITE( iout, * ) ' '
        WRITE( iout, * )                                                       &
     ' ========================================================================'
        WRITE( iout, * ) ' '
        CALL NLPT_J_perm_from_C_to_Scol( problem, perm, row, ptr )
        WRITE( iout,  * ) ' perm =', perm
        WRITE( iout,  * ) ' row  =', row
        WRITE( iout,  * ) ' ptr  =', ptr
     CASE ( 8 )
        EXIT
     END SELECT
  END DO
! Cleanup the problem.
  CALL NLPT_cleanup( problem )
  STOP
END PROGRAM GALAHAD_NLPT_EXAMPLE

