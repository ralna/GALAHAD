! THIS VERSION: GALAHAD 3.3 - 03/06/2021 AT 11:45 GMT.
   PROGRAM GALAHAD_BQPB_SECOND_EXAMPLE
   USE GALAHAD_BQPB_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( BQPB_reverse_type ) :: reverse
   TYPE ( BQPB_data_type ) :: data
   TYPE ( BQPB_control_type ) :: control        
   TYPE ( BQPB_inform_type ) :: inform
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: nflag, i, j, k, l
   REAL ( KIND = wp ) :: v_j
   INTEGER, PARAMETER :: n = 3, h_ne = 4, h_all = 5
!  INTEGER, PARAMETER :: n = 3, h_ne = 3, h_all = 3
   INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat, FLAG, ROW, PTR
   REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VAL
! start problem data
   ALLOCATE( p%G( n ), p%X_l( n ), p%X_u( n ) )
   ALLOCATE( p%X( n ), p%Z( n ) )
   ALLOCATE( B_stat( n ), FLAG( n ) )
   ALLOCATE( VAL( h_all ), ROW( h_all ), PTR( n + 1 ) )
   p%new_problem_structure = .TRUE.           ! new structure
   p%n = n ; p%f = 1.0_wp                     ! dimensions & objective constant
   p%G = (/ 0.0_wp, 2.0_wp, 1.0_wp /)         ! objective gradient
   p%X_l = (/ - 1.0_wp, - infinity, 0.0_wp /) ! variable lower bound
   p%X_u = (/ infinity, 1.0_wp, 2.0_wp /)     ! variable upper bound
   p%X = 0.0_wp ; p%Z = 0.0_wp ! start from zero
   PTR = (/ 1, 3, 5, 6 /)                      ! whole Hessian by rows
   ROW = (/ 1, 2, 1, 2, 3 /)                   ! for matrix-vector products
   VAL = (/ 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp, 3.0_wp /)
! problem data complete   
   CALL BQPB_initialize( data, control, inform )  ! Initialize controls
   control%infinity = 0.1_wp * infinity           ! Set infinity
   control%print_level = 1                    ! print one line/iteration
   control%maxit = 100                        ! limit the # iterations
!  control%print_gap = 100                    ! print every 100 terations
!  control%exact_gcp = .FALSE.
   nflag = 0 ; FLAG = 0
   inform%status = 1
10 CONTINUE            ! Solve problem - reverse commmunication loop
     CALL BQPB_solve( p,  B_stat, data, control, inform, userdata, reverse )  
     SELECT CASE ( inform%status )
     CASE ( 0 )          !  Successful return
       WRITE( 6, "( ' BQPB: ', I0, ' iterations  ', /,                         &
      &     ' Optimal objective value =',                                      &
      &       ES12.4, /, ' Optimal solution = ', ( 5ES12.4 ) )" )              &
       inform%iter, inform%obj, p%X
     CASE ( 2 )          ! compute H * v
       reverse%PROD = 0.0_wp
       DO j = 1, p%n
         v_j = reverse%V( j )
         DO k = PTR( j ), PTR( j + 1 ) - 1
           i = ROW( k )
           reverse%PROD( i ) = reverse%PROD( i ) + VAL( k ) * v_j
         END DO
       END DO
       GO TO 10
     CASE DEFAULT        ! Error returns
       WRITE( 6, "( ' BQPB_solve exit status = ', I6 ) " ) inform%status
     END SELECT
   CALL BQPB_terminate( data, control, inform, reverse )  !  delete workspace
   DEALLOCATE( p%G, p%X, p%X_l, p%X_u, p%Z, B_stat, FLAG, PTR, ROW, VAL )
   END PROGRAM GALAHAD_BQPB_SECOND_EXAMPLE
