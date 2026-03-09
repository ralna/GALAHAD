! THIS VERSION: GALAHAD 5.5 - 2026-02-19 AT 10:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_SLLS_TEST_PROGRAM
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SLLS_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20
   TYPE ( QPT_problem_type ) :: p
   TYPE ( SLLS_data_type ) :: data
   TYPE ( SLLS_control_type ) :: control
   TYPE ( SLLS_inform_type ) :: inform
   TYPE ( REVERSE_type ) :: reverse
   TYPE ( USERDATA_type ) :: userdata, userdata2
   INTEGER ( KIND = ip_ ) :: i, j, k, l, weight, shift, mode
   INTEGER ( KIND = ip_ ) :: s, status, exact_arc_search
   REAL ( KIND = rp_ ) :: val
! problem 1
   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, o = 4, m = 1, a_ne = 5
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_col, A_ptr
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr_row
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val
! problem 2
   INTEGER ( KIND = ip_ ), PARAMETER :: n2 = 5, o2 = n2 + 1, a2_ne = 2 * n2 + 2
   INTEGER ( KIND = ip_ ), PARAMETER :: m2 = 2
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A2_row, A2_ptr
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A2_val
! both problems
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: DIAG
   INTEGER ( KIND = ip_ ), PARAMETER :: nflag = 3, st_flag = 3, st_val = 0
! problem 1: partition userdata%integer so that it holds
!   o n nflag  flag          a_ptr          a_row
!  |1|2|  3  |4 to on+3 |on+4 to on+n+4|on+n+5 to on+n+4+a_ne|, with on=max(o,n)
! partition userdata%real so that it holds
!     a_val
!  |1 to a_ne|
   INTEGER ( KIND = ip_ ), PARAMETER :: on = MAX( o, n )
   INTEGER ( KIND = ip_ ), PARAMETER :: st_ptr = st_flag + on
   INTEGER ( KIND = ip_ ), PARAMETER :: st_row = st_ptr + n + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: len_integer = st_row + a_ne + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: len_real = a_ne
! problem 2
   INTEGER ( KIND = ip_ ), PARAMETER :: on2 = MAX( o2, n2 )
   INTEGER ( KIND = ip_ ), PARAMETER :: st_ptr2 = st_flag + on2
   INTEGER ( KIND = ip_ ), PARAMETER :: st_row2 = st_ptr2 + n2 + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: len_integer2 = st_row2 + a2_ne + 1
   INTEGER ( KIND = ip_ ), PARAMETER :: len_real2 = a2_ne
   EXTERNAL :: APROD, ASCOL, AFPROD, PREC
   EXTERNAL :: APROD_broken, ASCOL_broken, AFPROD_broken
! set up problem 1 data
   ALLOCATE( p%B( o ), p%X( n ) )
   p%n = n ; p%o = o ; p%m = m ! dimensions
   p%B = (/ 0.0_rp_, 2.0_rp_, 1.0_rp_, 2.0_rp_ /)  ! right-hand side
   p%X = 0.0_rp_ ! start from zero
   ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_col( a_ne ) )
   ALLOCATE( A_ptr( n + 1 ), A_ptr_row( o + 1 ) )
   A_val = (/ 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ /) ! Jacobian A by col
   A_row = (/ 1, 2, 2, 3, 4 /)                     ! row indices
   A_col = (/ 1, 1, 2, 3, 3 /)                     ! column indices
   A_ptr = (/ 1, 3, 4, 6 /)                        ! pointers to column starts
   A_ptr_row = (/ 1, 2, 4, 5, 6 /)                 ! pointers to row starts

! problem data complete

   ALLOCATE( DIAG( n ) )
   DO j = 1, n
     val = 0.0_rp_
     DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
       val = val + A_val( k ) ** 2
     END DO
     DIAG( j ) = val
   END DO

! generic runs to test each mode

! weight (0 = no weight, 1 = weight of one)

!  DO weight = 0, 0
!  DO weight = 1, 1
   DO weight = 0, 1
     p%regularization_weight = REAL( weight, KIND = rp_ )
     IF ( weight == 1 ) THEN
       ALLOCATE( p%W( o ) )
       p%W = 1.0_rp_
     END IF

! shift (0 = no shift, 1 = shift of one)

!    DO shift = 0, 0
!    DO shift = 1, 1
     DO shift = 0, 1
       IF ( weight == 0 .AND. shift == 1 ) CYCLE
       IF ( shift == 1 ) THEN
         ALLOCATE( p%X_s( n2 ) )
         p%X_s = [ 1.0_rp_, 1.0_rp_, 1.0_rp_ ]
       END IF

       WRITE( 6, "( /, ' run tests (cohort = 1, shift = ', I0, ', weight = ',  &
      & I0, ')', / )" ) shift, weight

! search used (0 = inexat, 1 = exact)

!      DO exact_arc_search = 0, 0
!      DO exact_arc_search = 1, 1
       DO exact_arc_search = 0, 1

! mode (1 = A explicit, iterative subproblem, 2 = A explicit, direct subproblem,
!       3 = A products via subroutine, 4 = A products via reverse communication,
!       5 = preconditioner via subroutine, 6 = preconditioner via reverse com.)
!        DO mode = 1, 1
!        DO mode = 1, 2
!        DO mode = 2, 2
!        DO mode = 3, 3
!        DO mode = 3, 4
!        DO mode = 4, 4
!        DO mode = 5, 5
!        DO mode = 6, 6
!        DO mode = 5, 6
         DO mode = 1, 6
           CALL SLLS_initialize( data, control, inform )
           CALL WHICH_sls( control )
!          control%print_level = 1                   ! print one line/iteration
           control%exact_arc_search = exact_arc_search == 1
!          control%exact_arc_search = .FALSE.
           p%X = 0.0_rp_ ! start from zero
           SELECT CASE ( mode ) ! matrix access
           CASE ( 1, 2 ) ! A explicitly available (iterative and direct solves)
             CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', s )
             ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%ptr( n + 1 ) )
             p%Ao%m = o ; p%Ao%n = n
             p%Ao%val( : a_ne ) = A_val( : a_ne )
             p%Ao%row( : a_ne ) = A_row( : a_ne )
             p%Ao%ptr( : n + 1 ) = A_ptr( : n + 1 )
             control%direct_subproblem_solve = mode == 2
             inform%status = 1
             CALL SLLS_solve( p, data, control, inform, userdata )
             WRITE( 6, "( ' SLLS_solve argument mode = ', I0, ', search = ',   &
            &  I0, ', status = ', I0,', objective = ', F6.4 ) " )              &
                mode, exact_arc_search, inform%status, inform%obj
             DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%ptr, p%Ao%type )
           CASE ( 3 ) ! A available by external subroutines
             ALLOCATE( userdata%integer( len_integer ),                        &
                       userdata%real( len_real ) )
             userdata%integer( 1 ) = o   ! load Jacobian data into userdata
             userdata%integer( 2 ) = n
             userdata%integer( st_ptr + 1 : st_ptr + n + 1 ) = A_ptr( : n + 1 )
             userdata%integer( st_row + 1 : st_row + a_ne ) = A_row( : a_ne )
             userdata%real( st_val + 1 : st_val + a_ne ) = A_val( : a_ne )
             userdata%integer( nflag ) = 0
             userdata%integer( st_flag + 1 : st_flag + on ) = 0
             inform%status = 1
             CALL SLLS_solve( p, data, control, inform, userdata,              &
                              eval_APROD = APROD, eval_ASCOL = ASCOL,          &
                              eval_AFPROD = AFPROD )
             WRITE( 6, "( ' SLLS_solve argument mode = ', I0, ', search = ',   &
            &  I0, ', status = ', I0,', objective = ', F6.4 ) " )              &
                mode, exact_arc_search, inform%status, inform%obj
             DEALLOCATE( userdata%integer, userdata%real )
           CASE ( 4 ) ! A available by reverse matrix-vector products
             inform%status = 1
             DO ! Solve problem - reverse commmunication loop
               CALL SLLS_solve( p, data, control, inform, userdata,            &
                                reverse = reverse )
               SELECT CASE ( inform%status )
               CASE ( : 0 ) !  termination return
                 WRITE( 6, "( ' SLLS_solve argument mode = ',I0,', search = ', &
                &  I0, ', status = ', I0,', objective = ', F6.4 ) " )          &
                    mode, exact_arc_search, inform%status, inform%obj
                 EXIT
               CASE ( 2 ) ! compute A * v
                 reverse%P( : o ) = 0.0_rp_
                 DO j = 1, n
                   val = reverse%V( j )
                   DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
                     i = A_row( k )
                     reverse%P( i ) = reverse%P( i ) + A_val( k ) * val
                   END DO
                 END DO
               CASE ( 3 ) ! compute A^T * v
                 reverse%P( : n ) = 0.0_rp_
                 DO j = 1, n
                   val = 0.0_rp_
                   DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
                     val = val + A_val( k ) * reverse%V( A_row( k ) )
                   END DO
                   reverse%P( j ) = val
                 END DO
               CASE ( 4 ) ! compute a sparse column of A
                 reverse%lp = 0
                 DO k = A_ptr( reverse%index ), A_ptr( reverse%index + 1 ) - 1
                   reverse%lp = reverse%lp + 1
                   reverse%P( reverse%lp ) = A_val( k )
                   reverse%IP( reverse%lp ) = A_row( k )
                 END DO
               CASE ( 5 ) ! compute A * sparse v
                 reverse%P( : o ) = 0.0_rp_
                 DO l = reverse%lvl, reverse%lvu
                   j = reverse%IV( l )
                   val = reverse%V( j )
                   DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
                     i = A_row( k )
                     reverse%P( i ) = reverse%P( i ) + A_val( k ) * val
                   END DO
                 END DO
               CASE ( 6 ) ! compute sparse( A^T * v )
                 reverse%P( : n ) = 0.0_rp_
                 DO l = reverse%lvl, reverse%lvu
                   j = reverse%IV( l )
                   val = 0.0_rp_
                   DO k = A_ptr( j ), A_ptr( j + 1 ) - 1
                     val = val + A_val( k ) * reverse%V( A_row( k ) )
                   END DO
                   reverse%P( j ) = val
                 END DO
               END SELECT
             END DO
           CASE ( 5 ) ! preconditioner explicitly available
             CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', s )
             ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%ptr( n + 1 ) )
             p%Ao%m = o ; p%Ao%n = n
             p%Ao%val( : a_ne ) = A_val( : a_ne )
             p%Ao%row( : a_ne ) = A_row( : a_ne )
             p%Ao%ptr( : n + 1 ) = A_ptr( : n + 1 )
             ALLOCATE( userdata%integer( 1 ), userdata%real( n ) )
             userdata%integer( 1 ) = n  ! load preconditioner data into userdata
             userdata%real( : n ) = DIAG( : n )
!            control%print_level = 1
             control%preconditioner = 2
             control%direct_subproblem_solve = .FALSE.
             inform%status = 1
             CALL SLLS_solve( p, data, control, inform, userdata,              &
                              eval_PREC = PREC )
             WRITE( 6, "( ' SLLS_solve argument mode = ', I0, ', search = ',   &
            &  I0, ', status = ', I0,', objective = ', F6.4 ) " )              &
                mode, exact_arc_search, inform%status, inform%obj
             DEALLOCATE( userdata%integer, userdata%real )
           CASE ( 6 ) ! preconditioner available by reverse communication
!            control%print_level = 3
             control%preconditioner = 2
             control%direct_subproblem_solve = .FALSE.
             inform%status = 1
             DO ! Solve problem - reverse commmunication loop
               CALL SLLS_solve( p, data, control, inform, userdata,            &
                                reverse = reverse )
               SELECT CASE ( inform%status )
               CASE ( : 0 ) !  termination return
                 WRITE( 6, "( ' SLLS_solve argument mode = ',I0,', search = ', &
                &  I0, ', status = ', I0,', objective = ', F6.4 ) " )          &
                    mode, exact_arc_search, inform%status, inform%obj
                 EXIT
               CASE ( 7 ) !
                 reverse%P( : n ) = reverse%V( : n ) / DIAG( : n )
               END SELECT
             END DO
             DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%ptr, p%Ao%type )
           END SELECT
           CALL SLLS_terminate( data, control, inform )  !  delete workspace
         END DO
       END DO
!      CYCLE

! generic runs to test each storage mode

       WRITE( 6, "( '' )" )
       DO exact_arc_search = 0, 0
!      DO exact_arc_search = 1, 1
!      DO exact_arc_search = 0, 1
!        DO mode = 3, 5
!        DO mode = 1, 1
         DO mode = 1, 5
           CALL SLLS_initialize( data, control, inform )
           CALL WHICH_sls( control )
!          control%print_level = 1                    ! print one line/iteration
           control%exact_arc_search = exact_arc_search == 1
!          control%exact_arc_search = .FALSE.
           p%regularization_weight = REAL( weight, KIND = rp_ )
           p%X = 0.0_rp_ ! start from zero
           SELECT CASE ( mode )
           CASE ( 1 ) ! A by columns
             CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', s )
             ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%ptr( n + 1 ) )
             p%Ao%m = o ; p%Ao%n = n
             p%Ao%val( : a_ne ) = A_val( : a_ne )
             p%Ao%row( : a_ne ) = A_row( : a_ne )
             p%Ao%ptr( : n + 1 ) = A_ptr( : n + 1 )
           CASE ( 2 ) ! A by rows
             CALL SMT_put( p%Ao%type, 'SPARSE_BY_ROWS', s )
             ALLOCATE( p%Ao%val( a_ne ), p%Ao%col( a_ne ), p%Ao%ptr( o + 1 ) )
             p%Ao%m = o ; p%Ao%n = n
             p%Ao%val( : a_ne ) = A_val( : a_ne )
             p%Ao%col( : a_ne ) = A_col( : a_ne )
             p%Ao%ptr( : o + 1 ) = A_ptr_row( : o + 1 )
           CASE ( 3 ) ! A coordinate
             CALL SMT_put( p%Ao%type, 'COORDINATE', s )
             ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( a_ne ) )
             p%Ao%m = o ; p%Ao%n = n ; p%Ao%ne = a_ne
             p%Ao%val( : a_ne ) = A_val( : a_ne )
             p%Ao%row( : a_ne ) = A_row( : a_ne )
             p%Ao%col( : a_ne ) = A_col( : a_ne )
           CASE ( 4 ) ! A dense by columns
             CALL SMT_put( p%Ao%type, 'DENSE_BY_COLUMNS', s )
             ALLOCATE( p%Ao%val( o * n ) )
             p%Ao%m = o ; p%Ao%n = n
             p%Ao%val( : o * n ) = (/ 1.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,      &
                                      0.0_rp_, 1.0_rp_, 0.0_rp_, 0.0_rp_,      &
                                      0.0_rp_, 0.0_rp_, 1.0_rp_, 1.0_rp_ /)
           CASE ( 5 ) ! A dense by rows
             CALL SMT_put( p%Ao%type, 'DENSE_BY_ROWS', s )
             ALLOCATE( p%Ao%val( o * n ) )
             p%Ao%m = o ; p%Ao%n = n
             p%Ao%val( : o * n ) = (/ 1.0_rp_, 0.0_rp_, 0.0_rp_,               &
                                      1.0_rp_, 1.0_rp_, 0.0_rp_,               &
                                      0.0_rp_, 0.0_rp_, 1.0_rp_,               &
                                      0.0_rp_, 0.0_rp_, 1.0_rp_ /)
           END SELECT
           inform%status = 1
           CALL SLLS_solve( p, data, control, inform, userdata )
           WRITE( 6, "( ' SLLS_solve argument storage = ', I0, ', search = ',  &
          &  I0, ', status = ', I0, ', objective = ', F6.4 ) " )               &
             mode, exact_arc_search, inform%status, inform%obj
           SELECT CASE ( mode )
           CASE ( 1 ) ! A by columns
             DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%ptr, p%Ao%type )
           CASE ( 2 ) ! A by rows
             DEALLOCATE( p%Ao%val, p%Ao%col, p%Ao%ptr, p%Ao%type )
           CASE ( 3 ) ! A coordinate
             DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )
           CASE ( 4, 5 ) ! A dense_by_rows or A dense_by_columns
             DEALLOCATE( p%Ao%val, p%Ao%type )
           END SELECT
           CALL SLLS_terminate( data, control, inform )  !  delete workspace
         END DO
       END DO
       IF ( shift == 1 ) DEALLOCATE( p%X_s )
     END DO ! end of shift loop
     IF ( weight == 1 ) DEALLOCATE( p%W )
   END DO ! end of weight loop
   DEALLOCATE( p%B, p%X, p%Z, DIAG )

! set up problem 2 data

   ALLOCATE( p%B( o2 ), p%X( n2 ), p%COHORT( n2 ) )
   p%n = n2 ; p%o = o2 ; p%m = m2 ! dimensions
   p%X = 0.0_rp_ ! start from zero
   ALLOCATE( A2_val( a2_ne ), A2_row( a2_ne ), A2_ptr( n2 + 1 ) )
   p%COHORT = [ 1, 2, 0, 1, 2 ]
   l = 0
   DO j = 1, n2
      l = l + 1
      A2_ptr( j ) = l
      A2_row( l ) = j
      A2_val( l ) = 1.0_rp_
      l = l + 1
      A2_row( l ) = j + 1
      A2_val( l ) = - 1.0_rp_
    END DO
    A2_ptr( n2 + 1 ) = l + 1
    p%B = [ 0.0_rp_, 2.0_rp_, 1.0_rp_, 2.0_rp_, 1.0_rp_, 2.0_rp_ ] 

! problem data complete

   ALLOCATE( DIAG( n2 ) )
   DO j = 1, n2
     val = 0.0_rp_
     DO k = A2_ptr( j ), A2_ptr( j + 1 ) - 1
       val = val + A2_val( k ) ** 2
     END DO
     DIAG( j ) = val
   END DO

! generic runs to test each mode

! weight (0 = no weight, 1 = weight of one)

!  DO weight = 0, 0
!  DO weight = 1, 1
   DO weight = 0, 1
     p%regularization_weight = REAL( weight, KIND = rp_ )
     IF ( weight == 1 ) THEN
       ALLOCATE( p%W( o ) )
       p%W = 1.0_rp_
     END IF

! shift (0 = no shift, 1 = shift of one)

!    DO shift = 0, 0
!    DO shift = 1, 1
     DO shift = 0, 1
       IF ( weight == 0 .AND. shift == 1 ) CYCLE
       IF ( shift == 1 ) THEN
         ALLOCATE( p%X_s( n ) )
         p%X_s = [ 1.0_rp_, 1.0_rp_, 1.0_rp_ ]
       END IF

       WRITE( 6, "( /, ' run tests (cohort = 2, shift = ', I0, ', weight = ',  &
      & I0, ')', / )" ) shift, weight

! s  earch used (0 = inexat, 1 = exact)

!      DO exact_arc_search = 0, 0
!      DO exact_arc_search = 1, 1
       DO exact_arc_search = 0, 1

! mode (1 = A explicit, iterative subproblem, 2 = A explicit, direct subproblem,
!       3 = A products via subroutine, 4 = A products via reverse communication,
!       5 = preconditioner via subroutine, 6 = preconditioner via reverse com.)
!        DO mode = 1, 1
!        DO mode = 1, 2
!        DO mode = 2, 2
!        DO mode = 3, 3
!        DO mode = 3, 4
!        DO mode = 4, 4
!        DO mode = 5, 5
!        DO mode = 6, 6
!        DO mode = 5, 6
         DO mode = 1, 6
           CALL SLLS_initialize( data, control, inform )
           CALL WHICH_sls( control )
!          control%print_level = 3                   ! print one line/iteration
           control%exact_arc_search = exact_arc_search == 1
!          control%exact_arc_search = .FALSE.
!          control%maxit = 3
           p%X = 0.0_rp_ ! start from zero
           SELECT CASE ( mode ) ! matrix access
           CASE ( 1, 2 ) ! A explicitly available (iterative and direct solves)
             CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', s )
             ALLOCATE( p%Ao%val( a2_ne ), p%Ao%row( a2_ne ), p%Ao%ptr( n2 + 1 ))
             p%Ao%m = o2 ; p%Ao%n = n2
             p%Ao%val( : a2_ne ) = A2_val( : a2_ne )
             p%Ao%row( : a2_ne ) = A2_row( : a2_ne )
             p%Ao%ptr( : n2 + 1 ) = A2_ptr( : n2 + 1 )
             control%direct_subproblem_solve = mode == 2
             inform%status = 1
             CALL SLLS_solve( p, data, control, inform, userdata2 )
             WRITE( 6, "( ' SLLS_solve argument mode = ', I0, ', search = ',   &
            &  I0, ', status = ', I0,', objective = ', F6.4 ) " )              &
                mode, exact_arc_search, inform%status, inform%obj
             DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%ptr, p%Ao%type )
           CASE ( 3 ) ! A available by external subroutines
             ALLOCATE( userdata2%integer( len_integer2 ),                      &
                       userdata2%real( len_real2 ))
             userdata2%integer( 1 ) = o2   ! load Jacobian data into userdata
             userdata2%integer( 2 ) = n2
             userdata2%integer( st_ptr2 + 1 : st_ptr2 + n2 + 1 )               &
               = A2_ptr( : n2 + 1 )
             userdata2%integer( st_row2 + 1 : st_row2 + a2_ne )                &
               = A2_row( : a2_ne )
             userdata2%real( st_val + 1 : st_val + a2_ne ) = A2_val( : a2_ne )
             userdata2%integer( nflag ) = 0
             userdata2%integer( st_flag + 1 : st_flag + on2 ) = 0
             inform%status = 1
             CALL SLLS_solve( p, data, control, inform, userdata2,             &
                              eval_APROD = APROD, eval_ASCOL = ASCOL,          &
                              eval_AFPROD = AFPROD )
             WRITE( 6, "( ' SLLS_solve argument mode = ', I0, ', search = ',   &
            &  I0, ', status = ', I0,', objective = ', F6.4 ) " )              &
                mode, exact_arc_search, inform%status, inform%obj
             DEALLOCATE( userdata2%integer, userdata2%real )
           CASE ( 4 ) ! A available by reverse matrix-vector products
             inform%status = 1
             DO ! Solve problem - reverse commmunication loop
               CALL SLLS_solve( p, data, control, inform, userdata2,           &
                                reverse = reverse )
               SELECT CASE ( inform%status )
               CASE ( : 0 ) !  termination return
                 WRITE( 6, "( ' SLLS_solve argument mode = ',I0,', search = ', &
                &  I0, ', status = ', I0,', objective = ', F6.4 ) " )          &
                    mode, exact_arc_search, inform%status, inform%obj
                 EXIT
               CASE ( 2 ) ! compute Ao * v
                 reverse%P( : o2 ) = 0.0_rp_
                 DO j = 1, n2
                   val = reverse%V( j )
                   DO k = A2_ptr( j ), A2_ptr( j + 1 ) - 1
                     i = A2_row( k )
                     reverse%P( i ) = reverse%P( i ) + A2_val( k ) * val
                   END DO
                 END DO
               CASE ( 3 ) ! compute Ao^T * v
                 reverse%P( : n2 ) = 0.0_rp_
                 DO j = 1, n2
                   val = 0.0_rp_
                   DO k = A2_ptr( j ), A2_ptr( j + 1 ) - 1
                     val = val + A2_val( k ) * reverse%V( A2_row( k ) )
                   END DO
                   reverse%P( j ) = val
                 END DO
               CASE ( 4 ) ! compute a sparse column of Ao
                 reverse%lp = 0
                 DO k = A2_ptr( reverse%index ), A2_ptr( reverse%index + 1 ) - 1
                   reverse%lp = reverse%lp + 1
                   reverse%P( reverse%lp ) = A2_val( k )
                   reverse%IP( reverse%lp ) = A2_row( k )
                 END DO
               CASE ( 5 ) ! compute Ao * sparse v
                 reverse%P( : o2 ) = 0.0_rp_
                 DO l = reverse%lvl, reverse%lvu
                   j = reverse%IV( l )
                   val = reverse%V( j )
                   DO k = A2_ptr( j ), A2_ptr( j + 1 ) - 1
                     i = A2_row( k )
                     reverse%P( i ) = reverse%P( i ) + A2_val( k ) * val
                   END DO
                 END DO
               CASE ( 6 ) ! compute sparse( Ao^T * v )
                 reverse%P( : n2 ) = 0.0_rp_
                 DO l = reverse%lvl, reverse%lvu
                   j = reverse%IV( l )
                   val = 0.0_rp_
                   DO k = A2_ptr( j ), A2_ptr( j + 1 ) - 1
                     val = val + A2_val( k ) * reverse%V( A2_row( k ) )
                   END DO
                   reverse%P( j ) = val
                 END DO
               END SELECT
             END DO
           CASE ( 5 ) ! preconditioner explicitly available
             CALL SMT_put( p%Ao%type, 'SPARSE_BY_COLUMNS', s )
             ALLOCATE( p%Ao%val( a2_ne ), p%Ao%row( a2_ne ),p%Ao%ptr( n2 + 1 ) )
             p%Ao%m = o2 ; p%Ao%n = n2
             p%Ao%val( : a2_ne ) = A2_val( : a2_ne )
             p%Ao%row( : a2_ne ) = A2_row( : a2_ne )
             p%Ao%ptr( : n2 + 1 ) = A2_ptr( : n2 + 1 )
             ALLOCATE( userdata2%integer( 1 ), userdata2%real( n2 ) )
             userdata2%integer( 1 ) = n2 ! load preconditioner data to userdata
             userdata2%real( : n2 ) = DIAG( : n2 )
!            control%print_level = 1
             control%preconditioner = 2
             control%direct_subproblem_solve = .FALSE.
             inform%status = 1
             CALL SLLS_solve( p, data, control, inform, userdata2,             &
                              eval_PREC = PREC )
             WRITE( 6, "( ' SLLS_solve argument mode = ', I0, ', search = ',   &
            &  I0, ', status = ', I0,', objective = ', F6.4 ) " )              &
                mode, exact_arc_search, inform%status, inform%obj
             DEALLOCATE( userdata2%integer, userdata2%real )
           CASE ( 6 ) ! preconditioner available by reverse communication
!            control%print_level = 3
             control%preconditioner = 2
             control%direct_subproblem_solve = .FALSE.
             inform%status = 1
             DO ! Solve problem - reverse commmunication loop
               CALL SLLS_solve( p, data, control, inform, userdata2,           &
                                reverse = reverse )
               SELECT CASE ( inform%status )
               CASE ( : 0 ) !  termination return
                 WRITE( 6, "( ' SLLS_solve argument mode = ',I0,', search = ', &
                &  I0, ', status = ', I0,', objective = ', F6.4 ) " )          &
                    mode, exact_arc_search, inform%status, inform%obj
                 EXIT
               CASE ( 7 ) !
                 reverse%P( : n2 ) = reverse%V( : n2 ) / DIAG( : n2 )
               END SELECT
             END DO
             DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%ptr, p%Ao%type )
           END SELECT
           CALL SLLS_terminate( data, control, inform )  !  delete workspace
         END DO
       END DO
!      CYCLE
       IF ( shift == 1 ) DEALLOCATE( p%X_s )
     END DO
   END DO
   DEALLOCATE( p%B, p%X, p%Z, DIAG, p%COHORT )

!  ================
!  error exit tests
!  ================

   WRITE( 6, "( /, ' error exit tests ', / )" )

   ALLOCATE( p%B( o ), p%X( n ) )
   p%n = n ; p%o = o
   p%B = (/ 0.0_rp_,  2.0_rp_,  1.0_rp_,  2.0_rp_ /)
   CALL SMT_put( p%Ao%type, 'COORDINATE', s )
   ALLOCATE( p%Ao%val( a_ne ), p%Ao%row( a_ne ), p%Ao%col( a_ne ) )
   p%Ao%m = o ; p%Ao%n = n ; ; p%Ao%ne = a_ne
   p%Ao%val( : a_ne ) = A_val( : a_ne )
   p%Ao%row( : a_ne ) = A_row( : a_ne )
   p%Ao%col( : a_ne ) = A_col( : a_ne )

!  tests for status = - 1 ... - 24

   DO status = 1, 24

     CALL SLLS_initialize( data, control, inform )
     CALL WHICH_sls( control )
     SELECT CASE ( status )
     CASE ( - GALAHAD_error_allocate,                                          &
            - GALAHAD_error_deallocate,                                        &
            - GALAHAD_error_bad_bounds,                                        &
            - GALAHAD_error_dual_infeasible,                                   &
            - GALAHAD_error_unbounded,                                         &
            - GALAHAD_error_primal_infeasible,                                 &
            - GALAHAD_error_no_center,                                         &
            - GALAHAD_error_analysis,                                          &
            - GALAHAD_error_factorization,                                     &
            - GALAHAD_error_solve,                                             &
            - GALAHAD_error_uls_analysis,                                      &
            - GALAHAD_error_uls_factorization,                                 &
            - GALAHAD_error_uls_solve,                                         &
            - GALAHAD_error_preconditioner,                                    &
            - GALAHAD_error_ill_conditioned,                                   &
            - GALAHAD_error_inertia,                                           &
            - GALAHAD_error_file,                                              &
            - GALAHAD_error_io,                                                &
            - GALAHAD_error_sort,                                              &
            - GALAHAD_error_tiny_step,                                         &
            - GALAHAD_error_upper_entry )
       CYCLE
     CASE ( - GALAHAD_error_restrictions )
       p%n = 0 ; p%o = - 1
     CASE ( - GALAHAD_error_max_iterations )
       control%maxit = 0
!      control%print_level = 1
     CASE ( - GALAHAD_error_cpu_limit )
       control%cpu_time_limit = 0.0
!      control%print_level = 1
     END SELECT

     p%X = 0.0_rp_
     inform%status = 1
     CALL SLLS_solve( p, data, control, inform, userdata )
     WRITE( 6, "( ' SLLS_solve (check = ', I2, ') exit status = ', I0 ) " )    &
       status, inform%status

     SELECT CASE ( status )
     CASE ( - GALAHAD_error_restrictions )
       p%n = n ; p%o = o
     CASE ( - GALAHAD_error_max_iterations )
       control%maxit = 100
!      control%print_level = 1
     CASE ( - GALAHAD_error_cpu_limit )
       control%cpu_time_limit = 100.0
!      control%print_level = 1
     END SELECT
     CALL SLLS_terminate( data, control, inform )  !  delete workspace
   END DO
   DEALLOCATE( p%Ao%val, p%Ao%row, p%Ao%col, p%Ao%type )

   ALLOCATE( userdata%integer( len_integer ), userdata%real( len_real ) )
   userdata%integer( 1 ) = o   ! load Jacobian data into userdata
   userdata%integer( 2 ) = n
   userdata%integer( st_ptr + 1 : st_ptr + n + 1 ) = A_ptr( : n + 1 )
   userdata%integer( st_row + 1 : st_row + a_ne ) = A_row( : a_ne )
   userdata%real( st_val + 1 : st_val + a_ne ) = A_val( : a_ne )
   userdata%integer( nflag ) = 0
   userdata%integer( st_flag + 1 : st_flag + on ) = 0

   p%X = 0.0_rp_
   control%exact_arc_search = .FALSE.
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata,                        &
                    eval_APROD = APROD_broken, eval_ASCOL = ASCOL,             &
                    eval_AFPROD = AFPROD )
   WRITE( 6, "( ' SLLS_solve bad APROD exit status = ', I0 ) " ) inform%status

   p%X(1) = 0.2_rp_
   p%X(2) = 0.4_rp_
   p%X(3) = 0.4_rp_
   control%exact_arc_search = .TRUE.
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata,                        &
                    eval_APROD = APROD, eval_ASCOL = ASCOL_broken,             &
                    eval_AFPROD = AFPROD )
   WRITE( 6, "( ' SLLS_solve bad ASCOL exit status = ', I0 ) " ) inform%status

   p%X = 0.0_rp_
   control%exact_arc_search = .TRUE.
   inform%status = 1
   CALL SLLS_solve( p, data, control, inform, userdata,                        &
                    eval_APROD = APROD, eval_ASCOL = ASCOL,                    &
                    eval_AFPROD = AFPROD_broken )
   WRITE( 6, "( ' SLLS_solve bad AFPROD exit status = ', I0 ) " ) inform%status
   DEALLOCATE( userdata%integer, userdata%real )
   DEALLOCATE( p%B, p%X, p%Z, p%R, p%G )
   DEALLOCATE( A_val, A_row, A_col, A_ptr, A_ptr_row )
   WRITE( 6, "( /, ' tests completed' )" )

   CONTAINS
     SUBROUTINE WHICH_sls( control )
     TYPE ( SLLS_control_type ) :: control
#include "galahad_sls_defaults_ls.h"
     control%SBLS_control%symmetric_linear_solver = symmetric_linear_solver
     control%SBLS_control%definite_linear_solver = definite_linear_solver
     END SUBROUTINE WHICH_sls

   END PROGRAM GALAHAD_SLLS_TEST_PROGRAM

   SUBROUTINE APROD( status, userdata, transpose, V, P )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
   INTEGER ( KIND = ip_ ) :: i, j, k
   REAL ( KIND = rp_ ) :: val
!  recover problem data from userdata
   INTEGER ( KIND = ip_ ) :: o, n, nflag, st_flag, st_ptr, st_row, st_val
   o = userdata%integer( 1 )
   n = userdata%integer( 2 )
   nflag = 3
   st_flag = 3
   st_ptr = st_flag + MAX( o, n )
   st_row = st_ptr + n + 1
   st_val = 0
   IF ( transpose ) THEN
     DO j = 1, n
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         P( j ) = P( j ) + userdata%real( st_val + k ) *                       &
                     V( userdata%integer( st_row + k ) )
       END DO
     END DO
   ELSE
     DO j = 1, n
       val = V( j )
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         i = userdata%integer( st_row + k )
         P( i ) = P( i ) + userdata%real( st_val + k ) * val
       END DO
     END DO
   END IF
   status = 0
   RETURN
   END SUBROUTINE APROD

   SUBROUTINE ASCOL( status, userdata, col, P, IP, lp )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: col
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: lp
   INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: IP
   INTEGER ( KIND = ip_ ) :: i, k
!  recover problem data from userdata
   INTEGER ( KIND = ip_ ) :: o, n, nflag, st_flag, st_ptr, st_row, st_val
   o = userdata%integer( 1 )
   n = userdata%integer( 2 )
   nflag = 3
   st_flag = 3
   st_ptr = st_flag + MAX( o, n )
   st_row = st_ptr + n + 1
   st_val = 0
   userdata%integer( nflag ) = userdata%integer( nflag ) + 1
   lp = 0
   DO k = userdata%integer( st_ptr + col ),                                    &
          userdata%integer( st_ptr + col + 1 ) - 1
     i = userdata%integer( st_row + k )
     IF ( userdata%integer( st_flag + i ) < userdata%integer( nflag ) ) THEN
       userdata%integer( st_flag + i ) = userdata%integer( nflag )
       P( i ) = userdata%real( st_val + k )
       lp = lp + 1
       IP( lp ) = i
     ELSE
       P( i ) = P( i ) + userdata%real( st_val + k )
     END IF
   END DO
   status = 0
   RETURN
   END SUBROUTINE ASCOL

   SUBROUTINE AFPROD( status, userdata, transpose, V, P, FREE, n_free )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER ( KIND = ip_ ) :: i, j, k, l
   REAL ( KIND = rp_ ) :: val
!  recover problem data from userdata
   INTEGER ( KIND = ip_ ) :: o, n, nflag, st_flag, st_ptr, st_row, st_val
   o = userdata%integer( 1 )
   n = userdata%integer( 2 )
   nflag = 3
   st_flag = 3
   st_ptr = st_flag + MAX( o, n )
   st_row = st_ptr + n + 1
   st_val = 0
   IF ( transpose ) THEN
     DO l = 1, n_free
       j = FREE( l )
       val = 0.0_rp_
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         val = val + userdata%real( st_val + k ) *                             &
                 V( userdata%integer( st_row + k ) )
       END DO
       P( j ) = val
     END DO
   ELSE
     P( : o ) = 0.0_rp_
     DO l = 1, n_free
       j = FREE( l )
       val = V( j )
       DO k = userdata%integer( st_ptr + j ),                                  &
              userdata%integer( st_ptr + j + 1 ) - 1
         i = userdata%integer( st_row + k )
         P( i ) = P( i ) + userdata%real( st_val + k ) * val
       END DO
     END DO
   END IF
   status = 0
   RETURN
   END SUBROUTINE AFPROD

   SUBROUTINE APROD_broken( status, userdata, transpose, V, P )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
   status = - 1
   RETURN
   END SUBROUTINE APROD_broken

   SUBROUTINE ASCOL_broken( status, userdata, col, P, IP, lp )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: col
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
   INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: lp
   INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( INOUT ) :: IP
   P( 1 ) = 0.0_rp_
   status = - 1
   RETURN
   END SUBROUTINE ASCOL_broken

   SUBROUTINE AFPROD_broken( status, userdata, transpose, V, P, FREE, n_free )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   LOGICAL, INTENT( IN ) :: transpose
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: n_free
   INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( : ) :: FREE
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: P
   P( 1 ) = 0.0_rp_
   status = - 1
   RETURN
   END SUBROUTINE AFPROD_broken

   SUBROUTINE PREC( status, userdata, V, P )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   TYPE ( USERDATA_type ), INTENT( INOUT ) :: userdata
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P
!  recover problem data from userdata
   INTEGER ( KIND = ip_ ) :: n
   n = userdata%integer( 1 )
   P( : n ) = V( : n ) / userdata%real( : n )
   status = 0
   RETURN
   END SUBROUTINE PREC
