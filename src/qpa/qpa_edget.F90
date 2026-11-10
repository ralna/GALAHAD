! THIS VERSION: GALAHAD 5.6.1 - 2026-11-10 AT 07:43 GMT.
!
! Edge-case and reliability tests for QPA. Every solution is checked
! independently of QPA through the KKT conditions of the original problem
!
!   minimize 1/2 x'Hx + g'x + f  s.t.  c_l <= Ax <= c_u,  x_l <= x <= x_u,
!
! using QPA's convention Hx + g = A'y + z, with y_i >= 0 (<= 0) when the i-th
! constraint is active at its lower (upper) bound, and similarly for z.
! Random problems are generated with a planted solution (x*,y*,z*) and a
! positive-definite H, so that x* is the unique minimizer and x can be
! compared with it directly. Each check prints "ok" or "FAIL"; the program
! stops with exit code 1 if any check fails.

#include "galahad_modules.h"

   PROGRAM GALAHAD_QPA_edge_test_program
   USE GALAHAD_KINDS_precision
   USE GALAHAD_SYMBOLS
   USE GALAHAD_QPA_precision
   IMPLICIT NONE

   REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_, one = 1.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: eps = EPSILON( one )
   REAL ( KIND = rp_ ), PARAMETER :: tol = eps ** ( 1.0_rp_ / 3.0_rp_ )
   REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 19
   REAL ( KIND = rp_ ), PARAMETER :: big_bound = 10.0_rp_ ** 20

!  a problem in co-ordinate form, kept separately from the copy given to QPA

   TYPE :: qp_type
     INTEGER ( KIND = ip_ ) :: n = 0, m = 0, h_ne = 0, a_ne = 0
     REAL ( KIND = rp_ ) :: f = zero
     INTEGER ( KIND = ip_ ), ALLOCATABLE :: h_row( : ), h_col( : )
     INTEGER ( KIND = ip_ ), ALLOCATABLE :: a_row( : ), a_col( : )
     REAL ( KIND = rp_ ), ALLOCATABLE :: h_val( : ), a_val( : ), g( : )
     REAL ( KIND = rp_ ), ALLOCATABLE :: x_l( : ), x_u( : ), c_l( : ), c_u( : )
     REAL ( KIND = rp_ ), ALLOCATABLE :: x_star( : )   ! planted solution
     LOGICAL :: has_star = .FALSE.
   END TYPE qp_type

!  the outcome of a solve

   TYPE :: result_type
     INTEGER ( KIND = ip_ ) :: status = 0, iter = 0
     INTEGER ( KIND = ip_ ) :: num_g_infeas = 0, num_b_infeas = 0
     REAL ( KIND = rp_ ) :: obj = zero, infeas_g = zero, infeas_b = zero
     REAL ( KIND = rp_ ), ALLOCATABLE :: x( : ), y( : ), z( : )
     INTEGER ( KIND = ip_ ), ALLOCATABLE :: c_stat( : ), b_stat( : )
   END TYPE result_type

   INTEGER ( KIND = ip_ ) :: nfail, npass
   INTEGER ( KIND = i8_ ) :: seed

   nfail = 0 ; npass = 0 ; seed = 1234567_i8_
#ifdef REAL_32
   WRITE( 6, "( '  qpa edge-case tests, single precision', / )" )
#elif REAL_128
   WRITE( 6, "( '  qpa edge-case tests, quadruple precision', / )" )
#else
   WRITE( 6, "( '  qpa edge-case tests, double precision', / )" )
#endif

   WRITE( 6, "( ' -- invalid input --' )" )
   CALL case_invalid_input( )
   CALL case_upper_entry( )

   WRITE( 6, "( /, ' -- small and special problems --' )" )
   CALL case_small_problems( )

   WRITE( 6, "( /, ' -- infeasible and unbounded problems --' )" )
   CALL case_infeasible_unbounded( )

   WRITE( 6, "( /, ' -- fixed variables --' )" )
   CALL case_fixed_variables( )

   WRITE( 6, "( /, ' -- randomized problems with planted solutions --' )" )
   CALL case_planted_sweep( )

   WRITE( 6, "( /, ' -- duplicated entries in H and A --' )" )
   CALL case_duplicates( )

   WRITE( 6, "( /, ' -- extreme scaling and infinite bounds --' )" )
   CALL case_scaling_and_bounds( )

   WRITE( 6, "( /, ' -- starting points and warm starts --' )" )
   CALL case_starts( )

   WRITE( 6, "( /, ' -- non-default control parameters --' )" )
   CALL case_controls( )

   WRITE( 6, "( /, ' -- l_1 penalty problems and larger problems --' )" )
   CALL case_l1_and_large( )

   WRITE( 6, "( /, '  ', I0, ' checks passed, ', I0, ' failed' )" )         &
     npass, nfail
   IF ( nfail /= 0 ) THEN
     WRITE( 6, "( '  qpa edge-case tests FAILED' )" )
     STOP 1
   END IF
   WRITE( 6, "( '  qpa edge-case tests completed' )" )

 CONTAINS

!  =========================================================================
!  reporting
!  =========================================================================

   SUBROUTINE report( cond, label )
   LOGICAL, INTENT( IN ) :: cond
   CHARACTER ( * ), INTENT( IN ) :: label
   IF ( cond ) THEN
     WRITE( 6, "( '  ok   : ', A )" ) label
     npass = npass + 1
   ELSE
     WRITE( 6, "( '  FAIL : ', A )" ) label
     nfail = nfail + 1
   END IF
   FLUSH( 6 )
   END SUBROUTINE report

!  =========================================================================
!  random numbers (a simple 64-bit LCG, identical in every precision)
!  =========================================================================

   REAL ( KIND = rp_ ) FUNCTION urand( )      ! uniform on [0,1)
   seed = seed * 6364136223846793005_i8_ + 1442695040888963407_i8_
   urand = REAL( ISHFT( seed, - 11 ), rp_ ) / 2.0_rp_ ** 53
   urand = MIN( MAX( urand, zero ), one - eps )
   END FUNCTION urand

   INTEGER ( KIND = ip_ ) FUNCTION irand( k )   ! uniform on 1..k
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: k
   irand = MIN( k, 1_ip_ + INT( urand( ) * REAL( k, rp_ ), ip_ ) )
   END FUNCTION irand

!  =========================================================================
!  products with the stored (co-ordinate) matrices
!  =========================================================================

   SUBROUTINE h_prod( qp, x, hx )             ! hx = H x, H symmetric
   TYPE ( qp_type ), INTENT( IN ) :: qp
   REAL ( KIND = rp_ ), INTENT( IN ) :: x( : )
   REAL ( KIND = rp_ ), INTENT( OUT ) :: hx( : )
   INTEGER ( KIND = ip_ ) :: k, i, j
   hx( : qp%n ) = zero
   DO k = 1, qp%h_ne
     i = qp%h_row( k ) ; j = qp%h_col( k )
     hx( i ) = hx( i ) + qp%h_val( k ) * x( j )
     IF ( i /= j ) hx( j ) = hx( j ) + qp%h_val( k ) * x( i )
   END DO
   END SUBROUTINE h_prod

   SUBROUTINE a_prod( qp, x, ax )             ! ax = A x
   TYPE ( qp_type ), INTENT( IN ) :: qp
   REAL ( KIND = rp_ ), INTENT( IN ) :: x( : )
   REAL ( KIND = rp_ ), INTENT( OUT ) :: ax( : )
   INTEGER ( KIND = ip_ ) :: k
   ax( : qp%m ) = zero
   DO k = 1, qp%a_ne
     ax( qp%a_row( k ) ) = ax( qp%a_row( k ) )                                 &
                         + qp%a_val( k ) * x( qp%a_col( k ) )
   END DO
   END SUBROUTINE a_prod

   SUBROUTINE at_prod( qp, y, aty )           ! aty = A' y
   TYPE ( qp_type ), INTENT( IN ) :: qp
   REAL ( KIND = rp_ ), INTENT( IN ) :: y( : )
   REAL ( KIND = rp_ ), INTENT( OUT ) :: aty( : )
   INTEGER ( KIND = ip_ ) :: k
   aty( : qp%n ) = zero
   DO k = 1, qp%a_ne
     aty( qp%a_col( k ) ) = aty( qp%a_col( k ) )                               &
                          + qp%a_val( k ) * y( qp%a_row( k ) )
   END DO
   END SUBROUTINE at_prod

   REAL ( KIND = rp_ ) FUNCTION objective( qp, x )
   TYPE ( qp_type ), INTENT( IN ) :: qp
   REAL ( KIND = rp_ ), INTENT( IN ) :: x( : )
   REAL ( KIND = rp_ ) :: hx( qp%n )
   CALL h_prod( qp, x, hx )
   objective = 0.5_rp_ * DOT_PRODUCT( x( : qp%n ), hx ) +                      &
               DOT_PRODUCT( qp%g, x( : qp%n ) ) + qp%f
   END FUNCTION objective

!  =========================================================================
!  KKT check of (x,y,z) for the original problem. Returns the largest of
!  the relative primal infeasibility, the relative dual residual
!  || Hx + g - A'y - z ||, and the relative violation of the sign and
!  complementarity conditions; the individual measures are returned too
!  =========================================================================

   REAL ( KIND = rp_ ) FUNCTION kkt_error( qp, x, y, z, pinf, dres, comp )
   TYPE ( qp_type ), INTENT( IN ) :: qp
   REAL ( KIND = rp_ ), INTENT( IN ) :: x( : ), y( : ), z( : )
   REAL ( KIND = rp_ ), INTENT( OUT ) :: pinf, dres, comp
   REAL ( KIND = rp_ ) :: hx( qp%n ), ax( MAX( qp%m, 1 ) ), aty( qp%n )
   REAL ( KIND = rp_ ) :: r( qp%n ), xs, gs, d
   INTEGER ( KIND = ip_ ) :: i
   CALL h_prod( qp, x, hx )
   CALL a_prod( qp, x, ax )
   CALL at_prod( qp, y, aty )
   xs = MAX( one, MAXVAL( ABS( x( : qp%n ) ) ) )
   gs = MAX( TINY( one ), MAXVAL( ABS( qp%g ) ), MAXVAL( ABS( hx ) ),          &
             MAXVAL( ABS( aty ) ), MAXVAL( ABS( z( : qp%n ) ) ) )
!  primal infeasibility, relative to the size of x (and of Ax)
   pinf = zero
   DO i = 1, qp%n
     IF ( qp%x_l( i ) > - infinity ) pinf = MAX( pinf, qp%x_l( i ) - x( i ) )
     IF ( qp%x_u( i ) < infinity ) pinf = MAX( pinf, x( i ) - qp%x_u( i ) )
   END DO
   DO i = 1, qp%m
     IF ( qp%c_l( i ) > - infinity ) pinf = MAX( pinf, qp%c_l( i ) - ax( i ) )
     IF ( qp%c_u( i ) < infinity ) pinf = MAX( pinf, ax( i ) - qp%c_u( i ) )
   END DO
   IF ( qp%m > 0 ) xs = MAX( xs, MAXVAL( ABS( ax( : qp%m ) ) ) )
   pinf = pinf / xs
!  dual residual
   r = hx + qp%g - aty - z( : qp%n )
   dres = MAXVAL( ABS( r ) ) / gs
!  signs and complementarity: a positive (negative) multiplier needs a finite
!  lower (upper) bound that is (nearly) active
   comp = zero
   DO i = 1, qp%n
     d = bound_gap( z( i ), x( i ), qp%x_l( i ), qp%x_u( i ), xs )
     comp = MAX( comp, d )
   END DO
   DO i = 1, qp%m
     d = bound_gap( y( i ), ax( i ), qp%c_l( i ), qp%c_u( i ), xs )
     comp = MAX( comp, d )
   END DO
   comp = comp / ( gs * xs )
   kkt_error = MAX( pinf, dres, comp )
   END FUNCTION kkt_error

!  |multiplier| times the distance to the bound it belongs to. A nonzero
!  multiplier for an infinite bound has the wrong sign; it is measured by its
!  size times scale_inf (the scale of x), so rounding-level values pass

   REAL ( KIND = rp_ ) FUNCTION bound_gap( mult, v, lo, up, scale_inf )
   REAL ( KIND = rp_ ), INTENT( IN ) :: mult, v, lo, up, scale_inf
   bound_gap = zero
   IF ( mult > zero ) THEN
     IF ( lo > - infinity ) THEN
       bound_gap = mult * ABS( v - lo )
     ELSE
       bound_gap = mult * scale_inf
     END IF
   ELSE IF ( mult < zero ) THEN
     IF ( up < infinity ) THEN
       bound_gap = - mult * ABS( up - v )
     ELSE
       bound_gap = - mult * scale_inf
     END IF
   END IF
   END FUNCTION bound_gap

!  =========================================================================
!  solve a stored problem with QPA (on a fresh copy of the data)
!  =========================================================================

   SUBROUTINE solve( qp, control, res, x0, c_stat0, b_stat0, n_override,     &
                     m_override, rho, restored )
   TYPE ( qp_type ), INTENT( IN ) :: qp
   TYPE ( QPA_control_type ), INTENT( IN ) :: control
   TYPE ( result_type ), INTENT( OUT ) :: res
   REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: x0( : )
   INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: c_stat0( : ), b_stat0( : )
   INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: n_override, m_override
   REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: rho
   LOGICAL, OPTIONAL, INTENT( OUT ) :: restored
   TYPE ( QPT_problem_type ) :: p
   TYPE ( QPA_data_type ) :: data
   TYPE ( QPA_control_type ) :: ctrl
   TYPE ( QPA_inform_type ) :: inform
   INTEGER ( KIND = ip_ ) :: s, mm
   mm = MAX( qp%m, 1_ip_ )
   ALLOCATE( p%G( qp%n ), p%X_l( qp%n ), p%X_u( qp%n ), p%X( qp%n ),           &
             p%Z( qp%n ), p%C( mm ), p%C_l( mm ), p%C_u( mm ), p%Y( mm ) )
   ALLOCATE( res%x( qp%n ), res%z( qp%n ), res%y( mm ),                        &
             res%c_stat( mm ), res%b_stat( qp%n ) )
   p%new_problem_structure = .TRUE.
   p%n = qp%n ; p%m = qp%m ; p%f = qp%f
   p%G = qp%g ; p%X_l = qp%x_l ; p%X_u = qp%x_u
   p%C_l = zero ; p%C_u = zero
   IF ( qp%m > 0 ) THEN
     p%C_l( : qp%m ) = qp%c_l ; p%C_u( : qp%m ) = qp%c_u
   END IF
   p%rho_g = 10.0_rp_ ; p%rho_b = 10.0_rp_
   IF ( PRESENT( rho ) ) THEN
     p%rho_g = rho ; p%rho_b = rho
   END IF
   p%X = zero ; p%Y = zero ; p%Z = zero ; p%C = zero
   IF ( PRESENT( x0 ) ) p%X = x0( : qp%n )
   CALL SMT_put( p%H%type, 'COORDINATE', s )
   CALL SMT_put( p%A%type, 'COORDINATE', s )
   ALLOCATE( p%H%val( MAX( qp%h_ne, 1 ) ), p%H%row( MAX( qp%h_ne, 1 ) ),       &
             p%H%col( MAX( qp%h_ne, 1 ) ) )
   ALLOCATE( p%A%val( MAX( qp%a_ne, 1 ) ), p%A%row( MAX( qp%a_ne, 1 ) ),       &
             p%A%col( MAX( qp%a_ne, 1 ) ) )
   p%H%ne = qp%h_ne ; p%A%ne = qp%a_ne
   IF ( qp%h_ne > 0 ) THEN
     p%H%val( : qp%h_ne ) = qp%h_val( : qp%h_ne )
     p%H%row( : qp%h_ne ) = qp%h_row( : qp%h_ne )
     p%H%col( : qp%h_ne ) = qp%h_col( : qp%h_ne )
   END IF
   IF ( qp%a_ne > 0 ) THEN
     p%A%val( : qp%a_ne ) = qp%a_val( : qp%a_ne )
     p%A%row( : qp%a_ne ) = qp%a_row( : qp%a_ne )
     p%A%col( : qp%a_ne ) = qp%a_col( : qp%a_ne )
   END IF
   res%c_stat = 0 ; res%b_stat = 0
   IF ( PRESENT( c_stat0 ) ) res%c_stat( : qp%m ) = c_stat0( : qp%m )
   IF ( PRESENT( b_stat0 ) ) res%b_stat = b_stat0( : qp%n )
   CALL QPA_initialize( data, ctrl, inform )
   ctrl = control
   IF ( PRESENT( n_override ) ) p%n = n_override
   IF ( PRESENT( m_override ) ) p%m = m_override
   CALL QPA_solve( p, res%c_stat, res%b_stat, data, ctrl, inform )
   res%status = inform%status ; res%iter = inform%iter ; res%obj = inform%obj
   res%infeas_g = inform%infeas_g ; res%infeas_b = inform%infeas_b
   res%num_g_infeas = inform%num_g_infeas
   res%num_b_infeas = inform%num_b_infeas
   res%x = p%X ; res%z = p%Z ; res%y = p%Y
!  check that the values of H and A given to QPA have been restored exactly
   IF ( PRESENT( restored ) ) THEN
     restored = p%H%ne == qp%h_ne .AND. p%A%ne == qp%a_ne
     IF ( restored .AND. qp%h_ne > 0 ) restored =                              &
       ALL( p%H%val( : qp%h_ne ) == qp%h_val( : qp%h_ne ) ) .AND.              &
       ALL( p%H%row( : qp%h_ne ) == qp%h_row( : qp%h_ne ) ) .AND.              &
       ALL( p%H%col( : qp%h_ne ) == qp%h_col( : qp%h_ne ) )
     IF ( restored .AND. qp%a_ne > 0 ) restored =                              &
       ALL( p%A%val( : qp%a_ne ) == qp%a_val( : qp%a_ne ) ) .AND.              &
       ALL( p%A%row( : qp%a_ne ) == qp%a_row( : qp%a_ne ) ) .AND.              &
       ALL( p%A%col( : qp%a_ne ) == qp%a_col( : qp%a_ne ) )
   END IF
   CALL QPA_terminate( data, ctrl, inform )
   END SUBROUTINE solve

!  default controls for the tests: quiet, and solving the QP itself

   SUBROUTINE set_controls( control )
   TYPE ( QPA_control_type ), INTENT( OUT ) :: control
   TYPE ( QPA_data_type ) :: data
   TYPE ( QPA_inform_type ) :: inform
   CALL QPA_initialize( data, control, inform )
   CALL QPA_terminate( data, control, inform )
   control%infinity = infinity
   control%solve_qp = .TRUE.
   control%print_level = 0
   END SUBROUTINE set_controls

!  check a result: status 0, KKT conditions and, for a planted problem, x*

   SUBROUTINE check( qp, res, label, ok_out )
   TYPE ( qp_type ), INTENT( IN ) :: qp
   TYPE ( result_type ), INTENT( IN ) :: res
   CHARACTER ( * ), INTENT( IN ) :: label
   LOGICAL, OPTIONAL, INTENT( OUT ) :: ok_out
   REAL ( KIND = rp_ ) :: e, pinf, dres, comp, dx
   CHARACTER ( LEN = 100 ) :: buf
   LOGICAL :: ok
   IF ( res%status /= 0 ) THEN
     WRITE( buf, "( ' (status ', I0, ', QPA infeas g ', ES8.1, ' in ', I0,    &
    &       ', b ', ES8.1, ' in ', I0, ')' )" ) res%status, res%infeas_g,      &
       res%num_g_infeas, res%infeas_b, res%num_b_infeas
     ok = .FALSE.
   ELSE
     e = kkt_error( qp, res%x, res%y, res%z, pinf, dres, comp )
     dx = zero
     IF ( qp%has_star ) dx = MAXVAL( ABS( res%x - qp%x_star ) ) /              &
                             MAX( one, MAXVAL( ABS( qp%x_star ) ) )
     ok = e <= tol .AND. dx <= tol
     WRITE( buf, "( ' (infeas ', ES8.1, ', dual ', ES8.1, ', comp ', ES8.1,    &
    &               ', |x-x*| ', ES8.1, ')' )" ) pinf, dres, comp, dx
   END IF
   IF ( PRESENT( ok_out ) ) THEN
     ok_out = ok
     IF ( .NOT. ok ) WRITE( 6, "( '         ', A, A )" ) label, TRIM( buf )
   ELSE
     IF ( ok ) THEN
       CALL report( .TRUE., label )
     ELSE
       CALL report( .FALSE., label // TRIM( buf ) )
     END IF
   END IF
   END SUBROUTINE check

!  =========================================================================
!  random problems with a planted solution
!  =========================================================================

!  kinds of degeneracy that can be requested
!   n_fixed      fraction of fixed variables (x_l = x_u)
!   weak         fraction of active bounds/constraints with zero multiplier
!   dependent    number of constraint rows that are copies of other rows
!   scale        the objective (H and g) and the multipliers are scaled by
!                scale, so that the problem is a consistently scaled copy
!                of the unscaled one

   SUBROUTINE planted( qp, n, m, weak, fixed, dependent, scale )
   TYPE ( qp_type ), INTENT( OUT ) :: qp
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m, dependent
   REAL ( KIND = rp_ ), INTENT( IN ) :: weak, fixed, scale
   INTEGER ( KIND = ip_ ) :: i, j, k, kk, per_row, nh, src
   REAL ( KIND = rp_ ) :: ystar( MAX( m, 1 ) ), zstar( n ), hx( n ), aty( n )
   REAL ( KIND = rp_ ) :: ax( MAX( m, 1 ) ), offsum( n ), u, mult
   qp%n = n ; qp%m = m ; qp%f = urand( ) - 0.5_rp_
!  H: random sparse symmetric, made positive definite by diagonal dominance
   nh = n + 2 * n
   ALLOCATE( qp%h_row( nh ), qp%h_col( nh ), qp%h_val( nh ) )
   offsum = zero ; k = 0
   DO kk = 1, 2 * n
     i = irand( n ) ; j = irand( n )
     IF ( i == j ) CYCLE
     k = k + 1
     qp%h_row( k ) = MAX( i, j ) ; qp%h_col( k ) = MIN( i, j )
     qp%h_val( k ) = 2.0_rp_ * urand( ) - one
     offsum( i ) = offsum( i ) + ABS( qp%h_val( k ) )
     offsum( j ) = offsum( j ) + ABS( qp%h_val( k ) )
   END DO
   DO i = 1, n
     k = k + 1
     qp%h_row( k ) = i ; qp%h_col( k ) = i
     qp%h_val( k ) = offsum( i ) + 0.1_rp_ + urand( )
   END DO
   qp%h_ne = k
   qp%h_val( : k ) = scale * qp%h_val( : k )
!  A: about three entries per row; the last "dependent" rows copy others
   per_row = MIN( 3_ip_, n )
   ALLOCATE( qp%a_row( MAX( per_row * m, 1 ) ),                                &
             qp%a_col( MAX( per_row * m, 1 ) ),                                &
             qp%a_val( MAX( per_row * m, 1 ) ) )
   k = 0
   DO i = 1, m
     IF ( i > m - dependent .AND. i > 1 ) THEN
       src = irand( i - 1 )
       DO kk = 1, k
         IF ( qp%a_row( kk ) == src ) THEN
           IF ( k + 1 > SIZE( qp%a_row ) ) EXIT
           k = k + 1
           qp%a_row( k ) = i ; qp%a_col( k ) = qp%a_col( kk )
           qp%a_val( k ) = qp%a_val( kk )
         END IF
       END DO
     ELSE
       DO kk = 1, per_row
         k = k + 1
         qp%a_row( k ) = i
         qp%a_col( k ) = MOD( i + ( kk - 1 ) * ( 1 + n / per_row ) - 1, n ) + 1
         qp%a_val( k ) = 2.0_rp_ * urand( ) - one
       END DO
     END IF
   END DO
   qp%a_ne = k
!  the planted solution and the status of every bound and constraint
   ALLOCATE( qp%x_star( n ), qp%g( n ), qp%x_l( n ), qp%x_u( n ),              &
             qp%c_l( MAX( m, 1 ) ), qp%c_u( MAX( m, 1 ) ) )
   DO i = 1, n
     qp%x_star( i ) = 2.0_rp_ * urand( ) - one
   END DO
   CALL a_prod( qp, qp%x_star, ax )
   zstar = zero
   DO i = 1, n
     u = urand( )
     mult = MERGE( zero, 0.1_rp_ + urand( ), urand( ) < weak )
     IF ( u < fixed ) THEN                          ! fixed variable
       qp%x_l( i ) = qp%x_star( i ) ; qp%x_u( i ) = qp%x_star( i )
       zstar( i ) = scale * ( 2.0_rp_ * urand( ) - one )
     ELSE IF ( u < fixed + 0.25_rp_ ) THEN          ! active lower bound
       qp%x_l( i ) = qp%x_star( i ) ; qp%x_u( i ) = qp%x_star( i ) + one
       IF ( urand( ) < 0.3_rp_ ) qp%x_u( i ) = big_bound
       zstar( i ) = scale * mult
     ELSE IF ( u < fixed + 0.5_rp_ ) THEN           ! active upper bound
       qp%x_u( i ) = qp%x_star( i ) ; qp%x_l( i ) = qp%x_star( i ) - one
       IF ( urand( ) < 0.3_rp_ ) qp%x_l( i ) = - big_bound
       zstar( i ) = - scale * mult
     ELSE IF ( u < fixed + 0.75_rp_ ) THEN          ! inactive, bounded
       qp%x_l( i ) = qp%x_star( i ) - 0.5_rp_ - urand( )
       qp%x_u( i ) = qp%x_star( i ) + 0.5_rp_ + urand( )
     ELSE                                           ! free
       qp%x_l( i ) = - big_bound ; qp%x_u( i ) = big_bound
     END IF
   END DO
   ystar = zero
   DO i = 1, m
     u = urand( )
     mult = MERGE( zero, 0.1_rp_ + urand( ), urand( ) < weak )
     IF ( i > m - dependent .AND. i > 1 ) THEN     ! copy: keep consistent,
       qp%c_l( i ) = ax( i ) - one ; qp%c_u( i ) = ax( i ) + one   ! inactive
       IF ( u < 0.5_rp_ ) THEN                     ! or an active equality
         qp%c_l( i ) = ax( i ) ; qp%c_u( i ) = ax( i )
       END IF
     ELSE IF ( u < 0.2_rp_ ) THEN                  ! equality
       qp%c_l( i ) = ax( i ) ; qp%c_u( i ) = ax( i )
       ystar( i ) = scale * ( 2.0_rp_ * urand( ) - one )
     ELSE IF ( u < 0.45_rp_ ) THEN                 ! active lower
       qp%c_l( i ) = ax( i ) ; qp%c_u( i ) = ax( i ) + one
       IF ( urand( ) < 0.3_rp_ ) qp%c_u( i ) = big_bound
       ystar( i ) = scale * mult
     ELSE IF ( u < 0.7_rp_ ) THEN                  ! active upper
       qp%c_u( i ) = ax( i ) ; qp%c_l( i ) = ax( i ) - one
       IF ( urand( ) < 0.3_rp_ ) qp%c_l( i ) = - big_bound
       ystar( i ) = - scale * mult
     ELSE                                          ! inactive
       qp%c_l( i ) = ax( i ) - 0.5_rp_ - urand( )
       qp%c_u( i ) = ax( i ) + 0.5_rp_ + urand( )
       IF ( urand( ) < 0.3_rp_ ) qp%c_l( i ) = - big_bound
     END IF
   END DO
!  g = A'y* + z* - H x* makes (x*,y*,z*) a KKT point, hence (H positive
!  definite) x* the unique solution
   CALL h_prod( qp, qp%x_star, hx )
   CALL at_prod( qp, ystar, aty )
   qp%g = aty + zstar - hx
   qp%has_star = .TRUE.
   END SUBROUTINE planted

!  =========================================================================
!  sweep: many random planted problems, with and without degeneracy
!  =========================================================================

   SUBROUTINE case_planted_sweep( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   INTEGER ( KIND = ip_ ) :: trial, kind, ntrial, nbad, n, m
   LOGICAL :: ok
   CHARACTER ( LEN = 80 ) :: lab
   CHARACTER ( LEN = 24 ), PARAMETER :: kinds( 4 ) =                           &
     [ CHARACTER ( LEN = 24 ) :: 'nondegenerate', 'weakly active',            &
       'fixed variables', 'dependent constraints' ]
   CALL set_controls( control )
   ntrial = 40
   DO kind = 1, 4
     nbad = 0
     DO trial = 1, ntrial
       n = 5 + irand( 40_ip_ ) ; m = irand( n )
       SELECT CASE ( kind )
       CASE ( 1 ) ; CALL planted( qp, n, m, zero, zero, 0_ip_, one )
       CASE ( 2 ) ; CALL planted( qp, n, m, 0.5_rp_, zero, 0_ip_, one )
       CASE ( 3 ) ; CALL planted( qp, n, m, zero, 0.3_rp_, 0_ip_, one )
       CASE ( 4 ) ; CALL planted( qp, n, m, zero, zero, MIN( 3_ip_, m / 2 ),   &
                                  one )
       END SELECT
       CALL solve( qp, control, res )
       WRITE( lab, "( A, ' trial ', I0, ' (n = ', I0, ', m = ', I0, ')' )" )   &
         TRIM( kinds( kind ) ), trial, n, m
       CALL check( qp, res, TRIM( lab ), ok )
       IF ( .NOT. ok ) nbad = nbad + 1
     END DO
     WRITE( lab, "( A, ': ', I0, ' random problems' )" ) TRIM( kinds( kind ) ),&
       ntrial
     IF ( nbad > 0 ) WRITE( lab, "( A, ' (', I0, ' failed)' )" )               &
       TRIM( lab ), nbad
     CALL report( nbad == 0, TRIM( lab ) )
   END DO
   END SUBROUTINE case_planted_sweep


!  =========================================================================
!  build a problem from dense data (H lower triangle taken from Hd)
!  =========================================================================

   SUBROUTINE make_dense( qp, n, m, hd, ad, g, xl, xu, cl, cu )
   TYPE ( qp_type ), INTENT( OUT ) :: qp
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m
   REAL ( KIND = rp_ ), INTENT( IN ) :: hd( n, n ), ad( MAX( m, 1 ), n )
   REAL ( KIND = rp_ ), INTENT( IN ) :: g( n ), xl( n ), xu( n )
   REAL ( KIND = rp_ ), INTENT( IN ) :: cl( MAX( m, 1 ) ), cu( MAX( m, 1 ) )
   INTEGER ( KIND = ip_ ) :: i, j, k
   qp%n = n ; qp%m = m ; qp%f = zero
   ALLOCATE( qp%h_row( MAX( n * n, 1 ) ), qp%h_col( MAX( n * n, 1 ) ),        &
             qp%h_val( MAX( n * n, 1 ) ) )
   ALLOCATE( qp%a_row( MAX( n * m, 1 ) ), qp%a_col( MAX( n * m, 1 ) ),        &
             qp%a_val( MAX( n * m, 1 ) ) )
   k = 0
   DO j = 1, n
     DO i = j, n
       IF ( hd( i, j ) /= zero ) THEN
         k = k + 1 ; qp%h_row( k ) = i ; qp%h_col( k ) = j
         qp%h_val( k ) = hd( i, j )
       END IF
     END DO
   END DO
   qp%h_ne = k
   k = 0
   DO i = 1, m
     DO j = 1, n
       IF ( ad( i, j ) /= zero ) THEN
         k = k + 1 ; qp%a_row( k ) = i ; qp%a_col( k ) = j
         qp%a_val( k ) = ad( i, j )
       END IF
     END DO
   END DO
   qp%a_ne = k
   ALLOCATE( qp%g( n ), qp%x_l( n ), qp%x_u( n ), qp%c_l( MAX( m, 1 ) ),      &
             qp%c_u( MAX( m, 1 ) ) )
   qp%g = g ; qp%x_l = xl ; qp%x_u = xu ; qp%c_l = cl ; qp%c_u = cu
   END SUBROUTINE make_dense

!  check that a solve ends with an expected status

   SUBROUTINE check_status( res, expected, label )
   TYPE ( result_type ), INTENT( IN ) :: res
   INTEGER ( KIND = ip_ ), INTENT( IN ) :: expected
   CHARACTER ( * ), INTENT( IN ) :: label
   CHARACTER ( LEN = 40 ) :: buf
   IF ( res%status == expected ) THEN
     CALL report( .TRUE., label )
   ELSE
     WRITE( buf, "( ' (status ', I0, ', expected ', I0, ')' )" )               &
       res%status, expected
     CALL report( .FALSE., label // TRIM( buf ) )
   END IF
   END SUBROUTINE check_status

!  identity-like dense helpers

   SUBROUTINE eye( a, d )
   REAL ( KIND = rp_ ), INTENT( OUT ) :: a( :, : )
   REAL ( KIND = rp_ ), INTENT( IN ) :: d
   INTEGER ( KIND = ip_ ) :: i
   a = zero
   DO i = 1, MIN( SIZE( a, 1 ), SIZE( a, 2 ) )
     a( i, i ) = d
   END DO
   END SUBROUTINE eye

!  =========================================================================
!  invalid input
!  =========================================================================

   SUBROUTINE case_invalid_input( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   REAL ( KIND = rp_ ) :: h( 2, 2 ), a( 1, 2 )
   CALL set_controls( control )
   CALL eye( h, one ) ; a = one
   CALL make_dense( qp, 2_ip_, 1_ip_, h, a, (/ one, one /), (/ - one, - one /),&
                    (/ one, one /), (/ zero /), (/ one /) )
   qp%n = 0
   CALL solve_raw( qp, control, res )
   CALL check_status( res, GALAHAD_error_restrictions, 'n = 0' )
   qp%n = 2 ; qp%m = - 1
   CALL solve_raw( qp, control, res )
   CALL check_status( res, GALAHAD_error_restrictions, 'm = -1' )
   qp%m = 1
   END SUBROUTINE case_invalid_input

!  in co-ordinate storage, an entry (i,j) of H with i < j is accepted and
!  treated as its mirror image (j,i); the result must match the lower form

   SUBROUTINE case_upper_entry( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res, res_low
   REAL ( KIND = rp_ ) :: h( 2, 2 ), a( 1, 2 )
   INTEGER ( KIND = ip_ ) :: k
   CALL set_controls( control )
   h = RESHAPE( (/ 2.0_rp_, one, one, 2.0_rp_ /), (/ 2, 2 /) ) ; a = zero
   CALL make_dense( qp, 2_ip_, 0_ip_, h, a, (/ - one, - one /),                &
                    (/ - big_bound, - big_bound /), (/ big_bound, big_bound /),&
                    (/ zero /), (/ zero /) )
   CALL solve( qp, control, res_low )
   DO k = 1, qp%h_ne
     IF ( qp%h_row( k ) /= qp%h_col( k ) ) THEN
       qp%h_row( k ) = qp%h_col( k ) ; qp%h_col( k ) = 2
     END IF
   END DO
   CALL solve( qp, control, res )
   CALL report( res%status == 0 .AND. res_low%status == 0 .AND.               &
                MAXVAL( ABS( res%x - res_low%x ) ) <= tol .AND.               &
                MAXVAL( ABS( res%x - one / 3.0_rp_ ) ) <= tol,                &
                'co-ordinate H entry above the diagonal used as its mirror' )
   END SUBROUTINE case_upper_entry

!  as solve, but passing n and m straight through (for invalid sizes)

   SUBROUTINE solve_raw( qp, control, res )
   TYPE ( qp_type ), INTENT( IN ) :: qp
   TYPE ( QPA_control_type ), INTENT( IN ) :: control
   TYPE ( result_type ), INTENT( OUT ) :: res
   TYPE ( qp_type ) :: q2
   q2 = qp
   q2%n = MAX( qp%n, 1_ip_ ) ; q2%m = MAX( qp%m, 0_ip_ )
   CALL solve( q2, control, res, n_override = qp%n, m_override = qp%m )
   END SUBROUTINE solve_raw

!  =========================================================================
!  small and special problems
!  =========================================================================

   SUBROUTINE case_small_problems( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   REAL ( KIND = rp_ ) :: h1( 1, 1 ), a1( 1, 1 ), h5( 5, 5 ), a5( 1, 5 )
   REAL ( KIND = rp_ ) :: h3( 3, 3 ), a3( 2, 3 ), g5( 5 )
   INTEGER ( KIND = ip_ ) :: i
   CALL set_controls( control )
!  n = 1, no constraints, no bounds: x = 2
   h1 = 2.0_rp_ ; a1 = zero
   CALL make_dense( qp, 1_ip_, 0_ip_, h1, a1, (/ - 4.0_rp_ /),                 &
                    (/ - big_bound /), (/ big_bound /), (/ zero /), (/ zero /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'n = 1 unconstrained' )
   CALL report( res%status == 0 .AND. ABS( res%x( 1 ) - 2.0_rp_ ) <= tol,      &
                'n = 1 unconstrained: x = 2' )
!  n = 1 with an active bound
   CALL make_dense( qp, 1_ip_, 0_ip_, h1, a1, (/ - 4.0_rp_ /),                 &
                    (/ - big_bound /), (/ one /), (/ zero /), (/ zero /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'n = 1, active upper bound' )
!  LP (H = 0) with a vertex solution: min -x1 - x2, x1 + x2 <= 1, 0 <= x
   h3 = zero ; a3 = zero ; a3( 1, 1 : 2 ) = one ; a3( 2, 3 ) = one
   CALL make_dense( qp, 3_ip_, 2_ip_, h3, a3, (/ - one, - 2.0_rp_, one /),     &
                    (/ zero, zero, zero /),                                    &
                    (/ big_bound, big_bound, 5.0_rp_ /),                       &
                    (/ - big_bound, zero /), (/ one, 3.0_rp_ /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'LP (H = 0) with a vertex solution' )
   CALL report( res%status == 0 .AND. ABS( res%x( 2 ) - one ) <= tol .AND.    &
                ABS( res%x( 1 ) ) <= tol .AND. ABS( res%x( 3 ) ) <= tol,       &
                'LP (H = 0): x = (0, 1, 0)' )
!  nonconvex, bounded: H = -I on a box; any KKT point is acceptable
   CALL eye( h5, - one ) ; a5 = zero
   DO i = 1, 5
     g5( i ) = 0.3_rp_ * REAL( i - 3, rp_ )
   END DO
   CALL make_dense( qp, 5_ip_, 0_ip_, h5, a5, g5, (/ ( - one, i = 1, 5 ) /),   &
                    (/ ( 2.0_rp_, i = 1, 5 ) /), (/ zero /), (/ zero /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'nonconvex (H = -I) on a box' )
!  a zero row of A with consistent bounds 0 <= 0 x <= 1
   CALL eye( h3, one ) ; a3 = zero ; a3( 2, : ) = one
   CALL make_dense( qp, 3_ip_, 2_ip_, h3, a3, (/ one, - one, 2.0_rp_ /),       &
                    (/ ( - big_bound, i = 1, 3 ) /),                           &
                    (/ ( big_bound, i = 1, 3 ) /),                             &
                    (/ zero, one /), (/ one, big_bound /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'zero row 0 <= 0 x <= 1' )
   END SUBROUTINE case_small_problems

!  =========================================================================
!  infeasible and unbounded problems
!  =========================================================================

   SUBROUTINE case_infeasible_unbounded( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   REAL ( KIND = rp_ ) :: h2( 2, 2 ), a2( 2, 2 ), h1( 1, 1 ), a1( 1, 1 )
   CALL set_controls( control )
   CALL eye( h2, one )
!  x1 + x2 >= 3 and x1 + x2 <= 1
   a2 = one
   CALL make_dense( qp, 2_ip_, 2_ip_, h2, a2, (/ zero, zero /),                &
                    (/ - big_bound, - big_bound /), (/ big_bound, big_bound /),&
                    (/ 3.0_rp_, - big_bound /), (/ big_bound, one /) )
   CALL solve( qp, control, res )
   CALL check_status( res, GALAHAD_error_primal_infeasible,                    &
                      'contradictory constraints' )
!  inconsistent dependent equalities x1 + x2 = 1, 2 x1 + 2 x2 = 3
   a2( 2, : ) = 2.0_rp_
   CALL make_dense( qp, 2_ip_, 2_ip_, h2, a2, (/ zero, zero /),                &
                    (/ - big_bound, - big_bound /), (/ big_bound, big_bound /),&
                    (/ one, 3.0_rp_ /), (/ one, 3.0_rp_ /) )
   CALL solve( qp, control, res )
   CALL check_status( res, GALAHAD_error_primal_infeasible,                    &
                      'inconsistent dependent equalities' )
!  constraints and bounds in conflict: x1 + x2 >= 3 with 0 <= x <= 1
   a2 = zero ; a2( 1, : ) = one
   CALL make_dense( qp, 2_ip_, 1_ip_, h2, a2( 1 : 1, : ), (/ zero, zero /),    &
                    (/ zero, zero /), (/ one, one /),                          &
                    (/ 3.0_rp_ /), (/ big_bound /) )
   CALL solve( qp, control, res )
   CALL check_status( res, GALAHAD_error_primal_infeasible,                    &
                      'constraints in conflict with bounds' )
!  a zero row with inconsistent bounds 1 <= 0 x
   a2 = zero ; a2( 1, 1 ) = one
   CALL make_dense( qp, 2_ip_, 2_ip_, h2, a2, (/ zero, zero /),                &
                    (/ - big_bound, - big_bound /), (/ big_bound, big_bound /),&
                    (/ zero, one /), (/ big_bound, big_bound /) )
   CALL solve( qp, control, res )
   CALL check_status( res, GALAHAD_error_primal_infeasible,                    &
                      'zero row with 1 <= 0 x' )
!  unbounded LP: min -x, x >= 0
   h1 = zero ; a1 = zero
   CALL make_dense( qp, 1_ip_, 0_ip_, h1, a1, (/ - one /), (/ zero /),         &
                    (/ big_bound /), (/ zero /), (/ zero /) )
   CALL solve( qp, control, res )
   CALL check_status( res, GALAHAD_error_unbounded, 'unbounded LP' )
!  unbounded nonconvex QP: min -x^2/2
   h1 = - one
   CALL make_dense( qp, 1_ip_, 0_ip_, h1, a1, (/ zero /), (/ - big_bound /),   &
                    (/ big_bound /), (/ zero /), (/ zero /) )
   CALL solve( qp, control, res, x0 = (/ 0.1_rp_ /) )
   CALL check_status( res, GALAHAD_error_unbounded, 'unbounded nonconvex QP' )
!  the same, starting from the stationary point x = 0 (a maximum)
   CALL solve( qp, control, res, x0 = (/ zero /) )
   CALL check_status( res, GALAHAD_error_unbounded,                            &
                      'unbounded nonconvex QP from a stationary point' )
   END SUBROUTINE case_infeasible_unbounded

!  =========================================================================
!  fixed variables (QPP removes them and shifts the constraint bounds)
!  =========================================================================

   SUBROUTINE case_fixed_variables( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   REAL ( KIND = rp_ ) :: h4( 4, 4 ), a4( 1, 4 ), rhs
   CALL set_controls( control )
!  three variables fixed at 1, constraint 0.1 x1 + 0.2 x2 + 0.3 x3 = rhs with
!  rhs the floating-point sum of the terms, so the fixed values satisfy it;
!  removing the fixed variables leaves a residual of order eps
   CALL eye( h4, zero ) ; h4( 4, 4 ) = one
   a4 = zero ; a4( 1, 1 : 3 ) = (/ 0.1_rp_, 0.2_rp_, 0.3_rp_ /)
   rhs = 0.1_rp_ + 0.2_rp_ + 0.3_rp_
   CALL make_dense( qp, 4_ip_, 1_ip_, h4, a4, (/ zero, zero, zero, - one /),   &
                    (/ one, one, one, - 10.0_rp_ /),                           &
                    (/ one, one, one, 10.0_rp_ /), (/ rhs /), (/ rhs /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'row whose variables are all fixed (equality)' )
!  the same as an inequality, active at the fixed values
   CALL make_dense( qp, 4_ip_, 1_ip_, h4, a4, (/ zero, zero, zero, - one /),   &
                    (/ one, one, one, - 10.0_rp_ /),                           &
                    (/ one, one, one, 10.0_rp_ /), (/ rhs /), (/ big_bound /) )
   CALL solve( qp, control, res )
   CALL check( qp, res, 'row whose variables are all fixed (lower bound)' )
   END SUBROUTINE case_fixed_variables

!  =========================================================================
!  duplicated entries in H and A, which should be summed: every entry of a
!  random problem with a planted solution is split into two halves, given as
!  separate entries; the solution must be that of the original problem, and
!  the data given to QPA must be restored on exit
!  =========================================================================

   SUBROUTINE case_duplicates( )
   TYPE ( qp_type ) :: qp, qpd
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   INTEGER ( KIND = ip_ ) :: trial, nbad, nunrestored, n, m
   LOGICAL :: ok, restored
   CHARACTER ( LEN = 80 ) :: lab
   CALL set_controls( control )
   nbad = 0 ; nunrestored = 0
   DO trial = 1, 10
     n = 5 + irand( 40_ip_ ) ; m = irand( n )
     CALL planted( qp, n, m, zero, zero, 0_ip_, one )
     CALL split_entries( qp, qpd )
     CALL solve( qpd, control, res, restored = restored )
     WRITE( lab, "( 'duplicates trial ', I0, ' (n = ', I0, ', m = ', I0,     &
    &               ')' )" ) trial, n, m
     CALL check( qp, res, TRIM( lab ), ok )
     IF ( .NOT. ok ) nbad = nbad + 1
     IF ( .NOT. restored ) nunrestored = nunrestored + 1
   END DO
   WRITE( lab, "( 'every entry duplicated: 10 problems' )" )
   IF ( nbad > 0 ) WRITE( lab, "( A, ' (', I0, ' failed)' )" ) TRIM( lab ), nbad
   CALL report( nbad == 0, TRIM( lab ) )
   WRITE( lab, "( 'duplicated entries restored on exit: 10 problems' )" )
   IF ( nunrestored > 0 )                                                      &
     WRITE( lab, "( A, ' (', I0, ' failed)' )" ) TRIM( lab ), nunrestored
   CALL report( nunrestored == 0, TRIM( lab ) )
   END SUBROUTINE case_duplicates

!  split every entry of H and A into two halves, the second halves being
!  appended after all of the original entries

   SUBROUTINE split_entries( qp, qpd )
   TYPE ( qp_type ), INTENT( IN ) :: qp
   TYPE ( qp_type ), INTENT( OUT ) :: qpd
   INTEGER ( KIND = ip_ ) :: hne, ane
   REAL ( KIND = rp_ ), PARAMETER :: frac = 0.3_rp_
   qpd = qp
   hne = qp%h_ne ; ane = qp%a_ne
   qpd%h_ne = 2 * hne ; qpd%a_ne = 2 * ane
   DEALLOCATE( qpd%h_row, qpd%h_col, qpd%h_val,                               &
               qpd%a_row, qpd%a_col, qpd%a_val )
   ALLOCATE( qpd%h_row( MAX( 2 * hne, 1 ) ), qpd%h_col( MAX( 2 * hne, 1 ) ),  &
             qpd%h_val( MAX( 2 * hne, 1 ) ), qpd%a_row( MAX( 2 * ane, 1 ) ),  &
             qpd%a_col( MAX( 2 * ane, 1 ) ), qpd%a_val( MAX( 2 * ane, 1 ) ) )
   qpd%h_row( : hne ) = qp%h_row( : hne )
   qpd%h_row( hne + 1 : 2 * hne ) = qp%h_row( : hne )
   qpd%h_col( : hne ) = qp%h_col( : hne )
   qpd%h_col( hne + 1 : 2 * hne ) = qp%h_col( : hne )
   qpd%h_val( : hne ) = frac * qp%h_val( : hne )
   qpd%h_val( hne + 1 : 2 * hne ) = qp%h_val( : hne ) - qpd%h_val( : hne )
   qpd%a_row( : ane ) = qp%a_row( : ane )
   qpd%a_row( ane + 1 : 2 * ane ) = qp%a_row( : ane )
   qpd%a_col( : ane ) = qp%a_col( : ane )
   qpd%a_col( ane + 1 : 2 * ane ) = qp%a_col( : ane )
   qpd%a_val( : ane ) = frac * qp%a_val( : ane )
   qpd%a_val( ane + 1 : 2 * ane ) = qp%a_val( : ane ) - qpd%a_val( : ane )
   END SUBROUTINE split_entries

!  =========================================================================
!  extreme scaling and infinite bounds
!  =========================================================================

   SUBROUTINE case_scaling_and_bounds( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   INTEGER ( KIND = ip_ ) :: k
   REAL ( KIND = rp_ ), PARAMETER :: scales( 4 ) =                             &
     (/ 1.0E-6_rp_, 1.0E-3_rp_, 1.0E+3_rp_, 1.0E+6_rp_ /)
   CHARACTER ( LEN = 60 ) :: lab
   CALL set_controls( control )
   DO k = 1, SIZE( scales )
     CALL planted( qp, 30_ip_, 15_ip_, zero, zero, 0_ip_, scales( k ) )
     CALL solve( qp, control, res, rho = 10.0_rp_ * scales( k ) )
     WRITE( lab, "( 'objective and multipliers scaled by ', ES8.1 )" )      &
       scales( k )
     CALL check( qp, res, TRIM( lab ) )
   END DO
!  infinite bounds given as +-1e30 (beyond control%infinity) and finite but
!  huge inactive bounds +-1e18
   CALL planted( qp, 30_ip_, 15_ip_, zero, zero, 0_ip_, one )
   WHERE ( qp%x_l <= - infinity ) qp%x_l = - 1.0E+30_rp_
   WHERE ( qp%x_u >= infinity ) qp%x_u = 1.0E+30_rp_
   CALL solve( qp, control, res )
   CALL check( qp, res, 'infinite bounds given as +-1e30' )
   WHERE ( qp%x_l <= - infinity ) qp%x_l = - 1.0E+18_rp_
   WHERE ( qp%x_u >= infinity ) qp%x_u = 1.0E+18_rp_
   CALL solve( qp, control, res )
   CALL check( qp, res, 'huge finite inactive bounds +-1e18' )
   END SUBROUTINE case_scaling_and_bounds

!  =========================================================================
!  starting points and warm starts
!  =========================================================================

   SUBROUTINE case_starts( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res, res0
   INTEGER ( KIND = ip_ ) :: cs
   CHARACTER ( LEN = 60 ) :: lab
   CALL set_controls( control )
   CALL planted( qp, 40_ip_, 25_ip_, 0.3_rp_, zero, 2_ip_, one )
   DO cs = 1, 4
     control%cold_start = cs
     CALL solve( qp, control, res )
     WRITE( lab, "( 'cold_start = ', I0 )" ) cs
     CALL check( qp, res, TRIM( lab ) )
   END DO
   control%cold_start = 1
   CALL solve( qp, control, res, x0 = qp%x_star )
   CALL check( qp, res, 'cold_start = 1 from the solution' )
   CALL solve( qp, control, res, x0 = qp%x_star + 1.0E+6_rp_ )
   CALL check( qp, res, 'cold_start = 1 from x* + 1e6' )
!  warm start (cold_start = 0) with the optimal and with an empty working set
   control%cold_start = 3
   CALL solve( qp, control, res0 )
   control%cold_start = 0
   CALL solve( qp, control, res, x0 = res0%x, c_stat0 = res0%c_stat,          &
               b_stat0 = res0%b_stat )
   CALL check( qp, res, 'cold_start = 0 with the optimal working set' )
   CALL solve( qp, control, res, c_stat0 = 0 * res0%c_stat,                    &
               b_stat0 = 0 * res0%b_stat )
   CALL check( qp, res, 'cold_start = 0 with an empty working set' )
   CALL solve( qp, control, res, c_stat0 = - SIGN( 1_ip_, res0%c_stat + 1 ),   &
               b_stat0 = SIGN( 1_ip_, res0%b_stat ) )
   CALL check( qp, res, 'cold_start = 0 with a wrong working set' )
   END SUBROUTINE case_starts

!  =========================================================================
!  non-default control parameters
!  =========================================================================

   SUBROUTINE case_controls( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control, base
   TYPE ( result_type ) :: res
   INTEGER ( KIND = ip_ ) :: k
   CHARACTER ( LEN = 60 ) :: lab
   CALL set_controls( base )
   CALL planted( qp, 40_ip_, 25_ip_, 0.3_rp_, zero, 2_ip_, one )
   DO k = 1, 22
     control = base
     SELECT CASE ( k )
     CASE ( 1 ) ; control%factor = 1 ; lab = 'factor = 1 (Schur complement)'
     CASE ( 2 ) ; control%factor = 2 ; lab = 'factor = 2 (augmented system)'
     CASE ( 3 ) ; control%precon = 1 ; lab = 'precon = 1 (none)'
     CASE ( 4 ) ; control%precon = 2 ; lab = 'precon = 2 (full)'
     CASE ( 5 ) ; control%precon = 3 ; lab = 'precon = 3 (band)'
     CASE ( 6 ) ; control%precon = 3 ; control%nsemib = 0
                  lab = 'precon = 3, nsemib = 0'
     CASE ( 7 ) ; control%precon = 4 ; lab = 'precon = 4 (diagonal)'
     CASE ( 8 ) ; control%deletion_strategy = 1 ; lab = 'deletion_strategy = 1'
     CASE ( 9 ) ; control%deletion_strategy = 5 ; lab = 'deletion_strategy = 5'
     CASE ( 10 ) ; control%randomize = .FALSE. ; lab = 'randomize = .false.'
     CASE ( 11 ) ; control%max_sc = 1 ; lab = 'max_sc = 1'
     CASE ( 12 ) ; control%max_col = 1 ; lab = 'max_col = 1'
     CASE ( 13 ) ; control%inner_stop_relative = 0.5_rp_
                   lab = 'inner_stop_relative = 0.5'
     CASE ( 14 ) ; control%feas_tol = 1.0E-8_rp_ ; lab = 'feas_tol = 1e-8'
     CASE ( 15 ) ; control%treat_zero_bounds_as_general = .TRUE.
                   lab = 'treat_zero_bounds_as_general'
     CASE ( 16 ) ; control%full_max_fill = 1 ; lab = 'full_max_fill = 1'
     CASE ( 17 ) ; control%cg_maxit = 1 ; lab = 'cg_maxit = 1'
     CASE ( 18 ) ; control%infeas_check_interval = 1
                   lab = 'infeas_check_interval = 1'
     CASE ( 19 ) ; control%symmetric_linear_solver = 'sils'
                   lab = 'symmetric_linear_solver = sils'
     CASE ( 20 ) ; control%symmetric_linear_solver = 'sytr'
                   lab = 'symmetric_linear_solver = sytr'
     CASE ( 21 ) ; control%increase_rho_g_factor = 100.0_rp_
                   control%increase_rho_b_factor = 100.0_rp_
                   lab = 'increase_rho_g/b_factor = 100'
     CASE ( 22 ) ; control%multiplier_tol = 1.0E-6_rp_
                   lab = 'multiplier_tol = 1e-6'
     END SELECT
     CALL solve( qp, control, res )
     CALL check( qp, res, TRIM( lab ) )
   END DO
!  too few iterations
   control = base ; control%maxit = 1
   CALL solve( qp, control, res )
   CALL check_status( res, GALAHAD_error_max_iterations, 'maxit = 1' )
   END SUBROUTINE case_controls

!  =========================================================================
!  l_1 penalty problems (solve_qp = .false.) and larger problems
!  =========================================================================

   SUBROUTINE case_l1_and_large( )
   TYPE ( qp_type ) :: qp
   TYPE ( QPA_control_type ) :: control
   TYPE ( result_type ) :: res
   INTEGER ( KIND = ip_ ) :: trial, nbad
   LOGICAL :: ok
   CHARACTER ( LEN = 60 ) :: lab
   CALL set_controls( control )
!  with rho = 10 above every multiplier (all at most 2 here) the l_1 penalty
!  is exact, so the l_1 problem has the same solution as the QP
   control%solve_qp = .FALSE.
   nbad = 0
   DO trial = 1, 10
     CALL planted( qp, 10_ip_ + irand( 30_ip_ ), irand( 20_ip_ ), 0.2_rp_,     &
                   zero, 0_ip_, one )
     CALL solve( qp, control, res )
     WRITE( lab, "( 'l_1 problem trial ', I0 )" ) trial
     CALL check( qp, res, TRIM( lab ), ok )
     IF ( .NOT. ok ) nbad = nbad + 1
   END DO
   CALL report( nbad == 0,                                                     &
               'exact l_1 penalty (solve_qp = .false.): 10 problems' )
   control%solve_qp = .TRUE.
   nbad = 0
   DO trial = 1, 4
     CALL planted( qp, 300_ip_, 150_ip_ + 25_ip_ * trial, 0.2_rp_, 0.05_rp_,   &
                   3_ip_, one )
     CALL solve( qp, control, res )
     WRITE( lab, "( 'large problem trial ', I0 )" ) trial
     CALL check( qp, res, TRIM( lab ), ok )
     IF ( .NOT. ok ) nbad = nbad + 1
   END DO
   CALL report( nbad == 0,                                                     &
               'larger problems (n = 300, m = 175..250): 4 problems' )
   END SUBROUTINE case_l1_and_large

   END PROGRAM GALAHAD_QPA_edge_test_program
