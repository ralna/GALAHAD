! THIS VERSION: GALAHAD 5.6 - 2026-06-23 AT 13:30 GMT.

#include "galahad_modules.h"

  PROGRAM E04NQ_test

!  main program to test the NAG convex QP package E04NQF

!  Nick Gould, June 2026

  USE GALAHAD_KINDS_precision

  IMPLICIT NONE

!  Parameters

  INTEGER ( KIND = ip_ ), PARAMETER :: out = 6
! INTEGER ( KIND = ip_ ), PARAMETER :: spec = 29
  INTEGER ( KIND = ip_ ), PARAMETER :: len_c_w = 600
  INTEGER ( KIND = ip_ ), PARAMETER :: len_r_w = 600
  INTEGER ( KIND = ip_ ), PARAMETER :: len_i_w = 600
  INTEGER, PARAMETER :: n = 3, m = 2, neh = 4, nea = 4
  REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20

!  local variables

  INTEGER ( KIND = ip_ ) :: status, nname, lenc, ncolh, iobj
  INTEGER ( KIND = ip_ ) :: i, j, l, iter, ns, ninf, ifail
  REAL ( KIND = rp_ ) :: f, sinf, obj, res_p, res_d
! LOGICAL :: filexst
  CHARACTER ( LEN = 1 ) :: start
  CHARACTER ( LEN = 8 ) :: prob, c_dummy( 1 )
  CHARACTER ( LEN = 10 ) :: p_name = 'qptest    '
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: HELAST, HS, I_w, I_user
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr, A_row
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_ptr, H_row
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, X_0, X, X_l, X_u
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z, Y, C_l, C_u, C, G_l
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u, R_w, R_user
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, H_val
  CHARACTER ( LEN = 8 ), ALLOCATABLE, DIMENSION( : )  :: C_w, C_user

!  results summary output if required (set output_summary > 10) 

  INTEGER ( KIND = ip_ ) :: output_summary = 0
! INTEGER ( KIND = ip_ ) :: output_summary = 47
! CHARACTER ( LEN = 10 ) :: summary_filename = 'E04NQ.res'

  ALLOCATE( G( n ), X_l( n ), X_u( n ), STAT = status )
  ALLOCATE( C( m ), C_l( m ), C_u( m ), STAT = status )
  ALLOCATE( X_0( n ), Y( m ), Z( n ), STAT = status )
  ALLOCATE( H_val( neh ), H_row( neh ), H_ptr( n + 1 ), STAT = status )
  ALLOCATE( A_val( nea ), A_row( nea ), A_ptr( n + 1 ), STAT = status )

!  input the problem data per GALAHAD's standard QP format

  f = 1.0_rp_                              ! objective constant
  G = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]        ! objective gradient
  C_l = [ 1.0_rp_, 2.0_rp_ ]               ! constraint lower bound
  C_u = [ 2.0_rp_, 2.0_rp_ ]               ! constraint upper bound
  X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
  X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]     ! variable upper bound
  X_0 = 0.0_rp_; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
  H_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ] ! Hessian H, column storage
  H_row = [ 1, 2, 3, 3 ]                         ! NB lower triangle
  H_ptr = [ 1, 2, 4, 5 ] 
  A_val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ] ! Jacobian A, column storage
  A_row = [ 1, 1, 2, 2 ]
  A_ptr = [ 1, 2, 4, 5 ]

!  transfer data into the format required by E04NQ

  start = 'C'
  nname = 1
  lenc = n
  ncolh = n
  iobj = 0
  ns = 0
  prob = p_name( 1 : 8 )

!  manipulate vectors so that they conform to E04NQ's structures

  DEALLOCATE( Z, STAT = status )
  IF ( status /= 0 ) GO TO 990
  ALLOCATE( B_l( n + m ), B_u( n + m ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  B_l( : n ) = X_l( : n ) ; B_l( n + 1 : n + m ) = C_l( : m )
  B_u( : n ) = X_u( : n ) ; B_u( n + 1 : n + m ) = C_u( : m )
  DEALLOCATE( X_l, X_u, C_l, C_u, STAT = status )
  IF ( status /= 0 ) GO TO 990
  ALLOCATE( HELAST( n + m ), HS( n + m ), X( n + m ), Z( n + m ),              &
            STAT = status )
  IF ( status /= 0 ) GO TO 990
  HELAST( : n + m ) = 3 ; HS( : n + m ) = 0
  X( : n ) = X_0( : n ) ; X( n + 1 : n + m ) = 0.0_rp_
  DEALLOCATE( X_0, STAT = status )

!  record the sparse matrix H in E04NQ's user data structure

  ALLOCATE( C_w( len_c_w ), R_w( len_r_w ), I_w( len_i_w ) , STAT = status )
  IF ( status /= 0 ) GO TO 990
  ALLOCATE( C_user( 1 ), R_user( neh ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  R_user( : neh ) = H_val( : neh )
  DEALLOCATE( H_val, STAT = status )
  IF ( status /= 0 ) GO TO 990
  ALLOCATE( I_user( neh + n + 1 ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  I_user( : n + 1 ) = H_ptr( : n + 1 )
  I_user( n + 2 : neh + n + 1 ) = H_row( : neh )
  DEALLOCATE( H_ptr, H_row, STAT = status )
  IF ( status /= 0 ) GO TO 990

!  set up the internal structures

  ifail = - 1
  CALL E04NPF( C_w, len_c_w, I_w, len_i_w, R_w, len_r_w, ifail )
  SELECT CASE( ifail )
  CASE ( - 199 )
    WRITE( out, "( ' call to E04NQ failed, substitute dummy package called' )" )
    STOP
  CASE ( - 399 )
    WRITE( out, "( ' call to E04NQ failed, licence key expired' )" )
    STOP
  CASE ( - 999 )
    status = ifail ; GO TO 990
  END SELECT

!  read options file

! OPEN( spec, file = 'E04NQ.SPC', form = 'FORMATTED', status = 'OLD' )
! REWIND( spec )
! ifail = - 1
! CALL E04NRF( spec, C_w, I_w, R_w, ifail )
! IF ( ifail /= 0 ) GO TO 910
! CLOSE( spec )

!  solve the problem

  ifail = - 1
  CALL E04NQF( start, E04NQ_qphx, m, n, nea, nname, lenc, ncolh, iobj, f,     &
               prob, A_val, A_row, A_ptr, B_l, B_u, G, c_dummy, HELAST, HS,    &
               X, Y, Z, ns, ninf, sinf, obj, C_w, len_c_w, I_w, len_i_w,       &
               R_w, len_r_w, C_user, I_user, R_user, ifail )

!  write details

! WRITE( out, "(' Final objective value = ', ES11.3 )" ) obj
! WRITE( out, "(' Optimal X = ', 7F9.2 )" ) X( : n )

  WRITE( out, "( /, 24('*'), ' GALAHAD statistics ', 24('*') //                &
 &              ,' Package used            :  E04NQF',   /                     &
 &              ,' Problem                 :  ', A10,    /                     &
 &              ,' # variables             =      ', I10 /                     &
 &              ,' # constraints           =      ', I10 /                     &
 &              ,' Exit code               =      ', I10 /                     &
 &              ,' Final f                 = ', ES15.7 //                      &
 &               67('*') / )" ) p_name, n, m, ifail, obj

!  compute the primal and dual residuals if necessary

  IF ( output_summary > 10 ) THEN
    ALLOCATE( C( m ), G_l( n ), STAT = status )
    CALL E04NQ_QPHX( n, X, G_l, 0, C_user, I_user, R_user )
    G_l( : n ) = G_l( : n ) + G( : n ) - Z( : n )
    C( : m ) = 0.0_rp_
    DO j = 1, n
      DO l = A_ptr( j ), A_ptr( j + 1 ) - 1
        i = A_row( l )
        G_l( j ) = G_l( j ) - A_val( l ) * Y( i )
        C( i ) = C( i ) + A_val( l ) * X( j )
      END DO
    END DO
    C( : m ) = MIN( B_u( n + 1 : n + m ),                                      &
                    MAX( B_l( n + 1 : n + m ), C( : m ) ) ) - C( : m )
    res_p = MAXVAL( ABS( C( : m ) ) )
    res_d = MAXVAL( ABS( G_l( : n ) ) )
    DEALLOCATE( G_l, C, STAT = status )

!  output a summary of the results to a file if required

    BACKSPACE( output_summary )
    iter = 0 ! not available from e04nqf apparently!
    SELECT CASE ( ifail )
    CASE ( 0, 3, 4 )
      WRITE( out,                                                              &
        "( 1X, I8, 1X, I8, ES16.8, 2ES9.1, bn, I9, I6 )" )                     &
       n, m, obj, res_p, res_d, iter, ifail
    CASE DEFAULT
      WRITE( 6,                                                                &
        "( 1X, I8, 1X, I8, ES16.8, 2ES9.1, bn, I9, I6 )" )                     &
        n, m, obj, res_p, res_d, - iter, ifail
    END SELECT
  END IF

!  deallocate workspace

  DEALLOCATE( A_val, A_row, A_ptr, B_l, B_u, G, HELAST, HS, X, Y, Z,           &
               C_w, I_w, R_w, C_user, I_user, R_user, STAT = status )

  STOP

! 910 CONTINUE
! WRITE( out, "( ' call to E04NQ failed' )" )
! WRITE( out, "( ' GALAHAD error, ifail = ', i0, ', stopping' )") ifail
! STOP

  990 CONTINUE
  WRITE( out, "( ' Allocation error, status = ', i0 )" ) status
  STOP

  CONTAINS

    SUBROUTINE E04NQ_QPHX( ncolh, X, HX, nstate, C_user, I_user, R_user )
    USE GALAHAD_KINDS_precision

!  given x, compute hx = H*x

!  dummy arguments

    INTEGER ( KIND = ip_ ), INTENT ( IN ) :: ncolh, nstate
    INTEGER ( KIND = ip_ ), INTENT ( INOUT ) :: I_user( * )
    REAL ( KIND = rp_ ), INTENT( IN ) :: X( ncolh )
    REAL ( KIND = rp_ ), INTENT( INOUT ) :: R_user( * )
    REAL ( KIND = rp_ ), INTENT( OUT ) :: HX( ncolh )
    CHARACTER ( len = 8 ), INTENT( INOUT ) :: C_user( * )

!  local variables

    INTEGER ( KIND = ip_ ) :: i, j, l, n_row

!  initialize 

    n_row = ncolh + 1
    HX = 0.0_rp_

!  loop over the columns of H, remembering that only one triangle of H is stored

    DO j = 1, ncolh
      DO l = I_user( j ), I_user( j + 1 ) - 1 
        i = I_user( n_row + l )
        HX( i ) = HX( i ) + R_user( l ) * X( j )
        IF ( i /= j ) HX( j ) = HX( j ) + R_user( l ) * X( i )
      END DO
    END DO

    RETURN

!  end of subroutine E04NQ_QPHX

    END SUBROUTINE E04NQ_QPHX

  END PROGRAM E04NQ_test

