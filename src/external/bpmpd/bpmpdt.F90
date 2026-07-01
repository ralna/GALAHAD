! THIS VERSION: GALAHAD 5.6 - 2026-06-27 AT 11:00 GMT.

#include "galahad_modules.h"

  PROGRAM BPMPD_test

!  main program to test the convex QP package BPMPD

!  Nick Gould, June 2026

  USE GALAHAD_KINDS_precision

  IMPLICIT NONE

!  Parameters

  INTEGER ( KIND = ip_ ), PARAMETER :: out = 6
  INTEGER, PARAMETER :: n = 3, m = 2, nz = 4, qn = n, qnz = 4
  REAL ( KIND = rp_ ), PARAMETER :: infinity = 10.0_rp_ ** 20

!  local variables

  INTEGER ( KIND = ip_ ) :: alloc_status, code, msizi, msizr, iter, loglevel
  REAL ( KIND = rp_ ) :: f, big, opt
  CHARACTER ( LEN = 10 ) :: p_name = 'qptest    '
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ACOLCNT, ACOLIDX
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: QCOLCNT, QCOLIDX
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: STATUS
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: ACOLNZS, QCOLNZS
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RHS, OBJ
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: LBOUND, UBOUND
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: PRIMAL, DUAL

  ALLOCATE( ACOLCNT( n ), ACOLIDX( nz ), QCOLCNT( n ), QCOLIDX( qnz ),         &
            STATUS( n + m ), ACOLNZS( nz ), QCOLNZS( qnz ),                    &
            RHS( m ), OBJ( n ), LBOUND( n + m ), UBOUND( n + m ),              &
            PRIMAL( n + m ), DUAL( n + m ), STAT = alloc_status )
  IF ( alloc_status /= 0 ) GO TO 990

!  input the problem data

  f = 1.0_rp_                              ! objective constant
  OBJ = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]      ! objective gradient
  RHS = 0.0_rp_                            ! constraint rhs
  LBOUND = [ - 1.0_rp_, - infinity, - infinity,                                &
             1.0_rp_, 2.0_rp_]             ! variable and slack lower bound
  UBOUND = [ 1.0_rp_, infinity, 2.0_rp_,                                       &
             2.0_rp_, 2.0_rp_]             ! variable and slack upper bound
  ACOLCNT = [ 1, 2, 1 ]               ! Jacobian A, row storage
  ACOLIDX = [ 1, 1, 2, 2 ]
  ACOLNZS = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ]
  QCOLCNT = [ 1, 2, 1 ]              ! Hessian H, row storage,
  QCOLIDX = [ 1, 2, 3, 3 ]         ! lower triangle only
  QCOLNZS = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ] 
  PRIMAL = 0.0_rp_ ; DUAL = 0.0_rp_       ! start from zero

  msizi = 0 ; msizr = 0 ! make bpmpd choose appropriate sizes for workspace
  big = infinity ! set infinity values for bound constraints

!  solve the problem

  loglevel = 0 ! turn off the log file and output generation
  CALL bpmpdx( m, n, nz, qn, qnz, ACOLCNT, ACOLIDX, ACOLNZS,                   &
               QCOLCNT, QCOLIDX, QCOLNZS, RHS, OBJ, LBOUND, UBOUND,            &
               PRIMAL, DUAL, STATUS, big, code, opt, msizi, msizr,             &
               f, loglevel, iter )

!  write details

! WRITE( out, "(' Final objective value = ', ES11.3 )" ) opt + f
! WRITE( out, "(' Optimal X = ', 7F9.2 )" ) PRIMAL( : n )

  WRITE( out, "( /, 24('*'), ' GALAHAD statistics ', 24('*') //                &
 &              ,' Package used            :  BPMPD',   /                      &
 &              ,' Problem                 :  ', A10,    /                     &
 &              ,' # variables             =      ', I10 /                     &
 &              ,' # constraints           =      ', I10 /                     &
 &              ,' Exit code               =      ', I10 )" ) p_name, n, m, code
  IF ( code == 2 )                                                             &
    WRITE( out, "( ' Final f                 = ', ES15.7, /                    &
 &              ,' iterations              =      ', I10  )" ) opt, iter
  WRITE( out, "( /, 67('*') / )" ) 

!  deallocate workspace

  DEALLOCATE( ACOLCNT, ACOLIDX, QCOLCNT, QCOLIDX, STATUS, ACOLNZS, QCOLNZS,    &
              RHS, OBJ, LBOUND, UBOUND, PRIMAL, DUAL, STAT = alloc_status )

  STOP

  990 CONTINUE
  WRITE( out, "( ' Allocation error, status = ', I0 )" ) alloc_status
  STOP

  END PROGRAM BPMPD_test

