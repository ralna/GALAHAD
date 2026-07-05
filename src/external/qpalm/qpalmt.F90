! THIS VERSION: 25/04/2022 AT 13:45 GMT
! Nick Gould (nick.gould@stfc.ac.uk)

#include "galahad_modules.h"

PROGRAM QPALM_example

   USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
   USE QPALM_fiface

!  local variables

   INTEGER ( KIND = ip_ ) :: i, j, k, l, n_bnds, m_qp, a_ne_qp, status
   REAL ( KIND = rp_ ) :: f_calc
!  REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
   REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 20

!  problem parameters

   INTEGER ( KIND = ip_ ), PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4
!  INTEGER ( KIND = ip_ ), PARAMETER :: n = 2, m = 1, h_ne = 2, a_ne = 2
   INTEGER ( KIND = ip_ ), PARAMETER :: ane = 4
   INTEGER ( KIND = ip_ ), PARAMETER :: m_bnds = 3

!  set problem and solution arrays

   REAL ( KIND = rp_ ) :: f
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr, A_ptr_qp
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_row_qp
   INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_ptr, H_row
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, X, X_l, X_u
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u, C
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u
   REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, A_val_qp, H_val
   TYPE ( QPALM_settings ) :: settings
   TYPE ( QPALM_info ) :: info

!  input the problem data per GALAHAD's standard QP format

   ALLOCATE( G( n ), X_l( n ), X_u( n ), STAT = status )
   ALLOCATE( C( m ), C_l( m ), C_u( m ), STAT = status )
   ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_ptr( n + 1 ), STAT = status )
   ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_ptr( n + 1 ), STAT = status )

!IF ( .false. ) THEN
IF ( .true. ) THEN
   f = 1.0_rp_                              ! objective constant
   G = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]        ! objective gradient
   H_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ] ! Hessian H, column storage
   H_row = [ 1, 2, 2, 3 ]                         ! NB upper triangle
   H_ptr = [ 1, 2, 3, 5 ] 
   C_l = [ 1.0_rp_, 2.0_rp_ ]                  ! constraint lower bound
   C_u = [ 2.0_rp_, 2.0_rp_ ]                  ! constraint upper bound
   X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
   X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]        ! variable upper bound
   A_val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ] ! Jacobian A, column storage
   A_row = [ 1, 1, 2, 2 ]
   A_ptr = [ 1, 2, 4, 5 ]
 ELSE
   f = -100.0_rp_                              ! objective constant
   G = [ 0.0_rp_, 0.0_rp_ ]        ! objective gradient
   H_val = [ 0.02_rp_, 2.0_rp_ ] ! Hessian H, column storage
   H_row = [ 1, 2 ]                         ! NB upper triangle
   H_ptr = [ 1, 2, 3 ] 
   C_l = [ 10.0_rp_ ]                  ! constraint lower bound
   C_u = [ infinity ]                  ! constraint upper bound
   X_l = [ 2.0_rp_, - 50.0_rp_ ] ! variable lower bound
   X_u = [ 50.0_rp_, 50.0_rp_ ]        ! variable upper bound
   A_val = [ 10.0_rp_, -1.0_rp_ ] ! Jacobian A, column storage
   A_row = [ 1, 1 ]
   A_ptr = [ 1, 2, 3 ]
 END IF

!  transfer the data into QPALM's QP format

   n_bnds = COUNT( X_l > -infinity .OR. X_u < infinity )
   m_qp = m + n_bnds
   a_ne_qp = a_ne + n_bnds
   ALLOCATE( A_val_qp( a_ne_qp ), A_row_qp( a_ne_qp ), STAT = status )
   ALLOCATE( A_ptr_qp( n + 1 ), STAT = status )
   ALLOCATE( B_l( m_qp ), B_u( m_qp ), X( n ), Y( m_qp ), STAT = status )

!  reset problem constraint data (NB 1-based integer index arrays)

   B_l( : m ) = C_l( : m ) ; B_u( : m ) = C_u( : m )
   l = 1 ; k = m
   DO j = 1, n
     A_ptr_qp( j ) = l
     DO i = A_ptr( j ),  A_ptr( j + 1 ) - 1
       A_row_qp( l ) = A_row( i )
       A_val_qp( l ) = A_val( i )
       l = l + 1
     END DO
     IF ( X_l( j ) > -infinity .OR. X_u( j ) < infinity ) THEN
       k = k + 1
       A_row_qp( l ) = k
       A_val_qp( l ) = 1.0_rp_
       l = l + 1
       B_l( k ) = X_l( j ) ; B_u( k ) = X_u( j )
     END IF
   END DO
   A_ptr_qp( n + 1 ) = l
   DEALLOCATE( X_l, X_u, C_l, C_u, A_ptr, A_row, A_val, STAT = status )

!  assign non-default settings prior to solution

   settings%verbose = 0
!  settings%verbose = 1
  settings%eps_abs = 0.00000000001
  settings%eps_rel = 0.00000000001
! settings%scaling = 0

!  solve the problem

   CALL qpalm_fortran( n, m_qp, h_ne, H_ptr, H_row, H_val, g, f,               &
                       a_ne_qp, A_ptr_qp, A_row_qp, A_val_qp,                  &
                       B_l, B_u, settings, X, Y, info )

!  report the solution

   IF ( info%status_val == 1 ) THEN
      WRITE( 6, "( ' problem solved in ', I0, ' iterations, objective =',      &
      &  ES18.10 )" ) info%iter, info%objective
      WRITE( 6, "( ' X =', ( 5ES13.5 ) )" ) x
      WRITE( 6, "( ' Y =', ( 5ES13.5 ) )" ) y

   f_calc = f
   do j = 1, n
    f_calc = f_calc + G( j ) * X( j )
    do i = H_ptr( j ), H_ptr( j + 1 ) - 1
     IF ( H_row( i ) /= i ) THEN
       f_calc = f_calc + H_val( i ) * X( H_row( i ) ) * X( j )
     ELSE
       f_calc = f_calc + 0.5_rp_ * H_val( i ) * X( H_row( i ) ) * X( j )
     END IF
    end do
   end do
   write(6,*) ' f ', f_calc

   ELSE
      WRITE( 6, "( ' error return with status = ', I0 )" ) info%status_val
   END IF

   DEALLOCATE( X, Y, B_l, B_u, H_ptr, H_row, H_val, G, STAT = status )
   DEALLOCATE( A_ptr_qp, A_row_qp, A_val_qp, STAT = status )

END PROGRAM QPALM_example
