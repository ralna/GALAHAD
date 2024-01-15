! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.
#include "galahad_modules.h"
   PROGRAM GALAHAD_FDH_test_deck
   USE GALAHAD_KINDS_precision
   USE GALAHAD_FDH_precision
   USE GALAHAD_SYMBOLS
   IMPLICIT NONE
   TYPE ( FDH_control_type ) :: control
   TYPE ( FDH_inform_type ) :: inform
   TYPE ( FDH_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTERFACE
     SUBROUTINE GRAD( status, X, userdata, G )
     USE GALAHAD_USERDATA_precision
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     END SUBROUTINE GRAD
   END INTERFACE

   INTERFACE
     SUBROUTINE WRP( data, products ,out )
     USE GALAHAD_KINDS_precision
     USE GALAHAD_FDH_precision, ONLY: FDH_data_type
     TYPE ( FDH_data_type ) :: data
     INTEGER ( KIND = ip_ ) :: out, products
     END SUBROUTINE WRP
   END INTERFACE

   INTERFACE
     SUBROUTINE WRH( H, H_true, out )
     USE GALAHAD_KINDS_precision
     INTEGER ( KIND = ip_ ) :: out
     REAL ( KIND = rp_ ), DIMENSION( : ) :: H, H_true
     END SUBROUTINE WRH
   END INTERFACE

   INTEGER ( KIND = ip_ ) :: i, j, nz, n, out, status
   INTEGER ( KIND = ip_ ), PARAMETER :: n_true = 6, nz_true = 11
   LOGICAL :: ok, ok_overall
   REAL ( KIND = rp_ ) :: epsqrt
   INTEGER ( KIND = ip_ ) :: DIAG( n_true ), ROW( nz_true )
   REAL ( KIND = rp_ ) :: H_true( nz_true ), H( nz_true ), G( n_true )
   REAL ( KIND = rp_ ) :: X( n_true ), STEPSIZE( n_true )
   REAL ( KIND = rp_ ) :: epsmch = EPSILON( 1.0_rp_ )
   epsqrt = SQRT( epsmch )
   out = 6
   ok_overall = .TRUE.

!  set up space for internal parameters

    ALLOCATE( userdata%real( nz_true ) )
    ALLOCATE( userdata%integer( 2 + n_true + nz_true ) )

!  set up an initial structure

   n = n_true ; nz = nz_true
   DIAG( : n ) = (/ 1, 3, 5, 7, 9, 11 /)
   ROW( : nz ) = (/ 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6 /)

!  store the structure to generate an appropriate gradient

   userdata%integer( 1 ) = n
   userdata%integer( 2 ) = nz
   userdata%integer( 2 + 1 : 2 + n ) = DIAG( : n )
   userdata%integer( 2 + n + 1 : 2 + n + nz ) = ROW( : nz )

!  also set up an estimation point and step lengths

   X( : n ) = (/ 3.0_rp_, -4.0_rp_, 0.0_rp_, 5.0_rp_, 9.0_rp_, -2.0_rp_ /)
   STEPSIZE( : n ) = epsqrt

!  set up a true Hessian matrix

   H_true( : nz ) = (/ 2.0_rp_, 1.0_rp_, 1.0_rp_, 3.0_rp_, -3.0_rp_, 3.0_rp_,  &
                       -1.0_rp_, -1.0_rp_, 2.0_rp_, 4.0_rp_, -5.0_rp_ /)
   userdata%real( : nz_true ) = H_true( : nz )

   CALL FDH_initialize( data, control, inform )

!  build the corresponding gradient

   CALL GRAD( status, X( : n ), userdata, G( : n ) )

!  try to call FDH_estimate before FDH_analyse

   WRITE ( out,                                                                &
     "(/, ' error status ', I0, ' expected from FDH_estimate')" )              &
        GALAHAD_error_call_order

   CALL FDH_estimate( n, nz, ROW, DIAG, X, G, STEPSIZE, H,                     &
                      data, control, inform, userdata, eval_G = GRAD )
   IF ( inform%status /= 0 )                                                   &
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status

!  perturb the pattern

   n = - 3 ; nz = 11 ; ROW( 7 ) = 5

!  try to call FDH_analyse with incorrect pattern specifications

   WRITE (out,                                                                 &
     "(/, ' error status ', I0, ' expected from FDH_analyse')" )               &
        GALAHAD_error_restrictions
   CALL FDH_analyse( n, nz, ROW, DIAG, data, control, inform )
   IF ( inform%status /= 0 )                                                   &
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status

   n = 6 ; nz = 4
   WRITE (out,                                                                 &
     "(/, ' error status ', I0, ' expected from FDH_analyse')" )               &
        GALAHAD_error_restrictions
   CALL FDH_analyse( n, nz, ROW, DIAG, data, control, inform )
   IF ( inform%status /= 0 )                                                   &
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status

   nz = 11
   WRITE (out,                                                                 &
     "(/, ' error status ', I0, ' expected from FDH_analyse')" )               &
        GALAHAD_error_upper_entry
   CALL FDH_analyse( n, nz, ROW, DIAG, data, control, inform )
   IF ( inform%status /= 0 )                                                   &
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status
   ROW( 7 ) = 4

!  compute groups for the trdiagonal structure

   CALL FDH_analyse( n, nz, ROW, DIAG, data, control, inform )

!  check the analysis

   ok = inform%status == 0 .OR. ( inform%products == 0 .AND.                   &
          inform%status == GALAHAD_error_restrictions )

   DO i = 1, nz
     IF ( data%ROW_perm( i ) /= ROW( i ) ) ok = .FALSE.
     IF ( data%OLD( i ) /= i ) ok = .FALSE.
   END DO
   data%DIAG_perm( 1 ) = - data%DIAG_perm( 1 )
   DO I = 1, n
     IF ( data%PERM( i ) /= i ) ok = .FALSE.
     IF ( data%DIAG_perm( i ) /= DIAG( i ) ) ok = .FALSE.
     j = MOD( i, 2 )
     IF ( j == 0 ) j = 2
     IF ( data%GROUP( i )  /= j ) ok = .FALSE.
   END DO
   IF ( .NOT. ok ) THEN
     ok_overall = .FALSE.
     WRITE( out,                                                               &
       "(/, '*** error from FDH_analysis in the tridiagonal case' )" )
     CALL WRP( data, inform%products, out )
   END IF
   data%DIAG_perm( 1 ) = - data%DIAG_perm( 1 )

!  try to alter the dimensions and call FDH_estimate

   nz = 4
   WRITE (out,                                                                 &
     "(/, ' error status ', I0, ' expected from FDH_estimate')" )              &
        GALAHAD_error_restrictions
   CALL FDH_estimate( n, nz, ROW, DIAG, X, G, STEPSIZE, H,                     &
                      data, control, inform, userdata, eval_G = GRAD )
   IF ( inform%status /= 0 )                                                   &
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status

!  correct the dimension and obtain an approximation

   nz = 11
   CALL FDH_estimate( n, nz, ROW, DIAG, X, G, STEPSIZE, H,                     &
                      data, control, inform, userdata, eval_G = GRAD )

!  check the approximate Hessian

   ok = inform%status == 0
   DO i = 1, nz
     IF ( ABS( H( i ) - H_true( i ) ) > 100.0_rp_ * epsqrt ) ok = .FALSE.
   END DO
   IF ( .NOT. ok ) THEN
     ok_overall = .FALSE.
     WRITE( out, "( /, ' *** error wrong trdiagonal estimate' )" )
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status
     CALL WRH( H( : nz ), H_true( : nz ), out )
   END IF

!  now consider a more complicated pattern and call FDH_analyse

   DIAG( : n ) = (/ 1, 4, 6, 8, 10, 11 /)
   ROW( : nz ) = (/ 1, 3, 4, 2, 4, 3, 5, 4, 5, 5, 6 /)
   userdata%integer( 1 ) = n_true
   userdata%integer( 2 ) = nz_true
   userdata%integer( 2 + 1 : 2 + n_true ) = DIAG( : n )
   userdata%integer( 2 + n_true + 1 : 2 + n_true + nz_true ) = ROW( : nz )

   CALL FDH_analyse( n, nz, ROW, DIAG, data, control, inform )

!  check the analysis

   ok = inform%status == 0 .AND. inform%products == 3

   IF ( COUNT( data%GROUP( : n ) /=                                            &
                 (/ 1, 2, 3, 1, 1, 1 /) ) > 0 ) ok = .FALSE.
   IF ( COUNT( data%PERM( : n ) /=                                             &
          (/ 1, 3, 4, 5, 2, 6 /) ) > 0 ) ok = .FALSE.
   IF ( COUNT( data%DIAG_perm( : n ) /=                                        &
                 (/ 1, 4, 6, 9, 10, 11 /) ) > 0 ) ok = .FALSE.
   IF ( COUNT( data%ROW_perm( : nz ) /=                                        &
                 (/ 1, 2, 3, 2, 4, 5, 3, 4, 4, 5, 6 /) ) > 0 ) ok = .FALSE.
!                (/ 1, 2, 3, 2, 4, 4, 5, 3, 4, 5, 6 /) ) > 0 ) ok = .FALSE.
   IF ( COUNT( data%OLD( : nz ) /=                                             &
                 (/ 1, 2, 3, 6, 7, 5, 8, 9, 10, 4, 11 /) ) > 0 ) ok = .FALSE.
!                (/ 1, 2, 3, 6, 7, 9, 5, 8, 10, 4, 11 /) ) > 0 ) ok = .FALSE.

   IF ( .NOT. ok ) THEN
     ok_overall = .FALSE.
     WRITE( out, "(/, ' *** error wrong analysis for the general case' )" )
     WRITE( out, "( ' *** error status = ', I0 )" ) inform%status
     CALL WRP( data, inform%products, out )
   END IF

!   obtain an aproximation to the Hessian

   CALL GRAD( status, X( : n ), userdata, G( : n ) )
   CALL FDH_estimate( n, nz, ROW, DIAG, X, G, STEPSIZE, H,                     &
                      data, control, inform, userdata, eval_G = GRAD )

!  check the approximate Hessian

   ok = inform%status == 0
   DO i = 1, nz
     IF ( ABS( H( i ) - H_true( i ) ) > epsmch ** 0.25 ) ok = .FALSE.
   END DO
   IF ( .NOT. ok ) THEN
     ok_overall = .FALSE.
     WRITE ( out, "( /, '*** error  wrong random estimation' )" )
     CALL WRH( H( : nz ), H_true( : nz ), out )
   END IF
   IF ( ok_overall ) THEN
     WRITE( out, "( /, ' remaing tests of package FDH successful' )" )
   ELSE
     WRITE( out, "( /, ' unsuccessful test of package FDH' )" )
   END IF
   END PROGRAM GALAHAD_FDH_test_deck

! internal subroutine to evaluate the gradient of the objective
   SUBROUTINE GRAD( status, X, userdata, G )
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
   REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
   TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
   INTEGER ( KIND = ip_ ) :: i, ic, ir, n, nz, ldiag, lrow
   REAL ( KIND = rp_ ) :: hi
   n = userdata%integer( 1 ) ; nz = userdata%integer( 2 )
   ldiag = 2 ; lrow =  2 + n
   G( : n ) = 0.0_rp_
   ic = 1
   DO i = 1, nz
     ir = userdata%integer( lrow + i )
     IF ( i > 1 .AND. ic < n ) THEN
       IF ( i == userdata%integer( ldiag + ic + 1 ) ) ic = ic + 1
     END IF
     hi =  userdata%real( i )
     G( ir ) = G( ir ) + hi * X( ic )
     IF ( ir /= ic ) G( ic ) = G( ic ) + hi * X( ir )
   END DO
   status = 0
   RETURN
   END SUBROUTINE GRAD

   SUBROUTINE WRP( data, products ,out )
   USE GALAHAD_KINDS_precision
   USE GALAHAD_FDH_precision, ONLY: FDH_data_type
   TYPE ( FDH_data_type ) :: data
   INTEGER ( KIND = ip_ ) :: out, products
   WRITE( out, "( /, 5X, 'products  = ', I3 )" ) products
   WRITE( out, "( 5X, 'ROW_perm  = ', 11I3 )" ) data%ROW_perm( 1 : data%nz )
   WRITE( out, "( 5X, 'DIAG_perm = ', 6I3 )" ) data%DIAG_perm( 1 : data%n )
   WRITE( out, "( 5X, 'GROUP     = ', 6I3 )" ) data%GROUP( 1 : data%n )
   WRITE( out, "( 5X, 'PERM      = ', 6I3 )" ) data%PERM( 1 : data%n )
   WRITE( out, "( 5X, 'OLD       = ', 11I3, / )" ) data%OLD( 1 : data%nz )
   RETURN
   END SUBROUTINE WRP

   SUBROUTINE WRH( H, H_true, out )
   USE GALAHAD_KINDS_precision
   INTEGER ( KIND = ip_ ) :: out
   REAL ( KIND = rp_ ), DIMENSION( : ) :: H, H_true
   INTEGER ( KIND = ip_ ) :: l
   WRITE( out, "( /, 5X, 'True Hessian   = ', 5ES15.7 )" )                     &
      ( H_true( l ), l = 1, 11 )
   WRITE( out, "( 5X, 'approx Hessian = ', 5ES15.7 )" )                        &
      ( H( l ), l = 1, 11 )
   RETURN
   END SUBROUTINE WRH
