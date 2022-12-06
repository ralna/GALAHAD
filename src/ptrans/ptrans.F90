! THIS VERSION: GALAHAD 2.5 - 13/02/2013 AT 08:15 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ P T R A N S   M O D U L E  *-*-*-*-*-*-*-*-*-*

!  Suppose that x_trans = X_scale^-1 ( x - x_shift )
!               f_trans( x_trans ) = f_scale^-1 ( f( x ) - f_shift )
!          and  c_trans( x_trans ) = C_scale^-1 ( c( x ) - c_shift )
!  where f and c are available from calls to CUTEst subroutines.
!  Compute derivatives of f_trans( x_trans ) and c_trans( x_trans )

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 4th 2004

   MODULE GALAHAD_PTRANS_double

     USE GALAHAD_SPACE_double
     USE GALAHAD_TRANS_double, only :                                          &
       PTRANS_trans_type => TRANS_trans_type,                                  &
       PTRANS_data_type => TRANS_data_type,                                    &
       PTRANS_inform_type => TRANS_inform_type,                                &
       PTRANS_initialize => TRANS_initialize,                                  &
       PTRANS_terminate => TRANS_terminate,                                    &
       PTRANS_default => TRANS_default,                                        &
       PTRANS_trans => TRANS_trans,                                            &
       PTRANS_untrans => TRANS_untrans,                                        &
       PTRANS_s_trans => TRANS_s_trans,                                        &
       PTRANS_s_untrans => TRANS_s_untrans,                                    &
       PTRANS_v_trans => TRANS_v_trans,                                        &
       PTRANS_v_untrans => TRANS_v_untrans
     USE CUTEST_INTERFACE_double

     IMPLICIT NONE

     PRIVATE

     PUBLIC :: PTRANS_trans_type
     PUBLIC :: PTRANS_data_type
     PUBLIC :: PTRANS_inform_type
     PUBLIC :: PTRANS_initialize
     PUBLIC :: PTRANS_terminate
     PUBLIC :: PTRANS_default
     PUBLIC :: PTRANS_trans
     PUBLIC :: PTRANS_untrans
     PUBLIC :: PTRANS_s_trans
     PUBLIC :: PTRANS_s_untrans
     PUBLIC :: PTRANS_v_trans
     PUBLIC :: PTRANS_v_untrans

     PUBLIC :: PTRANS_ufn
     PUBLIC :: PTRANS_ugr
     PUBLIC :: PTRANS_uofg
     PUBLIC :: PTRANS_udh
     PUBLIC :: PTRANS_ugrdh
     PUBLIC :: PTRANS_ush
     PUBLIC :: PTRANS_ueh
     PUBLIC :: PTRANS_ugrsh
     PUBLIC :: PTRANS_ugreh
     PUBLIC :: PTRANS_uprod
     PUBLIC :: PTRANS_ubandh
     PUBLIC :: PTRANS_cfn
     PUBLIC :: PTRANS_cgr
     PUBLIC :: PTRANS_cofg
     PUBLIC :: PTRANS_csgr
     PUBLIC :: PTRANS_ccfg
     PUBLIC :: PTRANS_ccfsg
     PUBLIC :: PTRANS_cscfg
     PUBLIC :: PTRANS_ccifg
     PUBLIC :: PTRANS_ccifsg
     PUBLIC :: PTRANS_cscifg
     PUBLIC :: PTRANS_cdh
     PUBLIC :: PTRANS_cidh
     PUBLIC :: PTRANS_cgrdh
     PUBLIC :: PTRANS_csh
     PUBLIC :: PTRANS_cish
     PUBLIC :: PTRANS_ceh
     PUBLIC :: PTRANS_csgrsh
     PUBLIC :: PTRANS_csgreh
     PUBLIC :: PTRANS_cprod

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

  CONTAINS

!  --------------------------------------------
!  Interfaces to the unconstrained CUTEst tools
!  --------------------------------------------

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ufn  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_ufn( n, X, f, trans, data, inform )
     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( OUT ) :: f
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                              inform%alloc_status, exact_size = .TRUE.,        &
                              array_name = array_name,                         &
                              bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate f

     CALL CUTEST_ufn( inform%status, n, data%X_orig( : n ), f )
     IF ( inform%status /= 0 ) RETURN

!  Transform f

     CALL PTRANS_trans( n, 0, trans, 0.0_wp, f = f )

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ufn

     END SUBROUTINE PTRANS_ufn

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ugr  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_ugr( n, X, G, trans, data, inform )
     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                              inform%alloc_status, exact_size = .TRUE.,        &
                              array_name = array_name,                         &
                              bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate g

     CALL CUTEST_ugr( inform%status, n, data%X_orig( : n ), G )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradient

     G = ( trans%X_scale( : n ) / trans%f_scale ) * G

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN
     END SUBROUTINE PTRANS_ugr

     SUBROUTINE PTRANS_uofg( n, X, f, G, grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( OUT ) :: f
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     LOGICAL, INTENT( IN ) :: grad
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                  &
                               inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                      &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate f and psssibly g

     CALL CUTEST_uofg( inform%status, n, data%X_orig, f, G, grad )
     IF ( inform%status /= 0 ) RETURN

!  Transform f

     CALL PTRANS_trans( n, 0, trans, 0.0_wp, f = f )

!  Possibly transform the gradient

     IF ( grad ) G = ( trans%X_scale( : n ) / trans%f_scale ) * G

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_uofg

     END SUBROUTINE PTRANS_uofg

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_udh  S U B R O U T I N E  -*-*-*-*-*-*-*

     SUBROUTINE PTRANS_udh( n, X, lh1, H, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, lh1
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate H

     CALL CUTEST_udh( inform%status, n, data%X_orig( : n ), lh1, H )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     DO j = 1, n
       DO i = 1, n
         H( i, j ) = ( trans%X_scale( i ) * trans%X_scale( j ) /               &
                       trans%f_scale ) * H( i, j )
       END DO
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_udh

     END SUBROUTINE PTRANS_udh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_ugrdh  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_ugrdh( n, X, G, lh1, H, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, lh1
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate g and H

     CALL CUTEST_ugrdh( inform%status, n, data%X_orig( : n ), G, lh1, H )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradient

     G = ( trans%X_scale( : n ) / trans%f_scale ) * G

!  Transform the Hessian

     DO j = 1, n
       DO i = 1, n
         H( i, j ) = ( trans%X_scale( i ) * trans%X_scale( j ) /               &
                       trans%f_scale ) * H( i, j )
       END DO
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ugrdh

     END SUBROUTINE PTRANS_ugrdh

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ush  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_ush( n, X, nnzh, lh, H, IRNH, ICNH, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, lh
     INTEGER, INTENT( OUT ) :: nnzh
     INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate H

     CALL CUTEST_ush( inform%status, n, data%X_orig( : n ),                    &
                      nnzh, lh, H, IRNH, ICNH )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     DO l = 1, nnzh
       i = IRNH( l ) ; j = ICNH( l )
       H( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /                    &
                  trans%f_scale ) * H( l )
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ush

     END SUBROUTINE PTRANS_ush

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_ueh  S U B R O U T I N E    -*-*-*-*-*-*

     SUBROUTINE PTRANS_ueh( n, X, ne, IRNHI, lirnhi, le, IPRNHI, HI, lhi,      &
                            IPRHI, byrows, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, le, lirnhi, lhi
     INTEGER, INTENT( INOUT ) :: ne
     LOGICAL, INTENT( IN ) :: byrows
     INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
     INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, ie, ii, is, j, jj, l, ni
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate H

     CALL CUTEST_ueh( inform%status, n, data%X_orig( : n ), ne, le, IPRNHI,    &
                      IPRHI, lirnhi, IRNHI, lhi, HI, byrows )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     l = 1
     DO ie = 1, ne
       ni = IPRNHI( ie + 1 ) - IPRNHI( ie )
       is = IPRNHI( ie ) - 1
       IF ( byrows ) THEN
         DO ii = 1, ni
           i = IRNHI( is + ii )
           DO jj = i, ni
             j = IRNHI( is + jj )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       ELSE
         DO jj = 1, ni
           j = IRNHI( is + jj )
           DO ii = 1, jj
             i = IRNHI( is + ii )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       END IF
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ueh

     END SUBROUTINE PTRANS_ueh

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ugrsh  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_ugrsh( n, X, G, nnzh, lh, H, IRNH, ICNH, trans, data,    &
                              inform )
     INTEGER, INTENT( IN ) :: n, lh
     INTEGER, INTENT( OUT ) :: nnzh
     INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate g and H

     CALL CUTEST_ugrsh( inform%status, n, data%X_orig( : n ), G,               &
                        nnzh, lh, H, IRNH, ICNH )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradient

     G = ( trans%X_scale( : n ) / trans%f_scale ) * G

!  Transform the Hessian

     DO l = 1, nnzh
       i = IRNH( l ) ; j = ICNH( l )
       H( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /                    &
                  trans%f_scale ) * H( l )
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ugrsh

     END SUBROUTINE PTRANS_ugrsh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_ugreh  S U B R O U T I N E    -*-*-*-*-*

     SUBROUTINE PTRANS_ugreh( n, X, G, ne, IRNHI, lirnhi, le, IPRNHI,          &
                              HI, lhi, IPRHI, byrows, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, le, lirnhi, lhi
     INTEGER, INTENT( OUT ) :: ne
     LOGICAL, INTENT( IN ) :: byrows
     INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
     INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, ie, ii, is, j, jj, l, ni
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate g and H

     CALL CUTEST_ugreh( inform%status, n, data%X_orig( : n ), G, ne, le,       &
                        IPRNHI, IPRHI, lirnhi, IRNHI, lhi, HI, byrows )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradient

     G = ( trans%X_scale( : n ) / trans%f_scale ) * G

!  Transform the Hessian

     l = 1
     DO ie = 1, ne
       ni = IPRNHI( ie + 1 ) - IPRNHI( ie )
       is = IPRNHI( ie ) - 1
       IF ( byrows ) THEN
         DO ii = 1, ni
           i = IRNHI( is + ii )
           DO jj = i, ni
             j = IRNHI( is + jj )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       ELSE
         DO jj = 1, ni
           j = IRNHI( is + jj )
           DO ii = 1, jj
             i = IRNHI( is + ii )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       END IF
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ugreh

     END SUBROUTINE PTRANS_ugreh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_uprod  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_uprod( n, goth, X, P, RESULT, trans, data, inform )
     INTEGER, INTENT( IN ) :: n
     LOGICAL, INTENT( IN ) :: goth
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%P_orig'
     CALL SPACE_resize_array( n, data%P_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x and v, and unscale p

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )
     data%P_orig( : n ) = trans%X_scale( : n ) * P

!  Form the product between H and the unscaled p

     CALL CUTEST_uhprod( inform%status, n, goth, data%X_orig( : n ),           &
                         data%P_orig( : n ), RESULT )
     IF ( inform%status /= 0 ) RETURN

!  Transform the result

     RESULT = ( trans%X_scale( : n ) / trans%f_scale ) * RESULT

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_uprod

     END SUBROUTINE PTRANS_uprod

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_ubandh  S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE PTRANS_ubandh( n, X, nsemib, BANDH, lbandh, trans, data,       &
                               inform )
     INTEGER, INTENT( IN ) :: n, nsemib, lbandh
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 0 : lbandh, n ) :: BANDH
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, j_max, l, maxsbw
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate a band submatrix of H

     CALL CUTEST_ubandh( inform%status, n, data%X_orig( : n ), nsemib,         &
                         BANDH, lbandh, maxsbw )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     j_max = n
     DO l = 0, nsemib
       i = l + 1
       DO j = 1, j_max
         BANDH( l, j ) = ( trans%X_scale( i ) * trans%X_scale( j ) /           &
                           trans%f_scale ) * BANDH( l, j )
       END DO
       j_max = j_max - 1
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ubandh

     END SUBROUTINE PTRANS_ubandh

!  ------------------------------------------
!  Interfaces to the constrained CUTEst tools
!  ------------------------------------------

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_cfn  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_cfn( n, m, X, f, lc, C, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lc
     REAL ( KIND = wp ), INTENT( OUT ) :: f
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lc ) :: C
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate f and c

     CALL CUTEST_cfn( inform%status, n, m, data%X_orig( : n ), f, C )
     IF ( inform%status /= 0 ) RETURN

!  Transform f and c

     CALL PTRANS_trans( n, m, trans, 0.0_wp, f = f, C = C )

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cfn

     END SUBROUTINE PTRANS_cfn

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_cgr  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_cgr( n, m, X, GRLAGF, lv, V, G, JTRANS, lcjac1, lcjac2, &
                            CJAC, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lcjac1, lcjac2
     LOGICAL, INTENT( IN ) :: GRLAGF, JTRANS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac1, lcjac2 ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     IF ( grlagf )                                                             &
       data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate g and J

     CALL CUTEST_cgr( inform%status, n, m, data%X_orig( : n ),                 &
                      data%V_orig( : m ), grlagf, G, jtrans,                   &
                      lcjac1, lcjac2, CJAC )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradients

     G = ( trans%X_scale( : n ) / trans%f_scale ) * G

     IF ( jtrans ) THEN
       DO i = 1, m
         DO j = 1, n
           CJAC( j, i )                                                        &
             = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( j, i )
         END DO
       END DO
     ELSE
       DO j = 1, n
         DO i = 1, m
           CJAC( i, j )                                                        &
             = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( i, j )
         END DO
       END DO
     END IF

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cgr

     END SUBROUTINE PTRANS_cgr

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_cofg  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_cofg( n, X, f, G, grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( OUT ) :: f
     LOGICAL, INTENT( IN ) :: grad
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate f and possibly g

     CALL CUTEST_cofg( inform%status, n, data%X_orig( : n ), f, G, grad )
     IF ( inform%status /= 0 ) RETURN

!  Transform f

     CALL PTRANS_trans( n, 0, trans, 0.0_wp, f = f )

!  Possibly transform the gradients

     IF ( grad ) G = ( trans%X_scale( : n ) / trans%f_scale ) * G

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cofg

     END SUBROUTINE PTRANS_cofg

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_csgr  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_csgr( n, m, grlagf, lv, V, X, nnzj, lcjac, CJAC,        &
                             INDVAR, INDFUN, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lcjac
     INTEGER, INTENT( OUT ) :: nnzj
     LOGICAL, INTENT( IN ) :: grlagf
     INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     IF ( grlagf )                                                             &
       data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate g and J

     CALL CUTEST_csgr( inform%status, n, m,                                    &
                       data%X_orig( : n ), data%V_orig( : m ), GRLAGF,         &
                       nnzj, lcjac, CJAC, INDVAR, INDFUN )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradients

     DO l = 1, nnzj
       i = INDFUN( l ) ; j = INDVAR( l )
       IF ( i > 0 ) THEN
         CJAC( l ) = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( l )
       ELSE
         CJAC( l ) = ( trans%X_scale( j ) / trans%f_scale ) * CJAC( l )
       END IF
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_csgr

     END SUBROUTINE PTRANS_csgr

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ccfg  S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE PTRANS_ccfg( n, m, X, lc, C, jtrans, lcjac1, lcjac2, CJAC,     &
                             grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lc, lcjac1, lcjac2
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lc ) :: C
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac1, lcjac2 ) :: CJAC
     LOGICAL, INTENT( IN ) :: jtrans, grad
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate c and possibly J

     CALL CUTEST_ccfg( inform%status, n, m, data%X_orig( : n ), C,             &
                       jtrans, lcjac1, lcjac2, CJAC, grad )
     IF ( inform%status /= 0 ) RETURN

!  Transform c

     CALL PTRANS_trans( n, m, trans, 0.0_wp, C = C )

!  Possibly transform J

     IF ( grad ) THEN
       IF ( jtrans ) THEN
         DO i = 1, m
           DO j = 1, n
             CJAC( j, i )                                                      &
               = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( j, i )
           END DO
         END DO
       ELSE
         DO j = 1, n
           DO i = 1, m
             CJAC( i, j )                                                      &
               = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( i, j )
           END DO
         END DO
       END IF
     END IF

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ccfg

     END SUBROUTINE PTRANS_ccfg

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ccfsg  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_ccfsg( n, m, X, lc, C, nnzj, lcjac, CJAC,               &
                       INDVAR, INDFUN, grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, M, lc, lcjac
     INTEGER, INTENT( OUT ) :: nnzj
     LOGICAL, INTENT( IN ) :: grad
     INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lc ) :: C
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate c and possibly J

     CALL CUTEST_ccfsg( inform%status, n, m, data%X_orig( : n ), C,            &
                        nnzj, lcjac, CJAC, INDVAR, INDFUN, grad )
     IF ( inform%status /= 0 ) RETURN

!  Transform c

     CALL PTRANS_trans( n, m, trans, 0.0_wp, C = C )

!  Possibly transform J

     IF ( grad ) THEN
       DO l = 1, nnzj
         i = INDFUN( l ) ; j = INDVAR( l )
         CJAC( l ) = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( l )
       END DO
     END IF

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ccfsg

     END SUBROUTINE PTRANS_ccfsg

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_cscfg  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_cscfg( n, m, X, lc, C, nnzj, lcjac, CJAC,                &
                       INDVAR, INDFUN, grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, M, lc, lcjac
     INTEGER, INTENT( OUT ) :: nnzj
     LOGICAL, INTENT( IN ) :: grad
     INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lc ) :: C
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  This is an alias for PTRANS_ccfsg, maintained for backwards compatibility

     CALL PTRANS_ccfsg( n, m, X, lc, C, nnzj, lcjac, CJAC, INDVAR, INDFUN,     &
                        grad, trans, data, inform )

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cscfg

     END SUBROUTINE PTRANS_cscfg

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_ccifg  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_ccifg( n, icon, X, ci, GCI, grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, icon
     LOGICAL, INTENT( IN ) :: grad
     REAL ( KIND = wp ), INTENT( OUT ) :: ci
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GCI
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate c_i and possibly grad c_i

     CALL CUTEST_ccifg( inform%status, n, icon, data%X_orig( : n ),            &
                        ci, GCI, grad )
     IF ( inform%status /= 0 ) RETURN

!  Transform ci

     ci = ( ci - trans%C_shift( icon ) ) / trans%C_scale( icon )

!  Possibly transform the gradient

     IF ( grad ) GCI = ( trans%X_scale( : n ) / trans%C_scale( icon ) ) * GCI

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ccifg

     END SUBROUTINE PTRANS_ccifg

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_ccifsg  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_ccifsg( n, icon, X, ci, nnzgci, lgci, GCI, INDVAR,      &
                               grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, icon, lgci
     INTEGER, INTENT( OUT ) :: nnzgci
     REAL ( KIND = wp ), INTENT( OUT ) :: ci
     LOGICAL, INTENT( IN ) :: grad
     INTEGER, INTENT( OUT ), DIMENSION( lgci ) :: INDVAR
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lgci ) :: GCI
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate c_i and possibly grad c_i

     CALL CUTEST_ccifsg( inform%status, n, icon, data%X_orig( : n ),           &
                         ci, nnzgci, lgci, GCI, INDVAR, grad )
     IF ( inform%status /= 0 ) RETURN

!  Transform ci

     ci = ( ci - trans%C_shift( icon ) ) / trans%C_scale( icon )

!  Possibly transform its gradient

     IF ( grad ) THEN
       DO l = 1, nnzgci
         i = INDVAR( l )
         GCI( l ) = ( trans%X_scale( i ) / trans%C_scale( icon ) ) * GCI( l )
       END DO
     END IF

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ccifsg

     END SUBROUTINE PTRANS_ccifsg

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_csifg  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_cscifg( n, icon, X, ci, nnzgci, lgci, GCI, INDVAR,      &
                               grad, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, icon, lgci
     INTEGER, INTENT( OUT ) :: nnzgci
     REAL ( KIND = wp ), INTENT( OUT ) :: ci
     LOGICAL, INTENT( IN ) :: grad
     INTEGER, INTENT( OUT ), DIMENSION( lgci ) :: INDVAR
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lgci ) :: GCI
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  This is an alias for PTRANS_ccifsg, maintained for backwards compatibility

     CALL PTRANS_ccifsg( n, icon, X, ci, nnzgci, lgci, GCI, INDVAR,           &
                         grad, trans, data, inform )

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cscifg

     END SUBROUTINE PTRANS_cscifg

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_cdh  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_cdh( n, m, X, lv, V, lh1, H, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lh1
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Un-transform V

     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate H

     CALL CUTEST_cdh( inform%status, n, m, data%X_orig( : n ),                 &
                      data%V_orig( : m ), lh1, H )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     DO j = 1, n
       DO i = 1, n
         H( i, j ) = ( trans%X_scale( i ) * trans%X_scale( j ) /               &
                       trans%f_scale ) * H( i, j )
       END DO
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cdh

     END SUBROUTINE PTRANS_cdh

!  *-*-*-*-*-*-*-*-  P T R A N S   PTRANS_cidh  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE PTRANS_cidh( n, X, iprob, lh1, H, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, iprob, lh1
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate H

     CALL CUTEST_cidh( inform%status, n, data%X_orig( : n ), iprob, lh1, H )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     DO j = 1, n
       DO i = 1, n
         H( i, j ) = ( trans%X_scale( i ) * trans%X_scale( j ) /               &
                       trans%f_scale ) * H( i, j )
       END DO
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cidh

     END SUBROUTINE PTRANS_cidh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_cgrdh  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_cgrdh( n, m, X, grlagf, lv, V, G, jtrans, lcjac1,       &
                              lcjac2, CJAC, lh1, H, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lh1, lcjac1, lcjac2
     LOGICAL, INTENT( IN ) :: grlagf, jtrans
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh1, n ) :: H
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac1, lcjac2 ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate g, J and H

     CALL CUTEST_cgrdh( inform%status, n, m, data%X_orig( : n ),               &
                        data%V_orig( : m ), grlagf, G, jtrans,                 &
                        lcjac1, lcjac2, CJAC, lh1, H )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradients

     G = ( trans%X_scale( : n ) / trans%f_scale ) * G

     IF ( jtrans ) THEN
       DO i = 1, m
         DO j = 1, n
           CJAC( j, i )                                                        &
             = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( j, i )
         END DO
       END DO
     ELSE
       DO j = 1, n
         DO i = 1, m
           CJAC( i, j )                                                        &
             = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( i, j )
         END DO
       END DO
     END IF

!  Transform the Hessian

     DO j = 1, n
       DO i = 1, n
         H( i, j ) = ( trans%X_scale( i ) * trans%X_scale( j ) /               &
                       trans%f_scale ) * H( i, j )
       END DO
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cgrdh

     END SUBROUTINE PTRANS_cgrdh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_csh  S U B R O U T I N E  -*-*-*-*-*-*-*

     SUBROUTINE PTRANS_csh( n, m, X, lv, V, nnzh, lh, H, IRNH, ICNH, trans,    &
                            data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lh
     INTEGER, INTENT( OUT ) :: nnzh
     INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate H

     CALL CUTEST_csh( inform%status, n, m, data%X_orig( : n ),                 &
                      data%V_orig( : m ), nnzh, lh, H, IRNH, ICNH )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     DO l = 1, nnzh
       i = IRNH( l ) ; j = ICNH( l )
       H( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /                    &
                  trans%f_scale ) * H( l )
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_csh

     END SUBROUTINE PTRANS_csh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_cish  S U B R O U T I N E   -*-*-*-*-*-*

     SUBROUTINE PTRANS_cish( n, X, iprob, nnzh, lh, H, IRNH, ICNH, trans,      &
                             data, inform )
     INTEGER, INTENT( IN ) :: n, iprob, lh
     INTEGER, INTENT( OUT ) :: nnzh
     INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  Evaluate H

     CALL CUTEST_cish( inform%status, n, data%X_orig( : n ), iprob,            &
                       nnzh, lh, H, IRNH, ICNH )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     DO l = 1, nnzh
       i = IRNH( l ) ; j = ICNH( l )
       H( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /                    &
                  trans%f_scale ) * H( l )
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cish

     END SUBROUTINE PTRANS_cish

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_ceh  S U B R O U T I N E    -*-*-*-*-*-*

     SUBROUTINE PTRANS_ceh( n, m, X, lv, V, ne, IRNHI, lirnhi, le, IPRNHI,     &
                            HI, lhi, IPRHI, byrows, trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, le, lirnhi, lhi
     INTEGER, INTENT( OUT ) :: ne
     LOGICAL, INTENT( IN ) :: byrows
     INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
     INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, ie, ii, is, j, jj, l, ni
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate H in element format

     CALL CUTEST_ceh( inform%status, n, m, data%X_orig( : n ),                 &
                      data%V_orig( : m ), ne, le,                              &
                      IPRNHI, IPRHI, lirnhi, IRNHI, lhi, HI, byrows )
     IF ( inform%status /= 0 ) RETURN

!  Transform the Hessian

     l = 1
     DO ie = 1, ne
       ni = IPRNHI( ie + 1 ) - IPRNHI( ie )
       is = IPRNHI( ie ) - 1
       IF ( byrows ) THEN
         DO ii = 1, ni
           i = IRNHI( is + ii )
           DO jj = i, ni
             j = IRNHI( is + jj )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       ELSE
         DO jj = 1, ni
           j = IRNHI( is + jj )
           DO ii = 1, jj
             i = IRNHI( is + ii )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       END IF
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_ceh

     END SUBROUTINE PTRANS_ceh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_csgrsh  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_csgrsh( n, m, X, grlagf, lv, V, nnzj, lcjac, CJAC,       &
                               INDVAR, INDFUN, nnzh, lh, H, IRNH, ICNH,         &
                               trans, data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lcjac, lh
     INTEGER, INTENT( OUT ) :: nnzj, nnzh
     LOGICAL, INTENT( IN ) :: grlagf
     INTEGER, INTENT( OUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
     INTEGER, INTENT( OUT ), DIMENSION( lh ) :: IRNH, ICNH
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lh ) :: H
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate g, J and H

     CALL CUTEST_csgrsh( inform%status, n, m, data%X_orig( : n ),              &
                         data%V_orig( : m ), grlagf, nnzj, lcjac,              &
                         CJAC, INDVAR, INDFUN, nnzh, lh, H, IRNH, ICNH )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradients

     DO l = 1, nnzj
       i = INDFUN( l ) ; j = INDVAR( l )
       IF ( i > 0 ) THEN
         CJAC( l ) = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( l )
       ELSE
         CJAC( l ) = ( trans%X_scale( j ) / trans%f_scale ) * CJAC( l )
       END IF
     END DO

!  Transform the Hessian

     DO l = 1, nnzh
       i = IRNH( l ) ; j = ICNH( l )
       H( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /                    &
                  trans%f_scale ) * H( l )
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_csgrsh

     END SUBROUTINE PTRANS_csgrsh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_csgreh  S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE PTRANS_csgreh( n, m, X, grlagf, lv, V, nnzj, lcjac,            &
                               CJAC, INDVAR, INDFUN, ne, IRNHI, lirnhi,        &
                               le, IPRNHI, HI, lhi, IPRHI, byrows, trans,      &
                               data, inform )
     INTEGER, INTENT( IN ) :: n, m, lv, lcjac, le, lirnhi, lhi
     INTEGER, INTENT( OUT ) :: ne, nnzj
     LOGICAL, INTENT( IN ) :: grlagf, byrows
     INTEGER, INTENT( INOUT ), DIMENSION( lcjac ) :: INDVAR, INDFUN
     INTEGER, INTENT( OUT ), DIMENSION( lirnhi ) :: IRNHI
     INTEGER, INTENT( OUT ), DIMENSION( le ) :: IPRNHI, IPRHI
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lhi ) :: HI
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( lcjac ) :: CJAC
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     INTEGER :: i, ie, ii, is, j, jj, l, ni
     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                                inform%alloc_status, exact_size = .TRUE.,      &
                                array_name = array_name,                       &
                                bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )

!  If the gradient of the Lagrangian is required, un-transform V

     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V

!  Evaluate g, J and H

     CALL CUTEST_csgreh( inform%status, n, m, data%X_orig( : n ),              &
                         data%V_orig( : m ), grlagf, nnzj, lcjac,              &
                         CJAC, INDVAR, INDFUN, ne, le,                         &
                         IPRNHI, IPRHI, lirnhi, IRNHI, lhi, HI, byrows )
     IF ( inform%status /= 0 ) RETURN

!  Transform the gradients

     DO l = 1, nnzj
       i = INDFUN( l ) ; j = INDVAR( l )
       IF ( i > 0 ) THEN
         CJAC( l ) = ( trans%X_scale( j ) / trans%C_scale( i ) ) * CJAC( l )
       ELSE
         CJAC( l ) = ( trans%X_scale( j ) / trans%f_scale ) * CJAC( l )
       END IF
     END DO

!  Transform the Hessian

     l = 1
     DO ie = 1, ne
       ni = IPRNHI( ie + 1 ) - IPRNHI( ie )
       is = IPRNHI( ie ) - 1
       IF ( byrows ) THEN
         DO ii = 1, ni
           i = IRNHI( is + ii )
           DO jj = i, ni
             j = IRNHI( is + jj )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       ELSE
         DO jj = 1, ni
           j = IRNHI( is + jj )
           DO ii = 1, jj
             i = IRNHI( is + ii )
             HI( l ) = ( trans%X_scale( i ) * trans%X_scale( j ) /             &
                         trans%f_scale ) * HI( l )
             l = l + 1
           END DO
         END DO
       END IF
     END DO

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_csgreh

     END SUBROUTINE PTRANS_csgreh

!  *-*-*-*-*-*-*-  P T R A N S   PTRANS_cprod  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE PTRANS_cprod( n, m, goth, X, lv, V, P, RESULT, trans, data,    &
                              inform )
     INTEGER, INTENT( IN ) :: n, m, lv
     LOGICAL, INTENT( IN ) :: goth
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, P
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lv ) :: V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: RESULT
     TYPE ( PTRANS_trans_type ), INTENT( IN ) :: trans
     TYPE ( PTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PTRANS_inform_type ), INTENT( OUT ) :: inform

!  Local variables

     CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that workspace arrays are large enough

     array_name = 'ptrans: data%X_orig'
     CALL SPACE_resize_array( n, data%X_orig, inform%status,                   &
                              inform%alloc_status, exact_size = .TRUE.,        &
                              array_name = array_name,                         &
                              bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%V_orig'
     CALL SPACE_resize_array( m, data%V_orig, inform%status,                   &
                              inform%alloc_status, exact_size = .TRUE.,        &
                              array_name = array_name,                         &
                              bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     array_name = 'ptrans: data%P_orig'
     CALL SPACE_resize_array( n, data%P_orig, inform%status,                   &
                              inform%alloc_status, exact_size = .TRUE.,        &
                              array_name = array_name,                         &
                              bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

!  Un-transform x and v, and unscale p

     CALL PTRANS_v_untrans( n, trans%X_scale, trans%X_shift, X, data%X_orig )
     data%V_orig( : m ) = ( trans%f_scale / trans%C_scale( : m ) ) * V
     data%P_orig( : n ) = trans%X_scale( : n ) * P

!  Form the product between H and the unscaled p

     CALL CUTEST_chprod( inform%status, n, m, goth, data%X_orig( : n ),        &
                        data%V_orig( : m ), data%P_orig( : n ), RESULT )
     IF ( inform%status /= 0 ) RETURN

!  Transform the result

     RESULT = ( trans%X_scale( : n ) / trans%f_scale ) * RESULT

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN

!  End of subroutine PTRANS_cprod

     END SUBROUTINE PTRANS_cprod

!  End of module GALAHAD_PTRANS_double

   END MODULE GALAHAD_PTRANS_double

