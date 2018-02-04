! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ Q T R A N S   M O D U L E  *-*-*-*-*-*-*-*-*-*

!  Suppose that x_t = X_s^-1 ( x - x_s )
!               f_t( x_t ) = F_s^-1 ( q( x ) - f_s )
!          and  A_t x_t = C_s^-1 ( A x - c_s )
!
!  Compute suitable shifts (x_s,f_s) and scale factors (X_s,F_s,C_s)
!  and apply these transformations (and their inversess) to the data 
!  for the quadratic programming (QP) problem
!
!      min f(x) = 1/2 x^T H x + x^T g + f
!
!      s.t.        c_l <= A x <= c_u, 
!      and         x_l <=  x  <= x_u
!
!  (or optionally to the parametric problem
!
!      min  1/2 x^T H x + x^T g + theta x^T dg + f + theta df
!
!      s.t. c_l + theta dc_l <= A x <= c_u + theta dc_u,
!      and  x_l + theta dx_l <=  x  <= x_u + theta dx_u )
!
!  to derive the transformed problem
!
!      min f_t(x_t) = 1/2 x_t^T H_t x_t + x_t^T g_t + f_t
!
!      s.t.           c_t_l <= A_t x_t <= c_t_u, 
!                     x_t_l <=   x_t   <= x_t_u
!
!  (or optionally for the parametric problem
!
!      min  1/2 x_t^T H_t x_t + x_t^T g_t + theta x_t^T dg_t + f_t + theta df_t
!
!      s.t. c_t_l + theta dc_t_l <= A_t x_t <= c_t_u + theta dc_t_u,
!      and  x_t_l + theta dx_t_l <=    x_t  <= x_t_u + theta dx_t_u )
!
!  where H_t = X_s^T H X_s / F_s
!        g_t = X_s ( H x_s + g ) / F_s
!        dg_t = X_s dg / F_s
!        f_t = 1/2 x_s^T H x_s + x_s^T g + f - f_s ) / F_s
!        df_t = x_s^T dg / F_s
!        A_t = C_s^-1 A X_s
!        c_s = A x_s
!        c_t_l = C_s^-1 ( c_l - c_s )
!        dc_t_l = C_s^-1 dc_l
!        c_t_u = C_s^-1 ( c_u - c_s )
!        dc_t_u = C_s^-1 d_u
!        x_t_l = X_s^-1 ( c_l - x_s )
!        dx_t_l = X_s^-1 d_l
!        x_t_u = X_s^-1 ( c_u - x_s )
!  and   dx_t_u = X_s^-1 dc_u

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  December 20th 2004

   MODULE GALAHAD_QTRANS_double

     USE GALAHAD_SPACE_double
     USE GALAHAD_TRANS_double, only :                                          &
       QTRANS_trans_type => TRANS_trans_type,                                  &
       QTRANS_data_type => TRANS_data_type,                                    &
       QTRANS_inform_type => TRANS_inform_type,                                &
       TRANS_initialize,                                                       &
       QTRANS_terminate => TRANS_terminate,                                    &
       TRANS_default,                                                          &
       QTRANS_trans => TRANS_trans,                                            &
       QTRANS_untrans => TRANS_untrans,                                        &
       QTRANS_v_trans_inplace => TRANS_v_trans_inplace,                        &
       QTRANS_v_untrans_inplace => TRANS_v_untrans_inplace
     USE CUTEr_interface_double

     IMPLICIT NONE     

     PRIVATE

     PUBLIC :: QTRANS_trans_type
     PUBLIC :: QTRANS_data_type
     PUBLIC :: QTRANS_inform_type
     PUBLIC :: QTRANS_initialize
     PUBLIC :: QTRANS_get_factors
     PUBLIC :: QTRANS_apply
     PUBLIC :: QTRANS_apply_inverse
     PUBLIC :: QTRANS_terminate
     PUBLIC :: QTRANS_trans
     PUBLIC :: QTRANS_untrans

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  ====================================
!  The QTRANS_control_type derived type
!  ====================================

     TYPE, PUBLIC :: QTRANS_control_type
       INTEGER :: out, print_level
       INTEGER :: shift_x, shift_f, scale_x, scale_c, scale_f
       REAL ( KIND = wp ) :: infinity, scale_x_min, scale_c_min
       LOGICAL :: deallocate_error_fatal
     END TYPE QTRANS_control_type

!  Set parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp

  CONTAINS

!  *-*-*-*-*-*-  Q T R A N S  QTRANS_initialize  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE QTRANS_initialize( control, inform )
     TYPE ( QTRANS_control_type ), INTENT( OUT ) :: control
     TYPE ( QTRANS_inform_type ), INTENT( OUT ) :: inform

!  Nullify pointers

     CALL TRANS_initialize( inform )

!  Set default control parameters

     control%out = 6
     control%print_level = 0
     control%infinity = HUGE( one )
     control%scale_x_min = one
     control%scale_c_min = one
     control%shift_x = 1
     control%scale_x = 1
     control%shift_f = 1
     control%scale_f = 1
     control%scale_c = 1
     control%deallocate_error_fatal = .FALSE.

     RETURN

!  End of subroutine QTRANS_initialize

     END SUBROUTINE QTRANS_initialize

!  *-*-*-*-*-  Q T R A N S  QTRANS_get_factors  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE QTRANS_get_factors( trans, data, control, inform,              &
                                    m, n, f, X, X_l, X_u, C, C_l, C_u,         &
                                    G, A_col, A_ptr, A_val )

     TYPE ( QTRANS_trans_type ), INTENT( INOUT ) :: trans
     TYPE ( QTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( QTRANS_control_type ), INTENT( IN ) :: control
     TYPE ( QTRANS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER, INTENT( IN ) :: m, n
     INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
     INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
     REAL ( KIND = wp ), INTENT( IN ) :: f
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, X_l, X_u, G
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, C_l, C_u
     REAL ( KIND = wp ), INTENT( IN ),                                         &
       DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val

     INTEGER :: i, l, out
     LOGICAL :: printi, printd

     out = control%out
!    printi = out > 0 .AND. control%print_level > 0
     printi = .TRUE.
     printd = out > 0 .AND. control%print_level > 5

     CALL TRANS_default( n, m, trans, inform )
     IF ( inform%status /= 0 ) RETURN

!  Scale and/or shift the variables

     IF ( control%shift_x > 0 .OR. control%scale_x > 0 ) THEN
       DO i = 1, n
         IF ( X_l( i ) < X_u( i ) ) THEN
           IF ( X_u( i ) < control%infinity ) THEN
             IF ( X_l( i ) > - control%infinity ) THEN
               IF ( control%shift_x > 0 )                                      &
                 trans%X_shift( i ) = half * ( X_u( i ) + X_l( i ) )
               IF ( control%scale_x > 0 ) trans%X_scale( i )                   &
                 = MAX( control%scale_x_min, half * ( X_u( i ) - X_l( i ) ) )
             ELSE
               IF ( control%shift_x > 0 )                                      &
                 trans%X_shift( i ) = X_u( i )
               IF ( control%scale_x > 0 ) trans%X_scale( i )                   &
                 = MAX( control%scale_x_min, X_u( i ) - X( i ) )
             END IF
           ELSE IF ( X_l( i ) > - control%infinity ) THEN
             IF ( control%shift_x > 0 )                                        &
               trans%X_shift( i ) = X_l( i )
             IF ( control%scale_x > 0 ) trans%X_scale( i )                     &
               = MAX( control%scale_x_min, X( i ) - X_l( i ) )
           END IF
         END IF
       END DO

       IF ( printd ) THEN
         WRITE( out, "( '  shift_x ', /, ( 3ES22.14 ) )" )                     &
           trans%X_shift( 1 : n )
         WRITE( out, "( '  scale_x ', /, ( 3ES22.14 ) )" )                     &
           trans%X_scale( 1 : n )
       ELSE IF ( printi ) THEN
         WRITE( out, "( '  max shift_x ', /, ES22.14 )" )                      &
           MAXVAL( ABS( trans%X_shift( 1 : n ) ) )
         WRITE( out, "( '  max scale_x ', /, ES22.14 )" )                      &
           MAXVAL( ABS( trans%X_scale( 1 : n ) ) )
       END IF
     END IF

!  If the variables have been shifted, make sure that the shift
!  is reflected in a shift in c

     IF ( control%shift_x > 0 ) THEN
       DO i = 1, m
         trans%C_shift( i ) = zero
         DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
           trans%C_shift( i ) =                                                &
             trans%C_shift( i ) + A_val( l ) * trans%X_shift( A_col( l ) ) 
         END DO
       END DO

       IF ( printd ) THEN
         WRITE( out, "( '  shift_c ', /, ( 3ES22.14 ) )" )                     &
           trans%C_shift( 1 : m )
       ELSE IF ( printi ) THEN
         WRITE( out, "( '  max shift_c ', /, ES22.14 )" )                      &
           MAXVAL( ABS( trans%C_shift( 1 : m ) ) )
       END IF
     END IF

!  Scale the constraints

     IF ( control%scale_c > 0 ) THEN

!  Scale and shift so that shifts try to make c of O(1)

       IF ( control%scale_c == 2 ) THEN
         DO i = 1, m
           IF ( C_l( i ) < C_u( i ) ) THEN
             IF ( C_u( i ) < control%infinity ) THEN
               IF ( C_l( i ) > - control%infinity ) THEN
                 trans%C_scale( i ) = MAX( control%scale_c_min,                &
                                           half * ( C_u( i ) - C_l( i ) ) )
               ELSE
                 trans%C_scale( i ) = MAX( control%scale_c_min,                &
                                           ABS( C_u( i ) - C( i ) ) )
               END IF
             ELSE IF ( C_l( i ) > - control%infinity ) THEN
               trans%C_scale( i ) = MAX( control%scale_c_min,                  &
                                          ABS( C( i ) - C_l( i ) ) )
             END IF
           END IF
         END DO

!  Scale and shift so that shifts try to make O(1) changes to x make O(1)
!  changes to c, using the (scaled) infinity norms of the gradients of 
!  the constraints.

       ELSE
         DO i = 1, m
           trans%C_scale( i ) = one
           IF ( C_u( i ) < control%infinity )                                  &
             trans%C_scale( i ) = MAX( trans%C_scale( i ), ABS( C_u( i ) ) ) 
           IF ( C_l( i ) > - control%infinity )                                &
             trans%C_scale( i ) = MAX( trans%C_scale( i ), ABS( C_l( i ) ) )
           IF ( control%scale_x > 0 ) THEN
             DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
               trans%C_scale( i ) = MAX( trans%C_scale( i ),                   &
                 ABS( trans%X_scale( A_col( l ) ) * A_val( l ) ) )
             END DO
           ELSE
             DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
               trans%C_scale( i ) = MAX( trans%C_scale( i ),                   &
                 ABS( A_val( l ) ) )
             END DO
           END IF
         END DO
       END IF

       IF ( printd ) THEN
         WRITE( out, "( '  scale_c ', /, ( 3ES22.14 ) )" )                     &
           trans%C_scale( 1 : m )
       ELSE IF ( printi ) THEN
         WRITE( out, "( '  max scale_c ', /, ES22.14 )" )                      &
           MAXVAL( ABS( trans%C_scale( 1 : m ) ) )
       END IF
!        WRITE( out, "( '  scale_c ', /, ( 3ES22.14 ) )" )                     &
!          trans%C_scale( 1 : m )
     END IF

!  Scale the objective

     IF ( control%scale_f > 1 ) THEN

!  Scale and shift so that shifts try to make f of O(1)

!      trans%f_shift = f
       IF ( control%scale_f == 2 ) THEN
         trans%f_scale = one

!  Scale and shift so that shifts try to make O(1) changes to x make O(1)
!  changes to f, using the (scaled) infinity norm of the gradients of 
!  the objective

       ELSE
         IF ( control%scale_x > 0 ) THEN
           DO i = 1, n
             trans%f_scale = MAX( trans%f_scale,                               &
               ABS( trans%X_scale( i ) * G( i ) ) )
           END DO
         ELSE
           DO i = 1, n
             trans%f_scale = MAX( trans%f_scale, ABS( G( i ) ) )
           END DO
         END IF
       END IF
       IF ( printi ) THEN
         WRITE( out, "( '  shift_f ', /, ES22.14 )" ) trans%f_shift
         WRITE( out, "( '  scale_f ', /, ES22.14 )" ) trans%f_scale
       END IF
     END IF

     RETURN

!  End of subroutine QTRANS_get_factors

     END SUBROUTINE QTRANS_get_factors

!  *-*-*-*-*-*-  Q T R A N S  QTRANS_apply  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE QTRANS_apply( trans, data, control, inform,                    &
                              m, n, f, G, X, X_l, X_u, C_l, C_u,               &
                              A_col, A_ptr, A_val, H_col, H_ptr, H_val,        &
                              df, DG, DX_l, DX_u, DC_l, DC_u )

!  Apply the scale and shift factors to the problem data

     TYPE ( QTRANS_trans_type ), INTENT( INOUT ) :: trans
     TYPE ( QTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( QTRANS_control_type ), INTENT( IN ) :: control
     TYPE ( QTRANS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER, INTENT( IN ) :: m, n
     INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
     INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
     INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
     INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
     REAL ( KIND = wp ), INTENT( INOUT ) :: f
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: G, X, X_l, X_u
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C_l, C_u
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
       DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
       DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
     REAL ( KIND = wp ), OPTIONAL :: df
     REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: DG, DX_l, DX_u
     REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: DC_l, DC_u

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: point_name

!  Compute H x_s

     point_name = 'qtrans: H_x'
     CALL SPACE_resize_array( n, data%H_x, inform%status,                      &
                      inform%alloc_status, exact_size = .TRUE.,                &
                      deallocate_error_fatal = control%deallocate_error_fatal, &
                      array_name = point_name, bad_alloc = inform%bad_alloc,   &
                      out = control%out )
     IF ( inform%status /= 0 ) RETURN

     data%H_x = zero
     DO i = 1, n
       DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
         j = H_col( l )
         data%H_x( i ) = data%H_x( i ) + H_val( l ) * trans%X_shift( j )
         IF ( i /= j )                                                         &
           data%H_x( j ) = data%H_x( j ) + H_val( l ) * trans%X_shift( i )
       END DO
     END DO

!  Compute f <- 1/2 x_s^T H x_s + x_s^T g + f - f_s ) / F_s

     f = ( half * DOT_PRODUCT( data%H_x, trans%X_shift ) +                     &
           DOT_PRODUCT( g, trans%X_shift ) + f - trans%f_shift ) / trans%f_scale

!  Compute g <- X_s ( H x_s + G ) / F_s
  
     G = trans%X_scale * ( data%H_x + G ) / trans%f_scale

!  Compute df <- x_s^T dg / F_s

     IF ( PRESENT( df ) .AND. PRESENT( DG ) )                                  &
       df = DOT_PRODUCT( DG, trans%X_shift ) / trans%f_scale

!  Compute dg <- X_s dg / F_s

     IF ( PRESENT( DG ) ) DG = trans%X_scale * DG / trans%f_scale

!  Compute H <- X_s^T H X_s / F_s

     DO i = 1, n
       DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
         j = H_col( l )
         H_val( l ) = H_val( l ) * ( trans%X_scale( i ) *                      &
           trans%X_scale( H_col( l ) ) / trans%f_scale ) 
       END DO
     END DO

!  Compute A <- C_s^-1 A X_s

     DO i = 1, m
       DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
         A_val( l ) = A_val( l ) *                                             &
           ( trans%X_scale( A_col( l ) ) / trans%C_scale( i ) )
       END DO
     END DO

!  Compute c_l <- C_s^-1 ( c_l - c_s )

     CALL QTRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C_l,        &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute dc_l <- C_s^-1 dc_l

     IF ( PRESENT( DC_l ) )                                                    &
       WHERE ( DC_l >  - control%infinity ) DC_l = DC_l / trans%C_scale

!  Compute c_u <- C_s^-1 ( c_u - c_s )

     CALL QTRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C_u,        &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dc_u <- C_s^-1 d_u

     IF ( PRESENT( DC_u ) )                                                    &
       WHERE ( DC_u <  control%infinity ) DC_u = DC_u / trans%C_scale

!  Compute x <- X_s^-1 ( x - x_s )

     CALL QTRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X )

!  Compute x_l <- X_s^-1 ( x_l - x_s )

     CALL QTRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X_l,        &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute dx_l <- X_s^-1 d_l

     IF ( PRESENT( DX_l ) )                                                    &
       WHERE ( DX_l >  - control%infinity ) DX_l = DX_l / trans%X_scale

!  Compute x_u <- X_s^-1 ( x_u - x_s )

     CALL QTRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X_u,        &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dx_u <- X_s^-1 dc_u

     IF ( PRESENT( DX_u ) )                                                    &
       WHERE ( DX_u <  control%infinity ) DX_u = DX_u / trans%X_scale

     RETURN

!  End of subroutine QTRANS_apply

     END SUBROUTINE QTRANS_apply

!  *-*-*-*-*-  Q T R A N S  QTRANS_apply_inverse  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE QTRANS_apply_inverse(                                          &
                              trans, data, control, inform,                    &
                              m, n, f, G, X, X_l, X_u, C, C_l, C_u,            &
                              A_col, A_ptr, A_val, H_col, H_ptr, H_val,        &
                              Y, Z, df, DG, DX_l, DX_u, DC_l, DC_u )

!  Recover the problem data from its scaled and shifted version

     TYPE ( QTRANS_trans_type ), INTENT( INOUT ) :: trans
     TYPE ( QTRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( QTRANS_control_type ), INTENT( IN ) :: control
     TYPE ( QTRANS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER, INTENT( IN ) :: m, n
     INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
     INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
     INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
     INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
     REAL ( KIND = wp ), INTENT( INOUT ) :: f
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: G, X, X_l, X_u, Z
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, C_l, C_u, Y
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
       DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
       DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
     REAL ( KIND = wp ), OPTIONAL :: df
     REAL ( KIND = wp ), OPTIONAL, DIMENSION( n ) :: DG, DX_l, DX_u
     REAL ( KIND = wp ), OPTIONAL, DIMENSION( m ) :: DC_l, DC_u

     INTEGER :: i, j, l
     CHARACTER ( LEN = 80 ) :: point_name

!  Compute dx_u <- X_s dc_u

     IF ( PRESENT( DX_u ) )                                                    &
       WHERE ( DX_u <  control%infinity ) DX_u = DX_u * trans%X_scale

!  Compute x_u <- X_s x_u + x_s

     CALL QTRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X_u,      &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dx_l <- X_s d_l

     IF ( PRESENT( DX_l ) )                                                    &
       WHERE ( DX_l >  - control%infinity ) DX_l = DX_l * trans%X_scale

!  Compute x_l <- X_s x_l + x_s

     CALL QTRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X_l,      &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute x <- X_s x + x_s

     CALL QTRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X )

!  Compute dc_u <- C_s d_u

     IF ( PRESENT( DC_u ) )                                                    &
       WHERE ( DC_u <  control%infinity ) DC_u = DC_u * trans%C_scale

!  Compute c_u <- C_s c_u + c_s

     CALL QTRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C_u,      &
                                 lower = .FALSE., infinity = control%infinity )

!  Compute dc_l <- C_s dc_l

     IF ( PRESENT( DC_l ) )                                                    &
       WHERE ( DC_l >  - control%infinity ) DC_l = DC_l * trans%C_scale

!  Compute c_l <- C_s c_l + c_s

     CALL QTRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C_l,      &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute c <- C_s c + c_s

     CALL QTRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C,        &
                                 lower = .TRUE., infinity = control%infinity )

!  Compute A <- C_s A X_s^-1

     DO i = 1, m
       DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
         A_val( l ) = A_val( l ) *                                             &
           ( trans%C_scale( i ) / trans%X_scale( A_col( l ) ) )
       END DO
     END DO

!  Compute H <- X_s^-T H X_s^-1 * F_s

     DO i = 1, n
       DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
         j = H_col( l )
         H_val( l ) = H_val( l ) * ( trans%f_scale  /                          &
           ( trans%X_scale( i ) * trans%X_scale( H_col( l ) ) ) )
       END DO
     END DO

!  Compute dg <- X_s^-1 dg * F_s

     IF ( PRESENT( DG ) ) DG = ( DG / trans%X_scale ) * trans%f_scale

!  Compute H x_s

     point_name = 'qtrans: H_x'
     CALL SPACE_resize_array( n, data%H_x, inform%status,                      &
                      inform%alloc_status, exact_size = .TRUE.,                &
                      deallocate_error_fatal = control%deallocate_error_fatal, &
                      array_name = point_name, bad_alloc = inform%bad_alloc,   &
                      out = control%out )
     IF ( inform%status /= 0 ) RETURN

     data%H_x = zero
     DO i = 1, n
       DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
         j = H_col( l )
         data%H_x( i ) = data%H_x( i ) + H_val( l ) * trans%X_shift( j )
         IF ( i /= j )                                                         &
           data%H_x( j ) = data%H_x( j ) + H_val( l ) * trans%X_shift( i )
       END DO
     END DO

!  Compute g <- X_s^{-1} ( G - H x_s  ) * F_s
  
     G = trans%f_scale * G / trans%X_scale - data%H_x

!  Compute y <- C_s^{-1} y
  
     Y = trans%f_scale * Y / trans%C_scale

!  Compute z <- X_s^{-1} z
  
     Z = trans%f_scale * Z / trans%X_scale

!  Compute df <- 0

     IF ( PRESENT( df ) .AND. PRESENT( DG ) ) df = zero

!  Compute f <- F_S * ( f + f_s - x_s^T g -  1/2 x_s^T H x_s )

     f = trans%f_scale * ( f + trans%f_shift - DOT_PRODUCT( G, trans%X_shift ) &
          - half * DOT_PRODUCT( data%H_x, trans%X_shift ) )

     RETURN

!  End of subroutine QTRANS_apply_inverse

     END SUBROUTINE QTRANS_apply_inverse

!  End of module GALAHAD_QTRANS_double

   END MODULE GALAHAD_QTRANS_double
      
