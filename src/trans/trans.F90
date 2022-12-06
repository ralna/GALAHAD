! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _ T R A N S   M O D U L E  *-*-*-*-*-*-*-*-*-*

!  Suppose that x_trans = X_scale^-1 ( x - x_shift )
!               f_trans( x_trans ) = f_scale^-1 ( f( x_trans ) - f_shift )
!          and  c_trans( x_trans ) = C_scale^-1 ( c( x_trans ) - c_shift )
!  Provide tools to apply these transformations and their inverses to
!  various kinds of optimization problems. This package is primarily
!  for use by other GALAHAD routines.

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  December 20th 2004

   MODULE GALAHAD_TRANS_double

     USE GALAHAD_SPACE_double

     IMPLICIT NONE     

     PRIVATE
     PUBLIC :: TRANS_initialize
     PUBLIC :: TRANS_terminate
     PUBLIC :: TRANS_default
     PUBLIC :: TRANS_trans
     PUBLIC :: TRANS_untrans
     PUBLIC :: TRANS_s_trans
     PUBLIC :: TRANS_s_untrans
     PUBLIC :: TRANS_v_trans
     PUBLIC :: TRANS_v_untrans
     PUBLIC :: TRANS_v_trans_inplace
     PUBLIC :: TRANS_v_untrans_inplace

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  =================================
!  The TRANS_trans_type derived type
!  =================================

     TYPE, PUBLIC :: TRANS_trans_type
       REAL ( KIND = wp ) :: f_scale, f_shift
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_scale
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_shift
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_scale
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_shift
     END TYPE TRANS_trans_type

!  =================================
!  The TRANS_data_type derived type
!  =================================

     TYPE, PUBLIC :: TRANS_data_type
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_orig
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V_orig
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P_orig
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_x
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_x
     END TYPE TRANS_data_type

!  =====================================
!  The TRANS_inform_type derived type
!  =====================================

     TYPE, PUBLIC :: TRANS_inform_type
       INTEGER :: status, alloc_status
       CHARACTER ( LEN = 80 ) :: bad_alloc
     END TYPE TRANS_inform_type

  CONTAINS

!  ------------------------------------------
!  Initialization and termination subroutines
!  ------------------------------------------

!  *-*-*-*-*-*-  T R A N S   TRANS_initialize  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE TRANS_initialize( inform )
     TYPE ( TRANS_inform_type ), INTENT( OUT ) :: inform
     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
     RETURN
     END SUBROUTINE TRANS_initialize

!  *-*-*-*-*-*-  T R A N S   TRANS_default  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE TRANS_default( n, m, trans, inform )
     INTEGER, INTENT( IN ) :: n, m
     TYPE ( TRANS_trans_type ), INTENT( INOUT ) :: trans
     TYPE ( TRANS_inform_type ), INTENT( OUT ) :: inform
     CHARACTER ( LEN = 80 ) :: point_name

!  Initialize scale and shift factors

     point_name = 'trans: X_scale'
     CALL SPACE_resize_array( n, trans%X_scale, inform%status,                 &
                               inform%alloc_status, exact_size = .TRUE.,       &
                               array_name = point_name,                        &
                               bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     point_name = 'trans: X_shift'
     CALL SPACE_resize_array( n, trans%X_shift, inform%status,                 &
                               inform%alloc_status, exact_size = .TRUE.,       &
                               array_name = point_name,                        &
                               bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     point_name = 'trans: C_scale'
     CALL SPACE_resize_array( m, trans%C_scale, inform%status,                 &
                               inform%alloc_status, exact_size = .TRUE.,       &
                               array_name = point_name,                        &
                               bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     point_name = 'trans: C_shift'
     CALL SPACE_resize_array( m, trans%C_shift, inform%status,                 &
                               inform%alloc_status, exact_size = .TRUE.,       &
                               array_name = point_name,                        &
                               bad_alloc = inform%bad_alloc )
     IF ( inform%status /= 0 ) RETURN

     trans%f_scale = 1.0_wp ; trans%f_shift = 0.0_wp
     trans%X_scale = 1.0_wp ; trans%X_shift = 0.0_wp
     trans%C_scale = 1.0_wp ; trans%C_shift = 0.0_wp

     RETURN
     END SUBROUTINE TRANS_default

!  *-*-*-*-*-*-  T R A N S   TRANS_terminate  S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE TRANS_terminate( trans, data, inform )
     TYPE ( TRANS_trans_type ), INTENT( INOUT ) :: trans
     TYPE ( TRANS_data_type ), INTENT( INOUT ) :: data
     TYPE ( TRANS_inform_type ), INTENT( OUT ) :: inform
     INTEGER :: status, alloc_status

     inform%status = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''

     CALL SPACE_dealloc_array( trans%X_scale, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: trans%X_scale'
     END IF

     CALL SPACE_dealloc_array( trans%X_shift, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: trans%X_shift'
     END IF

     CALL SPACE_dealloc_array( trans%C_scale, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: trans%C_scale'
     END IF

     CALL SPACE_dealloc_array( trans%C_shift, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: trans%C_shift'
     END IF

     CALL SPACE_dealloc_array( data%X_orig, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: data%X_orig'
     END IF

     CALL SPACE_dealloc_array( data%P_orig, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: data%P_orig'
     END IF

     CALL SPACE_dealloc_array( data%V_orig, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: data%V_orig'
     END IF

     CALL SPACE_dealloc_array( data%H_x, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: data%H_x'
     END IF

     CALL SPACE_dealloc_array( data%A_x, status, alloc_status )
     IF ( status /= 0 ) THEN
       inform%status = status ; inform%alloc_status = alloc_status 
       inform%bad_alloc = 'trans: data%A_x'
     END IF

     RETURN
     END SUBROUTINE TRANS_terminate

!  --------------------------------------------------------------------------
!  Subroutines to apply the transformation or its inverse to constant problem
!  data such as variable and function values, their bounds and dual variables
!  --------------------------------------------------------------------------

!  *-*-*-*-*-*-  T R A N S   TRANS_trans  S U B R O U T I N E  -*-*-*-*-*-*-*

     SUBROUTINE TRANS_trans( n, m, trans, infinity, f,                         &
                              X, X_l, X_u, Z_l, Z_u, V_n,                      &
                              C, C_l, C_u, Y_l, Y_u, V_m )
     INTEGER, INTENT( IN ) :: n, m 
     TYPE ( TRANS_trans_type ), INTENT( IN ) :: trans
     REAL ( KIND = wp ), INTENT( IN ) :: infinity
     REAL ( KIND = wp ), OPTIONAL :: f
     REAL ( KIND = wp ), DIMENSION( n ), OPTIONAL :: X, X_l, X_u, Z_l, Z_u, V_n
     REAL ( KIND = wp ), DIMENSION( m ), OPTIONAL :: C, C_l, C_u, Y_l, Y_u, V_m
     IF ( PRESENT( f ) )                                                       &
       CALL TRANS_s_trans_inplace( trans%f_scale, trans%f_shift, f )
     IF ( PRESENT( X ) )                                                       &
       CALL TRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X )
     IF ( PRESENT( X_l ) )                                                     &
       CALL TRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X_l,       &
                                    lower = .TRUE., infinity = infinity )
     IF ( PRESENT( X_u ) )                                                     &
       CALL TRANS_v_trans_inplace( n, trans%X_scale, trans%X_shift, X_u,       &
                                    lower = .FALSE., infinity = infinity )
     IF ( PRESENT( Z_l ) )                                                     &
       CALL TRANS_d_trans_inplace( n, trans%X_scale, trans%f_scale, Z_l )
     IF ( PRESENT( Z_u ) )                                                     &
       CALL TRANS_d_trans_inplace( n, trans%X_scale, trans%f_scale, Z_u )
     IF ( PRESENT( V_n ) )                                                     &
       CALL TRANS_d_trans_inplace( n, trans%X_scale, trans%f_scale, V_n )
     IF ( PRESENT( C ) )                                                       &
       CALL TRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C )
     IF ( PRESENT( C_l ) )                                                     &
       CALL TRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C_l,       &
                                    lower = .TRUE., infinity = infinity )
     IF ( PRESENT( C_u ) )                                                     &
       CALL TRANS_v_trans_inplace( m, trans%C_scale, trans%C_shift, C_u,       &
                                    lower = .FALSE., infinity = infinity )
     IF ( PRESENT( Y_l ) )                                                     &
       CALL TRANS_d_trans_inplace( m, trans%C_scale, trans%f_scale, Y_l )
     IF ( PRESENT( Y_u ) )                                                     &
       CALL TRANS_d_trans_inplace( m, trans%C_scale, trans%f_scale, Y_u )
     IF ( PRESENT( V_m ) )                                                     &
       CALL TRANS_d_trans_inplace( m, trans%C_scale, trans%f_scale, V_m )
     RETURN
     END SUBROUTINE TRANS_trans

!  *-*-*-*-*-*-  T R A N S   TRANS_untrans  S U B R O U T I N E  -*-*-*-*-*-*

     SUBROUTINE TRANS_untrans( n, m, trans, infinity,                          &
                                f, X, X_l, X_u, Z_l, Z_u, V_n,                 &
                                C, C_l, C_u, Y_l, Y_u, V_m )
     INTEGER, INTENT( IN ) :: n, m 
     TYPE ( TRANS_trans_type ), INTENT( IN ) :: trans
     REAL ( KIND = wp ), INTENT( IN ) :: infinity
     REAL ( KIND = wp ), OPTIONAL :: f
     REAL ( KIND = wp ), DIMENSION( n ), OPTIONAL :: X, X_l, X_u, Z_l, Z_u, V_n
     REAL ( KIND = wp ), DIMENSION( m ), OPTIONAL :: C, C_l, C_u, Y_l, Y_u, V_m
     IF ( PRESENT( f ) )                                                       &
       CALL TRANS_s_untrans_inplace( trans%f_scale, trans%f_shift, f )
     IF ( PRESENT( X ) )                                                       &
       CALL TRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X )
     IF ( PRESENT( X_l ) )                                                     &
       CALL TRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X_l,     &
                                      lower = .TRUE., infinity = infinity )
     IF ( PRESENT( X_u ) )                                                     &
       CALL TRANS_v_untrans_inplace( n, trans%X_scale, trans%X_shift, X_u,     &
                                      lower = .FALSE., infinity = infinity )
     IF ( PRESENT( Z_l ) )                                                     &
       CALL TRANS_d_untrans_inplace( n, trans%X_scale, trans%f_scale, Z_l )
     IF ( PRESENT( Z_u ) )                                                     &
       CALL TRANS_d_untrans_inplace( n, trans%X_scale, trans%f_scale, Z_u )
     IF ( PRESENT( V_n ) )                                                     &
       CALL TRANS_d_untrans_inplace( n, trans%X_scale, trans%f_scale, V_n )
     IF ( PRESENT( C ) )                                                       &
       CALL TRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C )
     IF ( PRESENT( C_l ) )                                                     &
       CALL TRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C_l,     &
                                      lower = .TRUE., infinity = infinity )
     IF ( PRESENT( C_u ) )                                                     &
       CALL TRANS_v_untrans_inplace( m, trans%C_scale, trans%C_shift, C_u,     &
                                    lower = .FALSE., infinity = infinity )
     IF ( PRESENT( Y_l ) )                                                     &
       CALL TRANS_d_untrans_inplace( m, trans%C_scale, trans%f_scale, Y_l )
     IF ( PRESENT( Y_u ) )                                                     &
       CALL TRANS_d_untrans_inplace( m, trans%C_scale, trans%f_scale, Y_u )
     IF ( PRESENT( V_m ) )                                                     &
       CALL TRANS_d_untrans_inplace( m, trans%C_scale, trans%f_scale, V_m )
     RETURN
     END SUBROUTINE TRANS_untrans

!  ---------------------------------------------------------
!  Subroutines to compute the transformation and its inverse
!  ---------------------------------------------------------

!  *-*-*-*-*-*-*-  T R A N S   TRANS_s_trans  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE TRANS_s_trans( scale, shift, s, s_trans )
     REAL ( KIND = wp ), INTENT( IN ) :: scale, shift, s
     REAL ( KIND = wp ), INTENT( OUT ) :: s_trans
     s_trans = ( s - shift ) / scale
     RETURN
     END SUBROUTINE TRANS_s_trans
     
!  *-*-*-*-*-*-  T R A N S   TRANS_s_untrans  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE TRANS_s_untrans( scale, shift, s_trans, s )
     REAL ( KIND = wp ), INTENT( IN ) :: scale, shift, s_trans
     REAL ( KIND = wp ), INTENT( OUT ) :: s
     s = scale * s_trans + shift
     RETURN
     END SUBROUTINE TRANS_s_untrans
     
!  *-*-*-*-*-*-*-  T R A N S   TRANS_v_trans  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE TRANS_v_trans( n, SCALE, SHIFT, V, V_trans )
     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SCALE, SHIFT, V
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: V_trans
     V_trans = ( V - SHIFT ) / SCALE
     RETURN
     END SUBROUTINE TRANS_v_trans
     
!  *-*-*-*-*-*-  T R A N S   TRANS_v_untrans  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE TRANS_v_untrans( n, SCALE, SHIFT, V_trans, V )
     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SCALE, SHIFT, V_trans
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: V
     V = SCALE * V_TRANS + SHIFT
     RETURN
     END SUBROUTINE TRANS_v_untrans
     
!  ------------------------------------------------------------------
!  Subroutines to compute the transformation and its inverse in place
!  ------------------------------------------------------------------

!  *-*-*-*- T R A N S   TRANS_s_trans_inplace  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE TRANS_s_trans_inplace( scale, shift, s )

!  shift and scale a scalar variable

     REAL ( KIND = wp ), INTENT( IN ) :: scale, shift
     REAL ( KIND = wp ), INTENT( INOUT ) :: s
     s = ( s - shift ) / scale
     RETURN
     END SUBROUTINE TRANS_s_trans_inplace
     
!  *-*-*- T R A N S   TRANS_s_untrans_inplace  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE TRANS_s_untrans_inplace( scale, shift, s )

!  unscale and unshift a scalar variable

     REAL ( KIND = wp ), INTENT( IN ) :: scale, shift
     REAL ( KIND = wp ), INTENT( INOUT ) :: s
     s = scale * s + shift
     RETURN
     END SUBROUTINE TRANS_s_untrans_inplace
     
!  *-*-*-*- T R A N S   TRANS_v_trans_inplace  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE TRANS_v_trans_inplace( n, SCALE, SHIFT, V, lower, infinity )

!  shift and scale a vector of variables

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SCALE, SHIFT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: V
     REAL ( KIND = wp ), OPTIONAL :: infinity
     LOGICAL, OPTIONAL :: lower
     IF ( PRESENT( lower ) .AND. PRESENT( infinity ) ) THEN
       IF ( lower ) THEN
         WHERE( V > - infinity ) V = ( V - SHIFT ) / SCALE
       ELSE
         WHERE( V < infinity ) V = ( V - SHIFT ) / SCALE
       END IF
     ELSE
       V = ( V - SHIFT ) / SCALE
     END IF
     RETURN
     END SUBROUTINE TRANS_v_trans_inplace
     
!  *-*-*- T R A N S   TRANS_v_untrans_inplace  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE TRANS_v_untrans_inplace( n, SCALE, SHIFT, V, lower, infinity )

!  unscale and unshift  a vector of variables

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SCALE, SHIFT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: V
     REAL ( KIND = wp ), OPTIONAL :: infinity
     LOGICAL, OPTIONAL :: lower
     IF ( PRESENT( lower ) .AND. PRESENT( infinity ) ) THEN
       IF ( lower ) THEN
         WHERE( V > - infinity ) V = SCALE * V + SHIFT
       ELSE
         WHERE( V < infinity ) V = SCALE * V + SHIFT
       END IF
     ELSE
       V = SCALE * V + SHIFT
     END IF
     RETURN
     END SUBROUTINE TRANS_v_untrans_inplace
     
!  *-*-*-*- T R A N S   TRANS_d_trans_inplace  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE TRANS_d_trans_inplace( n, SCALE, scale_f, V )

!  scale a vector of dual variables

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ) :: scale_f
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: V
     V = V * ( SCALE / scale_f )
     RETURN
     END SUBROUTINE TRANS_d_trans_inplace
     
!  *-*-*- T R A N S   TRANS_d_untrans_inplace  S U B R O U T I N E  -*-*-*-*

     SUBROUTINE TRANS_d_untrans_inplace( n, SCALE, scale_f, V )

!  unscale a vector of dual variables

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ) :: scale_f
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: SCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: V
     V = V * ( scale_f / SCALE )
     RETURN
     END SUBROUTINE TRANS_d_untrans_inplace
     
!  End of module GALAHAD_TRANS_double

   END MODULE GALAHAD_TRANS_double
      
