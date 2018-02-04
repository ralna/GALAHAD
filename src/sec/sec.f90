! THIS VERSION: GALAHAD 3.0 - 17/08/2017 AT 10:45 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ S E C   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.2. May 16th 2008

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SEC_double

!     ---------------------------
!    |                           |
!    | SEC: Build secant updates |
!    |                           |
!     ---------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_SPECFILE_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SEC_initialize, SEC_read_specfile, SEC_initial_approximation,   &
               SEC_bfgs_update, SEC_sr1_update

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SEC_control_type

!   error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, >= 1 warning messages

       INTEGER :: print_level = 0

!   the initial Hessian approximation will be h_initial I

       REAL ( KIND = wp ) :: h_initial = one

!   an update is skipped if the resulting matrix would have grown too much

       REAL ( KIND = wp ) :: update_skip_tol = epsmch

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE SEC_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SEC_inform_type

!  return status. See SEC_solve for details

       INTEGER :: status = 0

     END TYPE SEC_inform_type

   CONTAINS

!-*-*-  G A L A H A D -  S E C _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE SEC_initialize( control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SEC controls

!   Arguments:

!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SEC_control_type ), INTENT( OUT ) :: control
     TYPE ( SEC_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     control%update_skip_tol = SQRT( epsmch )
     inform%status = GALAHAD_ok

     RETURN

!  End of subroutine SEC_initialize

     END SUBROUTINE SEC_initialize

!-*-*-*-*-   S E C _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE SEC_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by SEC_initialize could (roughly)
!  have been set as:

! BEGIN SEC SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     0
!  initial-hessian-scaling                         1.0
!  update-skip-tolerance                           1.0D-8
!  output-line-prefix                              ""
! END SEC SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SEC_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: h_initial = print_level + 1
     INTEGER, PARAMETER :: update_skip_tol = h_initial + 1
     INTEGER, PARAMETER :: prefix = update_skip_tol + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'SEC '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'

!  Real key-words

     spec( h_initial )%keyword = 'initial-hessian-scaling'
     spec( update_skip_tol )%keyword = 'update-skip-tolerance'

!  Character key-words

     spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( h_initial ),                            &
                                 control%h_initial,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( update_skip_tol ),                      &
                                 control%update_skip_tol,                      &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

     RETURN

!  end of subroutine SEC_read_specfile

   END SUBROUTINE SEC_read_specfile

!-  G A L A H A D -  S E C _ i n i t i a l _ a p p r o i m a t i o n   S U B -

     SUBROUTINE SEC_initial_approximation( n, H, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*

!  initialize the matrix H as the n by n identiity matrix

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n * ( n + 1 ) / 2 ) :: H
     TYPE ( SEC_control_type ), INTENT( IN ) :: control
     TYPE ( SEC_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k

!  set H <- I

     k = 0
     DO i = 1, n
       DO j = 1, i
         H( k + 1 : k + i - 1 ) = zero
         H( k + i ) = control%h_initial
       END DO
       k = k + i
     END DO

     inform%status = GALAHAD_ok
     RETURN

!  end of subroutine SEC_initial_approximation

     END SUBROUTINE SEC_initial_approximation

!-*-  G A L A H A D -  S E C _ b f g s _ u p d a t e   S U B R O U T I N E -*-

     SUBROUTINE SEC_bfgs_update( n, S, Y, H, W, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*

!  perform the Broyden-Fletcher-Goldfarb-Shanno (BFGS) update

!    H -> H − H s s^T H / s^T H s + y y^T / yT s, where y = g^+ - g

!  The lower triangle of H is stored by rows in a rank-1 array

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S, Y
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n * ( n + 1 ) / 2 ) :: H
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: W
     TYPE ( SEC_control_type ), INTENT( IN ) :: control
     TYPE ( SEC_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k
     REAL ( KIND = wp ) :: sj, wj, yj, wts, yts

!  skip update if y^Ts is too small

     yts = DOT_PRODUCT( Y, S )
     IF ( yts <= control%update_skip_tol * DOT_PRODUCT( Y, Y ) ) THEN
       IF ( control%print_level > 0 ) WRITE( control%out,                      &
         "( ' BFGS update skipped, y^Ts too small' )" )
       inform%status = GALAHAD_warning_skip_update
       RETURN
     END IF

!  compute w = H s

     W = zero
     DO j = 1, n
       sj = S( j )
       k = j * ( j - 1 ) / 2
       W( : j ) = W( : j ) + sj * H( k + 1 : k + j )
       k = k + j
       DO i = j + 1, n
         k = k + i - 1
         W( i ) = W( i ) + sj * H( k )
       END DO
     END DO

!  update H

     wts = DOT_PRODUCT( W, S )
     IF ( wts > control%update_skip_tol * DOT_PRODUCT( S, S ) ) THEN
       k = 0
       DO j = 1, n
         yj = Y( j ) / yts ;  wj = W( j ) / wts
         H( k + 1 : k + j ) = H( k + 1 : k + j ) - wj * W( : j ) + yj * Y( : j )
         k = k + j
       END DO
       inform%status = GALAHAD_ok

!  skip update if curvature along s is too small

     ELSE
       IF ( control%print_level > 0 ) WRITE( control%out,                      &
         "( ' BFGS update skipped, y^Ts too small' )" )
       inform%status = GALAHAD_warning_skip_update
     END IF

!  check residual (turned off)

     IF ( .FALSE. ) THEN
       W = - Y
       DO j = 1, n
         sj = S( j )
         k = j * ( j - 1 ) / 2

!  add H s to w

         W( : j ) = W( : j ) + sj * H( k + 1 : k + j )
         k = k + j
         DO i = j + 1, n
           k = k + i - 1
           W( i ) = W( i ) + sj * H( k )
         END DO
       END DO
       IF ( control%print_level > 0 ) WRITE( control%out,                      &
         "( ' BFGS residual = ', ES12.4 )" ) SQRT( DOT_PRODUCT( W, W ) )
     END IF

     RETURN

!  End of subroutine SEC_bfgs_update

     END SUBROUTINE SEC_bfgs_update

!-*-*-  G A L A H A D -  S E C _ s r 1 _ u p d a t e   S U B R O U T I N E -*-*-

     SUBROUTINE SEC_sr1_update( n, S, Y, H, W, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*

!  perform the Symmetric Rank-one (SR1) update

!    H -> H - w w^T / w^T s, where w = H s - y and y = g^+ - g

!  The lower triangle of H is stored by rows in a rank-1 array

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S, Y
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n * ( n + 1 ) / 2 ) :: H
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: W
     TYPE ( SEC_control_type ), INTENT( IN ) :: control
     TYPE ( SEC_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k
     REAL ( KIND = wp ) :: sj, wj, wts, wtw

!  initiaize w = - y

     W = - Y

!  add H s to w

     DO j = 1, n
       sj = S( j )
       k = j * ( j - 1 ) / 2
       W( : j ) = W( : j ) + sj * H( k + 1 : k + j )
       k = k + j
       DO i = j + 1, n
         k = k + i - 1
         W( i ) = W( i ) + sj * H( k )
       END DO
     END DO

!  update H

     wts = DOT_PRODUCT( W, S ) ; wtw = DOT_PRODUCT( W, W )
     IF ( ABS( wts ) >  control%update_skip_tol * wtw ) THEN
       k = 0
       DO j = 1, n
         wj = W( j ) / wts
         H( k + 1 : k + j ) = H( k + 1 : k + j ) - wj * W( : j )
         k = k + j
       END DO
       inform%status = GALAHAD_ok

!  skip update if w^Ts is too small

     ELSE
       IF ( control%print_level > 0 ) WRITE( control%out,                      &
         "( ' SR1 update skipped, w^Ts too small' )" )
       inform%status = GALAHAD_warning_skip_update
     END IF

!  check residual (turned off)

     IF ( .FALSE. ) THEN
       W = - Y
       DO j = 1, n
         sj = S( j )
         k = j * ( j - 1 ) / 2

!  add H s to w

         W( : j ) = W( : j ) + sj * H( k + 1 : k + j )
         k = k + j
         DO i = j + 1, n
           k = k + i - 1
           W( i ) = W( i ) + sj * H( k )
         END DO
       END DO
       IF ( control%print_level > 0 ) WRITE( control%out,                      &
         "( ' SR1 residual = ', ES12.4 )" ) SQRT( DOT_PRODUCT( W, W ) )
     END IF
     RETURN

!  End of subroutine SEC_sr1_update

     END SUBROUTINE SEC_sr1_update

!  End of module GALAHAD_SEC

   END MODULE GALAHAD_SEC_double
