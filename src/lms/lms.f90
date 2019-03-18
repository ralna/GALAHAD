! THIS VERSION: GALAHAD 2.6 - 12/06/2014 AT 07:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L M S    M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started June 12th 2014
!   originally released GALAHAD Version 2.6. June 12th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LMS_double

!    ----------------------------------------------------------------------
!   |                                                                      |
!   | Construct and apply limited-memory secant Hessian approximations     |
!   |                                                                      |
!   | Based on Byrd, Nodedal and Schnabel, Math. Prog. 63:2 (1994) 129-156 |
!   |                                                                      |
!    ----------------------------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_LAPACK_interface, ONLY : POTRF, POTRS, SYTRF, SYTRS
      USE GALAHAD_BLAS_interface, ONLY : SWAP, GEMV, TRSV, GER
      USE GALAHAD_LMT_double, LMS_control_type => LMT_control_type,            &
                              LMS_time_type => LMT_time_type,                  &
                              LMS_inform_type => LMT_inform_type,              &
                              LMS_data_type => LMT_data_type

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LMS_control_type, LMS_time_type, LMS_inform_type,              &
                LMS_data_type, LMS_initialize, LMS_read_specfile, LMS_setup,   &
                LMS_form, LMS_form_shift, LMS_change_method, LMS_apply,        &
                LMS_apply_lbfgs, LMS_terminate

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: real_bytes = 8
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

   CONTAINS

!-*-*-*-*-*-   L M S  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LMS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LMS. This routine should be called before
!  LMS_setup
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LMS_control_type ), INTENT( OUT ) :: control
      TYPE ( LMS_inform_type ), INTENT( OUT ) :: inform

      data%any_method = .FALSE.
      data%restricted = 0

      inform%status = GALAHAD_ok

      RETURN

!  End of LMS_initialize

      END SUBROUTINE LMS_initialize

!-*-*-*-   L M S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LMS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LMS_initialize could (roughly)
!  have been set as:

! BEGIN LMS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  limited-memory-length                             10
!  limited-memory-method                             1
!  allow-any-method                                  F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END LMS SPECIFICATIONS

!  Dummy arguments

      TYPE ( LMS_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: memory_length = print_level + 1
      INTEGER, PARAMETER :: method = memory_length + 1
      INTEGER, PARAMETER :: any_method = method + 1
      INTEGER, PARAMETER :: space_critical = any_method + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LMS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( print_level )%keyword = 'print-level'
      spec( memory_length )%keyword = 'limited-memory-length'
      spec( method )%keyword = 'limited-memory-method'

!  Real key-words

!  Logical key-words

      spec( any_method )%keyword = 'allow-any-method'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

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

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( memory_length ),                       &
                                  control%memory_length,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( method ),                              &
                                  control%method,                              &
                                  control%error )

!  Set real values

!  Set logical values

      CALL SPECFILE_assign_value( spec( any_method ),                          &
                                  control%any_method,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

      RETURN

!  End of LMS_read_specfile

      END SUBROUTINE LMS_read_specfile

!-*-*-*-*-*-*-*-*-   L M S _ S E T U P   S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE LMS_setup( n, data, control, inform )

!  set up the data structures required to hold the limited memory secant
!  approximation B

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: nb, len2_qp, len2_qp_perm
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now
      CHARACTER ( LEN = 6 ) :: method
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER :: ILAENV
      EXTERNAL :: ILAENV

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' n must be +ve' )") prefix
        GO TO 900
      END IF

!  set initial values and method-specific workspace

      data%n = n ; data%m = MAX( control%memory_length, 1 ) ; data%latest = 0
      data%n_restriction = n
      data%length = 0 ; inform%length = 0 ; data%full = .FALSE.
      data%len_c = data%m ; len2_qp = 2 ; len2_qp_perm = 2
      data%lwork = 0

      SELECT CASE( control%method )
      CASE ( 2 )
        method = 'SR1   '
        nb = ILAENV( 1, 'DSYTRF', 'L', data%len_c, - 1, - 1, - 1 )
        data%lwork = data%len_c * nb
        len2_qp = 1 ; len2_qp_perm = 1
      CASE ( 3 )
        method = 'IBFGS '
      CASE ( 4 )
        method = 'ISBFGS'
        data%len_c = 2 * data%m ; len2_qp_perm = 1
        nb = ILAENV( 1, 'DSYTRF', 'L', data%len_c, - 1, - 1, - 1 )
        data%lwork = data%len_c * nb
      CASE DEFAULT
        method = 'BFGS  '
        data%lwork = data%n
      END SELECT
      data%method = method ; data%any_method = control%any_method

      IF ( data%any_method ) THEN
        method = 'ANY   '
        data%len_c = 2 * data%m
        nb = ILAENV( 1, 'DSYTRF', 'L', data%len_c, - 1, - 1, - 1 )
        data%lwork = MAX( data%len_c * nb, data%lwork )
      END IF

!  initialize workspace

      array_name = 'lms: data%ORDER'
      CALL SPACE_resize_array( data%m, data%ORDER,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lms: data%S'
      CALL SPACE_resize_array( n, data%m, data%S,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lms: data%Y'
      CALL SPACE_resize_array( n, data%m, data%Y,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lms: data%YTS'
      CALL SPACE_resize_array( data%m, data%m, data%YTS,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900
      data%YTS = zero

      SELECT CASE( TRIM( method ) )
      CASE ( 'BFGS', 'ISBFGS', 'SR1', 'ANY' )
        array_name = 'lms: data%STS'
        CALL SPACE_resize_array( data%m, data%m, data%STS,                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        data%STS = zero

        array_name = 'lms: data%PIVOTS'
        CALL SPACE_resize_array( data%len_c, data%PIVOTS,                      &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'lms: data%WORK'
        CALL SPACE_resize_array( data%lwork, data%WORK,                        &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END SELECT

      SELECT CASE( TRIM( method ) )
      CASE ( 'IBFGS', 'ISBFGS', 'ANY' )
        array_name = 'lms: data%YTY'
        CALL SPACE_resize_array( data%m, data%m, data%YTY,                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        data%YTY = zero
      END SELECT

      SELECT CASE( TRIM( method ) )
      CASE ( 'BFGS', 'ANY' )
        array_name = 'lms: data%L_scaled'
        CALL SPACE_resize_array( data%m, data%m, data%L_scaled,                &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END SELECT

      SELECT CASE( TRIM( method ) )
      CASE ( 'IBFGS', 'ANY' )
        array_name = 'lms: data%R'
        CALL SPACE_resize_array( data%m, data%m, data%R,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END SELECT

      array_name = 'lms: data%C'
      CALL SPACE_resize_array( data%len_c, data%len_c, data%C,                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900
      data%C = zero

      array_name = 'lms: data%QP'
      CALL SPACE_resize_array( data%m, len2_qp, data%QP,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lms: data%QP_perm'
      CALL SPACE_resize_array( data%len_c, len2_qp_perm, data%QP_perm,         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%setup = inform%time%setup + time_now - time_start
      inform%time%clock_setup                                                  &
        = inform%time%clock_setup + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%setup = inform%time%setup + REAL( time_now - time_start, wp )
      inform%time%clock_setup                                                  &
        = inform%time%clock_setup + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( A, '    ** Error return ', I0,                &
       &  ' from LMS_setup ' )" ) prefix, inform%status
      RETURN

!  end of subroutine LMS_setup

      END SUBROUTINE LMS_setup

!-*-*-*-*-*-*-*-   L M S _ U P D A T E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LMS_form( S, Y, delta, data, control, inform, lambda )

!  update the limited memory secant approximation B to account for
!  the incoming data (s,y,delta)

!  Dummy arguments

      REAL ( KIND = wp ), INTENT( IN ) :: delta
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( data%n ) :: S, Y
      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: lambda

!  Local variables

      INTEGER :: i, j, oj
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now
      REAL ( KIND = wp ) :: yts
      CHARACTER ( LEN = 6 ) :: method

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  check to see if the update is allowed

      inform%updates_skipped = .FALSE.
      IF ( delta <= zero ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' delta <= 0, skip update ' )" ) prefix
        inform%updates_skipped = .FALSE.
        inform%status = GALAHAD_error_restrictions
        GO TO 900
      END IF

      SELECT CASE( TRIM( data%method ) )
      CASE ( 'BFGS', 'IBFGS', 'ISBFGS'  )
        yts = DOT_PRODUCT( S, Y )
        IF ( yts <= zero ) THEN
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( A, ' y^T s <= 0, skip update ' )" ) prefix
          inform%updates_skipped = .FALSE.
          inform%status = GALAHAD_error_restrictions
          GO TO 900
        END IF
      END SELECT

!  if lambda is given, record it, otherwise set it zero

      IF ( PRESENT( lambda ) ) THEN
        IF ( lambda < zero ) THEN
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( A, ' lambda <= 0, skip shift ' )" ) prefix
          inform%status = GALAHAD_error_restrictions
          GO TO 900
        END IF
        data%lambda = lambda
      ELSE
        data%lambda = zero
      END IF

!  increase the number of vectors up to a limit of m

      data%latest = data%latest + 1

!  record that the limit has been reached

      IF ( data%latest > data%m ) THEN
        data%latest = 1
        data%full = .TRUE.
      END IF

!  once the limit has been reached, shuffle the order so that the vectors
!  appear in the order that they first arrived

      IF ( data%full ) THEN
        data%length = data%m
        DO i = 1, data%m - 1
          data%ORDER( i ) = data%ORDER( i + 1 )
        END DO
        data%ORDER( data%m ) = data%latest

!  if the limit has not yet been reached, order the latest last

      ELSE
        data%length = data%latest
        data%ORDER( data%latest ) = data%latest
      END IF
      inform%length = data%length

!  record the latest s, y and delta (overwriting the oldest if necessary)

      data%S( : data%n, data%latest ) = S( : data%n )
      data%Y( : data%n, data%latest ) = Y( : data%n )
      data%delta = delta

!  store the inner products of the latest s with the accumulated previous s

      SELECT CASE( TRIM( data%method ) )
      CASE ( 'BFGS', 'ISBFGS', 'SR1' )
        IF ( data%full ) THEN
          DO j = 1, data%length
            oj = data%ORDER( j )
            data%STS( data%latest, oj )                                        &
              = DOT_PRODUCT( S, data%S( : data%n, oj ) )
          END DO
        ELSE
          DO j = 1, data%length
            data%STS( data%latest, j ) = DOT_PRODUCT( S, data%S( : data%n, j ) )
          END DO
        END IF
      END SELECT

!  store the inner products of the latest y with the accumulated previous y

      SELECT CASE( TRIM( data%method ) )
      CASE ( 'IBFGS', 'ISBFGS' )
        IF ( data%full ) THEN
          DO j = 1, data%length
            oj = data%ORDER( j )
            data%YTY( data%latest, oj )                                        &
              = DOT_PRODUCT( Y, data%Y( : data%n, oj ) )
          END DO
        ELSE
          DO j = 1, data%length
            data%YTY( data%latest, j ) = DOT_PRODUCT( Y, data%Y( : data%n, j ) )
          END DO
        END IF

!  store gamma = 1 / delta

        data%gamma = one / delta
      END SELECT

      IF ( data%any_method ) THEN
        method = 'ANY   '
      ELSE
        method = data%method
      END IF

!  store the inner products of the latest s with the accumulated previous y
!  and the latest y with the accumulated previous s as required

      SELECT CASE( TRIM( method ) )

!  store the inner products of the latest s with the accumulated previous y

      CASE ( 'BFGS' )
        IF ( data%any_method ) THEN
          IF ( data%full ) THEN
            DO j = 1, data%length
              oj = data%ORDER( j )
              IF ( oj /= data%latest ) THEN
                data%YTS( data%latest, oj )                                    &
                 = DOT_PRODUCT( S, data%Y( : data%n, oj ) )
              ELSE
                data%YTS( data%latest, data%latest ) = yts
              END IF
            END DO
          ELSE
            DO j = 1, data%length
              IF ( j /= data%latest ) THEN
                data%YTS( data%latest, j )                                     &
                  = DOT_PRODUCT( S, data%Y( : data%n, j ) )
              ELSE
                data%YTS( data%latest, data%latest ) = yts
              END IF
            END DO
          END IF

!  store the inner products of the latest s with the accumulated  previous y,
!  scaling the s^T y products so that the diagonal ones are square rooted
!  and the off diagonal ones are divided by the sqaure roots of the previous
!  diagonals

        ELSE
          IF ( data%full ) THEN
            DO j = 1, data%length
              oj = data%ORDER( j )
              IF ( oj /= data%latest ) THEN
                data%YTS( data%latest, oj )                                    &
                 = DOT_PRODUCT( S, data%Y( : data%n, oj ) ) / data%YTS( oj, oj )
              ELSE
                data%YTS( data%latest, data%latest ) = SQRT( yts )
              END IF
            END DO
          ELSE
            DO j = 1, data%length
              IF ( j /= data%latest ) THEN
                data%YTS( data%latest, j )                                     &
                  = DOT_PRODUCT( S, data%Y( : data%n, j ) ) / data%YTS( j, j )
              ELSE
                data%YTS( data%latest, data%latest ) = SQRT( yts )
              END IF
            END DO
          END IF
        END IF

!  store the inner products of the latest y with the accumulated previous s

      CASE ( 'IBFGS' )
        IF ( data%full ) THEN
          DO j = 1, data%length
            oj = data%ORDER( j )
            IF ( oj /= data%latest ) THEN
              data%YTS( j, data%latest )                                       &
                = DOT_PRODUCT( Y, data%S( : data%n, oj ) )
            ELSE
              data%YTS( data%latest, data%latest ) = yts
            END IF
          END DO
        ELSE
          DO j = 1, data%length
            IF ( j /= data%latest ) THEN
              data%YTS( j, data%latest )                                       &
                = DOT_PRODUCT( Y, data%S( : data%n, j ) )
            ELSE
              data%YTS( data%latest, data%latest ) = yts
            END IF
          END DO
        END IF

!  store the inner products of the latest s with the accumulated previous y

      CASE ( 'SR1' )
        IF ( data%full ) THEN
          DO j = 1, data%length
            oj = data%ORDER( j )
            data%YTS( data%latest, oj )                                        &
              = DOT_PRODUCT( S, data%Y( : data%n, oj ) )
          END DO
        ELSE
          DO j = 1, data%length
            data%YTS( data%latest, j ) = DOT_PRODUCT( S, data%Y( : data%n, j ) )
          END DO
        END IF

!  store the inner products of the latest y with the accumulated previous s
!  and that of s with the accumulated previous y

      CASE DEFAULT
        IF ( data%full ) THEN
          DO j = 1, data%length
            oj = data%ORDER( j )
            IF ( oj /= data%latest ) THEN
              data%YTS( oj, data%latest )                                      &
                = DOT_PRODUCT( Y, data%S( : data%n, oj ) )
              data%YTS( data%latest, oj )                                      &
                = DOT_PRODUCT( S, data%Y( : data%n, oj ) )
            ELSE
              data%YTS( data%latest, data%latest ) = yts
            END IF
          END DO
        ELSE
          DO j = 1, data%length
            IF ( j /= data%latest ) THEN
              data%YTS( j, data%latest )                                       &
                = DOT_PRODUCT( Y, data%S( : data%n, j ) )
              data%YTS( data%latest, j )                                       &
                = DOT_PRODUCT( S, data%Y( : data%n, j ) )
            ELSE
              data%YTS( data%latest, data%latest ) = yts
            END IF
          END DO
        END IF
      END SELECT

!  assemble and, if necessary, compute the factorizations of the core
!  limited-memory matrices

      CALL LMS_factor( data, control, inform, lambda = lambda )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + REAL( time_now - time_start, wp )
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + REAL( time_now - time_start, wp )
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( A, '    ** Error return ', I0,                &
       &  ' from LMS_form ' )" ) prefix, inform%status
      RETURN

!  end of subroutine LMS_form

      END SUBROUTINE LMS_form

!-*-*-*-*-*-*-*-   L M S _ F A C T O R   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LMS_factor( data, control, inform, lambda )

!  assemble and, if necessary, compute the factorizations of the core
!  limited-memory matrices

!  Dummy arguments

      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: lambda

!  Local variables

      INTEGER :: i, j, k, oi, oj
      REAL ( KIND = wp ) :: val

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      SELECT CASE( TRIM( data%method ) )

!  ----------------------------------------------------------------------------
!  the limited-memory BFGS (L-BFGS) secant approximation:
!
!  B = delta I - [ Y : delta S ] [ -D       L^T    ]^{-1} [     Y^T   ],
!                                [ L   delta S^T S ]      [ delta S^T ]
!
!  where D = diag(s_i^T y_i) and L_ij = s_{i-1}^T y_{j-1} (i>j) or 0 otherwise
!  ----------------------------------------------------------------------------

      CASE ( 'BFGS' )

!  build the scaled matrix L with diagonal D^{1/2} and sub-diagonal L D^{-1/2}

        IF ( data%any_method ) THEN
          DO j = 1, data%length
            oj = data%ORDER( j )
            val = SQRT( data%YTS( oj, oj ) )
            data%L_scaled( j, j ) = val
            DO i = j + 1, data%length
              data%L_scaled( i, j ) = data%YTS( data%ORDER( i ), oj ) / val
            END DO
          END DO
        ELSE
          DO j = 1, data%length
            DO i = j, data%length
              data%L_scaled( i, j )                                            &
                = data%YTS( data%ORDER( i ), data%ORDER( j ) )
            END DO
          END DO
        END IF

!  construct the Schur complement C = delta S^T S + L D^{-1} L^T

        DO j = 1, data%length
          oj = data%ORDER( j )
          DO i = j, data%length
            oi = data%ORDER( i )
            val = data%delta * data%STS( oi, oj )
            DO k = 1, MIN( i - 1, j - 1 )
              val = val + data%L_scaled( i, k ) * data%L_scaled( j, k )
            END DO
            data%C( i, j ) = val
          END DO
        END DO

!  find the Cholesky factors of C

        i = 0
        CALL POTRF( 'L', data%length, data%C, data%len_c, i )
        IF ( i /= 0 ) THEN
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( A, ' Cholesky error ', I0 )" ) prefix, i
          inform%status = GALAHAD_error_factorization
          RETURN
        END IF

!  ----------------------------------------------------------------------------
!  the inverse limited-memory BFGS (IL-BFGS) secant approximation:
!
!  B^-1 = gamma I - [ gamma Y : S ] [ D + gamma Y^T Y  R^T ]^{-1} [ gamma Y^T ],
!                                   [       R           0  ]      [     S^T   ]
!
!  where gamma = 1/ delta, D = diag(s_i^T y_i) and
!  R_ij = s_{i-1}^T y_{j-1} (i<=j) or 0 otherwise
!  ----------------------------------------------------------------------------

      CASE ( 'IBFGS' )

!  build the matrix R

        DO j = 1, data%length
          data%R( j, j : data%length )                                         &
            = data%YTS( data%ORDER( j ), data%ORDER( j :  data%length ) )
        END DO

!  build the matrix C = D + gamma Y^T Y

        DO j = 1, data%length
          oj = data%ORDER( j )
          DO i = j, data%length
            oi = data%ORDER( i )
            val = data%gamma * data%YTY( oi, oj )
            IF ( i == j ) THEN
              data%C( i, i ) = val + data%R( i, i )
            ELSE
              data%C( i, j ) = val ; data%C( j, i ) = val
            END IF
          END DO
        END DO

!  ----------------------------------------------------------------------------
!  if required, form the inverse shifted limited-memory BFGS (IL-BFGS) secant
!  approximation
!  ----------------------------------------------------------------------------

      CASE ( 'ISBFGS' )
        IF ( PRESENT( lambda ) ) THEN
          CALL LMS_factor_ilbfgs( data, control, inform )
          IF ( inform%status /= GALAHAD_ok ) RETURN
          data%need_form_shift = .FALSE.
        ELSE
          data%need_form_shift = .TRUE.
        END IF

!  ----------------------------------------------------------------------------
!  the limited-memory symmetric rank-one (L-SR1) secant approximation:
!
!  B = delta I + [Y - delta S] [D + L + L^T - delta S^T S]^{-1} [Y - delta S]^T
!
!  where D = diag(s_i^T y_i) and  L_ij = s_{i-1}^T y_{j-1} (i>j) or 0 otherwise
!  ----------------------------------------------------------------------------

      CASE ( 'SR1' )

!  construct the matrix C = D + L + L^T - delta S^T S

        DO j = 1, data%length
          oj = data%ORDER( j )
          DO i = j, data%length
            oi = data%ORDER( i )
            data%C( i, j )                                                     &
              = - data%delta * data%STS( oi, oj ) + data%YTS( oi, oj )
          END DO
        END DO

!  find the Bunch-Kaufman factors of C

        i = 0
        CALL SYTRF( 'L', data%length, data%C, data%len_c, data%PIVOTS,         &
                    data%WORK, data%lwork, i )
        IF ( i == 0 ) THEN
          data%sr1_singular = .FALSE.
        ELSE IF ( i > 0 ) THEN
          data%sr1_singular = .TRUE.
          inform%updates_skipped = .TRUE.
        ELSE
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, "( A, ' Bunch-Kaufman error ', I0 )" )       &
              prefix, i
          inform%status = GALAHAD_error_factorization
          RETURN
        END IF
      END SELECT
      inform%status = GALAHAD_ok

      RETURN

!  end of subroutine LMS_factor

      END SUBROUTINE LMS_factor

!-*-*-*-*-   L M S _ F A C T O R _ I L B F G S   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LMS_factor_ilbfgs( data, control, inform )

!  ----------------------------------------------------------------------------
!  form the inverse shifted limited-memory BFGS (IL-BFGS) secant approximation:
!
!  (B+lambdaI)^-1 = 1/dl I - 1/dl^2 [ Y : delta S ] *
!
! [   D + 1/dl Y^T Y      -L^T + delta/dl Y^T S    ]^{-1} [    Y^T    ],
! [  -L + delta/dl S^T Y  -delta lambda / dl S^T S ]      [ delta S^T ]
!
!  where dl = delta + lambda, D = diag(s_i^T y_i) and
!  L_ij = s_{i-1}^T y_{j-1} (i>j) or 0 otherwise
!  ----------------------------------------------------------------------------

!  Dummy arguments

      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      INTEGER :: i, j, oi, oj
      REAL ( KIND = wp ) :: val, l_over_dpl, dl_over_dpl

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      data%delta_plus_lambda = data%delta + data%lambda
      data%one_over_dpl = one / data%delta_plus_lambda
      data%d_over_dpl = data%delta * data%one_over_dpl
      l_over_dpl = - data%lambda / data%delta_plus_lambda
      dl_over_dpl = - data%lambda * data%d_over_dpl

!  form the matrix C = [  D + 1/dl Y^T Y      -L^T + delta/dl Y^T S    ]
!                      [ -L + delta/dl S^T Y  -delta lambda / dl S^T S ]

      DO j = 1, data%length
        oj = data%ORDER( j )
        DO i = 1, j
          oi = data%ORDER( i )
          data%C( data%length + i, j ) = data%d_over_dpl * data%YTS( oi, oj )
        END DO
        DO i = j, data%length
          oi = data%ORDER( i )
          val = data%YTY( oi, oj ) / data%delta_plus_lambda
          IF ( i == j ) THEN
            data%C( i, i ) = val + data%YTS( oi, oi )
            data%C( data%length + i, data%length + i )                         &
              = dl_over_dpl * data%STS( oi, oi )
          ELSE
            data%C( i, j ) = val
            data%C( data%length + i, data%length + j )                         &
              = dl_over_dpl * data%STS( oi, oj )
            data%C( data%length + i, j ) = l_over_dpl * data%YTS( oi, oj )
          END IF
        END DO
      END DO

!  find the Bunch-Kaufman factors of C

      i = 0
      CALL SYTRF( 'L', 2 * data%length, data%C, data%len_c, data%PIVOTS,       &
                  data%WORK, data%lwork, i )
      IF ( i /= 0 ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' Bunch-Kaufman error ', I0 )" ) prefix, i
        inform%status = GALAHAD_error_factorization
        RETURN
      END IF

!  record that the shift update has happened

      data%need_form_shift = .FALSE.
      inform%status = GALAHAD_ok

      RETURN

!  end of subroutine LMS_factor_ilbfgs

      END SUBROUTINE LMS_factor_ilbfgs

!-*-*-*-*-*-*-*-   L M S _ U P D A T E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LMS_form_shift( lambda, data, control, inform )

!  update an inverse limited memory secant approximation B to account for
!  a positive shift lamda I

!  Dummy arguments

      REAL ( KIND = wp ), INTENT( IN ) :: lambda
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  check to see if the shift is allowed

      IF ( lambda < zero ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' lambda <= 0, skip shift' )" ) prefix
        inform%status = GALAHAD_error_restrictions
        GO TO 900
      END IF

      SELECT CASE( TRIM( data%method ) )
      CASE ( 'BFGS', 'SR1' )
        data%lambda = lambda
      CASE ( 'IBFGS' )
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' shift ignored as method = 3' )" ) prefix
        inform%status = GALAHAD_error_factorization
        GO TO 900
      CASE ( 'ISBFGS' )
        data%lambda = lambda
        CALL LMS_factor_ilbfgs( data, control, inform )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END SELECT

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + REAL( time_now - time_start, wp )
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + REAL( time_now - time_start, wp )
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( A, '    ** Error return ', I0,                &
       &  ' from LMS_form ' )" ) prefix, inform%status
      RETURN

!  end of subroutine LMS_form_shift

      END SUBROUTINE LMS_form_shift

!-*-*-*-*-   L M S _ C H A N G E _ M E T H O D   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LMS_change_method( data, control, inform, lambda )

!  change the limited memory secant approximation B used

!  Dummy arguments

      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: lambda

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      IF ( data%any_method .AND. data%length > 0 ) THEN
        CALL LMS_factor( data, control, inform, lambda = lambda )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      ELSE IF ( data%length < 0 .OR. .NOT. data%any_method ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' incorrect call order' )" ) prefix
        inform%status = GALAHAD_error_call_order
        GO TO 900
      END IF

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + REAL( time_now - time_start, wp )
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + REAL( time_now - time_start, wp )
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( A, '    ** Error return ', I0,                &
       &  ' from LMS_change_method ' )" ) prefix, inform%status
      RETURN

!  end of subroutine LMS_change_method

      END SUBROUTINE LMS_change_method

!-*-*-*-*-*-*-*-   L M S _ A P P L Y   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LMS_apply( V, U, data, control, inform )

!  obtain the product u = B v between the limited memory secant
!  approximation B and the vector v

!  Dummy arguments

      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( data%n ) :: V
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( data%n ) :: U
      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, oi
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  check that the limited-memory matrix has been initialized

      IF ( data%length < 0 ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' incorrect call order' )" ) prefix
        inform%status = GALAHAD_error_call_order
        GO TO 900

!  if the length is zero, use B = I

      ELSE IF ( data%length == 0 ) THEN
        U = V

!  otherwise, use the appropriate limited-memory B

      ELSE
       SELECT CASE( TRIM( data%method ) )

!  ----------------------------------------------------------------------------
!  apply the limited-memory BFGS (L-BFGS) secant approximation formula to v:
!
!   B v = delta v - [ Y : delta S ] [ -D      L^T     ]^{-1} [     Y^T   ] v.
!                                   [ L   delta S^T S ]      [ delta S^T ]
!  ----------------------------------------------------------------------------

        CASE ( 'BFGS' )
          data%n_restriction = data%n
          data%restricted = 0
          CALL LMS_apply_lbfgs( V, data, inform%status, RESULT = U )
          IF ( inform%status /= 0 ) THEN
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( A, ' Cholesky solve error ', I0 )" )    &
                prefix, inform%status
            inform%status = GALAHAD_error_factorization
            GO TO 900
          END IF

!  include any non-zero shift

          IF ( data%lambda > zero ) U = U + data%lambda * V

!  ----------------------------------------------------------------------------
!  applyv the inverse limited-memory BFGS (IL-BFGS) secant approximation
!  formula to v:
!
!  B^-1 v = gamma v - [ gamma Y : S ] [ D + gamma Y^T Y R^T ]^-1 [ gamma Y^T v ]
!                                     [       R          0  ]    [     S^T v   ]
! i.e.,
!
!  B^-1 v = gamma v - [ gamma Y : S R^-T ] [ 0        I       ] [ gamma Y^T v ]
!                                          [ I -D-gamma Y^T Y ] [  R^-1 S^T v ]
!
!    = gamma v - [ gamma Y : S R^-T ] [              R^-1 S^T v               ]
!                                     [ gammaY^Tv -(D+gamma Y^T Y) R^-1 S^T v ]
!
!  where gamma = 1/ delta, D = diag(s_i^T y_i) and
!  R_ij = s_{i-1}^T y_{j-1} (i<=j) or 0 otherwise
!  ----------------------------------------------------------------------------

        CASE ( 'IBFGS' )

!  form q <- gamma Y^Tv and p <- S^T v, where the vector (q,p) is stored
!  as the two columns of the matrix QP

          CALL GEMV( 'T', data%n, data%length, data%gamma, data%Y, data%n,     &
                      V, 1, zero, data%QP( : , 1 ), 1 )
          CALL GEMV( 'T', data%n, data%length, one, data%S, data%n,            &
                      V, 1, zero, data%QP( : , 2 ), 1 )

!  permute p and q

          DO i = 1, data%length
            oi = data%ORDER( i )
            data%QP_perm( i, 1 ) = data%QP( oi, 1 )
            data%QP_perm( i, 2 ) = data%QP( oi, 2 )
          END DO

!  solve R p <- p

          CALL TRSV( 'U', 'N', 'N',data%length, data%R, data%m,                &
                     data%QP_perm( : , 2 ), 1 )

!  form q <- q - C p

          CALL GEMV( 'N', data%length, data%length, - one, data%C,             &
                     data%len_c, data%QP_perm( : , 2 ), 1, one,                &
                     data%QP_perm( : , 1 ), 1 )

!  solve R^T q <- q

          CALL TRSV( 'U', 'T', 'N',data%length, data%R, data%m,                &
                     data%QP_perm( : , 1 ), 1 )

!  unpermute p and q

          DO i = 1, data%length
            oi = data%ORDER( i )
            data%QP( oi, 1 ) = data%QP_perm( i, 1 )
            data%QP( oi, 2 ) = data%QP_perm( i, 2 )
          END DO

!  apply u <- gamma ( v - Y p ) - S q

          U = V
          CALL GEMV( 'N', data%n, data%length, - one, data%Y, data%n,          &
                     data%QP( : , 2 ), 1, one, U, 1 )
          U = data%gamma * U
          CALL GEMV( 'N', data%n, data%length, - one, data%S, data%n,          &
                     data%QP( : , 1 ), 1, one, U, 1 )

!  ----------------------------------------------------------------------------
!  apply the inverse shifted limited-memory BFGS (IL-BFGS) secant approximation
!  formula to v:
!
!  (B+lambdaI)^-1 v = 1/dl v - 1/dl^2 [ Y : delta S ] *
!
!    [   D + 1/dl Y^T Y      -L^T + delta/dl Y^T S    ]^{-1} [    Y^T v    ],
!    [  -L + delta/dl S^T Y  -delta lambda / dl S^T S ]      [ delta S^T v ]
!
!  where dl = delta + lambda, D = diag(s_i^T y_i) and
!  L_ij = s_{i-1}^T y_{j-1} (i>j) or 0 otherwise
!  ----------------------------------------------------------------------------

        CASE ( 'ISBFGS' )
          IF ( data%need_form_shift ) THEN
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( A, ' lambda unset when method = 4' )" ) &
                prefix
            inform%status = GALAHAD_error_call_order
            GO TO 900
          END IF

!  form q <- Y^Tv / dl and p <- delta / dl S^T v, where the vector (q,p) is
!  stored as the two columns of the matrix QP

          CALL GEMV( 'T', data%n, data%length, data%one_over_dpl,              &
                      data%Y, data%n, V, 1, zero, data%QP( : , 1 ), 1 )
          CALL GEMV( 'T', data%n, data%length,  data%d_over_dpl,               &
                      data%S, data%n, V, 1, zero, data%QP( : , 2 ), 1 )

!  permute p and q

          DO i = 1, data%length
            oi = data%ORDER( i )
            data%QP_perm( i, 1 ) = data%QP( oi, 1 )
            data%QP_perm( data%length + i, 1 ) = data%QP( oi, 2 )
          END DO

!  solve C [ q ] <- [ q ] (using the Bunch-Kaufman factors of C)
!          [ p ]    [ p ]

          CALL SYTRS( 'L', 2 * data%length, 1, data%C, data%len_c,             &
                      data%PIVOTS, data%QP_PERM,  data%len_c, i )
          IF ( i /= 0 ) THEN
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( A, ' Bunch-Kaufman error ', I0 )" )     &
                prefix, i
            inform%status = GALAHAD_error_factorization
            GO TO 900
          END IF

!  unpermute p and q

          DO i = 1, data%length
            oi = data%ORDER( i )
            data%QP( oi, 1 ) = data%QP_perm( i, 1 )
            data%QP( oi, 2 ) = data%QP_perm( data%length + i, 1 )
          END DO

!  apply u <- 1/dl ( v - Y q - delta S p )

          U = V
          CALL GEMV( 'N', data%n, data%length, - one, data%Y, data%n,          &
                     data%QP( : , 1 ), 1, one, U, 1 )
          CALL GEMV( 'N', data%n, data%length, - data%delta, data%S, data%n,   &
                     data%QP( : , 2 ), 1, one, U, 1 )
          U = data%one_over_dpl * U

!  ----------------------------------------------------------------------------
!  apply the limited-memory symmetric rank 1 (L-SR1) secant approximation
!  formula to v:
!
!  B v = delta v + [ Y - delta S ] C^{-1} [ Y - delta S ]^T v.
!
!  where C = D + L + L^T - delta S^T S
!  ----------------------------------------------------------------------------

        CASE ( 'SR1' )

!  apply q <- Y^T v - delta S^T v, where the vector q is stored as the first
!  column of the matrix QP

          CALL GEMV( 'T', data%n, data%length, one, data%Y, data%n,            &
                      V, 1, zero, data%QP( : , 1 ), 1 )
          CALL GEMV( 'T', data%n, data%length, - data%delta, data%S, data%n,   &
                      V, 1, one, data%QP( : , 1 ), 1 )

!  permute q

          DO i = 1, data%length
            oi = data%ORDER( i )
            data%QP_perm( i, 1 ) = data%QP( oi, 1 )
          END DO

!  apply q <- C^{-1} q (using the Bunch-Kaufman factors of C)

!         IF ( .TRUE. ) THEN
          IF ( data%sr1_singular ) THEN
            CALL SYTRS_singular( 'L', data%length, 1, data%C, data%len_c,      &
                                 data%PIVOTS, data%QP_PERM, data%m, i )
          ELSE
            CALL SYTRS( 'L', data%length, 1, data%C, data%len_c,               &
                       data%PIVOTS, data%QP_PERM, data%m, i )
          END IF
          IF ( i /= 0 ) THEN
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( A, ' Bunch-Kaufman error ', I0 )" )     &
                prefix, i
            inform%status = GALAHAD_error_factorization
            GO TO 900
          END IF

!  unpermute p and q

          DO i = 1, data%length
            oi = data%ORDER( i )
            data%QP( oi, 1 ) = data%QP_perm( i, 1 )
          END DO

!  apply u <- delta [ v - S q ] + Y q

          U = V
          CALL GEMV( 'N', data%n, data%length, - one, data%S, data%n,          &
                     data%QP( : , 1 ), 1, one, U, 1 )
          U = data%delta * U
          CALL GEMV( 'N', data%n, data%length, one, data%Y, data%n,            &
                     data%QP( : , 1 ), 1, one, U, 1 )

!  include any non-zero shift

          IF ( data%lambda > zero ) U = U + data%lambda * V
        END SELECT
      END IF

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%apply = inform%time%apply + REAL( time_now - time_start, wp )
      inform%time%clock_apply                                                  &
        = inform%time%clock_apply + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%apply = inform%time%apply + REAL( time_now - time_start, wp )
      inform%time%clock_apply                                                  &
        = inform%time%clock_apply + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( A, '    ** Error return ', I0,                &
       &  ' from LMS_apply ' )" ) prefix, inform%status
      RETURN

!  end of subroutine LMS_apply

      END SUBROUTINE LMS_apply

!-*-*-*-*-*-   L M S _ A P P L Y  _ L B F G S   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LMS_apply_lbfgs( V, B, status, RESULT, ADD_TO_RESULT,         &
                                  SUBTRACT_FROM_RESULT )

!  obtain the product B v

!  if ADD_TO_RESULT is present replace ADD_TO_RESULT by ADD_TO_RESULT + B v,
!  otherwise if SUBTRACT_FROM_RESULT is present replace SUBTRACT_FROM_RESULT by
!  SUBTRACT_FROM_RESULT - B v, otherwise if RESULT is present, set RESULT = Bv,
!  and otherwise overwrite v by Bv.

!  Dummy arguments

      TYPE ( LMS_data_type ), INTENT( INOUT ) :: B
      REAL ( KIND = wp ), DIMENSION( B%n_restriction ) :: V
      INTEGER, INTENT( OUT ) :: status
      REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ),                             &
        DIMENSION( B%n_restriction ) :: RESULT
      REAL ( KIND = wp ), OPTIONAL, INTENT( INOUT ),                           &
        DIMENSION( B%n_restriction ) :: ADD_TO_RESULT
      REAL ( KIND = wp ), OPTIONAL, INTENT( INOUT ),                           &
        DIMENSION( B%n_restriction ) :: SUBTRACT_FROM_RESULT

!  Local variables

      INTEGER :: i, j, oi
      REAL ( KIND = wp ) :: val

      status = GALAHAD_ok

!  if the length is zero, use B = I

      IF ( B%length <= 0 ) THEN
        IF ( PRESENT( ADD_TO_RESULT ) ) THEN
          ADD_TO_RESULT = ADD_TO_RESULT + V
        ELSE IF ( PRESENT( SUBTRACT_FROM_RESULT ) ) THEN
          SUBTRACT_FROM_RESULT = SUBTRACT_FROM_RESULT - V
        ELSE IF ( PRESENT( RESULT ) ) THEN
          RESULT = V
        END IF
        RETURN
      END IF

!  ----------------------------------------------------------------------------
!  apply the limited-memory BFGS (L-BFGS) secant approximation formula to v:
!
!   B v = delta v - R [ Y : delta S ] [ -D    L^T     ]^{-1} [     Y^T   ] R^T v
!                                     [ L delta S^T S ]      [ delta S^T ]
!
!  Since
!
!   [ -D      L^T    ] = [   D^{1/2}   0 ] [ -I 0 ] [ D^{1/2} -D^{-1/2} L^T ]
!   [ L  delta S^T S ]   [ -L D^{-1/2} I ] [  0 C ] [    0          I       ]
!
!
!  with C = delta S^T S + L D^{-1} L^T,
!
!   B v =  delta v - R [ Y : delta S ] [ p ], where            (L-BFGS-1)
!                                      [ q ]
!
!   [ D^{1/2} -D^{-1/2} L^T ] [ p ] = [ p2 ],                  (L-BFGS-2)
!   [    0          I       ] [ q ]   [ q2 ]
!
!   [ -I 0 ] [ p2 ] = [ p1 ],                                  (L-BFGS-3)
!   [  0 C ] [ q2 ]   [ q1 ]
!
!   [   D^{1/2}   0 ] [ p1 ] = [ p0 ] and                      (L-BFGS-4)
!   [ -L D^{-1/2} I ] [ q1 ]   [ q0 ]
!
!   [ p0 ] = [     Y^T R^T v   ]                               (L-BFGS-5)
!   [ q0 ]   [ delta S^T R^T v ]
!
!  ----------------------------------------------------------------------------

!  the vector (q,p) is stored as the two columns of the matrix QP

!  unrestricted case (L-BFGS-5): [ p ] = [     Y^T v   ]
!                                [ q ]   [ delta S^T v ]

      IF ( B%restricted == 0 ) THEN
        CALL GEMV( 'T', B%n, B%length, B%delta, B%S, B%n,                      &
                    V, 1, zero, B%QP( : , 1 ), 1 )
        CALL GEMV( 'T', B%n, B%length, one, B%Y, B%n,                          &
                    V, 1, zero, B%QP( : , 2 ), 1 )

!  restricted case (L-BFGS-5): [ p ] = [     Y^T R^T v   ]
!                              [ q ]   [ delta S^T R^T v ]

      ELSE
        B%WORK( : B%n ) = zero
        DO i = 1, B%n_restriction
          j = B%RESTRICTION( i )
          IF ( j <= B%n ) B%WORK( j ) = V( i )
        END DO
!       B%WORK( B%RESTRICTION( : B%n_restriction ) ) = V
        CALL GEMV( 'T', B%n, B%length, B%delta, B%S, B%n,                      &
                    B%WORK, 1, zero, B%QP( : , 1 ), 1 )
        CALL GEMV( 'T', B%n, B%length, one, B%Y, B%n,                          &
                    B%WORK, 1, zero, B%QP( : , 2 ), 1 )
      END IF

!  permute p and q

      DO i = 1, B%length
        oi = B%ORDER( i )
        B%QP_perm( i, 1 ) = B%QP( oi, 1 )
        B%QP_perm( i, 2 ) = B%QP( oi, 2 )
      END DO

!  apply (L-BFGS-4) p -> D^{-1/2} p and q -> q + L D^{-1/2} p

      DO i = 1, B%length
        B%QP_PERM( i, 2 ) = B%QP_PERM( i, 2 ) / B%L_scaled( i, i )
      END DO

      DO i = 2, B%length
        val = B%QP_PERM( i, 1 )
        DO j = 1, i - 1
          val = val + B%L_scaled( i, j ) * B%QP_PERM( j, 2 )
        END DO
        B%QP_PERM( i, 1 ) = val
      END DO

!  apply (L-BFGS-3) q -> C^{-1} q (using the Cholesky factors of C)

      CALL POTRS( 'L', B%length, 1, B%C, B%len_c, B%QP_PERM, B%m, status )
      IF ( status /= GALAHAD_ok ) RETURN

!  apply (L-BFGS-2) p -> D^{-1/2} ( - p + D^{-1/2} L^T q )

      DO i = 1, B%length - 1
        val = - B%QP_PERM( i, 2 )
        DO j = i + 1, B%length
          val = val + B%L_scaled( j, i ) * B%QP_PERM( j, 1 )
        END DO
        B%QP_PERM( i, 2 ) = val
      END DO
      B%QP_PERM( B%length, 2 ) = - B%QP_PERM( B%length, 2 )

      DO i = 1, B%length
        B%QP_PERM( i, 2 ) = B%QP_PERM( i, 2 ) / B%L_scaled( i, i )
      END DO

!  unpermute p and q

      DO i = 1, B%length
        oi = B%ORDER( i )
        B%QP( oi, 1 ) = B%QP_perm( i, 1 )
        B%QP( oi, 2 ) = B%QP_perm( i, 2 )
      END DO

!  unrestricted case: apply (L-BFGS-1)

      IF ( B%restricted == 0 ) THEN

!  r <- r + delta v - Y p - delta S q

        IF ( PRESENT( ADD_TO_RESULT ) ) THEN
          ADD_TO_RESULT = ADD_TO_RESULT + B%delta * V
          CALL GEMV( 'N', B%n, B%length, - one, B%Y, B%n,                      &
                     B%QP( : , 2 ), 1, one, ADD_TO_RESULT, 1 )
          CALL GEMV( 'N', B%n, B%length, - B%delta, B%S, B%n,                  &
                     B%QP( : , 1 ), 1, one, ADD_TO_RESULT, 1 )

!  r <- r - delta v + Y p + delta S q

        ELSE IF ( PRESENT( SUBTRACT_FROM_RESULT ) ) THEN
          SUBTRACT_FROM_RESULT = SUBTRACT_FROM_RESULT - B%delta * V
          CALL GEMV( 'N', B%n, B%length, one, B%Y, B%n,                        &
                     B%QP( : , 2 ), 1, one, SUBTRACT_FROM_RESULT, 1 )
          CALL GEMV( 'N', B%n, B%length, B%delta, B%S, B%n,                    &
                     B%QP( : , 1 ), 1, one, SUBTRACT_FROM_RESULT, 1 )

!  r = delta ( v - S q ) - Y p

        ELSE IF ( PRESENT( RESULT ) ) THEN
          RESULT = V
          CALL GEMV( 'N', B%n, B%length, - one, B%S, B%n,                      &
                     B%QP( : , 1 ), 1, one, RESULT, 1 )
          RESULT = B%delta * RESULT
          CALL GEMV( 'N', B%n, B%length, - one, B%Y, B%n,                      &
                     B%QP( : , 2 ), 1, one, RESULT, 1 )

!  v <- delta ( v - S q ) - Y p

        ELSE
          CALL GEMV( 'N', B%n, B%length, - one, B%S, B%n,                      &
                     B%QP( : , 1 ), 1, one, V, 1 )
          V = B%delta * V
          CALL GEMV( 'N', B%n, B%length, - one, B%Y, B%n,                      &
                     B%QP( : , 2 ), 1, one, V, 1 )
        END IF

!  restricted case: apply (L-BFGS-1)

      ELSE

!  compute Y p + delta S q

        CALL GEMV( 'N', B%n, B%length, B%delta, B%S, B%n,                      &
                   B%QP( : , 1 ), 1, zero, B%WORK( : B%n ), 1 )
        CALL GEMV( 'N', B%n, B%length, one, B%Y, B%n,                          &
                   B%QP( : , 2 ), 1, one, B%WORK( : B%n ), 1 )

!  r <- r + delta v - R ( Y p + delta S q )

        IF ( PRESENT( ADD_TO_RESULT ) ) THEN
          DO i = 1, B%n_restriction
            j = B%RESTRICTION( i )
            IF ( j <= B%n ) ADD_TO_RESULT( i ) =                               &
              ADD_TO_RESULT( i ) + B%delta * V( i ) - B%WORK( j )
          END DO
!         ADD_TO_RESULT( : B%n_restriction ) =                                 &
!           ADD_TO_RESULT( : B%n_restriction )                                 &
!             + B%delta * V( : B%n_restriction )                               &
!             - B%WORK( B%RESTRICTION( : B%n_restriction ) )

!  r <- r - delta v + R ( Y p + delta S q )

        ELSE IF ( PRESENT( SUBTRACT_FROM_RESULT ) ) THEN
          DO i = 1, B%n_restriction
            j = B%RESTRICTION( i )
            IF ( j <= B%n ) SUBTRACT_FROM_RESULT( i ) =                        &
              SUBTRACT_FROM_RESULT( i ) - B%delta * V( i ) + B%WORK( j )
          END DO
!         SUBTRACT_FROM_RESULT( : B%n_restriction ) =                          &
!           SUBTRACT_FROM_RESULT( : B%n_restriction )                          &
!             - B%delta * V( : B%n_restriction )                               &
!             + B%WORK( B%RESTRICTION( : B%n_restriction ) )

!  r = delta v - R ( Y p + delta S q )

        ELSE IF ( PRESENT( RESULT ) ) THEN
          DO i = 1, B%n_restriction
            j = B%RESTRICTION( i )
            IF ( j <= B%n ) THEN
              RESULT( i ) = B%delta * V( i ) - B%WORK( j )
            ELSE
              RESULT( i ) = zero
            END IF
          END DO
!         RESULT( : B%n_restriction ) = B%delta * V( : B%n_restriction )       &
!           - B%WORK( B%RESTRICTION( : B%n_restriction ) )

!  v <- delta v - R ( Y p + delta S q )

        ELSE
          DO i = 1, B%n_restriction
            j = B%RESTRICTION( i )
            IF ( j <= B%n ) THEN
              V( i ) = B%delta * V( i ) - B%WORK( j )
            ELSE
              V( i ) = zero
            END IF
          END DO
!         V( : B%n_restriction ) = B%delta * V( : B%n_restriction )            &
!           - B%WORK( B%RESTRICTION( : B%n_restriction ) )
        END IF
      END IF

      RETURN

!  end of subroutine LMS_apply_lbfgs

      END SUBROUTINE LMS_apply_lbfgs

!-*-*-*-*-*-   S Y T R S _ s i n g u l a r   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE SYTRS_singular( uplo, n, nrhs, A, lda, IPIV, B, ldb, info )

!  based upon -- DSYTRS, but skips any diagonal solves with zero diagonals
!  -- LAPACK routine (version 3.3.1) --
!  -- LAPACK is a software package provided by Univ. of Tennessee,    --
!  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
!  -- April 2011                                                      --

!  Dummy arguments

      CHARACTER ( len = 1 ) :: uplo
      INTEGER :: info, lda, ldb, n, nrhs
      INTEGER :: IPIV( n )
      REAL ( KIND = wp ) :: A( lda, n ), B( ldb, nrhs )

!  Purpose
!  =======

!  DSYTRS solves a system of linear equations A*X = B with a real
!  symmetric matrix A using the factorization A = U*D*U**T or
!  A = L*D*L**T computed by DSYTRF.

!  Arguments
!  =========

!  uplo    (input) CHARACTER*1
!          Specifies whether the details of the factorization are stored
!          as an upper or lower triangular matrix.
!          = 'U':  Upper triangular, form is A = U*D*U**T;
!          = 'L':  Lower triangular, form is A = L*D*L**T.

!  n       (input) INTEGER
!          The order of the matrix A.  n >= 0.

!  nrhs    (input) INTEGER
!          The number of right hand sides, i.e., the number of columns
!          of the matrix B.  nrhs >= 0.

!  A       (input) DOUBLE PRECISION array, dimension (lda,n)
!          The block diagonal matrix D and the multipliers used to
!          obtain the factor U or L as computed by DSYTRF.

!  lda     (input) INTEGER
!          The leading dimension of the array A.  lda >= max(1,n).

!  IPIV    (input) INTEGER array, dimension (N)
!          Details of the interchanges and the block structure of D
!          as determined by DSYTRF.

!  B       (input/output) DOUBLE PRECISION array, dimension (ldb,nrhs)
!          On entry, the right hand side matrix B.
!          On exit, the solution matrix X.

!  ldb     (input) INTEGER
!          The leading dimension of the array B.  ldb >= max(1,n).

!  info    (output) INTEGER
!          = 0:  successful exit
!          < 0:  if info = -i, the i-th argument had an illegal value

!  =====================================================================

!  local variables

      LOGICAL :: upper
      INTEGER :: j, k, kp
      DOUBLE PRECISION   ak, akm1, akm1k, bk, bkm1, denom
      LOGICAL :: LSAME
      EXTERNAL :: LSAME
!     EXTERNAL :: DGEMV, DGER, DSCAL, XERBLA
      EXTERNAL :: XERBLA

      info = 0
      upper = LSAME( uplo, 'U' )
      IF ( .NOT. upper .AND. .NOT. LSAME( uplo, 'L' ) ) THEN
        info = - 1
      ELSE IF ( n < 0 ) THEN
        info = - 2
      ELSE IF ( nrhs < 0 ) THEN
        info = - 3
      ELSE IF ( lda < MAX( 1, n ) ) THEN
        info = - 5
      ELSE IF ( ldb < MAX( 1, n ) ) THEN
        info = - 8
      END IF
      IF ( info /= 0 ) THEN
        CALL XERBLA( 'DSYTRS', - info )
        RETURN
      END IF

!  quick return if possible

      IF ( n == 0 .OR. nrhs == 0 ) RETURN

      IF ( upper ) THEN

!  solve A*X = B, where A = U*D*U**T
!  =================================

!  first solve U*D*X = B, overwriting B with X
!  --------------------------------------------

!  k is the main loop index, decreasing from N to 1 in steps of
!  1 or 2, depending on the size of the diagonal blocks

        k = n
        DO

!  If k < 1, exit from loop

          IF ( k < 1 ) EXIT
          IF ( IPIV( k ) > 0 ) THEN

!  1 x 1 diagonal block: interchange rows k and IPIV(K)

            kp = IPIV( k )
            IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), 1, B( kp, : ), 1 )

!  multiply by inv(U(K)), where U(K) is the transformation stored in
!  column k of A

            CALL GER( k - 1, nrhs, - one, A( : , k ), 1, B( k, : ), 1,         &
                      B, ldb )

!  multiply by the inverse of the diagonal block

!           CALL DSCAL( nrhs, one / A( k, k ), B( k, 1 ), ldb )
            IF (  A( k, k ) /= zero ) THEN
              B( k, 1 : nrhs ) = B( k, 1 : nrhs ) / A( k, k )
            ELSE
              B( k, 1 : nrhs ) = zero
            END IF
            k = k - 1

!  2 x 2 diagonal block: Interchange rows k - 1 and - IPIV(K)

          ELSE
            kp = - IPIV( k )
            IF ( kp /= k - 1 ) CALL SWAP( nrhs, B( k - 1, : ), 1,              &
                                          B( kp, : ), 1 )

!  multiply by inv(U(K)), where U(K) is the transformation stored in
!  columns k - 1 and k of A

            CALL GER( k - 2, nrhs, - one, A( : , k ), 1, B( k, : ), 1,         &
                      B, ldb )
            CALL GER( k - 2, nrhs, - one, A( : , k - 1 ), 1, B( k - 1, : ),    &
                      1, B, ldb )

!  multiply by the inverse of the diagonal block

            akm1k = A( k - 1, k )
            akm1 = A( k - 1, k - 1 ) / akm1k
            ak = A( k, k ) / akm1k
            denom = akm1 * ak - one
            DO j = 1, nrhs
              bkm1 = B( k - 1, j ) / akm1k
              bk = B( k, j ) / akm1k
              B( k - 1, j ) = ( ak * bkm1 - bk ) / denom
              B( k, j ) = ( akm1 * bk - bkm1 ) / denom
            END DO
            k = k - 2
          END IF
        END DO

!  next solve U**T *X = B, overwriting B with X
!  --------------------------------------------

!         k is the main loop index, increasing from 1 to n in steps of
!         1 or 2, depending on the size of the diagonal blocks.

        k = 1
        DO
          IF ( k > n ) EXIT
          IF ( IPIV( k ) > 0 ) THEN

!  1 x 1 diagonal block: multiply by inv(U**T(K)), where U(K) is the
!  transformation stored in column k of A

            CALL GEMV( 'T', k - 1, nrhs, - one, B, ldb, A( :, k ),             &
                       1, one, B( k, : ), 1 )

!  interchange rows k and IPIV(K)

            kp = IPIV( k )
            IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), 1, B( kp, : ), 1 )
            k = k + 1

!   2 x 2 diagonal block: multiply by inv(U**T(k + 1)), where U(k + 1) is the
!   transformation stored in columns k and k + 1 of A

          ELSE
            CALL GEMV( 'T', k - 1, nrhs, - one, B, ldb, A( : , k ),            &
                       1, one, B( k, : ), 1 )
            CALL GEMV( 'T', k - 1, nrhs, - one, B, ldb,                        &
                       A( :, k + 1 ), 1, one, B( k + 1, : ), 1 )

!   interchange rows k and - IPIV(K)

            kp = - IPIV( k )
            IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), 1, B( kp, : ), 1 )
            k = k + 2
          END IF
        END DO

!  Solve A*X = B, where A = L*D*L**T
!  =================================

!  First solve L*D*X = B, overwriting B with X
!  --------------------------------------------
!  k is the main loop index, increasing from 1 to n in steps of
!  1 or 2, depending on the size of the diagonal blocks.

      ELSE
        k = 1
        DO
          IF ( k > n ) EXIT
          IF ( IPIV( k ) > 0 ) THEN

!   1 x 1 diagonal block: interchange rows k and IPIV(K)

            kp = IPIV( k )
            IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), 1, B( kp, : ), 1 )

!   multiply by inv(L(K)), where L(K) is the transformation stored in
!   column k of A

            IF ( k < n ) CALL GER( n - k, nrhs, - one, A( k + 1 : , k ), 1,    &
                                   B( k : , 1 ), 1, B( k + 1 : , : ), n - k )

!   Multiply by the inverse of the diagonal block

!           CALL DSCAL( nrhs, one / A( k, k ), B( k, 1 ), ldb )
            IF (  A( k, k ) /= zero ) THEN
              B( k, 1 : nrhs ) = B( k, 1 : nrhs ) / A( k, k )
            ELSE
              B( k, 1 : nrhs ) = zero
            END IF
            k = k + 1

!   2 x 2 diagonal block: interchange rows k + 1 and - IPIV(K)

          ELSE
            kp = - IPIV( k )
            IF ( kp /= k + 1 ) CALL SWAP( nrhs, B( k + 1, : ), 1,              &
                                          B( kp, : ), 1 )

!   multiply by inv(L(K)), where L(K) is the transformation stored in columns
!   k and k + 1 of A

            IF ( k < n - 1 ) THEN
              CALL GER( n - k - 1, nrhs, - one, A( k + 2 : , k ), 1,           &
                        B( k : , 1 ), 1, B( k + 2 : , : ), n - k - 1 )
              CALL GER( n - k - 1, nrhs, - one, A( k + 2 : , k + 1 ), 1,       &
                        B( k + 1 : , 1 ), 1, B( k + 2 : , : ), n - k - 1 )
            END IF

!   multiply by the inverse of the diagonal block

             akm1k = A( k + 1, k )
             akm1 = A( k, k ) / akm1k
             ak = A( k + 1, k + 1 ) / akm1k
             denom = akm1 * ak - one
             DO j = 1, nrhs
               bkm1 = B( k, j ) / akm1k
               bk = B( k + 1, j ) / akm1k
               B( k, j ) = ( ak * bkm1 - bk ) / denom
               B( k + 1, j ) = ( akm1 * bk - bkm1 ) / denom
             END DO
             k = k + 2
          END IF
        END DO

!  next solve L**T *X = B, overwriting B with X
!  --------------------------------------------

!  k is the main loop index, decreasing from n to 1 in steps of
!  1 or 2, depending on the size of the diagonal blocks.

        k = n
        DO
          IF ( k < 1 ) EXIT
          IF ( IPIV( k ) > 0 ) THEN

!  1 x 1 diagonal block: multiply by inv(L**T(K)), where L(K) is the
!  transformation stored in column k of A

            IF ( k < n ) CALL GEMV( 'T', n - k, nrhs, - one,                   &
               B( k + 1 : , : ), n - k, A( k + 1 : , k ), 1, one,              &
               B( k, : ), 1 )

!  interchange rows k and IPIV(K)

            kp = IPIV( k )
            IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), 1, B( kp, : ), 1 )
            k = k - 1

!  2 x 2 diagonal block: multiply by inv(L**T(k - 1)), where L(k - 1) is the
!  transformation stored in columns k - 1 and k of A

          ELSE
            IF ( k < n ) THEN
              CALL GEMV( 'T', n - k, nrhs, - one, B( k + 1 : , : ),            &
                         n - k, A( k + 1 : , k ), 1, one, B( k, : ), 1 )
              CALL GEMV( 'T', n - k, nrhs, - one, B( k + 1 : , : ), n - k,     &
                         A( k + 1 : , k - 1 ), 1, one, B( k - 1, : ), 1 )
            END IF

!  interchange rows k and - IPIV(K)

            kp = - IPIV( k )
            IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), 1, B( kp, : ), 1 )
            k = k - 2
          END IF
        END DO
      END IF

      RETURN

!  end of subroutine SYTRS_singular

      END SUBROUTINE SYTRS_singular

!-*-*-*-*-*-   L M S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LMS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see preface
!   control see preface
!   inform  see preface

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LMS_control_type ), INTENT( IN ) :: control
      TYPE ( LMS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  restore starting values

      data%any_method = .FALSE.
      data%restricted = 0

!  deallocate all remaining allocated arrays

      array_name = 'lms: data%ORDER'
      CALL SPACE_dealloc_array( data%ORDER,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%PIVOTS'
      CALL SPACE_dealloc_array( data%PIVOTS,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%RESTRICTION'
      CALL SPACE_dealloc_array( data%RESTRICTION,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%WORK'
      CALL SPACE_dealloc_array( data%WORK,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%YTS'
      CALL SPACE_dealloc_array( data%YTS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%STS'
      CALL SPACE_dealloc_array( data%STS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%YTY'
      CALL SPACE_dealloc_array( data%YTY,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%L_scaled'
      CALL SPACE_dealloc_array( data%L_scaled,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%QP'
      CALL SPACE_dealloc_array( data%QP,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lms: data%QP_perm'
      CALL SPACE_dealloc_array( data%QP_perm,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  end of subroutine LMS_terminate

      END SUBROUTINE LMS_terminate

!  end of module GALAHAD_LMS_double

    END MODULE GALAHAD_LMS_double
