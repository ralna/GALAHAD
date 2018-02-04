! THIS VERSION: GALAHAD 2.3 - 30/06/2008 AT 14:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ I R   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.3. June 30th 2008

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_IR_double

!      --------------------------------------------------
!     |                                                  |
!     | Given a factorization of the symmetric matrix A, |
!     ! solve A x = b using iterative refinement         |
!     |                                                  |
!      --------------------------------------------------

      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double, ONLY : QPT_keyword_H
      USE GALAHAD_SLS_double
      USE GALAHAD_SPECFILE_double
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: IR_initialize, IR_read_specfile, IR_terminate,                 &
                IR_solve, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: IR_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  maximum number of iterative refinements allowed

        INTEGER :: itref_max = 1

!  refinement will cease as soon as the residual ||Ax-b|| falls below
!    max( acceptable_residual_relative * ||b||, acceptable_residual_absolute )

        REAL ( KIND = wp ) :: acceptable_residual_relative  = ten * epsmch
        REAL ( KIND = wp ) :: acceptable_residual_absolute  = ten * epsmch

!  record the initial and final residual

        LOGICAL :: record_residuals = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix  = '""                            '
      END TYPE

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: IR_data_type
        INTEGER :: n = 0
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B, RES
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: IR_inform_type

!   reported return status:
!      0 the solution has been found
!     -1 an array allocation has failed
!     -2 an array deallocation has failed

        INTEGER :: status = 0

!  STAT value after allocate failure

        INTEGER :: alloc_status = 0

!  name of array which provoked an allocate failure

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  infinity norm of the initial residual

        REAL ( KIND = wp ) :: norm_initial_residual = HUGE( one )

!  infinity norm of the final residual

        REAL ( KIND = wp ) :: norm_final_residual = HUGE( one )

      END TYPE

   CONTAINS

!-*-*-*-*-*-   I R  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE IR_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for IR. This routine should be called before
!  IR_solve
!
!  Arguments:
!  =========
!
!   data     private internal data
!   control  a structure containing control information. See IR_control_type
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( IR_data_type ), INTENT( INOUT ) :: data
      TYPE ( IR_control_type ), INTENT( OUT ) :: control
      TYPE ( IR_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set initial data value

      data%n = 0

      RETURN

!  End of IR_initialize

      END SUBROUTINE IR_initialize

!-*-*-*-   I R _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE IR_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by IR_initialize could (roughly)
!  have been set as:

!  BEGIN IR SPECIFICATIONS (DEFAULT)
!   error-printout-device                          6
!   printout-device                                6
!   print-level                                    0
!   maximum-refinements                            1
!   acceptable-residual-relative                   2.0D-15
!   acceptable-residual-absolute                   2.0D-15
!   record-residuals                               F
!   space-critical                                 F
!   deallocate-error-fatal                         F
!   output-line-prefix                             ""
!  END IR SPECIFICATIONS

!  Dummy arguments

      TYPE ( IR_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: itref_max = print_level + 1
      INTEGER, PARAMETER :: acceptable_residual_relative = itref_max + 1
      INTEGER, PARAMETER :: acceptable_residual_absolute                       &
                              = acceptable_residual_relative + 1
      INTEGER, PARAMETER :: record_residuals = acceptable_residual_absolute + 1
      INTEGER, PARAMETER :: space_critical = record_residuals + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 2 ), PARAMETER :: specname = 'IR'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( itref_max )%keyword = 'maximum-refinements'

!  Real key-words

      spec( acceptable_residual_relative )%keyword                            &
        = 'acceptable-residual-relative'
      spec( acceptable_residual_absolute )%keyword                            &
        = 'acceptable-residual-absolute'

!  Logical key-words

      spec( record_residuals )%keyword = 'record-residuals'
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

      CALL SPECFILE_assign_value( spec( error ), control%error,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ), control%out,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ), control%print_level,    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( itref_max ), control%itref_max,        &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( acceptable_residual_relative ),        &
                                  control%acceptable_residual_relative,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( acceptable_residual_absolute ),        &
                                  control%acceptable_residual_absolute,        &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( record_residuals ),                    &
                                  control%record_residuals,                    &
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

!  End of IR_read_specfile

      END SUBROUTINE IR_read_specfile

!-*-*-*-*-*-*-*-*-   I R _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE IR_solve( A, X, data, SLS_data, control, SLS_control,         &
                           inform, SLS_inform )

!  Given a symmetric matrix A and its SLS factors, solve the system
!  A x = b. b is input in X, and the solution x overwrites X

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      REAL ( KIND = wp ), INTENT( INOUT ) , DIMENSION ( : ) :: X
      TYPE ( IR_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( IR_control_type ), INTENT( IN ) :: control
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( IR_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SLS_inform_type ), INTENT( INOUT ) :: SLS_inform

!  Local variables

      INTEGER :: i, j, l, iter, n
      REAL ( KIND = wp ) :: residual, residual_zero, val
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Allocate space if necessary

      IF ( control%itref_max > 0 .OR. control%record_residuals ) THEN
        n = A%n
        IF ( data%n < n .OR. ( control%space_critical .AND. data%n /= n ) ) THEN
          array_name = 'ir: data%RES'
          CALL SPACE_resize_array( n, data%RES,                                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = .TRUE.,                                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'ir: data%B'
          CALL SPACE_resize_array( n, data%B,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = .TRUE.,                                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          data%n = n
        END IF
        data%B( : n ) = X( : n )
      END IF

!  No refinement is required
!  -------------------------

      IF ( control%itref_max <= 0 ) THEN

!  record the initial residuals if required

        IF ( control%record_residuals )                                        &
          residual_zero = MAXVAL( ABS( data%B( : n ) ) )

!  Solve A x = b

        CALL SLS_solve( A, X, SLS_data, SLS_control, SLS_inform )

!  Record exit status

        inform%status = SLS_inform%status
        IF ( inform%status /= GALAHAD_ok ) THEN
          IF ( inform%status == GALAHAD_error_allocate .OR.                    &
               inform%status == GALAHAD_error_deallocate ) THEN
            inform%alloc_status = SLS_inform%alloc_status
            inform%bad_alloc = SLS_inform%bad_alloc
          END IF
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' IR_solve exit status = ', I3 )" )      &
              prefix, inform%status
          RETURN
        END IF

!  record the final residuals if required

        IF ( control%record_residuals ) THEN
          data%RES = data%B
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l )
            val = A%val( l )
            data%RES( i ) = data%RES( i ) - val * X( j )
            IF ( i /= j )                                                      &
              data%RES( j ) = data%RES( j ) - val * X( i )
          END DO
          residual = MAXVAL( ABS( data%RES( : n ) ) )
        END IF

!  Iterative refinement is required
!  --------------------------------

      ELSE

!  Compute the original residual

!       WRITE( 25,"( ' sol ', /, ( 5ES24.16 ) )" ) SOL
        data%RES( : n ) = X( : n )
        X( : n ) = zero
        residual_zero = MAXVAL( ABS( data%B( : n ) ) )

!  Solve the system with iterative refinement

        IF ( control%print_level > 1 .AND. control%out > 0 )                   &
          WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )   &
            prefix, MAXVAL( ABS( data%RES( : n ) ) ), zero

        DO iter = 0, control%itref_max

!  Use factors of the A to solve for the correction

          CALL SLS_SOLVE( A, data%RES( : n ), SLS_data, SLS_control,           &
                          SLS_inform )
          inform%status = SLS_inform%status
          IF ( inform%status /= GALAHAD_ok ) THEN
            IF ( inform%status == GALAHAD_error_allocate .OR.                  &
                 inform%status == GALAHAD_error_deallocate ) THEN
              inform%alloc_status = SLS_inform%alloc_status
              inform%bad_alloc = SLS_inform%bad_alloc
            END IF
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' IR_solve exit status = ', I3 )" )    &
                prefix, inform%status
            RETURN
          END IF

!  Update the estimate of the solution

          X( : n ) = X( : n ) + data%RES( : n )

!  Form the residuals

          IF ( iter < control%itref_max .OR. control%record_residuals ) THEN
            data%RES = data%B
            DO l = 1, A%ne
              i = A%row( l ) ; j = A%col( l )
              val = A%val( l )
              data%RES( i ) = data%RES( i ) - val * X( j )
              IF ( i /= j )                                                    &
                data%RES( j ) = data%RES( j ) - val * X( i )
            END DO
            residual = MAXVAL( ABS( data%RES( : n ) ) )

            IF ( control%print_level > 1 .AND. control%out > 0 )               &
              WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )")&
                prefix, residual, MAXVAL( ABS( X( : n ) ) )

            IF ( residual < MAX( control%acceptable_residual_absolute,         &
                   control%acceptable_residual_relative * residual_zero ) ) EXIT
          END IF
        END DO
      END IF

      IF ( control%record_residuals ) THEN
        inform%norm_initial_residual = residual_zero
        inform%norm_final_residual = residual
      END IF

      RETURN

!  End of subroutine IR_solve

      END SUBROUTINE IR_solve

!-*-*-*-*-*-   I R _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE IR_terminate( data, control, inform )

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
!   data    see Subroutine IR_initialize
!   control see Subroutine IR_initialize
!   inform  see Subroutine IR_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( IR_control_type ), INTENT( IN ) :: control
      TYPE ( IR_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( IR_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'ir: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'ir: data%RES'
      CALL SPACE_dealloc_array( data%RES,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      RETURN

!  End of subroutine IR_terminate

      END SUBROUTINE IR_terminate

!  End of module GALAHAD_IR

   END MODULE GALAHAD_IR_double
