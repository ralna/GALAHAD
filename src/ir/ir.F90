! THIS VERSION: GALAHAD 4.1 - 2023-05-29 AT 13:30 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ I R   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.3. June 30th 2008

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_IR_precision

!      --------------------------------------------------
!     |                                                  |
!     | Given a factorization of the symmetric matrix A, |
!     ! solve A x = b using iterative refinement         |
!     |                                                  |
!      --------------------------------------------------

      USE GALAHAD_KINDS_precision
      USE GALAHAD_SYMBOLS
      USE GALAHAD_MOP_precision, ONLY : MOP_Ax
      USE GALAHAD_SPACE_precision
      USE GALAHAD_SMT_precision
!     USE GALAHAD_QPT_precision, ONLY : QPT_keyword_H
      USE GALAHAD_SLS_precision
      USE GALAHAD_SPECFILE_precision
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: IR_initialize, IR_read_specfile, IR_terminate, IR_solve,       &
                IR_full_initialize, IR_full_terminate, IR_information,         &
                SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE IR_initialize
       MODULE PROCEDURE IR_initialize, IR_full_initialize
     END INTERFACE IR_initialize

     INTERFACE IR_terminate
       MODULE PROCEDURE IR_terminate, IR_full_terminate
     END INTERFACE IR_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: IR_control_type

!  unit for error messages

        INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

        INTEGER ( KIND = ip_ ) :: out = 6

!  controls level of diagnostic output

        INTEGER ( KIND = ip_ ) :: print_level = 0

!  maximum number of iterative refinements allowed

        INTEGER ( KIND = ip_ ) :: itref_max = 1

!  refinement will cease as soon as the residual ||Ax-b|| falls below
!    max( acceptable_residual_relative * ||b||, acceptable_residual_absolute )

        REAL ( KIND = rp_ ) :: acceptable_residual_relative  = ten * epsmch
        REAL ( KIND = rp_ ) :: acceptable_residual_absolute  = ten * epsmch

!  refinement will be judged to have failed if the residual
!   ||Ax-b|| >= required_residual_relative * ||b||
!  No checking if required_residual_relative < 0

!       REAL ( KIND = rp_ ) :: required_residual_relative = epsmch ** 0.2
        REAL ( KIND = rp_ ) :: required_residual_relative = ten ** ( - 3 )

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
      END TYPE IR_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: IR_inform_type

!   reported return status:
!      0 the solution has been found
!     -1 an array allocation has failed
!     -2 an array deallocation has failed

        INTEGER ( KIND = ip_ ) :: status = 0

!  STAT value after allocate failure

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  name of array which provoked an allocate failure

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  infinity norm of the initial residual

        REAL ( KIND = rp_ ) :: norm_initial_residual = HUGE( one )

!  infinity norm of the final residual

        REAL ( KIND = rp_ ) :: norm_final_residual = HUGE( one )

      END TYPE IR_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: IR_data_type
        INTEGER ( KIND = ip_ ) :: n = 0
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B, RES
      END TYPE IR_data_type

!  - - - - - - - - - - - -
!   full data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: IR_full_data_type
        LOGICAL :: f_indexing
        TYPE ( IR_data_type ) :: IR_data
        TYPE ( IR_control_type ) :: IR_control
        TYPE ( IR_inform_type ) :: IR_inform
      END TYPE IR_full_data_type

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

!  revise control parameters (not all compilers currently support fortran 2013)

      control%required_residual_relative = epsmch ** 0.2
      inform%status = GALAHAD_ok

!  Set initial data value

      data%n = 0

      RETURN

!  End of IR_initialize

      END SUBROUTINE IR_initialize

!- G A L A H A D -  I R _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE IR_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for IR controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( IR_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( IR_control_type ), INTENT( OUT ) :: control
     TYPE ( IR_inform_type ), INTENT( OUT ) :: inform

     CALL IR_initialize( data%ir_data, control, inform )

     RETURN

!  End of subroutine IR_full_initialize

     END SUBROUTINE IR_full_initialize

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
!   required-residual-relative                     1.0D-3
!   record-residuals                               F
!   space-critical                                 F
!   deallocate-error-fatal                         F
!   output-line-prefix                             ""
!  END IR SPECIFICATIONS

!  Dummy arguments

      TYPE ( IR_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: itref_max = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: acceptable_residual_relative        &
                                             = itref_max + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: acceptable_residual_absolute        &
                                             = acceptable_residual_relative + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: required_residual_relative          &
                                             = acceptable_residual_absolute + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: record_residuals                    &
                                              = required_residual_relative + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = record_residuals + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                              = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
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

      spec( acceptable_residual_relative )%keyword                             &
        = 'acceptable-residual-relative'
      spec( acceptable_residual_absolute )%keyword                             &
        = 'acceptable-residual-absolute'
      spec( required_residual_relative )%keyword                               &
        = 'required-residual-relative'

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
      REAL ( KIND = rp_ ), INTENT( INOUT ) , DIMENSION ( : ) :: X
      TYPE ( IR_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( IR_control_type ), INTENT( IN ) :: control
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( IR_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SLS_inform_type ), INTENT( INOUT ) :: SLS_inform

!  Local variables

      INTEGER ( KIND = ip_ ) :: iter, n
      REAL ( KIND = rp_ ) :: residual, residual_zero
      LOGICAL :: print_more
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
      print_more = control%print_level > 1 .AND. control%out > 0

!  No refinement is required
!  -------------------------

      IF ( control%itref_max <= 0 ) THEN

!  record the initial residuals if required

        IF ( control%record_residuals .OR. print_more ) THEN
          residual_zero = MAXVAL( ABS( data%B( : n ) ) )
          IF ( print_more )                                                    &
            WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" ) &
              prefix, residual_zero, zero
        END IF

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

!  record the final residuals r = b - A x if required

        IF ( control%record_residuals .OR. print_more .OR.                     &
             control%required_residual_relative >= zero ) THEN
          data%RES = data%B
          CALL MOP_Ax( - one, A, X, one, data%RES, symmetric = .TRUE. )
          residual = MAXVAL( ABS( data%RES( : n ) ) )

          IF ( print_more )                                                    &
            WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )")  &
              prefix, residual, MAXVAL( ABS( X( : n ) ) )

!  check that sufficient reduction occured

          IF ( residual > control%required_residual_relative * residual_zero ) &
            inform%status = GALAHAD_error_solve
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

        IF ( print_more )                                                      &
          WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )   &
            prefix, residual_zero, zero

        DO iter = 0, control%itref_max

!  Use factors of the A to solve for the correction

          CALL SLS_solve( A, data%RES( : n ), SLS_data, SLS_control,           &
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

!  Form the residuals r = b - A x

          IF ( iter < control%itref_max .OR. control%record_residuals .OR.     &
                 control%required_residual_relative >= zero ) THEN
            data%RES = data%B
            CALL MOP_Ax( - one, A, X, one, data%RES, symmetric = .TRUE. )
            residual = MAXVAL( ABS( data%RES( : n ) ) )

            IF ( print_more )                                                  &
              WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )")&
                prefix, residual, MAXVAL( ABS( X( : n ) ) )

            IF ( residual < MAX( control%acceptable_residual_absolute,         &
                   control%acceptable_residual_relative * residual_zero ) ) EXIT

!  check that sufficient reduction occured

            IF ( residual <=                                                   &
                control%required_residual_relative * residual_zero ) THEN
              inform%status = GALAHAD_ok
            ELSE
              inform%status = GALAHAD_error_solve
            END IF
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

! -  G A L A H A D -  I R _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE IR_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( IR_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( IR_control_type ), INTENT( IN ) :: control
     TYPE ( IR_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

     CALL IR_terminate( data%ir_data, control, inform )
     RETURN

!  End of subroutine IR_full_terminate

     END SUBROUTINE IR_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-  G A L A H A D -  I R _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE IR_information( data, inform, status )

!  return solver information during or after solution by IR
!  See IR_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( IR_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( IR_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%ir_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine IR_information

     END SUBROUTINE IR_information

!  End of module GALAHAD_IR

   END MODULE GALAHAD_IR_precision
