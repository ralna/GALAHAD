! THIS VERSION: GALAHAD 5.3 - 2025-10-23 AT 10:10 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ S S L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Fowkes/Gould/Montoison/Orban, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started September 10th 2024

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SSLS_precision

!      ----------------------------------------------
!     |                                              |
!     | Given matrices A and (symmetric) H and C,    |
!     | solve the symmetric structured linear system |
!     |                                              |
!     |    ( H   A^T ) ( x ) = ( a )                 |
!     |    ( A   -C  ) ( y )   ( b )                 |
!     |                                              |
!      ----------------------------------------------

      USE GALAHAD_KINDS_precision
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_precision
      USE GALAHAD_SMT_precision
      USE GALAHAD_QPT_precision, ONLY: QPT_keyword_H, QPT_keyword_A
      USE GALAHAD_SLS_precision
      USE GALAHAD_SPECFILE_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SSLS_initialize, SSLS_read_specfile, SSLS_analyse,             &
                SSLS_factorize, SSLS_solve, SSLS_terminate,                    &
                SSLS_full_initialize, SSLS_full_terminate,                     &
                SSLS_import, SSLS_factorize_matrix, SSLS_solve_system,         &
                SSLS_reset_control, SSLS_information,                          &
                SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE SSLS_initialize
       MODULE PROCEDURE SSLS_initialize, SSLS_full_initialize
     END INTERFACE SSLS_initialize

     INTERFACE SSLS_terminate
       MODULE PROCEDURE SSLS_terminate, SSLS_full_terminate
     END INTERFACE SSLS_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: SSLS_control_type

!  unit for error messages

        INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

        INTEGER ( KIND = ip_ ) :: out = 6

!  controls level of diagnostic output

        INTEGER ( KIND = ip_ ) :: print_level = 0

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  symmetric indefinite linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = "ssids" //         &
                                                             REPEAT( ' ', 25 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

      END TYPE SSLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: SSLS_time_type

!  total cpu time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0

!  cpu time spent analysing K

        REAL ( KIND = rp_ ) :: analyse = 0.0

!  cpu time spent factorizing K

        REAL ( KIND = rp_ ) :: factorize = 0.0

!  cpu time spent solving linear systems inolving K

        REAL ( KIND = rp_ ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  clock time spent analysing K

        REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  clock time spent factorizing K

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving K

        REAL ( KIND = rp_ ) :: clock_solve = 0.0

      END TYPE SSLS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: SSLS_inform_type

!  return status. See SSLS_form_and_factorize for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long_ ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long_ ) :: factorization_real = - 1

!  the computed rank of K

        INTEGER ( KIND = ip_ ) :: rank = - 1

!  is the matrix K rank defficient?

        LOGICAL :: rank_def = .FALSE.

!  timings (see above)

        TYPE ( SSLS_time_type ) :: time

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

      END TYPE SSLS_inform_type

!  ..................
!   data derived type
!  ..................

      TYPE, PUBLIC :: SSLS_data_type
!       PRIVATE
        INTEGER ( KIND = ip_ ) :: a_ne, h_ne, c_ne, k_c
        TYPE ( SMT_type ) :: K
        TYPE ( SLS_data_type ) :: SLS_data
      END TYPE SSLS_data_type

!  ====================================
!  The SSLS_full_data_type derived type
!  ====================================

      TYPE, PUBLIC :: SSLS_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( SSLS_data_type ) :: SSLS_data
        TYPE ( SSLS_control_type ) :: SSLS_control
        TYPE ( SSLS_inform_type ) :: SSLS_inform
        TYPE ( SMT_type ) :: H, A, C
      END TYPE SSLS_full_data_type

   CONTAINS

!-*-*-*-*-*-   S S L S  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE SSLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for SSLS. This routine should be called before
!  SSLS_solve
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

      TYPE ( SSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SSLS_control_type ), INTENT( OUT ) :: control
      TYPE ( SSLS_inform_type ), INTENT( OUT ) :: inform

!  initalize SLS components

      CALL SLS_initialize( control%symmetric_linear_solver,                    &
                           data%sls_data, control%SLS_control,                 &
                           inform%SLS_inform, check = .TRUE. )
      control%symmetric_linear_solver = inform%SLS_inform%solver

      inform%status = GALAHAD_ok
      RETURN

!  End of SSLS_initialize

      END SUBROUTINE SSLS_initialize

!- G A L A H A D -  S S L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE SSLS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SSLS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SSLS_control_type ), INTENT( OUT ) :: control
     TYPE ( SSLS_inform_type ), INTENT( OUT ) :: inform

     CALL SSLS_initialize( data%ssls_data, control, inform )

     RETURN

!  End of subroutine SSLS_full_initialize

     END SUBROUTINE SSLS_full_initialize

!-*-*-*-   S S L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE SSLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SSLS_initialize could (roughly)
!  have been set as:

! BEGIN SSLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  ssids
!  output-line-prefix                                ""
! END SSLS SPECIFICATIONS

!  Dummy arguments

      TYPE ( SSLS_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: symmetric_linear_solver             &
                                             = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = symmetric_linear_solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'SSLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'

!  Real key-words

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
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
!  Set real values

!  Set logical values

      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),             &
                                  control%symmetric_linear_solver,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read the specfile for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

      END SUBROUTINE SSLS_read_specfile

!-*-*-*-*-*-*-   S S L S _ A N A L Y S E   S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SSLS_analyse( n, m, H, A, C, data, control, inform )

!  build the structure and analyse the matrix

!     K = ( H   A^T )
!         ( A   -C  )

!  dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SSLS_control_type ), INTENT( IN ) :: control
      TYPE ( SSLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: i, j, l
      REAL :: time_start, time_now
      REAL ( KIND = rp_ ) :: clock_start, clock_now
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Check for faulty dimensions

      IF ( n <= 0 .OR. m < 0 .OR.                                              &
           .NOT. QPT_keyword_H( H%type ) .OR.                                  &
           .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2000 ) prefix, inform%status
        GO TO 900
      END IF

!  find the number of nonzeros in A

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        data%a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        data%a_ne = A%ptr( m + 1 ) - 1
      ELSE
        data%a_ne = A%ne
      END IF

!  find the number of nonzeros in the lower triangle of J

      IF ( SMT_get( H%type ) == 'ZERO' .OR.                                    &
           SMT_get( H%type ) == 'NONE' ) THEN
        data%h_ne = 0
      ELSE IF ( SMT_get( H%type ) == 'DIAGONAL' .OR.                           &
           SMT_get( H%type ) == 'SCALED_IDENTITY' .OR.                         &
           SMT_get( H%type ) == 'IDENTITY' ) THEN
        data%h_ne = n
      ELSE IF ( SMT_get( H%type ) == 'DENSE' ) THEN
        data%h_ne = ( n * ( n + 1 ) ) / 2
      ELSE IF ( SMT_get( H%type ) == 'SPARSE_BY_ROWS' ) THEN
        data%h_ne = H%ptr( n + 1 ) - 1
      ELSE
        data%h_ne = H%ne
      END IF
      data%k_c = data%a_ne + data%h_ne

!  find the number of nonzeros in c

      IF ( SMT_get( C%type ) == 'ZERO' .OR.                                    &
           SMT_get( C%type ) == 'NONE' ) THEN
        data%c_ne = 0
      ELSE IF ( SMT_get( C%type ) == 'DIAGONAL' .OR.                           &
                SMT_get( C%type ) == 'SCALED_IDENTITY' .OR.                    &
                SMT_get( C%type ) == 'IDENTITY' ) THEN
        data%c_ne = m
      ELSE IF ( SMT_get( C%type ) == 'DENSE' ) THEN
        data%c_ne = ( m * ( m + 1 ) ) / 2
      ELSE IF ( SMT_get( C%type ) == 'SPARSE_BY_ROWS' ) THEN
        data%c_ne = C%ptr( m + 1 ) - 1
      ELSE
        data%c_ne = C%ne
      END IF

      data%K%n = n + m ; data%K%ne = data%h_ne + data%a_ne + data%c_ne

!  allocate sufficient space to hold K

      CALL SMT_put( data%K%type, 'COORDINATE', i )

      array_name = 'ssls: data%K%row'
      CALL SPACE_resize_array( data%K%ne, data%K%row,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ssls: data%K%col'
      CALL SPACE_resize_array( data%K%ne, data%K%col,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ssls: data%K%val'
      CALL SPACE_resize_array( data%K%ne, data%K%val,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  start by storing A

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        l = 0
        DO i = 1, m
          DO j = 1, n
            l = l + 1
            data%K%row( l ) = i + n
            data%K%col( l ) = j
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            data%K%row( l ) = i + n
            data%K%col( l ) = A%col( l )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        data%K%row( : data%a_ne ) = A%row( : data%a_ne ) + n
        data%K%col( : data%a_ne ) = A%col( : data%a_ne )
      END SELECT
      
!  store H

      SELECT CASE ( SMT_get( H%type ) )
      CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
        DO i = 1, n
          data%K%row( data%a_ne + i ) = i
          data%K%col( data%a_ne + i ) = i
        END DO
      CASE ( 'DENSE' )
        l = 0
        DO i = 1, n
          DO j = 1, i
            l = l + 1
            data%K%row( data%a_ne + l ) = i
            data%K%col( data%a_ne + l ) = j
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, n
          DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
            data%K%row( data%a_ne + l ) = i
            data%K%col( data%a_ne + l ) = H%col( l )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        data%K%row( data%a_ne + 1 : data%k_c ) = H%row( : data%h_ne )
        data%K%col( data%a_ne + 1 : data%k_c ) = H%col( : data%h_ne )
      END SELECT

!  Finally store -C in K

      SELECT CASE ( SMT_get( C%type ) )
      CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
        DO i = 1, m
          data%K%row( data%k_c + i ) = n + i
          data%K%col( data%k_c + i ) = n + i
        END DO
      CASE ( 'DENSE' )
        l = 0
        DO i = 1, m
          DO j = 1, i
            l = l + 1
            data%K%row( data%k_c + l ) = n + i
            data%K%col( data%k_c + l ) = n + j
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = C%ptr( i ), C%ptr( i + 1 ) - 1
            data%K%row( data%k_c + l ) = n + i
            data%K%col( data%k_c + l ) = n + C%col( l )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        data%K%row( data%k_c + 1 : data%K%ne ) = n + C%row( : data%c_ne )
        data%K%col( data%k_c + 1 : data%K%ne ) = n + C%col( : data%c_ne )
      END SELECT

!  record the linear solver used

      CALL SLS_initialize_solver( control%symmetric_linear_solver,             &
                                  data%SLS_data, control%SLS_control%error,    &
                                  inform%SLS_inform, check = .TRUE. )
      IF ( inform%SLS_inform%status < 0 ) THEN
        inform%status = inform%SLS_inform%status ; RETURN ; END IF

!  analyse the structure of K

      CALL SLS_analyse( data%K, data%SLS_data, control%SLS_control,            &
                        inform%SLS_inform )
      IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
        inform%status = GALAHAD_ok
      ELSE
        inform%status = GALAHAD_error_analysis
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2000 ) prefix, inform%status
      END IF

!  record times

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      IF ( control%print_level >= 1 ) WRITE( control%out,                      &
         "( A, ' time to analyse K ', F6.2 )") prefix, time_now - time_start
      inform%time%factorize = inform%time%factorize + time_now - time_start
      inform%time%clock_analyse =                                              &
        inform%time%clock_analyse + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      RETURN

!  Non-executable statements

 2000 FORMAT( ' ', /, A, '   **  Error return ', I0,' from SSLS ' )

!  end of subroutine SSLS_analyse

      END SUBROUTINE SSLS_analyse

!-*-*-*-*-*-*-   S S L S _ F A C T O I Z E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE SSLS_factorize( n, m, H, A, C, data, control, inform )

!  assemble the real data and factorize the matrix

!     K = ( H   A^T )
!         ( A   -C  )

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SSLS_control_type ), INTENT( IN ) :: control
      TYPE ( SSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = rp_ ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  insert the values of A into K

      data%K%val( : data%a_ne ) = A%val( : data%a_ne )

!  insert the values of H into K

      SELECT CASE ( SMT_get( H%type ) )
      CASE ( 'SCALED_IDENTITY' )
        data%K%val( data%a_ne + 1 : data%k_c ) = H%val( 1 )
      CASE ( 'IDENTITY' )
        data%K%val( data%a_ne + 1 : data%k_c ) = one
      CASE ( 'ZERO', 'NONE' )
      CASE DEFAULT
        data%K%val( data%a_ne + 1 : data%k_c ) = H%val( : data%h_ne )
      END SELECT

!  insert the values of C into K

      SELECT CASE ( SMT_get( C%type ) )
      CASE ( 'SCALED_IDENTITY' )
        data%K%val( data%k_c + 1 : data%K%ne ) = - C%val( 1 )
      CASE ( 'IDENTITY' )
        data%K%val( data%k_c + 1 : data%K%ne ) = - one
      CASE ( 'ZERO', 'NONE' )
      CASE DEFAULT
        data%K%val( data%k_c + 1 : data%K%ne ) = - C%val( : data%c_ne )
      END SELECT

!  factorize K

      CALL SLS_factorize( data%K, data%SLS_data, control%SLS_control,          &
                          inform%SLS_inform )
      IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
        inform%status = GALAHAD_ok
        inform%factorization_integer = inform%SLS_inform%integer_size_factors
        inform%factorization_real = inform%SLS_inform%real_size_factors
        inform%rank = inform%SLS_inform%rank
        inform%rank_def = inform%SLS_inform%rank /= n + m
      ELSE
        inform%status = GALAHAD_error_factorization
      END IF

!  record times

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      IF ( control%print_level >= 1 ) WRITE( control%out,                      &
         "( A, ' time to factorize K ', F6.2 )") prefix, time_now - time_start
      inform%time%factorize = inform%time%factorize + time_now - time_start
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      RETURN

!  end of subroutine SSLS_factorize

      END SUBROUTINE SSLS_factorize

      SUBROUTINE SSLS_solve( n, m, SOL, data, control, inform )

!-*-*-*-*-*-*-*-   S S L S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

!  solve the symmetric structured linear system

!        ( H   A^T ) ( x ) = ( a )
!        ( A   -C  ) ( y )   ( b )

!  where (x,y) overwrites (a,b) on return

!  Dummy arguments

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL
      TYPE ( SSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SSLS_control_type ), INTENT( IN ) :: control
      TYPE ( SSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = rp_ ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      CALL SLS_solve( data%K, SOL, data%SLS_data, control%SLS_control,         &
                      inform%SLS_inform )
      IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
        inform%status = GALAHAD_ok
      ELSE
        inform%status = GALAHAD_error_solve
      END IF

!  record times

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      IF ( control%print_level >= 1 ) WRITE( control%out,                      &
         "( A, ' time to solve with K ', F6.2 )") prefix, time_now - time_start
      inform%time%solve = inform%time%solve + time_now - time_start
      inform%time%clock_solve =                                                &
        inform%time%clock_solve + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  end of subroutine SSLS_solve

      END SUBROUTINE SSLS_solve

!-*-*-*-*-*-   S S L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE SSLS_terminate( data, control, inform )

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
!   data    see Subroutine SSLS_initialize
!   control see Subroutine SSLS_initialize
!   inform  see Subroutine SSLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( SSLS_control_type ), INTENT( IN ) :: control
      TYPE ( SSLS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SSLS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'ssls: K%row'
      CALL SPACE_dealloc_array( data%K%row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ssls: K%col'
      CALL SPACE_dealloc_array( data%K%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ssls: K%val'
      CALL SPACE_dealloc_array( data%K%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ssls: K%type'
      CALL SPACE_dealloc_array( data%K%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine SSLS_terminate

      END SUBROUTINE SSLS_terminate

! -  G A L A H A D -  S S L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE SSLS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SSLS_control_type ), INTENT( IN ) :: control
     TYPE ( SSLS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

!  deallocate workspace

     CALL SSLS_terminate( data%ssls_data, control, inform )

     RETURN

!  End of subroutine SSLS_full_terminate

     END SUBROUTINE SSLS_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-  G A L A H A D -  S B L S _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE SSLS_import( control, data, status, n, m,                      &
                             H_type, H_ne, H_row, H_col, H_ptr,                &
                             A_type, A_ne, A_row, A_col, A_ptr,                &
                             C_type, C_ne, C_row, C_col, C_ptr )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SSLS_solve
!
!  data is a scalar variable of type SSLS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, m >= 0 or requirement that type contains
!       its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' 'SCALED_IDENTITY', 'IDENTITY', 'ZERO', or 'NONE'
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   rows and columns in the 1,1 block
!
!  m is a scalar variable of type default integer, that holds the number of
!   rows and columns in the 2,2 block
!
!  H_type is a character string that specifies the storage scheme used for H.
!   It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   'diagonal' 'scaled_identity', 'identity', 'zero' or 'none';
!   lower or upper case variants are allowed.
!
!  H_ne is a scalar variable of type default integer, that holds the number of
!   entries in the  lower triangular part of H in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other schemes.
!
!  H_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of H in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  H_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of H in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  H_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the  lower
!   triangular part of H, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!  A_type is a character string that specifies the storage scheme used for A.
!   It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   or 'absent', the latter if m = 0; lower or upper case variants are allowed
!
!  A_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  A_row is a rank-one array of type default integer, that holds the row
!   indices J in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  A_col is a rank-one array of type default integer, that holds the column
!   indices of J in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense scheme is used, and
!   in this case can be of length 0
!
!  A_ptr is a rank-one array of dimension n+1 and type default integer,
!   that holds the starting position of each row of J, as well as the total
!   number of entries plus one, in the sparse row-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!  C_type is a character string that specifies the Hessian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   'diagonal' 'scaled_identity', 'identity', 'zero' or 'none';
!   lower or upper case variants are allowed.
!
!  C_ne is a scalar variable of type default integer, that holds the number of
!   entries in the  lower triangular part of H in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other schemes.
!
!  C_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of H in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  C_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of H in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  C_ptr is a rank-one array of dimension m+1 and type default
!   integer, that holds the starting position of  each row of the  lower
!   triangular part of H, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SSLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, m, A_ne, H_ne, C_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: H_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_ptr
     CHARACTER ( LEN = * ), INTENT( IN ) :: A_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_ptr
     CHARACTER ( LEN = * ), INTENT( IN ) :: C_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: C_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: C_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: C_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     data%SSLS_control = control

     error = data%ssls_control%error
     space_critical = data%ssls_control%space_critical
     deallocate_error_fatal = data%ssls_control%space_critical

!  set H appropriately in the smt storage type

     data%H%n = n ; data%H%m = n
     SELECT CASE ( H_type )
     CASE ( 'coordinate', 'COORDINATE' )
      IF ( .NOT. ( PRESENT( H_row ) .AND. PRESENT( H_col ) ) ) THEN
         data%ssls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%H%type, 'COORDINATE',                                &
                     data%ssls_inform%alloc_status )
       data%H%ne = H_ne

       array_name = 'ssls: data%H%row'
       CALL SPACE_resize_array( data%H%ne, data%H%row,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%H%col'
       CALL SPACE_resize_array( data%H%ne, data%H%col,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%H%row( : data%H%ne ) = H_row( : data%H%ne )
         data%H%col( : data%H%ne ) = H_col( : data%H%ne )
       ELSE
         data%H%row( : data%H%ne ) = H_row( : data%H%ne ) + 1
         data%H%col( : data%H%ne ) = H_col( : data%H%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
      IF ( .NOT. ( PRESENT( H_ptr ) .AND. PRESENT( H_col ) ) ) THEN
         data%ssls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%H%type, 'SPARSE_BY_ROWS',                            &
                     data%ssls_inform%alloc_status )
       IF ( data%f_indexing ) THEN
         data%H%ne = H_ptr( n + 1 ) - 1
       ELSE
         data%H%ne = H_ptr( n + 1 )
       END IF

       array_name = 'ssls: data%H%ptr'
       CALL SPACE_resize_array( n + 1, data%H%ptr,                             &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%H%col'
       CALL SPACE_resize_array( data%H%ne, data%H%col,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%H%ptr( : n + 1 ) = H_ptr( : n + 1 )
         data%H%col( : data%H%ne ) = H_col( : data%H%ne )
       ELSE
         data%H%ptr( : n + 1 ) = H_ptr( : n + 1 ) + 1
         data%H%col( : data%H%ne ) = H_col( : data%H%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%H%type, 'DENSE',                                     &
                     data%ssls_inform%alloc_status )
       data%H%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'ssls: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%H%type, 'DIAGONAL',                                  &
                     data%ssls_inform%alloc_status )
       data%H%ne = n

       array_name = 'ssls: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE ( 'scaled_identity', 'SCALED_IDENTITY' )
       CALL SMT_put( data%H%type, 'SCALED_IDENTITY',                           &
                     data%ssls_inform%alloc_status )
       data%H%ne = 1

       array_name = 'ssls: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE ( 'identity', 'IDENTITY' )
       CALL SMT_put( data%H%type, 'IDENTITY',                                  &
                     data%ssls_inform%alloc_status )
       data%H%ne = 0

     CASE ( 'zero', 'ZERO', 'none', 'NONE' )
       CALL SMT_put( data%H%type, 'ZERO',                                      &
                     data%ssls_inform%alloc_status )
       data%H%ne = 0

     CASE DEFAULT
       data%ssls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

!  set A appropriately in the smt storage type

     data%A%n = n ; data%A%m = m
     SELECT CASE ( A_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( A_row ) .AND. PRESENT( A_col ) ) ) THEN
         data%ssls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%A%type, 'COORDINATE',                                &
                     data%ssls_inform%alloc_status )
       data%A%ne = A_ne

       array_name = 'ssls: data%A%row'
       CALL SPACE_resize_array( data%A%ne, data%A%row,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%A%col'
       CALL SPACE_resize_array( data%A%ne, data%A%col,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%A%val'
       CALL SPACE_resize_array( data%A%ne, data%A%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%A%row( : data%A%ne ) = A_row( : data%A%ne )
         data%A%col( : data%A%ne ) = A_col( : data%A%ne )
       ELSE
         data%A%row( : data%A%ne ) = A_row( : data%A%ne ) + 1
         data%A%col( : data%A%ne ) = A_col( : data%A%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( A_ptr ) .AND. PRESENT( A_col ) ) ) THEN
         data%ssls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%A%type, 'SPARSE_BY_ROWS',                            &
                     data%ssls_inform%alloc_status )
       IF ( data%f_indexing ) THEN
         data%A%ne = A_ptr( m + 1 ) - 1
       ELSE
         data%A%ne = A_ptr( m + 1 )
       END IF

       array_name = 'ssls: data%A%ptr'
       CALL SPACE_resize_array( m + 1, data%A%ptr,                             &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%A%col'
       CALL SPACE_resize_array( data%A%ne, data%A%col,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%A%val'
       CALL SPACE_resize_array( data%A%ne, data%A%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%A%ptr( : m + 1 ) = A_ptr( : m + 1 )
         data%A%col( : data%A%ne ) = A_col( : data%A%ne )
       ELSE
         data%A%ptr( : m + 1 ) = A_ptr( : m + 1 ) + 1
         data%A%col( : data%A%ne ) = A_col( : data%A%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%A%type, 'DENSE',                                     &
                     data%ssls_inform%alloc_status )
       data%A%ne = m * n

       array_name = 'ssls: data%A%val'
       CALL SPACE_resize_array( data%A%ne, data%A%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%ssls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

!  set C appropriately in the smt storage type

     data%C%m = m ;  data%C%n = m
     SELECT CASE ( C_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( C_row ) .AND. PRESENT( C_col ) ) ) THEN
         data%ssls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%C%type, 'COORDINATE',                                &
                     data%ssls_inform%alloc_status )
       data%C%ne = C_ne

       array_name = 'ssls: data%C%row'
       CALL SPACE_resize_array( data%C%ne, data%C%row,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%C%col'
       CALL SPACE_resize_array( data%C%ne, data%C%col,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%C%val'
       CALL SPACE_resize_array( data%C%ne, data%C%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%C%row( : data%C%ne ) = C_row( : data%C%ne )
         data%C%col( : data%C%ne ) = C_col( : data%C%ne )
       ELSE
         data%C%row( : data%C%ne ) = C_row( : data%C%ne ) + 1
         data%C%col( : data%C%ne ) = C_col( : data%C%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( C_ptr ) .AND. PRESENT( C_col ) ) ) THEN
         data%ssls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%C%type, 'SPARSE_BY_ROWS',                            &
                     data%ssls_inform%alloc_status )
       IF ( data%f_indexing ) THEN
         data%C%ne = C_ptr( m + 1 ) - 1
       ELSE
         data%C%ne = C_ptr( m + 1 )
       END IF

       array_name = 'ssls: data%C%ptr'
       CALL SPACE_resize_array( m + 1, data%C%ptr,                             &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%C%col'
       CALL SPACE_resize_array( data%C%ne, data%C%col,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       array_name = 'ssls: data%C%val'
       CALL SPACE_resize_array( data%C%ne, data%C%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%C%ptr( : m + 1 ) = C_ptr( : m + 1 )
         data%C%col( : data%C%ne ) = C_col( : data%C%ne )
       ELSE
         data%C%ptr( : m + 1 ) = C_ptr( : m + 1 ) + 1
         data%C%col( : data%C%ne ) = C_col( : data%C%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%C%type, 'DENSE',                                     &
                     data%ssls_inform%alloc_status )
       data%C%ne = ( m * ( m + 1 ) ) / 2

       array_name = 'ssls: data%C%val'
       CALL SPACE_resize_array( data%C%ne, data%C%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%C%type, 'DIAGONAL',                                  &
                     data%ssls_inform%alloc_status )
       data%C%ne = m

       array_name = 'ssls: data%C%val'
       CALL SPACE_resize_array( data%C%ne, data%C%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE ( 'scaled_identity', 'SCALED_IDENTITY' )
       CALL SMT_put( data%C%type, 'SCALED_IDENTITY',                           &
                     data%ssls_inform%alloc_status )
       data%C%ne = 1

       array_name = 'ssls: data%C%val'
       CALL SPACE_resize_array( data%C%ne, data%C%val,                         &
              data%ssls_inform%status, data%ssls_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%ssls_inform%bad_alloc, out = error )
       IF ( data%ssls_inform%status /= 0 ) GO TO 900

     CASE ( 'identity', 'IDENTITY' )
       CALL SMT_put( data%C%type, 'IDENTITY',                                  &
                     data%ssls_inform%alloc_status )
       data%C%ne = 0

     CASE ( 'zero', 'ZERO', 'none', 'NONE' )
       CALL SMT_put( data%C%type, 'ZERO',                                      &
                     data%ssls_inform%alloc_status )
       data%C%ne = 0

     CASE DEFAULT
       data%ssls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

!  analyse the structure of the matrix

     CALL SSLS_analyse( data%H%n, data%C%m, data%H, data%A, data%C,            &
                        data%ssls_data, data%ssls_control, data%ssls_inform )

     status = data%ssls_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%ssls_inform%status
     RETURN

!  End of subroutine SSLS_import

     END SUBROUTINE SSLS_import

!-  G A L A H A D -  S B L S _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE SSLS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See SSLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SSLS_control_type ), INTENT( IN ) :: control
     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%ssls_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SSLS_reset_control

     END SUBROUTINE SSLS_reset_control

! G A L A H A D - S B L S _ f a c t o r i z e _ m a t r i x  S U B R O U T I N E

     SUBROUTINE SSLS_factorize_matrix( data, status, H_val, A_val, C_val )

!  form and factorize the block matrix ( H A^T ).
!                                      ( A -C  )
!  See SSLS_form_and_factorize for a description of the required arguments

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: H_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: A_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: C_val

!  save the values of H, A and C

     IF ( data%H%ne > 0 ) data%H%val( : data%H%ne ) = H_val( : data%H%ne )
     IF ( data%A%ne > 0 ) data%A%val( : data%A%ne ) = A_val( : data%A%ne )
     IF ( data%C%ne > 0 ) data%C%val( : data%C%ne ) = C_val( : data%C%ne )

!  form and factorize the block matrix

     CALL SSLS_factorize( data%H%n, data%C%m, data%H, data%A, data%C,          &
                          data%ssls_data, data%ssls_control, data%ssls_inform )

     status = data%ssls_inform%status
     RETURN

!  end of subroutine SSLS_factorize_matrix

     END SUBROUTINE SSLS_factorize_matrix

!--  G A L A H A D -  S B L S _ s o l v e _ s y s t e m   S U B R O U T I N E  -

     SUBROUTINE SSLS_solve_system( data, status, SOL )

!  solve the linear system ( H A^T ) ( x ) = ( a ),
!                          ( A -C  ) ( y )   ( b )
!  where SOL holds the right-hand side on input, and the solution on output.
!  See SSLS_solve for a description of the required arguments

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: SOL

!  solve the block linear system

     CALL SSLS_solve( data%H%n, data%C%m, SOL,                                 &
                      data%ssls_data, data%ssls_control, data%ssls_inform )

     status = data%ssls_inform%status
     RETURN

!  end of subroutine SSLS_solve_system

     END SUBROUTINE SSLS_solve_system

!-  G A L A H A D -  S B L S _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE SSLS_information( data, inform, status )

!  return solver information during or after solution by SSLS
!  See SSLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SSLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SSLS_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%ssls_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SSLS_information

     END SUBROUTINE SSLS_information

!  End of module SSLS

    END MODULE GALAHAD_SSLS_precision
