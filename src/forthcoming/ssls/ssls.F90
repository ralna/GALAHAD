! THIS VERSION: GALAHAD 5.1 - 2024-10-04 AT 14:10 GMT.

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
      USE GALAHAD_SLS_precision
      USE GALAHAD_SPECFILE_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SSLS_initialize, SSLS_read_specfile, SSLS_analyse,             &
                SSLS_factorize, SSLS_solve, SSLS_terminate,                    &
                SSLS_full_initialize, SSLS_full_terminate,                     &
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

!  indefinite linear equation solver

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

!  cpu time spent forming the preconditioner K

        REAL ( KIND = rp_ ) :: form = 0.0

!  cpu time spent factorizing K_G

        REAL ( KIND = rp_ ) :: factorize = 0.0

!  cpu time spent solving linear systems inolving K

        REAL ( KIND = rp_ ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  clock time spent forming the preconditioner K

        REAL ( KIND = rp_ ) :: clock_form = 0.0

!  clock time spent factorizing K_G

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving K_G

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

!  how many of the diagonals in the factorization are positive

        INTEGER ( KIND = ip_ ) :: d_plus = - 1

!  the computed rank of A

        INTEGER ( KIND = ip_ ) :: rank = - 1

!  is the matrix A rank defficient?

        LOGICAL :: rank_def = .FALSE.

!  has the used preconditioner been perturbed to guarantee correct inertia?

        LOGICAL :: perturbed = .FALSE.

!  the norm of the residual

        REAL ( KIND = rp_ ) :: norm_residual = - 1.0_rp_

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

      CALL SLS_INITIALIZE( control%symmetric_linear_solver,                    &
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
      CHARACTER ( LEN = 80 ) :: array_name

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
      END IF

      RETURN

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
      CASE DEFAULT
        data%K%val( data%k_c + 1 : data%K%ne ) = - C%val( : data%c_ne )
      END SELECT

!  factorize K

      CALL SLS_factorize( data%K, data%SLS_data, control%SLS_control,          &
                          inform%SLS_inform )
      IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
        inform%status = GALAHAD_ok
      ELSE
        inform%status = GALAHAD_error_factorization
      END IF

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

      CALL SLS_solve( data%K, SOL, data%SLS_data, control%SLS_control,         &
                      inform%SLS_inform )
      IF ( inform%SLS_inform%status == GALAHAD_ok ) THEN
        inform%status = GALAHAD_ok
      ELSE
        inform%status = GALAHAD_error_solve
      END IF

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

!  End of module SSLS

    END MODULE GALAHAD_SSLS_precision
