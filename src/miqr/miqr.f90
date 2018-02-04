! THIS VERSION: GALAHAD 2.6 - 13/05/2014 AT 09:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ M I Q R    M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   partially based on the C package imqr by Na Li (nli@cs.umn.edu), 2005
!   development started May 1st 2014
!   originally released GALAHAD Version 2.6. May 1st 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_MIQR_double

!     ---------------------------------------------------
!     |                                                 |
!     | Given a real matrix A, compute a multilevel     |
!     | incomplete QR factorization A ~ Q R where       |
!     | Q is orthogonal and R upper triangular          |
!     |                                                 |
!     | Based on the paper: Na Li and Yousef Saad       |
!     | MIQR: A Multilevel Incomplete QR Preconditioner |
!     | for Large Sparse Least‐Squares Problems         |
!     | SIAM. J. Matrix Anal. & Appl., 28(2), (2006)    |
!     | pp. 524–550                                     |
!     |                                                 |
!     ---------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_NORMS_double
      USE GALAHAD_SMT_double
      USE GALAHAD_CONVERT_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: MIQR_initialize, MIQR_read_specfile, MIQR_form, MIQR_apply,    &
                MIQR_terminate, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: real_bytes = 8
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

!     INTEGER, PARAMETER :: max_miqr_levels = 10
      INTEGER, PARAMETER :: max_miqr_levels = 100
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

      TYPE, PUBLIC :: MIQR_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  the maximum level allowed in a multi-level method

        INTEGER :: max_level = 4

!  the maximum order per level (-ve = n)

        INTEGER :: max_order = - 1

!  the max number of elements allowed in each column of R will not exceed
!    max_fill (-ve = n)

        INTEGER :: max_fill = 100

!  the max number of elements allowed in each column of Q will not exceed
!    max_fill (-ve = m)

        INTEGER :: max_fill_q = 100

!  increase array sizes in chunks of this when needed

        INTEGER :: increase_size = 100

!  unit for any out-of-core writing when expanding arrays

        INTEGER :: buffer = 70

!  any diagonal entry in the R factor that is smaller than smallest_diag
!  will be judged to be zero, and modified accordingly

!       REAL ( KIND = wp ) :: smallest_diag = 100.0_wp * epsmch
!       REAL ( KIND = wp ) :: smallest_diag = ten ** ( - 14 )
        REAL ( KIND = wp ) :: smallest_diag = ten ** ( - 10 )

!  tolerance for stopping multi-level phase. Stop if
!    reduced size < tol_level * previous size

        REAL ( KIND = wp ) :: tol_level = 0.3_wp

!  orthogonal tolerance: if |u^T v| < tol_orthogonal*||u||*||v|| vectors
!    u and v are considered as orthogonal

        REAL ( KIND = wp ) :: tol_orthogonal = 0.0_wp

!  increase the orthogonality tolerance by tol_orthogonal_increase at
!    each level

        REAL ( KIND = wp ) :: tol_orthogonal_increase = 0.01_wp

!  the max number of elements allowed in each column of R will not exceed
!    average_max_fill * ne / n

        REAL ( KIND = wp ) :: average_max_fill = 6.0_wp

!  the max number of elements allowed in each column of Q will not exceed
!    average_max_fill * ne / m

        REAL ( KIND = wp ) :: average_max_fill_q = 24.0_wp

!  dropping tolerance for small generated entries

        REAL ( KIND = wp ) :: tol_drop = 0.01_wp

!   the maximum CPU time allowed when constructing the preconditioner
!    (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed when constructing the preconditioner
!    (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!  find the factorization of the transpose of the matrix?

        LOGICAL :: transpose = .FALSE.

!  use incomplete multi-level QR (IMQR) or not (IQR)

        LOGICAL :: multi_level = .TRUE.

!  sort graph data according to degree?

        LOGICAL :: sort = .TRUE.

!  deallocate any workspace after every factorization

        LOGICAL :: deallocate_after_factorization = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for CONVERT

        TYPE ( CONVERT_control_type ) :: CONVERT_control

      END TYPE MIQR_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_time_type

!  total cpu time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  cpu time spent in the multi-level phase when forming the preconditioner

        REAL ( KIND = wp ) :: levels = 0.0

!  cpu time spent in IQR phase when forming the preconditioner

        REAL ( KIND = wp ) :: iqr = 0.0

!  cpu time spent forming the preconditioner

        REAL ( KIND = wp ) :: form = 0.0

!  cpu time spent applying the preconditioner

        REAL ( KIND = wp ) :: apply = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  total clock time spent in the multi-level phase when forming
!  the preconditioner

        REAL ( KIND = wp ) :: clock_levels = 0.0

!  total clock time spent in the IQR phase when forming
!  the preconditioner

        REAL ( KIND = wp ) :: clock_iqr = 0.0

!  clock time spent forming the preconditioner

        REAL ( KIND = wp ) :: clock_form = 0.0

!  clock time spent applying the preconditioner

        REAL ( KIND = wp ) :: clock_apply = 0.0

      END TYPE MIQR_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_inform_type

!  return status. See MIQR_form_and_factorize for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  number of entries in factors

       INTEGER ( KIND = long ) :: entries_in_factors = - 1_long

!  the number of entries dropped

        INTEGER ( KIND = long ) :: drop = 0

!  the number of zero columns encountered

        INTEGER ( KIND = long ) :: zero_diagonals = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  timings (see above)

        TYPE ( MIQR_time_type ) :: time

!  inform parameters for CONVERT

        TYPE ( CONVERT_inform_type ) :: CONVERT_inform

      END TYPE MIQR_inform_type

!  - - - - - - - - - - - - - - -
!   sparse vector derived type
!  - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_sparse_vector_type
        INTEGER :: dim   !  dimension
        INTEGER :: ne    !  number of entries
        INTEGER, POINTER, DIMENSION( : ) :: ind             ! indices
        REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: val  ! values
        LOGICAL, POINTER, DIMENSION( : ) :: pat   ! true <=> 0 nonzero element
      END TYPE MIQR_sparse_vector_type

!  - - - - - - - - - - - - - - -
!   data workspace derived type
!  - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_data_workspace_type
        INTEGER, POINTER, DIMENSION( : ) :: ind_n => NULL( )
        INTEGER, POINTER, DIMENSION( : ) :: ind_m => NULL( )
        LOGICAL, POINTER, DIMENSION( : ) :: pat_n => NULL( )
        LOGICAL, POINTER, DIMENSION( : ) :: pat_m => NULL( )
        REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: val_n => NULL( )
        REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: val_m => NULL( )
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: PTR
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: Q_list_next
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: Q_list_col
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Q_list_val
        TYPE ( SMT_type ) :: A_by_rows
        TYPE ( SMT_type ) :: C
        TYPE ( SMT_type ) :: Q
      END TYPE MIQR_data_workspace_type

!  - - - - - - - - - - - -
!   data global derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_data_global_type
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: nodes_degree
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: nodes_index
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_norms
!       TYPE ( MIQR_graph_node_type ), ALLOCATABLE, DIMENSION( : ) :: nodes
      END TYPE MIQR_data_global_type

!  - - - - - - - - - - - -
!   data level derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_data_level_type
        INTEGER :: n
        INTEGER :: order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: PERM
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: INVERSE_PERM
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D_inverse
        TYPE ( SMT_type ) :: F
      END TYPE MIQR_data_level_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: MIQR_data_type
        INTEGER :: n
        INTEGER :: levels
        INTEGER, DIMENSION( 1 : max_miqr_levels + 1 ) :: pos
        TYPE ( SMT_type ) :: A_by_cols
        TYPE ( SMT_type ) :: A_new
        TYPE ( SMT_type ) :: R
        TYPE ( MIQR_data_workspace_type ) :: workspace
        TYPE ( MIQR_data_global_type ),                                        &
          DIMENSION( 1 : max_miqr_levels + 1 ) :: global
        TYPE ( MIQR_data_level_type ),                                         &
          DIMENSION( 1 : max_miqr_levels + 1 ) :: level
        REAL ( KIND = wp ) :: tm_levels
        REAL ( KIND = wp ) :: tm_iqr
        TYPE ( MIQR_control_type ) :: control
      END TYPE MIQR_data_type

   CONTAINS

!-*-*-*-*-*-   M I Q R  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE MIQR_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for MIQR. This routine should be called before
!  MIQR_form
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

      TYPE ( MIQR_data_type ), INTENT( INOUT ) :: data
      TYPE ( MIQR_control_type ), INTENT( OUT ) :: control
      TYPE ( MIQR_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of MIQR_initialize

      END SUBROUTINE MIQR_initialize

!-*-*-*-   M I Q R _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE MIQR_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by MIQR_initialize could (roughly)
!  have been set as:

! BEGIN MIQR SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  max-level-allowed                                 4
!  max-order-allowed-per-level                       -1
!  max-entries-per-column                            100
!  max-entries-per-column-of-q                       100
!  increase-array-size-by                            100
!  out-of-core-buffer                                70
!  smallest-diagonal-factor-allowed                  1.0D+10
!  level-stop-tolerance                              0.3
!  orthogonality-tolerance                           0.0
!  orthogonality-tolerance-increase                  0.01
!  dropping-tolerance                                0.01
!  proportion-max-entries-per-column                 6.0
!  proportion-max-entries-per-column-of-q            6.0
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  factorize-transpose                               F
!  use-multi-level                                   T
!  sort-vertices                                     T
!  deallocate-workspace-after-factorization          F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END MIQR SPECIFICATIONS

!  Dummy arguments

      TYPE ( MIQR_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: max_level = print_level + 1
      INTEGER, PARAMETER :: max_order = max_level + 1
      INTEGER, PARAMETER :: max_fill = max_order + 1
      INTEGER, PARAMETER :: max_fill_q = max_fill + 1
      INTEGER, PARAMETER :: increase_size = max_fill_q + 1
      INTEGER, PARAMETER :: buffer = increase_size + 1
      INTEGER, PARAMETER :: smallest_diag = buffer + 1
      INTEGER, PARAMETER :: tol_level = smallest_diag + 1
      INTEGER, PARAMETER :: tol_orthogonal = tol_level + 1
      INTEGER, PARAMETER :: tol_orthogonal_increase = tol_orthogonal + 1
      INTEGER, PARAMETER :: tol_drop = tol_orthogonal_increase + 1
      INTEGER, PARAMETER :: average_max_fill = tol_drop + 1
      INTEGER, PARAMETER :: average_max_fill_q = average_max_fill + 1
      INTEGER, PARAMETER :: cpu_time_limit = average_max_fill_q + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: transpose = clock_time_limit + 1
      INTEGER, PARAMETER :: multi_level = transpose + 1
      INTEGER, PARAMETER :: sort = multi_level + 1
      INTEGER, PARAMETER :: deallocate_after_factorization = sort + 1
      INTEGER, PARAMETER :: space_critical = deallocate_after_factorization + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'MIQR'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( max_level )%keyword = 'max-level-allowed'
      spec( max_order )%keyword = 'max-order-allowed-per-level'
      spec( max_fill )%keyword = 'max-entries-per-column'
      spec( max_fill_q )%keyword = 'max-entries-per-column-of-q'
      spec( increase_size )%keyword = 'increase-array-size-by'
      spec( buffer )%keyword = 'out-of-core-buffer'

!  Real key-words

      spec( smallest_diag )%keyword = 'smallest-diagonal-factor-allowed'
      spec( tol_level )%keyword = 'level-stop-tolerance'
      spec( tol_orthogonal )%keyword = 'orthogonality-tolerance'
      spec( tol_orthogonal_increase )%keyword                                  &
        = 'orthogonality-tolerance-increase'
      spec( tol_drop )%keyword = 'dropping-tolerance'
      spec( average_max_fill )%keyword = 'proportion-max-entries-per-column'
      spec( average_max_fill_q )%keyword                                       &
        = 'proportion-max-entries-per-column-of-q'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( transpose )%keyword = 'factorize-transpose'
      spec( multi_level )%keyword = 'use-multi-level'
      spec( sort )%keyword = 'sort-vertices'
      spec( deallocate_after_factorization )%keyword                           &
        = 'deallocate-workspace-after-factorization'
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
      CALL SPECFILE_assign_value( spec( max_level ),                           &
                                  control%max_level,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_order ),                           &
                                  control%max_order,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_fill ),                            &
                                  control%max_fill,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_fill_q ),                          &
                                  control%max_fill_q,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( increase_size ),                       &
                                  control%increase_size,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( buffer ),                              &
                                  control%buffer,                              &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( smallest_diag ),                       &
                                  control%smallest_diag,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( tol_level ),                           &
                                  control%tol_level,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( tol_orthogonal ),                      &
                                  control%tol_orthogonal,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( tol_orthogonal_increase ),             &
                                  control%tol_orthogonal_increase,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( tol_drop ),                            &
                                  control%tol_drop,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( average_max_fill ),                    &
                                  control%average_max_fill,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( average_max_fill_q ),                  &
                                  control%average_max_fill_q,                  &
                                  control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( transpose ),                           &
                                  control%transpose,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( multi_level ),                         &
                                  control%multi_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( sort ),                                &
                                  control%sort,                                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_after_factorization ),      &
                                  control%deallocate_after_factorization,      &
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

!  Read the specfile for CONVERT

      IF ( PRESENT( alt_specname ) ) THEN
        CALL CONVERT_read_specfile( control%CONVERT_control, device,           &
                            alt_specname = TRIM( alt_specname ) // '-CONVERT' )
      ELSE
        CALL CONVERT_read_specfile( control%CONVERT_control, device )
      END IF
      control%CONVERT_control%transpose = control%transpose
      control%CONVERT_control%order = .TRUE.

      RETURN

!  End of MIQR_read_specfile

      END SUBROUTINE MIQR_read_specfile

!-*-*-*-*-*-*-*-*-   M I Q R _ F O R M   S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE MIQR_form( A, data, control, inform )

!  form the incomplete multi-level QR factorization of the matrix A

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( MIQR_data_type ), INTENT( INOUT ) :: data
      TYPE ( MIQR_control_type ), INTENT( IN ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, level, m, n, min_unprocessed_columns
      REAL ( KIND = wp ) :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set initial values

      IF ( control%transpose ) THEN
        m = A%n ; n = A%m
      ELSE
        m = A%m ; n = A%n
      END IF

!  initialize workspace

      array_name = 'miqr: data%workspace%ind_m'
      CALL SPACE_resize_pointer( m, data%workspace%ind_m,                      &
         inform%status, inform%alloc_status, point_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900
      data%workspace%ind_m = 0

      array_name = 'miqr: data%workspace%pat_m'
      CALL SPACE_resize_pointer( m, data%workspace%pat_m,                      &
         inform%status, inform%alloc_status, point_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'miqr: data%workspace%val_m'
      CALL SPACE_resize_pointer( m, data%workspace%val_m,                      &
         inform%status, inform%alloc_status, point_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'miqr: data%workspace%ind_n'
      CALL SPACE_resize_pointer( n, data%workspace%ind_n,                      &
         inform%status, inform%alloc_status, point_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'miqr: data%workspace%pat_n'
      CALL SPACE_resize_pointer( n, data%workspace%pat_n,                      &
         inform%status, inform%alloc_status, point_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'miqr: data%workspace%val_n'
      CALL SPACE_resize_pointer( n, data%workspace%val_n,                      &
         inform%status, inform%alloc_status, point_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy control parameters so that they may be altered as required

      data%control = control

!  store A column-wise in A_by_cols with the row entries within each column
!  in increasing order

      data%control%CONVERT_control%order = .TRUE.
      data%control%CONVERT_control%transpose = data%control%transpose
      CALL CONVERT_to_column_format( A, data%A_by_cols,                        &
                                     data%control%CONVERT_control,             &
                                     inform%CONVERT_inform,                    &
                                     data%workspace%ind_m, m,                  &
                                     data%workspace%val_m, m )
      IF ( inform%CONVERT_inform%status /= GALAHAD_ok ) THEN
        inform%status = inform%CONVERT_inform%status
        GO TO 900
      END IF

!     m = data%A_by_cols%m ; n = data%A_by_cols%n

!  allocate global workspace

      DO i = 1, max_miqr_levels
        array_name = 'miqr: data%global%A_norms'
        CALL SPACE_resize_array( n, data%global( i )%A_norms,                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: data%global%nodes_degree'
        CALL SPACE_resize_array( n, data%global( i )%nodes_degree,             &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: data%global%nodes_index'
        CALL SPACE_resize_array( n, data%global( i )%nodes_index,              &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END DO

!  initilaise data and workspace

      data%levels = 1
      data%workspace%pat_m = .FALSE.
      data%workspace%ind_n = 0 ; data%workspace%pat_n = .FALSE.
      min_unprocessed_columns = INT( 0.01_wp * REAL( n ) )
      inform%entries_in_factors = 0
      inform%drop = 0 ; inform%zero_diagonals = 0

!  multilevel IQR iteration

      level = 1
      IF ( data%control%multi_level ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        DO

!  perform the level-th iteration on the submatrix A

          IF ( control%out > 0 .AND. control%print_level > 1 )                 &
            WRITE( control%out, "( A, ' forming level ', I0, ' factors' )" )   &
              prefix, level
          CALL MIQR_form_level( data%A_by_cols, data%level( level ),           &
                                data%A_new, data%workspace,                    &
                                data%global( level ),                          &
                                time_start, clock_start,                       &
                                data%control, inform )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
          IF ( control%out > 0 .AND. control%print_level > 1 )                 &
            WRITE( control%out, "( A, ' level ', I0, ' factors formed, order', &
           & ' = ', I0 )" )  prefix, level, data%level( level )%order

!  the remaining unprocessed submatrix is in A_new, so copy this to A

          CALL MIQR_copy_col_matrix( data%A_new, data%A_by_cols,               &
                                     data%control, inform )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  exit if the number of newly-independent columns is small

          IF ( data%level( level )%order <=                                    &
                data%control%tol_level * data%level( level )%n ) EXIT

!  exit if the remaining number of unprocessed columns is small

          IF ( data%A_new%n < min_unprocessed_columns ) EXIT

!  exit if max_level levels have been processed

          IF ( level >= data%control%max_level ) EXIT

!  increase the adaptive angle tolerance

          data%control%tol_orthogonal                                          &
            = data%control%tol_orthogonal + data%control%tol_orthogonal_increase

          level = level + 1
        END DO

!  remove unwanted data if desired

        IF ( data%control%deallocate_after_factorization ) THEN
          CALL MIQR_dealloc_col_mat( data%A_new, 'A_new', control, inform )
          IF ( control%deallocate_error_fatal .AND.                            &
               inform%status /= GALAHAD_ok ) GO TO 900

          CALL MIQR_dealloc_row_mat( data%workspace%A_by_rows, 'A_by_rows',    &
                                     control, inform )
          IF ( control%deallocate_error_fatal .AND.                            &
               inform%status /= GALAHAD_ok ) GO TO 900

          CALL MIQR_dealloc_row_mat( data%workspace%C, 'C', control, inform )
          IF ( control%deallocate_error_fatal .AND.                            &
               inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  record the time taken in the multi-level phase

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%levels = inform%time%levels + time_now - time_record
        inform%time%clock_levels                                               &
          = inform%time%clock_levels + clock_now - clock_record
      END IF

!  process the remaining matrix with ordinary IQR

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      IF ( control%out > 0 .AND. control%print_level > 1 )                     &
        WRITE( control%out, "( A, ' forming final IQR factors' )" )  prefix
      CALL MIQR_form_iqr( data%A_by_cols, data%R, data%workspace,              &
                          time_start, clock_start,                             &
                          data%control, inform )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900
      IF ( control%out > 0 .AND. control%print_level > 1 )                     &
        WRITE( control%out, "( A, ' final IQR factors formed, m = ', I0,       &
       &  ', n = ', I0 )" )   prefix, data%A_by_cols%m, data%A_by_cols%n

!  record the time taken in the IQR phase

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%iqr = inform%time%iqr + time_now - time_record
      inform%time%clock_iqr = inform%time%clock_iqr + clock_now - clock_record

!  remove unwanted data if desired

      IF ( data%control%deallocate_after_factorization ) THEN
        CALL MIQR_dealloc_col_mat( data%A_by_cols,'A_by_cols', control, inform )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) GO TO 900

        CALL MIQR_dealloc_col_mat( data%workspace%Q, 'Q', control, inform )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  record starting addresses for factorization data at each level

      data%n = n
      data%levels = level
      data%pos( 1 ) = 0
      DO i = 1, level
        data%pos( i + 1 ) = data%pos( i ) + data%level( i )%order
      END DO

!  record the total time taken

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + time_now - time_start
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%form = inform%time%form + time_now - time_start
      inform%time%clock_form = inform%time%clock_form + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from MIQR ' )" ) prefix, inform%status
      RETURN

!  internal procedures for subroutine MIQR_form

      CONTAINS

!-*-*-*-*-*-   M I Q R _ F O R M _ L E V E L   S U B R O U T I N E   -*-*-*-*-*-

        SUBROUTINE MIQR_form_level( A, level, A_new, workspace,                &
                                    global, start_time, start_clock,           &
                                    control, inform )

!  find A = ( A1 | A2 ), where the columns of A1 are orthogonal to each other,
!  compute the QR factors of A1, and apply to A2

!  Dummy arguments

        TYPE ( SMT_type ), INTENT( IN ) :: A
        TYPE ( SMT_type ), INTENT( OUT ) :: A_new
        TYPE ( MIQR_data_level_type ) :: level
        TYPE ( MIQR_data_workspace_type ), INTENT( INOUT ) :: workspace
        TYPE ( MIQR_data_global_type ), INTENT( INOUT ) :: global
        TYPE ( MIQR_control_type ), INTENT( INOUT ) :: control
        TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform
        REAL ( KIND = wp ), INTENT( IN ) :: start_time, start_clock

!  Local variables

        INTEGER :: i, j, l, m, n, order, col, id, in, array_size
        REAL ( KIND = wp ) :: alpha, diag, val
        REAL ( KIND = wp ) :: time_start, time_now, clock_start, clock_now
        CHARACTER ( LEN = 80 ) :: array_name
        TYPE ( MIQR_sparse_vector_type ) :: F_i, A_i

!  prefix for all output

        CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
        IF ( LEN( TRIM( control%prefix ) ) > 2 )                               &
          prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

        CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set initial values

        m = A%m ; n = A%n
        level%n = n ; level%order = 1

!  set space for the permutation and its inverse

        array_name = 'miqr: level%PERM'
        CALL SPACE_resize_array( n, level%PERM,                                &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: level%INVERSE_PERM'
        CALL SPACE_resize_array( n, level%INVERSE_PERM,                        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  find A = ( A1 | A2 ), where columns of A1 are orthogonal to each other

!  calculate Euclidean norm of each column of A

        DO i = 1, n
          val = zero
          DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = val + A%val( j ) ** 2
          END DO
          global%A_norms( i ) = SQRT( val )
        END DO

!  find a set of order maximal orthogonal columns to define A1

        IF ( control%out > 0 .AND. control%print_level > 1 )                   &
          WRITE( control%out, "( A, ' finding orthogonal columns' )" ) prefix
        CALL MIQR_find_max_orthogonal_cols( A, level%PERM, level%INVERSE_PERM, &
                                            order, workspace, global,          &
                                            control, inform )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        IF ( control%out > 0 .AND. control%print_level > 1 )                   &
          WRITE( control%out, "( A, 1X, I0, ' orthogonal columns found' )" )   &
            prefix, order

!  check time limits have not been exceeded

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now - start_time > control%cpu_time_limit ) .OR.           &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - start_clock > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit
          GO TO 900
        END IF

!  set D_inverse = diag( 1/||A1_1||, 1/||A1_2||, ..., 1/||A_order|| ),
!  c.f. Li/Saad equation (4.8)

        level%order = order
        array_name = 'miqr: level%perm'
        CALL SPACE_resize_array( order, level%D_inverse,                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        DO i = 1, order
          diag = global%A_norms( level%perm( i ) )
          IF ( diag >= control%smallest_diag ) THEN
            level%D_inverse( i ) = one / diag

!  handle small diagonals

          ELSE
            IF ( control%error > 0 .AND. control%print_level > 0 .AND.         &
                 inform%zero_diagonals == 0 )                                  &
              WRITE( control%error, "( A, ' zero columns encountered' )") prefix
            inform%zero_diagonals = inform%zero_diagonals + 1
            level%D_inverse( i ) = zero
!           level%D_inverse( i ) = one
          END IF
        END DO

!  calculate F = Q1^T * A2, where Q1 = A1 * D^-1, c.f. Li/Saad equations
!  (4.8)-(4.10)

!  set up space for F, stored by columns

        level%F%m = order ; level%F%n = n - order ; level%F%ne = 0
        array_size = MAX( SIZE( A%row ), SIZE( A%val ) ) / 2

        array_name = 'miqr: level%F%ptr'
        CALL SPACE_resize_array( level%F%n + 1, level%F%ptr,                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: level%F%row'
        CALL SPACE_resize_array( array_size, level%F%row,                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: level%F%val'
        CALL SPACE_resize_array( array_size, level%F%val,                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        level%F%ptr( 1 ) = level%F%ne + 1

!  hold the ith row of F in F_i

        F_i%ind => workspace%ind_n ; F_i%pat => workspace%pat_n
        F_i%val => workspace%val_n

        DO i = 1, n - order
          F_i%ne = 0
          in = level%PERM( order + i )
          DO j = workspace%C%ptr( in ), workspace%C%ptr( in + 1 ) - 1
            id = level%INVERSE_PERM( workspace%C%col( j ) )
            IF ( id <= order ) THEN
              F_i%ne = F_i%ne + 1
              F_i%ind( F_i%ne ) = id
              F_i%val( F_i%ne ) = workspace%C%val( j ) * level%D_inverse( id )
            END IF
          END DO

!  ensure that F can accomodate F_i

          CALL MIQR_increase_col_mat_space( level%F, 'level%F',                &
                                            F_i%ne, control, inform )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy F_i to the i-th column of F

          DO j = 1, F_i%ne
            level%F%ne = level%F%ne + 1
            level%F%row( level%F%ne ) = F_i%ind( j )
            level%F%val( level%F%ne ) = F_i%val( j )
          END DO
          level%F%ptr( i + 1 ) = level%F%ne + 1
        END DO

!  nullify pointers

        F_i%val => NULL( ) ; F_i%pat => NULL( ) ; F_i%ind => NULL( )

!  set up space for A_new = A2 - Q1 F, c.f. Li/Saad equation (4.11)

        A_new%m = m ; A_new%n = n - order ; A_new%ne = 0
        array_size = MAX( SIZE( A%row ), SIZE( A%val ) )

        array_name = 'miqr: A_new%ptr'
        CALL SPACE_resize_array( A_new%n + 1, A_new%ptr,                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: A_new%row'
        CALL SPACE_resize_array( array_size, A_new%row,                        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: A_new%val'
        CALL SPACE_resize_array( array_size, A_new%val,                        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        A_new%ptr( 1 ) = A_new%ne + 1

!  calculate A_new one column at a time in A_i

        A_i%val => workspace%val_m ; A_i%pat => workspace%pat_m
        A_i%ind => workspace%ind_m

        DO i = 1, n - order

!  check time limits have not been exceeded

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - start_time > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
                 clock_now - start_clock > control%clock_time_limit ) ) THEN
            inform%status = GALAHAD_error_cpu_limit
            GO TO 900
          END IF

          A_i%dim = A%m ; A_i%ne = 0
          in = level%PERM( order + i )
          DO j = A%ptr( in ), A%ptr( in + 1 ) - 1
            in = A%row( j )
            A_i%ne = A_i%ne + 1
            A_i%ind( A_i%ne ) = in
            A_i%pat( in ) = .TRUE.
            A_i%val( in ) = A%val( j )
          END DO

!  form A_i = A_i - A_i^T q_j * q_j

          DO j = level%F%ptr( i ), level%F%ptr( i + 1 ) - 1
            col = level%F%row( j )
            alpha = - level%F%val( j ) * level%D_inverse( col )
            in = level%perm( col )
            DO l = A%ptr( in ), A%ptr( in + 1 ) - 1
              in = A%row( l ) ; val = A%val( l )

!  there is already an entry in this row

              IF ( A_i%pat( in ) ) THEN
                A_i%val( in ) = A_i%val( in ) + alpha * val

!  there is a fill in

              ELSE
                A_i%val( in ) = alpha * val
                A_i%pat( in ) = .TRUE.
                A_i%ne = A_i%ne + 1
                A_i%ind( A_i%ne ) = in
              END IF
            END DO
          END DO

!  copy A_i to the i-th column of A_new

          CALL MIQR_increase_col_mat_space( A_new, 'A_new', A_i%ne,            &
                                        control, inform )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          DO j = 1, A_i%ne
            in = A_i%ind( j )
            A_new%ne = A_new%ne + 1
            A_new%row( A_new%ne ) = in
            A_new%val( A_new%ne ) = A_i%val( in )
            A_i%pat( in ) = .FALSE.
          END DO
          A_new%ptr( i + 1 ) = A_new%ne + 1
        END DO

        A_i%val => NULL( ) ; A_i%pat => NULL( ) ; A_i%ind => NULL( )

!  record the time forming the multi-level part of the preconditioner

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%levels = inform%time%levels + time_now - time_start
        inform%time%clock_levels                                               &
          = inform%time%clock_levels + clock_now - clock_start

        inform%entries_in_factors                                              &
          = inform%entries_in_factors + level%order + level%F%ne
        inform%status = GALAHAD_ok
        RETURN

!  error returns

 900    CONTINUE
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%levels = inform%time%levels + time_now - time_start
        inform%time%clock_levels                                               &
          = inform%time%clock_levels + clock_now - clock_start
        RETURN

!  end of internal subroutine MIQR_form_level

        END SUBROUTINE MIQR_form_level

!-*-  M I Q R _ F I N D _ M A X _ O R T H O G O N A L _ C O L S  SUBROUTINE  -*-

        SUBROUTINE MIQR_find_max_orthogonal_cols( A, PERM, INVERSE_PERM,       &
                                                  order, workspace, global,    &
                                                  control, inform )

!  find a maximal (independent) set of orthogonal columns of A

!  Dummy arguments

        TYPE ( SMT_type ), INTENT( IN ) :: A
        INTEGER, DIMENSION( 1 : A%n ) :: PERM
        INTEGER, DIMENSION( 1 : A%n ) :: INVERSE_PERM
        INTEGER, INTENT( OUT ) :: order
        TYPE ( MIQR_data_workspace_type ), INTENT( INOUT ) :: workspace
        TYPE ( MIQR_data_global_type ), INTENT( INOUT ) :: global
        TYPE ( MIQR_control_type ), INTENT( INOUT ) :: control
        TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

        INTEGER :: i, ind, j, k, l, m, n, row, col, pm, ipm, ne, array_size
        INTEGER :: max_order
        REAL ( KIND = wp ) :: angle, tol, val
        CHARACTER ( LEN = 80 ) :: array_name
        TYPE ( MIQR_sparse_vector_type ) :: C_i

!  set up local values

        n = A%n ; m = A%m

!  let C = B^T B where B = ( a_1/||a_1||, a_2/||a_2||, ..., a_n/||a_n|| )
!  and a_i is the ith column of A

!  C_i will hold the i-th row of C

        C_i%val => workspace%val_n ; C_i%pat => workspace%pat_n
        C_i%ind => workspace%ind_n

!  set up space for C, stored by rows

        workspace%C%m = n ; workspace%C%n = n ; workspace%C%ne = 0 ;
        array_size = 2 * MAX( SIZE( A%row ), SIZE( A%val ) )

        array_name = 'miqr: workspace%C%ptr'
        CALL SPACE_resize_array( workspace%C%m + 1, workspace%C%ptr,           &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: workspace%C%col'
        CALL SPACE_resize_array( array_size, workspace%C%col,                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: workspace%C%val'
        CALL SPACE_resize_array( array_size, workspace%C%val,                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        workspace%C%ptr( 1 ) = workspace%C%ne + 1

!  ensure that there is sufficient space to store A row-wise in A_by_rows

        workspace%A_by_rows%m = m ; workspace%A_by_rows%n = n
        workspace%A_by_rows%ne = A%ne
        array_size = MAX( SIZE( A%row ), SIZE( A%val ) )
        array_name = 'miqr: workspace%A_by_rows%ptr'
        CALL SPACE_resize_array( m + 1, workspace%A_by_rows%ptr,               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: workspace%A_by_rows%col'
        CALL SPACE_resize_array( array_size, workspace%A_by_rows%col,          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: workspace%A_by_rows%val'
        CALL SPACE_resize_array( array_size, workspace%A_by_rows%val,          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  store A row-wise in A_by_rows

        CALL CONVERT_transpose( A%m, A%n, A%ne, A%ptr, A%row, A%val,           &
                             workspace%A_by_rows%ptr, workspace%A_by_rows%col, &
                             workspace%A_by_rows%val )

!  store the degree of each vertex of the column interesction graph of A,
!  that is the adjacency graph defined by the nonzero pattern of C
!  (upper triangular part only)

        global%nodes_degree( 1 : n ) = 0

!  calculate the i-th row of C: C_i = a_i^T * a(:,i+1:n)

        DO i = 1, n
          C_i%ne = 0
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            row = A%row( l ) ; val = A%val( l )

!  row indices are in increasing order, so stop when index <= i

            DO k = workspace%A_by_rows%ptr( row + 1 ) - 1,                     &
                   workspace%A_by_rows%ptr( row ), - 1
              col = workspace%A_by_rows%col( k )
              IF ( col <= i ) EXIT

!  there is already an entry in this column

              IF ( C_i%pat( col ) ) THEN
                C_i%val( col )                                                 &
                  = C_i%val( col ) + val * workspace%A_by_rows%val( k )

!  there is a fill-in

              ELSE
                C_i%ne = C_i%ne + 1
                C_i%ind( C_i%ne ) = col
                C_i%pat( col ) = .TRUE.
                C_i%val( col ) = val * workspace%A_by_rows%val( k )
              END IF
            END DO
          END DO

!  copy C_i to the i-th row of C

          ne = C_i%ne + global%nodes_degree( i )

!  check that there is sufficient space in C

          IF ( MAX( SIZE(  workspace%C%col ), SIZE(  workspace%C%val ) )       &
                 <  workspace%C%ne + ne ) THEN
            IF ( control%print_level > 2 .AND. control%out > 0 )               &
              WRITE( control%out, "( ' increase space ', I0, ', row ', I0 )" ) &
                ne, i
            CALL MIQR_increase_row_mat_space( workspace%C, 'C', ne,            &
                                              control, inform )
            IF ( inform%status /= GALAHAD_ok ) RETURN
          END IF

          workspace%C%ne = workspace%C%ne + global%nodes_degree( i )

          IF ( C_i%ne > 0 ) THEN
            tol = control%tol_orthogonal * global%A_norms( i )
            DO l = 1, C_i%ne
              col = C_i%ind( l )
              angle = C_i%val( col )

!  if | a_i^T a_j | < tol_orthogonal * |a_i| * |a_j|, a_i is presumed to
!  be approximately orthogonal to a_j

              IF ( ABS( angle ) > tol * global%A_norms( col ) ) THEN
                global%nodes_degree( col ) = global%nodes_degree( col ) + 1
                workspace%C%ne = workspace%C%ne + 1
                workspace%C%col( workspace%C%ne ) = col
                workspace%C%val( workspace%C%ne ) = angle
              END IF

!  unset the pattern

              C_i%pat( col ) = .FALSE.
            END DO
          END IF
          workspace%C%ptr( i + 1 ) = workspace%C%ne + 1
        END DO

!  compute the next available position in each row

        DO i = 1, n
          C_i%ind( i ) = workspace%C%ptr( i ) + global%nodes_degree( i )
          global%nodes_degree( i ) = workspace%C%ptr( i )
        END DO

!  convert to full column intereserction graph from the upper part

        DO i = 1, n
          DO j = C_i%ind( i ), workspace%C%ptr( i + 1 ) - 1
            row = workspace%C%col( j )
            workspace%C%col( global%nodes_degree( row ) ) = i
            workspace%C%val( global%nodes_degree( row ) ) = workspace%C%val( j )
            global%nodes_degree( row ) = global%nodes_degree( row ) + 1
          END DO
        END DO

!  record the row degrees of the graph

        DO i = 1, n
          global%nodes_degree( i )                                             &
            = workspace%C%ptr( i + 1 ) - workspace%C%ptr( i )
        END DO

!  if required, sort the nodes in increaing order of size

        IF ( control%sort ) THEN
          array_name = 'miqr: workspace%PTR'
          CALL SPACE_resize_array( 0, n, workspace%PTR,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

!  compute the number of times each degree occurs (store in PTR, and
!  record the degree for row i in PERM)

          workspace%PTR( 0 : n ) = 0
          DO i = 1, n
            j = global%nodes_degree( i )
            workspace%PTR( j ) = workspace%PTR( j ) + 1
            PERM( i ) = j
          END DO

!  set starting addresses for the rows with each degree

          k = 1
          DO i = 0, n
            j = workspace%PTR( i )
            workspace%PTR( i ) = k
            k = k + j
          END DO

!  march through the rows, assigning each to its place in the degree order

          DO i = 1, n
            j = PERM( i )
            k = workspace%PTR( j )
            global%nodes_index( k ) = i
            global%nodes_degree( k ) = j
            workspace%PTR( j ) = k + 1
          END DO

!  if sort is not required, store the nodes in their natural order

        ELSE
          DO i = 1, n
            global%nodes_index( i ) = i
          END DO
        END IF

!  set the permutation and order (Algorithm 3.1 in Li and Saad). Initialize
!  the independent set as empty

        order = 0
        DO i = 1, n
          PERM( i ) = i ; INVERSE_PERM( i ) = i ; C_i%ind( i ) = 0
        END DO

        max_order = control%max_order
        IF ( max_order <= 0 ) max_order = n

!  loop over all verties in order

        DO i = 1, n

!  find the next vertex in the list

          ind = global%nodes_index( i )

!  see if the vertex has already been marked

          IF ( C_i%ind( ind ) == 1 ) CYCLE

!  mark the vertex

          order = order + 1
          pm = PERM( order ) ; ipm = INVERSE_PERM( ind )
          PERM( order ) = ind ; PERM( ipm ) = pm
          INVERSE_PERM( ind ) = order ; INVERSE_PERM( pm ) = ipm
          IF ( order == max_order ) EXIT

!  mark all adjacent vertices

          C_i%ind( workspace%C%col( workspace%C%ptr( ind ) :                   &
                                    workspace%C%ptr( ind + 1 ) - 1 ) ) = 1
        END DO

!  nullify pointers

        C_i%val => NULL( ) ; C_i%pat => NULL( ) ; C_i%ind => NULL( )

        inform%status = GALAHAD_ok
        RETURN

!  end of internal subroutine MIQR_find_max_orthogonal_cols

        END SUBROUTINE MIQR_find_max_orthogonal_cols

!-*-*-*-*-*-*-   M I Q R _ F O R M _ I Q R   S U B R O U T I N E   -*-*-*-*-*-*-

        SUBROUTINE MIQR_form_iqr( A, R, workspace, start_time, start_clock,    &
                                  control, inform )

!  form an incomplete QR factorization of A using Yousef Saad's ILQ method
!  (J. Comput. Appl. Math. 24 (1988) 89-105), c.f. Li/Saad, Algorithm 2.1

!  Dummy arguments

        TYPE ( SMT_type ), INTENT( IN ) :: A
        TYPE ( SMT_type ), INTENT( OUT ) :: R
        TYPE ( MIQR_data_workspace_type ), INTENT( INOUT ) :: workspace
        TYPE ( MIQR_control_type ), INTENT( INOUT ) :: control
        TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform
        REAL ( KIND = wp ), INTENT( IN ) :: start_time, start_clock

!  Local variables

        INTEGER :: i, k, l, m, n, row, q_next, q_free, in, ind
        INTEGER :: q_list_size, array_size, q_increase_size, q_array_size
        INTEGER :: new_length, old_length, used_length, min_length
        INTEGER :: max_fill, max_fill_q
        REAL ( KIND = wp ) :: r_val_inverse, one_norm, val
        REAL ( KIND = wp ) :: time_start, time_now, clock_start, clock_now
        CHARACTER ( LEN = 80 ) :: array_name
        TYPE ( MIQR_sparse_vector_type ) :: Q_i, R_i

!  prefix for all output

        CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
        IF ( LEN( TRIM( control%prefix ) ) > 2 )                               &
          prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

        CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set up local values

        m = A%m ; n = A%n

!  check to see if R is null

        IF ( n == 0 ) THEN
          R%m = n ; R%n = n ; R%ne = n
          GO TO 800
        END IF

!  compute the maximum number of nonzeros allowed in each column of R

        IF ( control%max_fill >= 0 ) THEN
          max_fill = MIN( control%max_fill, INT( control%average_max_fill      &
                            * ( REAL( A%ne ) / REAL( n ) ) ) )
        ELSE
          max_fill = MIN( n, INT( control%average_max_fill                     &
                            * ( REAL( A%ne ) / REAL( n ) ) ) )
        END IF

!  compute the maximum number of nonzeros allowed in each column of Q

        IF ( control%max_fill_q >= 0 ) THEN
          max_fill_q = MIN( control%max_fill_q,                                &
                            INT( control%average_max_fill_q                    &
                            * ( REAL( A%ne ) / REAL( m ) ) ) )
        ELSE
          max_fill_q = MIN( m, INT( control%average_max_fill_q                 &
                            * ( REAL( A%ne ) / REAL( m ) ) ) )
        END IF

! initialize Q and R stored column wise

        workspace%Q%m = m ; workspace%Q%n = n ; workspace%Q%ne = 0 ;
        array_size = MAX( SIZE( A%row ), SIZE( A%val ) )

        array_name = 'miqr: workspace%Q%ptr'
        CALL SPACE_resize_array( workspace%Q%n + 1, workspace%Q%ptr,           &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: workspace%Q%row'
        CALL SPACE_resize_array( array_size, workspace%Q%row,                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: workspace%Q%val'
        CALL SPACE_resize_array( array_size, workspace%Q%val,                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        R%m = n ; R%n = n ; R%ne = n ; array_size = array_size + n

        array_name = 'miqr: R%ptr'
        CALL SPACE_resize_array( R%n + 1, R%ptr,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: R%row'
        CALL SPACE_resize_array( array_size, R%row,                            &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: R%val'
        CALL SPACE_resize_array( array_size, R%val,                            &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        workspace%Q%ptr( 1 ) = workspace%Q%ne + 1 ; R%ptr( 1 ) = R%ne + 1

!  initialize buffers for q_i and r_i, the i-th columns of Q and R

        Q_i%val => workspace%val_m ; Q_i%pat => workspace%pat_m
        Q_i%ind => workspace%ind_m
        R_i%val => workspace%val_n ; R_i%pat => workspace%pat_n
        R_i%ind => workspace%ind_n

!  initialize the linked list for Q stored row wise, using the array
!  Q_list_col/next/val (see Saad's ILQ)
!  * Q_list_*(i) points to the i-th row of Q
!  * q_array_size is the current size of Q_list
!  * q_increase_size is the size to increase when the current Q_list is full
!  * q_free is the next available free position in Q_list

        q_list_size = m
        q_array_size = m + array_size - n
        q_increase_size = control%increase_size

        array_name = 'miqr: workspace%Q%list_col'
        CALL SPACE_resize_array( q_array_size, workspace%Q_list_col,           &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: workspace%Q%list_next'
        CALL SPACE_resize_array( q_array_size, workspace%Q_list_next,          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'miqr: workspace%Q%list_val'
        CALL SPACE_resize_array( q_array_size, workspace%Q_list_val,           &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  the end of the list for each row is signified by -1

        workspace%Q_list_next( 1 : m ) = - 1
        q_free = m + 1

!       DO i = 1, MIN( m, n )
        DO i = 1, n

!  check time limits have not been exceeded

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          IF ( ( control%cpu_time_limit >= zero .AND.                          &
                 time_now - start_time > control%cpu_time_limit ) .OR.         &
               ( control%clock_time_limit >= zero .AND.                        &
                 clock_now - start_clock > control%clock_time_limit ) ) THEN
            inform%status = GALAHAD_error_cpu_limit
            GO TO 900
          END IF

!  initialize the i-th columns of Q and R

          Q_i%dim = m ; Q_i%ne = 0
          R_i%dim = n ; R_i%ne = 0

!  use Saad's ILQ method to calculate entries in the i-th column r_i of R

          one_norm = zero
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            row = A%row( l ) ; val = A%val( l )
            one_norm = one_norm + ABS( val )
            Q_i%ne = Q_i%ne + 1
            Q_i%ind( Q_i%ne ) = row
            Q_i%val( row ) = val
            Q_i%pat( row ) = .TRUE.
            q_next = workspace%Q_list_next( row )

            DO
              IF ( q_next == - 1 ) EXIT
              ind = workspace%Q_list_col( q_next )

!  there is already an entry in this row

              IF ( R_i%pat( ind ) ) THEN
                R_i%val( ind )                                                 &
                  = R_i%val( ind ) + val * workspace%Q_list_val( q_next )

!  there is a fill-in

              ELSE
                R_i%ne = R_i%ne + 1
                R_i%ind( R_i%ne ) = ind
                R_i%pat( ind ) = .TRUE.
                R_i%val( ind ) = val * workspace%Q_list_val( q_next )
              END IF
              q_next = workspace%Q_list_next( q_next )
            END DO
          END DO
          one_norm = one_norm * control%tol_drop

!  reset the pattern

          R_i%pat( R_i%ind( 1 : R_i%ne ) ) = .FALSE.

!  keep only the max_fill largest elements (in absolute value) in r_i

          IF ( R_i%ne > max_fill ) THEN
            CALL SPLIT( R_i%val, n, R_i%ind, R_i%ne, max_fill )
            R_i%ne = max_fill
            inform%drop = inform%drop + R_i%ne - max_fill
          END IF

!  calculate a_i - sum_j ( a_i^T q_j ) q_j = a_i - sum_j r_j q_j, where
!  r_i = a_i^t q_j

          CALL MIQR_increase_col_mat_space( R, 'R', R_i%ne, control, inform )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
          DO l = 1, R_i%ne
            in = R_i%ind( l ) ; val = R_i%val( in )
            DO k = workspace%Q%ptr( in ), workspace%Q%ptr( in + 1 ) - 1
              ind = workspace%Q%row( k )

!  there is already an entry in this row

              IF ( Q_i%pat( ind ) ) THEN
                Q_i%val( ind ) = Q_i%val( ind ) - val * workspace%Q%val( k )

!  there is a fill in

              ELSE
                Q_i%val( ind ) = - val * workspace%Q%val( k )
                Q_i%pat( ind ) = .TRUE.
                Q_i%ne = Q_i%ne + 1
                Q_i%ind( Q_i%ne ) = ind
              END IF
            END DO

!  drop elements that are small relative to the one norm of a_i

            IF ( ABS( val ) >= one_norm ) THEN
              R%ne = R%ne + 1
              R%row( R%ne ) = in
              R%val( R%ne ) = val
            ELSE
              inform%drop = inform%drop + 1
            END IF
          END DO
          R%ptr( i + 1 ) = R%ne + 1

!  reset the pattern of q_i

          Q_i%pat( Q_i%ind( 1 : Q_i%ne ) ) = .FALSE.

!  keep only the max_fill_q largest elements (in absolute value) in q_i

          IF ( Q_i%ne > max_fill_q ) THEN
            CALL SPLIT( Q_i%val, m, Q_i%ind, Q_i%ne, max_fill_q )
            Q_i%ne = max_fill_q
!           inform%drop = inform%drop + Q_i%ne - max_fill_q
          END IF

!  compute the Euclidean norm of q_i

          IF ( Q_i%ne > 0 ) THEN
            val = SQRT( DOT_PRODUCT( Q_i%val( Q_i%ind( 1 : Q_i%ne ) ),         &
                                     Q_i%val( Q_i%ind( 1 : Q_i%ne ) ) ) )
          ELSE
            val = zero
          END IF

!  store the inverse of the diagonal of R as the first n elements of R
!  for more effficient solves

!         IF ( val /= zero ) THEN
          IF ( val >= control%smallest_diag ) THEN
            r_val_inverse = one / val
            R%val( i ) = r_val_inverse

!  take precautions if the diagonal of R is small

          ELSE
            IF ( control%error > 0 .AND. control%print_level > 0 .AND.         &
                 inform%zero_diagonals == 0 )                                  &
              WRITE( control%error, "( A, ' zero columns encountered' )") prefix
            inform%zero_diagonals = inform%zero_diagonals + 1
!           r_val = control%smallest_diag
!           r_val_inverse = one / val
            r_val_inverse = zero
            R%val( i ) = r_val_inverse
          END IF

!  if the current Q_list is not large enough, enlarge it

          IF ( q_list_size + Q_i%ne > q_array_size ) THEN
            new_length = q_array_size + q_increase_size
            DO
             IF ( q_list_size + Q_i%ne <= new_length ) EXIT
             new_length = new_length + q_increase_size
            END DO

            old_length = q_array_size ; used_length = q_free - 1
            min_length = q_list_size + Q_i%ne
            CALL SPACE_extend_array( workspace%Q_list_col,                     &
              old_length, used_length, new_length, min_length,                 &
              control%buffer, inform%status, inform%alloc_status )
            IF ( inform%status /= 0 ) THEN
              inform%bad_alloc = 'miqr: workspace%Q_list_col' ; GO TO 900
            END IF

            old_length = q_array_size ; used_length = q_free - 1
            min_length = q_list_size + Q_i%ne
            CALL SPACE_extend_array( workspace%Q_list_next,                    &
              old_length, used_length, new_length, min_length,                 &
              control%buffer, inform%status, inform%alloc_status )
            IF ( inform%status /= 0 ) THEN
              inform%bad_alloc = 'miqr: workspace%Q_list_next' ; GO TO 900
            END IF

            old_length = q_array_size ; used_length = q_free - 1
            min_length = q_list_size + Q_i%ne
            CALL SPACE_extend_array( workspace%Q_list_val,                     &
              old_length, used_length, new_length, min_length,                 &
              control%buffer, inform%status, inform%alloc_status )
            IF ( inform%status /= 0 ) THEN
              inform%bad_alloc = 'miqr: workspace%Q_list_next' ; GO TO 900
            END IF

            q_array_size = new_length
          END IF

!  do the same to Q

          CALL MIQR_increase_col_mat_space( workspace%Q, 'Q', Q_i%ne,          &
                                            control, inform )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  update Q to accommodate q_i

          DO l = 1, Q_i%ne
            ind = Q_i%ind( l )
            val = Q_i%val( ind ) * r_val_inverse
            workspace%Q%ne = workspace%Q%ne + 1
            workspace%Q%row( workspace%Q%ne ) = ind
            workspace%Q%val( workspace%Q%ne ) = val

!  update Q_list to add the row data from q_i

            workspace%Q_list_col( q_free ) = i
            workspace%Q_list_val( q_free ) = val
            workspace%Q_list_next( q_free ) = workspace%Q_list_next( ind )
            workspace%Q_list_next( ind ) = q_free
            q_list_size = q_list_size + 1
            q_free = q_free + 1
          END DO
          workspace%Q%ptr( i + 1 ) = workspace%Q%ne + 1
        END DO

!  tidy up the singular part if m < n

!       DO i = MIN( m, n ) + 1, n
!         R%val( i ) = one
!         R%ptr( i + 1 ) = R%ne + 1
!       END DO

!  nullify pointers

        Q_i%val => NULL( ) ; Q_i%pat => NULL( ) ; Q_i%ind => NULL( )
        R_i%val => NULL( ) ; R_i%pat => NULL( ) ; R_i%ind => NULL( )

!  record the time forming the IQR part of the preconditioner

 800    CONTINUE
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%iqr = inform%time%iqr + time_now - time_start
        inform%time%clock_iqr = inform%time%clock_iqr + clock_now - clock_start

        inform%entries_in_factors = inform%entries_in_factors + R%ne
        inform%status = GALAHAD_ok
        RETURN

!  error returns

 900    CONTINUE
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%iqr = inform%time%iqr + time_now - time_start
        inform%time%clock_iqr = inform%time%clock_iqr + clock_now - clock_start
        RETURN

! end of internal subroutine MIQR_form_iqr

        END SUBROUTINE MIQR_form_iqr

!-*-*-*-*-*-*-*-   S A A D ' S   S P L I T   S U B R O U T I N E   -*-*-*-*-*-*-

        SUBROUTINE SPLIT( A, m, IND, ne, cut )

!  permute an array IND so that its elements satisfy

!  ABS(A(IND(i))) >= ABS(A(IND(cut))) for i < cut and
!  ABS(A(IND(i))) <= ABS(A(IND(cut))) for i > cut

!  author: Youcef Saad, Oct 31, 1989 (qsplit2, fortran 77)
!  modified by Nick Gould, May 2014

!  Dummy arguments

        INTEGER :: m, ne, cut
        INTEGER, DIMENSION( ne ) :: IND
        REAL ( KIND = wp ), DIMENSION( m ) :: A

!  Local variables

        INTEGER :: i, j, first, mid, last
        REAL ( KIND = wp ) :: val

        first = 1 ; last = ne
        IF ( cut < first .OR. cut > last ) RETURN

!  loop until mid = cut

        DO
          mid = first
          val = ABS( A( IND( mid ) ) )
          DO j = first + 1, last
            IF ( ABS( A( IND( j ) ) ) > val ) THEN
              mid = mid + 1

!  interchange

              i = IND( mid ) ; IND( mid ) = IND( j ) ; IND( j ) = i
            END IF
          END DO

!  interchange

          i = IND( mid ) ; IND( mid ) = IND( first ) ; IND( first ) = i

!  test for termination

          IF ( mid == cut ) THEN
            RETURN
          ELSE IF ( mid > cut ) THEN
            last = mid - 1
          ELSE
            first = mid + 1
          END IF
        END DO
        RETURN

!  end of subroutine SPLIT

        END SUBROUTINE SPLIT

!  end of subroutine MIQR_form

      END SUBROUTINE MIQR_form

!-*-*-   M I Q R _ C O P Y _ C O L _ M A T R I X   S U B R O U T I N E   -*-*-

      SUBROUTINE MIQR_copy_col_matrix( A_in, A_out, control, inform )

!  initialize space and copy data from the matrix A_in to A_out stored
!  by columns

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A_in
      TYPE ( SMT_type ), INTENT( OUT ) :: A_out
      TYPE ( MIQR_control_type ), INTENT( INOUT ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: array_size
      CHARACTER ( LEN = 80 ) :: array_name

      A_out%m = A_in%m ; A_out%n = A_in%n ; A_out%ne = A_in%ne
      array_size = MAX( SIZE( A_in%row ), SIZE( A_in%val ) )

      array_name = 'miqr: A_out%ptr'
      CALL SPACE_resize_array( A_out%n + 1, A_out%ptr,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN
      A_out%ptr( 1 : A_out%n + 1 ) = A_in%ptr( 1 : A_in%n + 1 )

      array_name = 'miqr: A_out%row'
      CALL SPACE_resize_array( array_size, A_out%row,                          &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN
      A_out%row( 1 : A_out%ne ) = A_in%row( 1 : A_in%ne )

      array_name = 'miqr: A_out%val'
      CALL SPACE_resize_array( array_size, A_out%val,                          &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN
      A_out%val( 1 : A_out%ne ) = A_in%val( 1 : A_in%ne )

      RETURN

!  end of subroutine MIQR_copy_col_matrix

      END SUBROUTINE MIQR_copy_col_matrix

!-*-*-   M I Q R _ D E A L L O C _ R O W _ M A T   S U B R O U T I N E   -*-*-

      SUBROUTINE MIQR_dealloc_row_mat( mat, mat_name, control, inform )

!  deallocate the array components of a variable mat of SMT_type
!  held by rows named mat_name

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( INOUT ) :: mat
      CHARACTER ( LEN = * ) :: mat_name
      TYPE ( MIQR_control_type ), INTENT( IN ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'miqr: ' // TRIM( mat_name ) // '%ptr'
      CALL SPACE_dealloc_array( mat%ptr,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: ' // TRIM( mat_name ) // '%col'
      CALL SPACE_dealloc_array( mat%col,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: ' // TRIM( mat_name ) // '%val'
      CALL SPACE_dealloc_array( mat%val,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  end of subroutine MIQR_dealloc_row_mat

      END SUBROUTINE MIQR_dealloc_row_mat

!-*-*-   M I Q R _ D E A L L O C _ C O L _ M A T   S U B R O U T I N E   -*-*-

      SUBROUTINE MIQR_dealloc_col_mat( mat, mat_name, control, inform )

!  deallocate the array components of a variable mat of SMT_type
!  held by columns named mat_name

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( INOUT ) :: mat
      CHARACTER ( LEN = * ) :: mat_name
      TYPE ( MIQR_control_type ), INTENT( IN ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

      array_name = 'miqr: ' // TRIM( mat_name ) // '%ptr'
      CALL SPACE_dealloc_array( mat%ptr,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: ' // TRIM( mat_name ) // '%row'
      CALL SPACE_dealloc_array( mat%row,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: ' // TRIM( mat_name ) // '%val'
      CALL SPACE_dealloc_array( mat%val,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  end of subroutine MIQR_dealloc_col_mat

      END SUBROUTINE MIQR_dealloc_col_mat

!- M I Q R _ I N C R E A S E _ R O W _ M A T _ S P A C E   S U B R O U T I N E -

      SUBROUTINE MIQR_increase_row_mat_space( mat, name, extra, control,       &
                                              inform )

!  expand arrays for matrices stored in SMT_type format
!  to ensure that there is at least extra unused storage locations

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( INOUT ) :: mat
      CHARACTER ( len = * ), INTENT( IN ) :: name
      INTEGER, INTENT( IN ) :: extra
      TYPE ( MIQR_control_type ), INTENT( IN ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: old_length, used_length, min_length, new_length, new_size
      INTEGER :: array_size

      inform%status = GALAHAD_ok

!  compute the array size

      array_size = MAX( SIZE( mat%col ), SIZE( mat%val ) )

!  check to see if there is already sufficient space

      IF ( array_size < mat%ne + extra ) THEN

!  more space is needed, so compute how much (in multiples of increase_size)

        new_size = array_size
        DO
          new_size = new_size + control%increase_size
          IF ( new_size >= mat%ne + extra ) EXIT
        END DO

!  extend all available arrays

        old_length = array_size ; used_length = mat%ne
        new_length = new_size ; min_length = array_size + 1
        CALL SPACE_extend_array( mat%col,                                      &
          old_length, used_length, new_length, min_length,                     &
          control%buffer, inform%status, inform%alloc_status )
        IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'miqr: ' // TRIM( name ) // '%col' ; RETURN
        END IF

        old_length = array_size ; used_length = mat%ne
        new_length = new_size ; min_length = array_size + 1
        CALL SPACE_extend_array( mat%val,                                      &
          old_length, used_length, new_length, min_length,                     &
          control%buffer, inform%status, inform%alloc_status )
        IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'miqr: ' // TRIM( name ) // '%val' ; RETURN
        END IF
      END IF

!  end of subroutine MIQR_increase_row_mat_space

      END SUBROUTINE MIQR_increase_row_mat_space

!- M I Q R _ I N C R E A S E _ C O L _ M A T _ S P A C E   S U B R O U T I N E -

      SUBROUTINE MIQR_increase_col_mat_space( mat, name, extra, control,       &
                                              inform )

!  expand arrays for matrices stored in SMT_type format
!  to ensure that there is at least extra unused storage locations

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( INOUT ) :: mat
      CHARACTER ( len = * ), INTENT( IN ) :: name
      INTEGER, INTENT( IN ) :: extra
      TYPE ( MIQR_control_type ), INTENT( IN ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: old_length, used_length, min_length, new_length, new_size
      INTEGER :: array_size

      inform%status = GALAHAD_ok

!  compute the array size

      array_size = MAX( SIZE( mat%row ), SIZE( mat%val ) )

!  check to see if there is already sufficient space

      IF ( array_size < mat%ne + extra ) THEN

!  more space is needed, so compute how much (in multiples of increase_size)

        new_size = array_size
        DO
          new_size = new_size + control%increase_size
          IF ( new_size >= mat%ne + extra ) EXIT
        END DO

!  extend all available arrays

        old_length = array_size ; used_length = mat%ne
        new_length = new_size ; min_length = array_size + 1
        CALL SPACE_extend_array( mat%row,                                      &
          old_length, used_length, new_length, min_length,                     &
          control%buffer, inform%status, inform%alloc_status )
        IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'miqr: ' // TRIM( name ) // '%row' ; RETURN
        END IF

        old_length = array_size ; used_length = mat%ne
        new_length = new_size ; min_length = array_size + 1
        CALL SPACE_extend_array( mat%val,                                      &
          old_length, used_length, new_length, min_length,                     &
          control%buffer, inform%status, inform%alloc_status )
        IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'miqr: ' // TRIM( name ) // '%val' ; RETURN
        END IF
      END IF

!  end of subroutine MIQR_increase_col_mat_space

      END SUBROUTINE MIQR_increase_col_mat_space

!-*-*-*-*-*-*-*-   M I Q R _ A P P L Y   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE MIQR_apply( SOL, transpose, data, inform )

!  given the matrix MQR = ( D_1 <-    F_1   -> )
!                         (     D_2  <- F_2- > )
!                         (         .          )
!                         (             R      )

!  solve (MQR) * sol = rhs (transpose = false) or
!        (MQR)^T * sol = rhs (transpose = true)
!  rhs is input in SOL and SOL is subsequently overwritten by sol

!  Dummy arguments

      TYPE ( MIQR_data_type ), INTENT( INOUT ) :: data
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 1 : data%n ) :: SOL
      LOGICAL, INTENT( IN ) :: transpose
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, k, l, pos_r, pos_i, pos_i1
      REAL ( KIND = wp ) :: val, time_start, time_now, clock_start, clock_now
      REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: WORK

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      WORK => data%workspace%val_n

!  ---------------------
!  solve (MQR)^T * x = y
!  ---------------------

      IF ( transpose ) THEN

!  first consider the solution components arising from the multi-level blocks

        DO i = 1, data%levels
          pos_i = data%pos( i ) ; pos_i1 = data%pos( i + 1 )

!  permute the rhs at the i-th level

          DO j = 1, data%level( i )%n
            WORK( j ) = SOL( pos_i + data%level( i )%perm( j ) )
          END DO

!  compute the solution w = D_i^{-1} rhs at the i-th level

          DO j = 1, data%level( i )%order
            SOL( pos_i + j ) = WORK( j ) * data%level( i )%D_inverse( j )
          END DO

!  adjust the remaining rhs to account for F_i^T w

          DO j = 1, data%level( i )%F%n
            val = zero
            DO l = data%level( i )%F%ptr( j ),                                 &
                   data%level( i )%F%ptr( j + 1 ) - 1
              val = val + data%level( i )%F%val( l )                           &
                      * SOL( pos_i + data%level( i )%F%row( l ) )
            END DO
            SOL( pos_i1 + j ) = val
          END DO

          l = data%level( i )%order
          DO j = 1, data%level( i )%n - l
            SOL( pos_i1 + j ) = WORK( l + j ) - SOL( pos_i1 + j )
          END DO
        END DO

!  deal with the final block

        IF ( data%R%n > 0 ) THEN
          pos_r = data%pos( data%levels + 1 )

!  apply forward solution using R^T

          DO i = 1, data%R%n
            val = SOL( pos_r + i )
            DO j = data%R%ptr( i ), data%R%ptr( i + 1 ) - 1
              val = val - SOL( pos_r + data%R%row( j ) ) * data%R%val( j )
            END DO
            SOL( pos_r + i ) = val * data%R%val( i )
          END DO
        END IF

!  -------------------
!  solve (MQR) * x = y
!  -------------------

      ELSE

!  first deal with the final block

        IF ( data%R%n > 0 ) THEN
          pos_r = data%pos( data%levels + 1 )

!  apply backward solution using R

          DO i = data%R%n, 1, - 1
            SOL( pos_r + i ) = SOL( pos_r + i ) * data%R%val( i )
            val = SOL( pos_r + i )
            DO j = data%R%ptr( i ), data%R%ptr( i + 1 ) - 1
              SOL( pos_r + data%R%row( j ) )                                   &
                = SOL( pos_r + data%R%row( j ) ) - data%R%val( j ) * val
            END DO
          END DO
        END IF

!  now consider the solution components arising from the multi-level blocks

        DO i = data%levels, 1, - 1
          pos_i = data%pos( i ) ; pos_i1 = data%pos( i + 1 )

!  form w <- x_i - F_i x_i+1:n

          DO j = 1, data%level( i )%order
            WORK( j ) = SOL( pos_i + j )
          END DO

          DO j = 1, data%level( i )%F%n
            val = SOL( pos_i1 + j )
            DO l = data%level( i )%F%ptr( j ),                                 &
                   data%level( i )%F%ptr( j + 1 ) - 1
              k = data%level( i )%F%row( l )
              WORK( k ) = WORK( k ) - data%level( i )%F%val( l ) * val
            END DO
          END DO

!  form w <- D_i^{-1} w

          DO j = 1, data%level( i )%order
            WORK( j ) = WORK( j ) * data%level( i )%D_inverse( j )
          END DO

!  append the permuted solution at the i+1-st level onwards

          DO j = 1, data%level( i )%n - data%level( i )%order
            WORK( data%level( i )%order + j ) =  SOL( pos_i1 + j )
          END DO

!  un-permute the solution at the i-th level

          DO j = 1, data%level( i )%n
            SOL( pos_i + j ) = WORK( data%level( i )%inverse_perm( j ) )
          END DO
        END DO
      END IF
      WORK => NULL( )

!  record the time taken applying the preconditioner

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%apply = inform%time%apply + time_now - time_start
      inform%time%clock_apply                                                  &
        = inform%time%clock_apply + clock_now - clock_start

      inform%status = GALAHAD_ok
      RETURN

!  end of subroutine MIQR_apply

      END SUBROUTINE MIQR_apply

!-*-*-*-*-*-   M I Q R _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE MIQR_terminate( data, control, inform )

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

      TYPE ( MIQR_control_type ), INTENT( IN ) :: control
      TYPE ( MIQR_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( MIQR_data_type ), INTENT( INOUT ) :: data

!  Local variables

      INTEGER :: i
      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate all remaining allocated arrays

      CALL MIQR_dealloc_col_mat( data%A_by_cols, 'A_by_cols', control, inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL MIQR_dealloc_col_mat( data%A_new, 'A_new', control, inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL MIQR_dealloc_col_mat( data%R, 'R', control, inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL MIQR_dealloc_row_mat( data%workspace%A_by_rows, 'A_by_rows',        &
                                 control, inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL MIQR_dealloc_row_mat( data%workspace%C, 'C', control, inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL MIQR_dealloc_col_mat( data%workspace%Q, 'Q', control, inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%PTR'
      CALL SPACE_dealloc_array( data%workspace%PTR,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%Q_list_next'
      CALL SPACE_dealloc_array( data%workspace%Q_list_next,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%Q_list_col'
      CALL SPACE_dealloc_array( data%workspace%Q_list_col,                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%Q_list_val'
      CALL SPACE_dealloc_array( data%workspace%Q_list_val,                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%ind_m'
      CALL SPACE_dealloc_pointer( data%workspace%ind_m,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%pat_m'
      CALL SPACE_dealloc_pointer( data%workspace%pat_m,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%val_m'
      CALL SPACE_dealloc_pointer( data%workspace%val_m,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%ind_n'
      CALL SPACE_dealloc_pointer( data%workspace%ind_n,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%pat_n'
      CALL SPACE_dealloc_pointer( data%workspace%pat_n,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%val_n'
      CALL SPACE_dealloc_pointer( data%workspace%val_n,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'miqr: data%workspace%val_n'
      CALL SPACE_dealloc_pointer( data%workspace%val_n,                        &
         inform%status, inform%alloc_status, point_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      DO i = 1, data%levels
        array_name = 'miqr: data%global%A_norms'
        CALL SPACE_dealloc_array( data%global( i )%A_norms,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: data%global%nodes_degree'
        CALL SPACE_dealloc_array( data%global( i )%nodes_degree,               &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: data%global%nodes_index'
        CALL SPACE_dealloc_array( data%global( i )%nodes_index,                &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: data%level%PERM'
        CALL SPACE_dealloc_array( data%level( i )%PERM,                        &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: data%level%INVERSE_PERM'
        CALL SPACE_dealloc_array( data%level( i )%INVERSE_PERM,                &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'miqr: data%level%D_inverse'
        CALL SPACE_dealloc_array( data%level( i )%D_inverse,                   &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        CALL MIQR_dealloc_col_mat( data%level( i )%F, 'F', control, inform )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN
      END DO

      RETURN

!  end of subroutine MIQR_terminate

      END SUBROUTINE MIQR_terminate

!  end of module GALAHAD_MIQR_double

    END MODULE GALAHAD_MIQR_double

