! THIS VERSION: GALAHAD 3.0 - 04/04/2018 AT 09:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ C R O   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started August 5th 2010
!   added limited-memory H capability January 6th 2015
!   added optional refinement April 4th 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_CRO_double

!      -----------------------------------------------------------------
!     |                                                                |
!     | Given values x, y and z that satisfy                           |
!     |                                                                |
!     |   H x + g - A^T y - z = 0, c_l <= A x <= c_u, x_l <= x <= x_u  |
!     |     { >= 0 if A x = c_l             { >= 0 if x = x_l          |
!     |   y {  = 0 if c_l < A x < c_u and z {  = 0 if x_l < x < x_u    |
!     |     { <= 0 if A x = c_u             { <= 0 if x = x_u          |
!     |                                                                |
!     | adjust x, y and z to maintain these relationships but at the   |
!     | same time reduce the number of nonzero components of y and z.  |
!     | Ultimately a ``basic'' solution, i.e., one for which the       |
!     | submatrix of A formed from rows with nonzero y and columns     |
!     | from nonzero z is of full rank, is obtained                    |
!     |                                                                |
!      -----------------------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_SLS_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_IR_double
      USE GALAHAD_ULS_double
      USE GALAHAD_SCU_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_ies
      USE GALAHAD_LMS_double, ONLY: LMS_data_type, LMS_apply_lbfgs

      USE GALAHAD_MOP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CRO_initialize, CRO_read_specfile, CRO_crossover,              &
                CRO_terminate, SMT_type, SMT_put, SMT_get, LMS_data_type

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
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE CRO_crossover
       MODULE PROCEDURE CRO_crossover_h_sparse_by_rows, CRO_crossover_h_lm,    &
                        CRO_crossover_no_h
     END INTERFACE CRO_crossover

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CRO_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   the maximum permitted size of the Schur complement before a refactorization
!     is performed

        INTEGER :: max_schur_complement = 100

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   feasibility tolerance for KKT violation

        REAL ( KIND = wp ) :: feasibility_tolerance = SQRT( epsmch )

!   if %check_io is true, the input (x,y,z) will be fully tested for consistency

        LOGICAL :: check_io = .FALSE.

!   if %refine solution is true, attempt to satisfy the KKT conditions as
!    accurately as possible

        LOGICAL :: refine_solution = .FALSE.

!   if %space_critical is true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  indefinite linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "sils" // REPEAT( ' ', 26 )

!  unsymmetric linear equation solver

        CHARACTER ( LEN = 30 ) :: unsymmetric_linear_solver =                  &
           "gls" // REPEAT( ' ', 27 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for ULS

        TYPE ( ULS_control_type ) :: ULS_control

!  control parameters for iterative refinement

        TYPE ( IR_control_type ) :: IR_control

      END TYPE CRO_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CRO_time_type

!  the total CPU time spent in the package

        REAL :: total = 0.0

!  the CPU time spent reordering the matrix prior to factorization

        REAL :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL :: factorize = 0.0

!  the CPU time spent computing corrections

        REAL :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing corrections

        REAL ( KIND = wp ) :: clock_solve = 0.0

      END TYPE

      TYPE, PUBLIC :: CRO_inform_type

!  return status. See CRO_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the number of dependent active constraints

        INTEGER :: dependent = 0

!  timings (see above)

        TYPE ( CRO_time_type ) :: time

!  information from SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

!  information from SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  information from ULS

        TYPE ( ULS_inform_type ) :: ULS_inform

!  information from SCU

        INTEGER :: scu_status = 0
        TYPE ( SCU_info_type ) :: SCU_inform

!  information from IR

        TYPE ( IR_inform_type ) :: IR_inform

      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   data derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CRO_data_type
        PRIVATE
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_inorder
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_inorder
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_free
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_fixed
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: BASIS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SLS_SOL
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DC
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_p
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_d
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: W

        TYPE ( SMT_type ) :: AT
        TYPE ( SMT_type ) :: K_r
        TYPE ( SMT_type ) :: H_r
        TYPE ( SMT_type ) :: A_r
        TYPE ( SMT_type ) :: C_r

!  an editable copy of control

        TYPE ( CRO_control_type ) :: control

!  private data for SLS

        TYPE ( SLS_data_type ) :: SLS_data

!  private data for SBLS

        TYPE ( SBLS_data_type ) :: SBLS_data

!  private data for ULS

        TYPE ( ULS_data_type ) :: ULS_data

!  private data for SCU

        TYPE ( SCU_matrix_type ) :: SCU_matrix
        TYPE ( SCU_data_type ) :: SCU_data

!  private type for IR

        TYPE ( IR_data_type ) :: IR_data
      END TYPE

   CONTAINS

!-*-*-*-*-*-   C R O _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE CRO_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for CRO. This routine should be called before
!  CRO_solve
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

      TYPE ( CRO_data_type ), INTENT( INOUT ) :: data
      TYPE ( CRO_control_type ), INTENT( OUT ) :: control
      TYPE ( CRO_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok
!     control%feasibility_tolerance = SQRT( epsmch )

!  Initalize SLS components

      CALL SLS_initialize( control%symmetric_linear_solver, data%SLS_data,     &
                           control%SLS_control, inform%SLS_inform )
      control%SLS_control%ordering = 0
      control%SLS_control%prefix = '" - SLS:"                    '

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%prefix = '" - SBLS:"                    '

!  Initalize ULS components

      CALL ULS_initialize( control%unsymmetric_linear_solver, data%ULS_data,   &
                           control%ULS_control, inform%ULS_inform )
      control%ULS_control%prefix = '" - ULS:"                    '

!  Set initial values for solve/refinement controls

      CALL IR_initialize( data%IR_data, control%IR_control, inform%IR_inform )
      control%IR_control%prefix = '" - IR:"'

      RETURN

!  End of CRO_initialize

      END SUBROUTINE CRO_initialize

!-*-*-*-*-   C R O _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE CRO_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by CRO_initialize could (roughly)
!  have been set as:

!  BEGIN CRO SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   maximum-dimension-of-schur-complement           100
!   infinity-value                                  1.0D+19
!   feasibility-tolerance                           1.0D-8
!   check-input-output                              yes
!   refine-solution                                 no
!   space-critical                                  no
!   deallocate-error-fatal                          no
!   symmetric-linear-equation-solver                sils
!   unsymmetric-linear-equation-solver              gls
!   output-line-prefix                              ""
!  END CRO SPECIFICATIONS

!  Dummy arguments

      TYPE ( CRO_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: max_schur_complement = print_level + 1
      INTEGER, PARAMETER :: infinity = max_schur_complement + 1
      INTEGER, PARAMETER :: feasibility_tolerance = infinity + 1
      INTEGER, PARAMETER :: check_io = feasibility_tolerance + 1
      INTEGER, PARAMETER :: refine_solution = check_io + 1
      INTEGER, PARAMETER :: space_critical = refine_solution + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: symmetric_linear_solver = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: unsymmetric_linear_solver =                        &
                              symmetric_linear_solver + 1
      INTEGER, PARAMETER :: prefix = unsymmetric_linear_solver + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'CRO'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( max_schur_complement )%keyword =                                   &
        'maximum-dimension-of-schur-complement'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( feasibility_tolerance )%keyword = 'feasibility-tolerance'

!  Logical key-words

      spec( check_io )%keyword = 'check-input-output'
      spec( refine_solution )%keyword = 'refine-solution'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
      spec( unsymmetric_linear_solver )%keyword =                              &
        'unsymmetric-linear-equation-solver'
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
      CALL SPECFILE_assign_value( spec( max_schur_complement ),                &
                                  control%max_schur_complement,                &
                                  control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feasibility_tolerance ),                &
                                 control%feasibility_tolerance,                &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( check_io ),                            &
                                  control%check_io,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( refine_solution ),                     &
                                  control%refine_solution,                     &
                                  control%error )
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
      CALL SPECFILE_assign_value( spec( unsymmetric_linear_solver ),           &
                                  control%unsymmetric_linear_solver,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read the controls for the linear solvers

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS')
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-SBLS')
        CALL ULS_read_specfile( control%ULS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-ULS')
        CALL IR_read_specfile( control%IR_control, device,                     &
                               alt_specname = TRIM( alt_specname ) // '-IR' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
        CALL SBLS_read_specfile( control%SBLS_control, device )
        CALL ULS_read_specfile( control%ULS_control, device )
        CALL IR_read_specfile( control%IR_control, device )
      END IF

      RETURN

      END SUBROUTINE CRO_read_specfile

!-  C R O _ C R O S S O V E R _ H _ S P A R S E _ B Y _ R O W S   SUBROUTINE  -

      SUBROUTINE CRO_crossover_h_sparse_by_rows( n, m, m_equal, H_val, H_col,  &
                                                 H_ptr, A_val, A_col, A_ptr,   &
                                                 G, C_l, C_u, X_l, X_u, C, X,  &
                                                 Y, Z, C_stat, X_stat, data,   &
                                                 control, inform )

!  interface to CRO_crossover for H stored by rows. For argument details,
!  see the header for CRO_crossover_main

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, m_equal
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( H_ptr( n + 1 ) - 1 ) :: H_val
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, Z
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: X_stat
      TYPE ( CRO_data_type ), INTENT( INOUT ) :: data
      TYPE ( CRO_control_type ), INTENT( IN ) :: control
      TYPE ( CRO_inform_type ), INTENT( INOUT ) :: inform

      CALL CRO_crossover_main( n, m, m_equal, A_val, A_col, A_ptr,             &
                               G, C_l, C_u, X_l, X_u, C, X, Y, Z,              &
                               C_stat, X_stat, data, control, inform,          &
                               H_val = H_val, H_col = H_col, H_ptr = H_ptr )

      RETURN

!  end of subroutine CRO_crossover_h_sparse_by_rows

      END SUBROUTINE CRO_crossover_h_sparse_by_rows

!-  C R O _ C R O S S O V E R _ H _ L M   S U B R O U T I N E  -

      SUBROUTINE CRO_crossover_h_lm( n, m, m_equal, H_lm, A_val, A_col, A_ptr, &
                                     G, C_l, C_u, X_l, X_u, C, X, Y, Z,        &
                                     C_stat, X_stat, data, control, inform )

!  interface to CRO_crossover for H stored as a limited-memory matrix. For
!  argument details, see the header for CRO_crossover_main

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, m_equal
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: H_lm
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, Z
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: X_stat
      TYPE ( CRO_data_type ), INTENT( INOUT ) :: data
      TYPE ( CRO_control_type ), INTENT( IN ) :: control
      TYPE ( CRO_inform_type ), INTENT( INOUT ) :: inform

      CALL CRO_crossover_main( n, m, m_equal, A_val, A_col, A_ptr,             &
                               G, C_l, C_u, X_l, X_u, C, X, Y, Z,              &
                               C_stat, X_stat, data, control, inform,          &
                               H_lm = H_lm )

      RETURN

!  end of subroutine CRO_crossover_h_lm

      END SUBROUTINE CRO_crossover_h_lm

!-  C R O _ C R O S S O V E R _ N O _ H    S U B R O U T I N E  -

      SUBROUTINE CRO_crossover_no_h( n, m, m_equal, A_val, A_col, A_ptr,       &
                                     G, C_l, C_u, X_l, X_u, C, X, Y, Z,        &
                                     C_stat, X_stat, data, control, inform )

!  interface to CRO_crossover for problems with no H. For
!  argument details, see the header for CRO_crossover_main

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, m_equal
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, Z
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: X_stat
      TYPE ( CRO_data_type ), INTENT( INOUT ) :: data
      TYPE ( CRO_control_type ), INTENT( IN ) :: control
      TYPE ( CRO_inform_type ), INTENT( INOUT ) :: inform

      CALL CRO_crossover_main( n, m, m_equal, A_val, A_col, A_ptr,             &
                               G, C_l, C_u, X_l, X_u, C, X, Y, Z,              &
                               C_stat, X_stat, data, control, inform )

      RETURN

!  end of subroutine CRO_crossover_no_h

      END SUBROUTINE CRO_crossover_no_h

!-*-*-*-   C R O _ C R O S S O V E R _ M A I N   S U B R O U T I N E   -*-*-*-

      SUBROUTINE CRO_crossover_main( n, m, m_equal, A_val, A_col, A_ptr,       &
                                     G, C_l, C_u, X_l, X_u, C, X, Y, Z,        &
                                     C_stat, X_stat, data, control, inform,    &
                                     H_val, H_col, H_ptr, H_lm )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Given values x, y and z that satisfy the KKT conditions
!
!    H x + g - A^T y - z = 0, c_l <= A x <= c_u, x_l <= x <= x_u
!      { >= 0 if A x = c_l             { >= 0 if x = x_l
!    y {  = 0 if c_l < A x < c_u and z {  = 0 if x_l < x < x_u
!      { <= 0 if A x = c_u             { <= 0 if x = x_u
!
!  adjust x, y and z to maintain these relationships but at the same time
!  reduce the number of nonzero components of y and z. Ultimately a
!  ``basic'' solution, i.e., one for which the submatrix of A formed from rows
!  with nonzero y and columns from nonzero z is of full rank, is obtained
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  n is an INTEGER variable, that must be set by the user to the
!   number of optimization parameters, n.
!   RESTRICTION: n >= 1
!
!  m is an INTEGER variable, that must be set by the user to the
!   number of general linear constraints, m.
!   RESTRICTION: m >= 0
!
!  m_equal is an INTEGER variable, that must be set by the user to the
!   number of general linear equality constraints (i.e., c_l = c_u).
!   RESTRICTION: m_equal >= 0
!
!  A_val is an REAL array of length A_ptr(m+1)-1 that holds the values of
!    the nonzero components of A, stored row by row.
!    RESTRICTION: the first m_equal rows must be equality constraints
!
!  A_col is an INTEGER array of length A_ptr(m+1)-1 that holds the column
!    indices of the components of A corresponding to A_val
!
!  A_ptr is an INTEGER array of length m + 1 that holds pointers to the start
!    of each row of A, and 1 past the end of the last row
!
!  G is a REAL array of length n, that must be set by the user to the vector
!   of graidients g of the quadratic objective function
!
!  C_l is a REAL array of length n, that must be set by the user to the
!   vector of lower bounds on the constraints c_l
!
!  C_u is a REAL array of length n, that must be set by the user to the
!   vector of lower bounds on the constraints c_u
!
!  X_l is a REAL array of length n, that must be set by the user to the
!   vector of lower bounds on the variables x_l
!
!  X_u is a REAL array of length n, that must be set by the user to the
!   vector of lower bounds on the variables x_u
!
!  X is a REAL array of length n, that must be set by the user to the
!   variables x. On successful exit, it will contain the updated varuiables
!
!  Y is a REAL array of length m, that must be set by the user
!   to the Lagrange multipliers y. On successful exit, it will contain
!   the updated multipliers
!
!  Z is a REAL array of length n, that must be set by the user
!   to the dual variables z. On successful exit, it will contain
!   the updated dual variables.
!
!  C_stat is an INTEGER array of length m, that on entry should give the
!  status of the constraints. Possible values are
!    C_stat( i ) < 0, the i-th constraint is in the active set,
!                     on its lower bound,
!                > 0, the i-th constraint is in the active set
!                     on its upper bound, and
!                = 0, the i-th constraint is not in the active set
!
!   On exit these will be reset so that
!
!    C_stat( i ) = - 1, for basic active constraints on their lower bounds,
!                  - 2, for non-basic active constraints on their lower bounds,
!                    0, for inactive constraints,
!                    1, for basic active constraints on their upper bounds, and
!                    2, for non-basic active constraints on their upper bounds
!
!  X_stat is an INTEGER array of length m, that on entry should give the
!   status of the simple bounds on the variables. Possible values are
!    X_stat( i ) < 0, the i-th variable is in the active set,
!                     on its lower bound,
!                > 0, the i-th variable is in the active set
!                     on its upper bound, and
!                = 0, the i-th variable is not in the active set
!
!   On exit these will be reset so that
!
!    X_stat( i ) = - 1, for basic active variables on their lower bounds,
!                  - 2, for non-basic active variables on their lower bounds,
!                    0, for inactive (i.e., free) variables,
!                    1, for basic active variables on their upper bounds, and
!                    2, for non-basic active variables on their upper bounds
!
!  data is a structure of type CRO_data_type that holds private internal data
!
!  control is a structure of type CRO_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to CRO_initialize. See CRO_initialize
!   for details
!
!  inform is a structure of type CRO_inform_type that provides
!    information on exit from CRO_crossover. The component status
!    has possible values:
!
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions
!          n     >=  1
!          m     >=  0
!       has been violated.
!
!    -5 the constraints are likely inconsistent
!
!    -9 an error has occured in SLS_analyse
!
!   -10 an error has occured in SLS_factorize
!
!   -11 an error has occured in SLS_solve
!
!   -12 an error has occured in ULS_factorize
!
!   -14 an error has occured in ULS_solve
!
!   -16 the residuals are large; the factorization may be unsatisfactory
!
!  On exit from CRO_crossover, other components of inform are given in the
!   preamble
!
!  H_val is an optional REAL array of length H_ptr(n+1)-1 that holds the
!    values of the nonzero components of the lower triangular part of H,
!    stored row by row
!
!  H_col is an opional INTEGER array of length H_ptr(n+1)-1 that holds the
!    column indices of the components of the lower triangular part of H,
!    corresponding to H_val
!
!  H_ptr is an optional INTEGER array of length n + 1 that holds pointers to
!    the start of each row of H, and 1 past the end of the last row
!
!  H_lm is an optional argument of type LMS_data_type used to hold a
!   "limited-memory" representation of H as set up by the module GALAHAD_LMS
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, m_equal
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: C, Y
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X, Z
      INTEGER, INTENT( INOUT ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: X_stat
      TYPE ( CRO_data_type ), INTENT( INOUT ) :: data
      TYPE ( CRO_control_type ), INTENT( IN ) :: control
      TYPE ( CRO_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_col
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_val
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( INOUT ) :: H_lm

!  Local variables

      INTEGER :: i, ip, j, jp, l, m_fixed, n_free, n_fixed, a_ne, nb, lbd, K_r_n
      INTEGER :: incoming, outgoing, basic, nonbasic, out, row_out, len_sls_sol
      INTEGER :: all_basic, nb_start, all_basic_old, dim_w, dim_w_max, oi, oj
      INTEGER :: ii, jj
!     INTEGER :: k, nviol8_xs, nviol8_cs, sofar, basic_old
!     INTEGER :: nviol8_p, nviol8_d, nviol8_x, nviol8_y, nviol8_z, nviol8_c
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: dy_i, step, val
!     REAL ( KIND = wp ) :: tol
!     REAL ( KIND = wp ) :: viol8_p, viol8_d, viol8_x, viol8_y, viol8_z, viol8_c
!     LOGICAL :: b_fx
      LOGICAL :: b_fr, c_fr, c_fx, b_fr_neq_0, tryboth, lbfgs, is_h
      LOGICAL :: printi, printt, printm, printd, printa
      CHARACTER ( LEN = 80 ) :: array_name

!  temporary!!

      REAL ( KIND = wp ), DIMENSION( m ) :: C_new, Y_new
      REAL ( KIND = wp ), DIMENSION( n ) :: X_new, Z_new
      REAL ( KIND = wp ), DIMENSION( n + m ) :: V_new, R_new

!  insert into data

!     INTEGER, DIMENSION( m + n + 1 ) :: PTR
!     INTEGER, DIMENSION( 10 * ( m + n + 1 ) ) :: IND
!     REAL ( KIND = wp ), DIMENSION( 10 * ( m + n + 1 ) ) :: VAL
!     INTEGER, DIMENSION( n ) :: INDEPENDENT
!     INTEGER :: n_independent

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ); CALL CLOCK_time( clock_start )
      inform%status = GALAHAD_ok

!  basic single line of output per iteration

      out =  control%out
      printi = out > 0 .AND. control%print_level >= 1
      printt = out > 0 .AND. control%print_level >= 2
      printm = out > 0 .AND. control%print_level >= 3
      printd = out > 0 .AND. control%print_level >= 4
      printa = out > 0 .AND. control%print_level >= 101
      IF ( printt ) WRITE( out, "( A, ' -- entering CRO_crossover ' )" ) prefix

!  ensure that control variables are reasonable

      data%control = control
      data%control%max_schur_complement =                                      &
        MAX( data%control%max_schur_complement, 2 )

      lbfgs = PRESENT( H_lm )
      IF ( lbfgs ) THEN
        is_h = .FALSE.
      ELSE
        is_h = PRESENT( H_val ) .AND. PRESENT( H_col ) .AND.                   &
               PRESENT( H_ptr )
        IF ( .NOT. is_h ) THEN
!         inform%status = GALAHAD_error_optional ; GO TO 900
        END IF
      END IF

!  print out input data if required

      IF ( out > 0 .AND. control%print_level >= 101 ) THEN
        WRITE( out, * ) ' n ', n
        WRITE( out, * ) ' m ', m
        WRITE( out, * ) ' m_equal ', m_equal
        IF ( is_h ) THEN
          WRITE( out, * ) ' H_val ', H_val( : H_ptr( n + 1 ) - 1 )
          WRITE( out, * ) ' H_col ', H_col( : H_ptr( n + 1 ) - 1 )
          WRITE( out, * ) ' H_ptr ', H_ptr
        END IF
        WRITE( out, * ) ' A_val ', A_val
        WRITE( out, * ) ' A_col ', A_col
        WRITE( out, * ) ' A_ptr ', A_ptr
        WRITE( out, * ) ' G ', G
        WRITE( out, * ) ' C_l ', C_l
        WRITE( out, * ) ' C_u ', C_u
        WRITE( out, * ) ' X_l ', X_l
        WRITE( out, * ) ' X_u ', X_u
        WRITE( out, * ) ' C ', C
        WRITE( out, * ) ' X ', X
        WRITE( out, * ) ' Y ', Y
        WRITE( out, * ) ' Z ', Z
        WRITE( out, * ) ' C_stat ', C_stat
        WRITE( out, * ) ' X_stat ', X_stat
      END IF

!  if required, check the given primal-dual point is a KKT point

      IF ( data%control%check_io ) THEN
        array_name = 'cro: data%DX'
        CALL SPACE_resize_array( n, data%DX,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%DY'
        CALL SPACE_resize_array( m, data%DY,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        CALL CRO_check_status( n, m, m_equal, A_val, A_col, A_ptr, G, C_l,     &
                               C_u, X_l, X_u, C, X, Y, Z, C_stat, X_stat,      &
                               control, inform, data%DX, data%DY, prefix,      &
                               H_val = H_val, H_col = H_col, H_ptr = H_ptr,    &
                               H_lm = H_lm )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  if there are no fixed constraints, the current active set suffices

      m_fixed = COUNT( C_stat( 1 : m ) /= 0 )
      IF ( m_fixed == 0 ) GO TO 800

!  allocate space to record the lists of free variables and fixed constraints,
!  X_free and C_fixed. Variables  X_free( : n_free) are free while constrants
!  C_fixed( : m_fixed ) are fixed

      array_name = 'cro: data%C_fixed'
      CALL SPACE_resize_array( m_fixed, data%C_fixed,                          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      n_free = COUNT( X_stat( 1 : n ) == 0 )
      K_r_n = n_free + m_fixed
      array_name = 'cro: data%X_free'
      CALL SPACE_resize_array( n, data%X_free,                                 &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  allocate space to record which variables are free and which constraints
!  fixed, X_inorder and C_inorder. If X_inorder( i ) > 0, variable i is the
!  X_inorder( i )-th free variable, while if X_inorder( i ) < 0, i the variable
!  is the - X_inorder(i)-th fixed variable. Similarly, if C_inorder( i ) > 0,
!  constraint i is the C_inorder( i )-th fixed constraint, while if
!  C_inorder( i ) = 0, the constraint is free. Thus, for example,

!   --------------------------------------------------
!     j        1  2  3  4  5  6 ....       ....   n
!   X_stat     0  1  0  0 -1  0 ....              0
!   X_free     1  3  4  6  ....  n_free
!   X_inorder  1 -1  2  3 -2  4 ....       .... n_free
!   --------------------------------------------------

!   --------------------------------------------------
!     i        1  2  3  4  5  6 ....       ....   m
!   C_stat     1 -1  0  0  1  0 ...              -1
!   C_fixed    1  2  5     ....     m_fixed
!   C_inorder  1  2  0  0  3  0 ....       ... m_fixed
!   --------------------------------------------------

      array_name = 'cro: data%X_inorder'
      CALL SPACE_resize_array( n, data%X_inorder,                              &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%C_inorder'
      CALL SPACE_resize_array( m, data%C_inorder,                              &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  record the number and list of free variables ...

      i = n_free ; n_free = 0 ; n_fixed = 0
      DO j = 1, n
        IF ( X_stat( j ) == 0 ) THEN
          n_free = n_free + 1
          data%X_inorder( j ) = n_free
          data%X_free( n_free ) = j
        ELSE
          n_fixed = n_fixed + 1 ; i = i + 1
          data%X_inorder( j ) = - n_fixed
          data%X_free( i ) = j
          X_stat( j ) = SIGN( 1, X_stat( j ) )
        END IF
      END DO

!  ... and the number and list of fixed constraints

      m_fixed = 0
      DO i = 1, m
        IF ( C_stat( i ) /= 0 ) THEN
          m_fixed = m_fixed + 1
          data%C_inorder( i ) = m_fixed
          data%C_fixed( m_fixed ) = i
          C_stat( i ) = SIGN( 1, C_stat( i ) )
        ELSE
          data%C_inorder( i ) = 0
        END IF
      END DO

!  print the list of free variables and fixed constraints

      IF ( printi ) WRITE( out, "( A, ' nfree, m_fixed ', I0, 1X, I0 )" )      &
         prefix, n_free, m_fixed
      IF ( printd ) THEN
        IF ( n_free > 0 ) WRITE( out, "( A, ' X_free', /, ( 10I8 ) )" )        &
          prefix, data%X_free( : n_free )
        IF ( m_fixed > 0 ) WRITE( out, "( A, ' C_fixed', /, ( 10I8 ) )" )      &
          prefix, data%C_fixed( : m_fixed )
      END IF

!  these lists induce the partitions (fr = free, fx = fixed, of = off-diagonal)

!    H = ( H_fx  H_od^T ) and A = ( A_fx ) = ( A_fxx  A_frx ) ;
!        ( H_od   H_fr  )         ( A_fr )   ( A_fxr  A_frr )

!  we shall discard free constraints A_fr. We are then interested in
!  non-trivial solutions to the system

!    ( H_fx  H_od^T   I   A_fxx^T ) ( dx_fx )
!    ( H_od   H_fr    0   A_frx^T ) ( dx_fr ) = 0                    (dKKT)
!    (  I      0      0      0    ) ( dz_fx )
!    ( A_fxx  A_frx   0      0    ) ( dy_fx )

!  since for these ( x + t dx, y - t dy, z - t dz ) continues to satisfy the
!  KKT equations for all t. [the remaining free variables and inactive
!  constraints cannot be active and may be discarded as the initial
!  maximally ``optimal'' partition from interior-point methods s is unique]

!  We check to see if A_frx is of full (row) rank. If so, dKKT has the
!  unique solution 0 and the unique solution to QP is basic. If not, we have

!    A_frx = ( A_frxb ) and A_fxx = ( A_fxxb )
!            ( A_frxn )             ( A_fxxn )

!  where rank( A_frxb ) = rank( A_frx ). In this case, we may rewrite dKKT as

!    ( H_fx    H_od^T  A_fxxb^T  I ) ( dx_fx  )     ( A_fxxn^T )
!    ( H_od     H_fr   A_frxb^T  0 ) ( dx_fr  ) = - ( A_frxn^T ) dy_fxn
!    ( A_fxxb  A_frxb     0      0 ) ( dy_fxb )     (    0     )       (bKKT)
!    (  I        0        0      0 ) ( dz_fx  )     (    0     )

!  where the left-hand side coeficient matrix K_b is non-singular, and multiple
!  solutions to (dKKT) may be found by varying dy_fxn. [the equations
!  involving  A_frxb may be omited since they are linearly dependent].
!  We shall refer to the coefficient matrices of bKKT as K_b

      IF ( n_free > 0 ) THEN

!  count the number of entries in the sub-matrix A_frx

        a_ne = 0
        DO i = 1, m
          IF ( data%C_inorder( i ) /= 0 ) THEN
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              IF ( data%X_inorder( A_col( l ) ) > 0 ) a_ne = a_ne + 1
            END DO
          END IF
        END DO

!  now check to see if A_frx is of full (row) rank by factorizing A_frx^T

!  allocate space for A_frx^T

        data%AT%ne = a_ne
        array_name = 'cro: data%AT%row'
        CALL SPACE_resize_array( data%AT%ne, data%AT%row,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%AT%col'
        CALL SPACE_resize_array( data%AT%ne, data%AT%col,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%AT%val'
        CALL SPACE_resize_array( data%AT%ne, data%AT%val,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  record the sub-matrix A_frx^T

        CALL SMT_put( data%AT%type, 'COORDINATE', i )
!       data%AT%n = n_free ; data%AT%m = m_fixed
        data%AT%m = n_free ; data%AT%n = m_fixed

        data%AT%ne = 0
        DO i = 1, m
          IF ( data%C_inorder( i ) > 0 ) THEN       ! fixed constraint
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              j = A_col( l )
              IF ( data%X_inorder( j ) > 0 ) THEN   ! free variable
                data%AT%ne = data%AT%ne + 1
!               data%AT%row( data%AT%ne ) = data%C_inorder( i )
!               data%AT%col( data%AT%ne ) = data%X_inorder( j )
                data%AT%row( data%AT%ne ) = data%X_inorder( j )
                data%AT%col( data%AT%ne ) = data%C_inorder( i )
                data%AT%val( data%AT%ne ) = A_val( l )
              END IF
            END DO
          END IF
        END DO
!do l = 1, data%AT%ne
! write(6,"( 2I6, F10.4 )" ) data%AT%row( l ), data%AT%col( l ),               &
!                            data%AT%val( l )
!end do

        IF ( data%AT%ne > 0 ) THEN

!  factorize A_frx^T

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL ULS_initialize_solver( data%control%unsymmetric_linear_solver,  &
                                      data%ULS_data, inform%ULS_inform )

!  ensure that block-triangular form is NOT used as this may underestimate
!  the rank

          data%control%ULS_control%minimum_size_for_btf = MAX( n_free, m_fixed )
          CALL ULS_factorize( data%AT, data%ULS_data,                          &
                              data%control%ULS_control, inform%ULS_inform )
!write(6,*) ' ranks ', inform%ULS_inform%rank, inform%ULS_inform%structural_rank
!write(6,*) data%control%ULS_control%relative_pivot_tolerance, &
!           data%control%ULS_control%absolute_pivot_tolerance
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%factorize = inform%time%factorize + time_now - time_record
          inform%time%clock_factorize =                                        &
            inform%time%clock_factorize + clock_now - clock_record

          IF ( printi ) WRITE( out,                                            &
               "( A, ' AT nnz(matrix,factors) = ', I0, ', ', I0,               &
            &  /, A, ' ULS: factorization of AT complete: status = ', I0 )" )  &
                 prefix, data%AT%ne, inform%ULS_inform%entries_in_factors,     &
                 prefix, inform%ULS_inform%status
          IF ( inform%ULS_inform%status < 0 ) THEN
             inform%status = GALAHAD_error_uls_factorization ; GO TO 900
          END IF
!write(6,*) data%AT%m, data%AT%n, inform%ULS_inform%rank

!  check to see if dependent constraints should be removed

          inform%dependent = MAX( 0, data%AT%n - inform%ULS_inform%rank )
          IF ( inform%dependent == 0 ) GO TO 800
        ELSE
          inform%dependent = m_fixed
        END IF
      ELSE
        inform%dependent = m_fixed
      END IF

!  recall that we aim to satisfy

!    ( H_fx    H_od^T  A_fxxb^T   I ) ( dx_fx  )     ( A_fxxn^T )
!    ( H_od     H_fr   A_frxb^T   0 ) ( dx_fr  ) = - ( A_frxn^T ) dy_fxn
!    ( A_fxxb  A_frxb       0     0 ) ( dy_fxb )     (    0     )        (bKKT)
!    (  I        0          0     0 ) ( dz_fx  )     (    0     )

!  we now need to "activate" nonbasic constraints A_fxn one at a time
!  by choosing dy_fxn to have a single nonzero (unit) component, and then
!  monitoring ( x + t dx, y + t dy, z + t dz ) as t increases from 0.
!  We call the candidate column of A_fxn^T the ** incoming ** column.
!  As we do so, the set of optimal multipliers/dual variables will vary
!  until one hits its bound. If it is the one corresponding to the activated
!  non-basic constraint, we can remove this from consideration. If, converesly,
!  one of the multipliers/dual variables of the basic constraints/variables
!  hits its bound, we will interchange this basic constraint/variable with the
!  incoming one and remove the ** outgoing ** constraint/variable from
!  consideration

!  Note that the block system bKKT may be solved as

!    (  H_fr   A_frxb^T ) ( dx_fr  ) = - ( A_frxn^T dy_fxn )             (rKKT)
!    ( A_frxb     0     ) ( dy_fxb )     (        0        )

!  where  dx_fx = 0 and dz_fx = - H_od^T dx_fr - A_fxx^T dy_fx. [Also recall
!  that we refer to the coefficient matrices of bKKT and rKKT as K_b and K_r]

      IF ( printi ) WRITE( out,                                                &
        "( A, ' there are ', I0, ' dependent constraint', A, ' to remove' )" ) &
            prefix, inform%dependent, TRIM( STRING_pleural( inform%dependent ) )

!  Find the basis; the ``basis'' is a maximal non-singular sub-matrix

!    K_r = (  H_fr   A_frxb^T )
!          ( A_frxb     0     )

!  while the remaining dependent matrix of rows/columns A_frxn is known as the
!  ``non-basis''. Allocate space to record the basic and non-basic rows and
!  columns

      array_name = 'cro: data%X_basic'
      CALL SPACE_resize_array( n_free, data%X_basic,                           &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%C_basic'
      CALL SPACE_resize_array( m_fixed, data%C_basic,                          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  if A_fr is not null, compute which rows and columns of A_fr define a basis

      IF ( n_free > 0 .AND. data%AT%ne > 0 ) THEN
        CALL ULS_enquire( data%ULS_data, inform%ULS_inform,                    &
                          data%X_basic( : n_free ), data%C_basic( : m_fixed ) )
!write(6,*) ' rank ', inform%ULS_inform%rank

!  record the basic rows (C_basic); flag these by flipping the signs of C_fixed

        basic = inform%ULS_inform%rank
        data%C_fixed( data%C_basic( : basic ) ) =                             &
           - data%C_fixed( data%C_basic( : basic ) )

!  now consider all fixed rows. Record and count the non-basic rows (those with
!  +ve signs of C_fixed) directly after the basic ones in C_basic and flag the
!  j-th free constraint as the nonbasic-th non-basic with a -ve sign in
!  C_inorder. The basic rows have positive signs in C_inorder, and as before,
!  the free rows have the value 0

        nonbasic = basic ; basic = 0
        DO j = 1, m_fixed
          i = data%C_fixed( j )
          IF ( i > 0 ) THEN
            nonbasic = nonbasic + 1
            data%C_basic( nonbasic ) = i
            data%C_inorder( i ) = - nonbasic
          ELSE
            basic = basic + 1
            data%C_basic( basic ) = - i
            data%C_inorder( - i ) = basic
            data%C_fixed( j ) = - i
          END IF
        END DO

!  do the same when A_fr is null

      ELSE
        basic = 0 ; nonbasic = 0
        DO j = 1, m_fixed
          nonbasic = nonbasic + 1
          i = data%C_fixed( j )
          data%C_basic( nonbasic ) = i
          data%C_inorder( i ) = - nonbasic
        END DO
      END IF
      nonbasic = nonbasic - basic

      IF ( printd ) THEN
        IF ( basic > 0 )                                                       &
          WRITE( out, "( A, ' C_basic ', /, ( 10I8 ) )" )                      &
            prefix, data%C_basic( : basic )
        IF ( nonbasic > 0 )                                                    &
          WRITE( out, "( A, ' C_nonbasis ', /, ( 10I8 ) )" )                   &
            prefix, data%C_basic( basic + 1 : basic + nonbasic )
      END IF

! initialize the Schur complement

      data%SCU_matrix%n = n + basic + n_fixed
      data%SCU_matrix%m = 0
      data%SCU_matrix%class = 2
      data%SCU_matrix%m_max = data%control%max_schur_complement

      lbd = nonbasic
      DO nb = basic + 1, m_fixed
        i = data%C_basic( nb )
        lbd = lbd + A_ptr( i + 1 ) - A_ptr( i )
      END DO

      array_name = 'cro: data%SCU_matrix%BD_col_start'
      CALL SPACE_resize_array( data%control%max_schur_complement + 1,          &
          data%SCU_matrix%BD_col_start,                                        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%SCU_matrix%BD_val'
      CALL SPACE_resize_array( lbd, data%SCU_matrix%BD_val, inform%status,     &
          inform%alloc_status, array_name = array_name,                        &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%SCU_matrix%BD_row'
      CALL SPACE_resize_array( lbd, data%SCU_matrix%BD_row, inform%status,     &
          inform%alloc_status, array_name = array_name,                        &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%VECTOR'
      CALL SPACE_resize_array( data%SCU_matrix%n, data%VECTOR,                 &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      data%SCU_matrix%BD_col_start( 1 ) = 1
      inform%scu_status = 1
      CALL SCU_factorize( data%SCU_matrix, data%SCU_data,                      &
                          data%VECTOR( : data%SCU_matrix%n ),                  &
                          inform%scu_status, inform%SCU_inform )

! allocate further workspace

      IF ( .NOT. data%control%check_io ) THEN
        array_name = 'cro: data%DX'
        CALL SPACE_resize_array( n, data%DX,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%DY'
        CALL SPACE_resize_array( m, data%DY,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

      array_name = 'cro: data%DZ'
      CALL SPACE_resize_array( n, data%DZ,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%DC'
      CALL SPACE_resize_array( m, data%DC,                                     &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%RHS'
      CALL SPACE_resize_array( data%SCU_matrix%n +                             &
          data%control%max_schur_complement, data%RHS,                         &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'cro: data%SOL'
      CALL SPACE_resize_array( data%SCU_matrix%n +                             &
          data%control%max_schur_complement, data%SOL,                         &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      IF ( printa ) THEN
        array_name = 'cro: data%RES_d'
        CALL SPACE_resize_array( n, data%RES_d,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%RES_p'
        CALL SPACE_resize_array( m, data%RES_p,                                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

!  for the limited memory case, form the matrix W = [ -D       L^T    ]
!                                                   [ L   delta S^T S ]

      IF ( lbfgs ) THEN
        dim_w_max = 2 * H_lm%m ; dim_w = 2 * H_lm%length
        array_name = 'cro: data%W'
        CALL SPACE_resize_array( dim_w_max, dim_w_max, data%W,                 &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        data%W( : dim_w, : dim_w ) = zero
        DO j = 1, H_lm%length
          oj = H_lm%ORDER( j ) ; jp = H_lm%length + j
          val = H_lm%L_scaled( j, j )
          data%W( j, j ) = - val * val
          DO i = j, H_lm%length
            oi = H_lm%ORDER( i ) ; ip = H_lm%length + i
            IF ( i > j ) THEN
              data%W( ip, j ) = H_lm%L_scaled( i, j ) * val
              data%W( j, ip ) = data%W( ip, j )
            END IF
            data%W( ip, jp ) = H_lm%delta * H_lm%STS( oi, oj )
            data%W( jp, ip ) = data%W( ip, jp )
          END DO
        END DO
      END IF

!  main loop

      nb_start = basic + 1
!write(6,*) ' nonbasic ', data%C_basic( basic + 1 : m_fixed )
 100  CONTINUE
      IF ( printi ) WRITE( out, "( A, ' K n = ', I0 )" )                       &
        prefix, data%SCU_matrix%n

      array_name = 'cro: data%BASIS'
      CALL SPACE_resize_array( basic + n_fixed +                               &
          data%control%max_schur_complement + 2, data%BASIS,                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  record the current list of basic constraints and variables in BASIS.
!  Positive components are constraints, negative components are variables
!  and zero components are irrelevent

      all_basic = basic + n_fixed
      data%BASIS( : basic ) = data%C_basic( : basic )
      data%BASIS( basic + 1 : all_basic ) = - data%X_free( n_free + 1 : n )
!     write(6,*) ' BASIS ',  data%BASIS( : all_basic )

!write(6,*) ' all_basic ', all_basic

!  having found the basic rows and columns, discard the others and factorize K_r

      IF ( n_free > 0 ) THEN

!  count the number of nonzeros in the basis matrix

!     K_r = (  H_fr   A_frxb^T )
!           ( A_frxb     0     )

        data%K_r%ne = 0
        DO i = 1, m
          IF ( data%C_inorder( i ) > 0 ) THEN       !  basic constraint
            IF (  A_ptr( i + 1 ) > A_ptr( i ) ) data%K_r%ne = data%K_r%ne +    &
                COUNT( data%X_inorder( A_col( A_ptr( i ) :                     &
                                              A_ptr( i + 1 ) - 1 ) ) > 0 )
          END IF
        END DO

!  for the limited memory Hessian case, H = delta I - V W^-1 V^T, where
!  V = R [ Y : delta S ], and thus

!     K_r = (  delta I - V_fr W^-1 V_fr^T  A_frxb^T )
!           (             A_frxb               0    )

!  Since any system K_r v = rhs we may be required to solve can be expanded to

!   ( delta I   A_frxb^T  V_fr ) (  v  )   ( rhs )
!   (  A_frxb      0       0   ) (     ) = (     ),
!   (  V_fr^T      0       W   ) ( - z )   (  0  )

!  we instead factorize this expanded matrix, K_re ; both V and W are dense

        IF ( lbfgs ) THEN
          dim_w = 2 * H_lm%length
          data%K_r%ne = data%K_r%ne + n_free                                   &
            + dim_w * n_free + ( dim_w * ( dim_w + 1 ) ) / 2

!  for Hessians stored by rows

        ELSE IF ( is_h ) THEN
          DO i = 1, n
            IF ( data%X_inorder( i ) > 0 ) THEN
              IF (  H_ptr( i + 1 ) > H_ptr( i ) ) data%K_r%ne = data%K_r%ne +  &
                COUNT( data%X_inorder( H_col( H_ptr( i ) :                     &
                                              H_ptr( i + 1 ) - 1 ) ) > 0 )
            END IF
          END DO

!  for problems without Hessians

        ELSE
        END IF

!  allocate space for K_r

        array_name = 'cro: data%K_r%row'
        CALL SPACE_resize_array( data%K_r%ne, data%K_r%row,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%K_r%col'
        CALL SPACE_resize_array( data%K_r%ne, data%K_r%col,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        array_name = 'cro: data%K_r%val'
        CALL SPACE_resize_array( data%K_r%ne, data%K_r%val,                    &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

        data%K_r%ne = 0
        K_r_n = n_free + basic

!  form K_re

        IF ( lbfgs ) THEN

!  record the components of H_fr in K_re, ...

          IF ( H_lm%restricted == 0 ) THEN
            DO j = 1, n_free
              data%K_r%ne = data%K_r%ne + 1
              data%K_r%row( data%K_r%ne ) = j
              data%K_r%col( data%K_r%ne ) = j
              data%K_r%val( data%K_r%ne ) = H_lm%delta
            END DO
          ELSE
            DO j = 1, n
              l = data%X_inorder( j )
              IF ( l > 0 .AND. l <= H_lm%n ) THEN
                data%K_r%ne = data%K_r%ne + 1
                data%K_r%row( data%K_r%ne ) = l
                data%K_r%col( data%K_r%ne ) = l
                data%K_r%val( data%K_r%ne ) = H_lm%delta
              END IF
            END DO
          END IF

!  ... those of A_frxb, ...

          DO i = 1, m
            IF ( data%C_inorder( i ) > 0 ) THEN
              DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                j = A_col( l )
                IF ( data%X_inorder( j ) > 0 ) THEN
                  data%K_r%ne = data%K_r%ne + 1
                  data%K_r%row( data%K_r%ne ) = data%C_inorder( i ) + n_free
                  data%K_r%col( data%K_r%ne ) = data%X_inorder( j )
                  data%K_r%val( data%K_r%ne ) = A_val( l )
                END IF
              END DO
            END IF
          END DO

!  ... those of W ...

          DO j = 1, dim_w
            oj = K_r_n + j
            DO i = j, dim_w
              data%K_r%ne = data%K_r%ne + 1
              data%K_r%row( data%K_r%ne ) = K_r_n + i
              data%K_r%col( data%K_r%ne ) = oj
              data%K_r%val( data%K_r%ne ) = data%W( i, j )
            END DO
          END DO

!  ... and those of V_fr^T where V = R [ Y : delta S ]

          DO i = 1, H_lm%length
            K_r_n = K_r_n + 1
            oi = H_lm%ORDER( i )
            DO j = 1, n
              IF ( data%X_inorder( j ) > 0 ) THEN
                data%K_r%ne = data%K_r%ne + 1
                data%K_r%row( data%K_r%ne ) = K_r_n
                data%K_r%col( data%K_r%ne ) = data%X_inorder( j )
                IF ( H_lm%restricted == 0 ) THEN
                  data%K_r%val( data%K_r%ne ) = H_lm%Y( j, oi )
                ELSE
                  l = H_lm%RESTRICTION( j )
                  IF ( l <= H_lm%n ) THEN
                    data%K_r%val( data%K_r%ne ) = H_lm%Y( l, oi )
                  ELSE
                    data%K_r%ne = data%K_r%ne - 1
                  END IF
                END IF
              END IF
            END DO
          END DO

          DO i = 1, H_lm%length
            K_r_n = K_r_n + 1
            oi = H_lm%ORDER( i )
            DO j = 1, n
              IF ( data%X_inorder( j ) > 0 ) THEN
                data%K_r%ne = data%K_r%ne + 1
                data%K_r%row( data%K_r%ne ) = K_r_n
                data%K_r%col( data%K_r%ne ) = data%X_inorder( j )
                IF ( H_lm%restricted == 0 ) THEN
                  data%K_r%val( data%K_r%ne ) = H_lm%delta * H_lm%S( j, oi )
                ELSE
                  l = H_lm%RESTRICTION( j )
                  IF ( l <= H_lm%n ) THEN
                    data%K_r%val( data%K_r%ne ) = H_lm%delta * H_lm%S( l, oi )
                  ELSE
                    data%K_r%ne = data%K_r%ne - 1
                  END IF
                END IF
              END IF
            END DO
          END DO

!  form K_r

!  for problems with Hessians, record the components of H_fr in K_r ...

        ELSE IF ( is_h ) THEN
          DO i = 1, n
            IF ( data%X_inorder( i ) > 0 ) THEN
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                IF ( data%X_inorder( j ) > 0 ) THEN
                  data%K_r%ne = data%K_r%ne + 1
                  data%K_r%row( data%K_r%ne ) = data%X_inorder( i )
                  data%K_r%col( data%K_r%ne ) = data%X_inorder( j )
                  data%K_r%val( data%K_r%ne ) = H_val( l )
                END IF
              END DO
            END IF
          END DO

!  ... and those of A_frxb

          DO i = 1, m
            IF ( data%C_inorder( i ) > 0 ) THEN
              DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                j = A_col( l )
                IF ( data%X_inorder( j ) > 0 ) THEN
                  data%K_r%ne = data%K_r%ne + 1
                  data%K_r%row( data%K_r%ne ) = data%C_inorder( i ) + n_free
                  data%K_r%col( data%K_r%ne ) = data%X_inorder( j )
                  data%K_r%val( data%K_r%ne ) = A_val( l )
                END IF
              END DO
            END IF
          END DO

!  for those without Hessians, just record the components of A_frxb in K_r

        ELSE
          DO i = 1, m
            IF ( data%C_inorder( i ) > 0 ) THEN
              DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                j = A_col( l )
                IF ( data%X_inorder( j ) > 0 ) THEN
                  data%K_r%ne = data%K_r%ne + 1
                  data%K_r%row( data%K_r%ne ) = data%C_inorder( i ) + n_free
                  data%K_r%col( data%K_r%ne ) = data%X_inorder( j )
                  data%K_r%val( data%K_r%ne ) = A_val( l )
                END IF
              END DO
            END IF
          END DO
        END IF
        data%K_r%n = K_r_n ; data%K_r%m = K_r_n
        CALL SMT_put( data%K_r%type, 'COORDINATE', i )

!       WRITE( 6, "( ' K_sub: n, nnz ', 2I4 )" ) data%K_r%n, data%K_r%ne
!       WRITE( 6, "( A, /, ( 10I7) )" ) ' rows =', data%K_r%row( : data%K_r%ne )
!       WRITE( 6, "( A, /, ( 10I7) )" ) ' cols =', data%K_r%col( : data%K_r%ne )
!       WRITE( 6, "( A, /, ( F7.2) )" ) ' vals =', data%K_r%val( : data%K_r%ne )

!  order the rows/columns of K_r prior to factorization

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_initialize_solver( data%control%symmetric_linear_solver,      &
                                    data%SLS_data, inform%SLS_inform )
        CALL SLS_analyse( data%K_r, data%SLS_data, data%control%SLS_control,   &
                          inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse =                                            &
          inform%time%clock_analyse + clock_now - clock_record

        IF ( printi ) WRITE( out,                                              &
             "( A, ' K nnz(matrix,predicted factors) = ', I0, ', ', I0,        &
          &  /, A, ' SLS: analysis of K complete: status = ', I0 )" )          &
               prefix, data%K_r%ne, inform%SLS_inform%real_size_factors,       &
               prefix, inform%SLS_inform%status
        IF ( printi .AND. inform%SLS_inform%out_of_range > 0 ) WRITE( out,     &
            "( A, ' ** warning: ', I0, ' entr', A, ' of K out of range' )" )   &
               prefix, inform%SLS_inform%out_of_range,                         &
               STRING_ies( inform%SLS_inform%out_of_range )
        IF ( inform%SLS_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_analysis ; GO TO 900
        END IF

!  factorize the basic sub-block K_r

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( data%K_r, data%SLS_data, data%control%SLS_control, &
                            inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_start
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock_record

        IF ( printi ) WRITE( out,                                              &
             "( A, ' K nnz(matrix,factors) = ', I0, ', ', I0,                  &
          &  /, A, ' SLS: factorization of K complete: status = ', I0 )" )     &
               prefix, data%K_r%ne, inform%SLS_inform%entries_in_factors,      &
               prefix, inform%SLS_inform%status
        IF ( inform%SLS_inform%status < 0 ) THEN
           inform%status = GALAHAD_error_factorization ; GO TO 900
        END IF

        IF ( printm ) WRITE( out, "( A, ' K%n ', I0, ' rank ', I0 )" )         &
          prefix, data%K_r%n, inform%SLS_inform%rank
      ELSE
        data%K_r%n = 0
      END IF

! allocate further workspace

      len_sls_sol = MAX( data%K_r%n, n )
      array_name = 'cro: data%SLS_SOL'
      CALL SPACE_resize_array( len_sls_sol, data%SLS_SOL,                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  loop over the nonbasic constraints to solve

!    ( H_fx    H_od^T  A_fxxb^T   I ) ( dx_fx  )
!    ( H_od     H_fr   A_frxb^T   0 ) ( dx_fr  ) = rhs_nb
!    ( A_fxxb  A_frxb       0     0 ) ( dy_fxb )
!    (  I        0          0     0 ) ( dz_fx  )

!  where the nb-th right-hand side (nb = 1, ...,  nonbasic) is

!             ( - A_fxxn^T dy_fxn_nb )
!    rhs_nb = ( - A_frxn^T dy_fxn_nb )
!             (           0          )
!             (           0          )

! and dy_fxn_nb has one its nonzero in position nb

!     data%DX = zero ; data%DY = zero

        IF ( data%control%check_io ) THEN
          CALL CRO_check_status( n, m, m_equal, A_val, A_col, A_ptr, G, C_l,   &
                                 C_u, X_l, X_u, C, X, Y, Z, C_stat, X_stat,    &
                                 control, inform, data%DX, data%DY, prefix,    &
                                 H_val = H_val, H_col = H_col, H_ptr = H_ptr,  &
                                 H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  ----------------------------- main loop ----------------------------------

      DO nb = nb_start, m_fixed

!       data%DX = zero ; data%DY = zero ; data%DZ = zero

!  record the change in the incoming non-basic constraint dy_fxn. Also
!  record the step to a convenient bound, and the variable that achieves
!  this step (-ve for constraints)

        tryboth = .FALSE.
        i = data%C_basic( nb )
        incoming = - i
        IF ( C_l( i ) == C_u( i ) ) THEN
          step = infinity
          tryboth = .TRUE.
!         dy_i = - one
          dy_i = one
        ELSE IF ( C_stat( i ) < 0 ) THEN
          dy_i = one
          step = Y( i )
        ELSE
          dy_i = - one
          step = - Y( i )
        END IF
        outgoing = - i
! write(6,*) ' i, C_stat, Y ', i, C_stat( i ), Y( i )
!write(6,*) ' step ', step
!  set up the right-hand side ( b  c ) in VECTOR

        data%RHS( : data%SCU_matrix%n + data%SCU_matrix%m ) = zero
!       b_fx = .FALSE.
        b_fr = .FALSE. ; c_fr = .FALSE. ; c_fx = .FALSE.
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          j = A_col( l )
          data%RHS( j ) = - dy_i * A_val( l )
          IF ( data%X_inorder( j ) >= 0 ) THEN
            b_fr = .TRUE.
!         ELSE
!           b_fx = .TRUE.
          END IF
!         j = j + 1
        END DO
        b_fr_neq_0 = b_fr

!  write(6,*) ' rhs ',  data%RHS( : data%SCU_matrix%n + data%SCU_matrix%m )

!  use SCU to solve the linear system

!    ( H_fx    H_od^T  A_fxxb^T  I ) (  dx_fx  )   ( b_fx )
!    ( H_od     H_fr   A_frxb^T  0 ) (  dx_fr  ) = ( b_fr )
!    ( A_fxxb  A_frxb       0    0 ) (  dy_fxb )   ( c_fr )
!    (  I        0          0    0 ) (  dz_fx  )   ( c_fx )

        IF ( data%SCU_matrix%m > 0 ) THEN

!  the basic variables/constraints have changed and have been accommodated
!  by a Schur complement (see SCU documentation)

          inform%scu_status = 1
          DO
            CALL SCU_solve( data%SCU_matrix, data%SCU_data, data%RHS,          &
                            data%SOL, data%VECTOR, inform%scu_status )

!  check return status from SCU

            IF ( inform%scu_status == 0 ) THEN
              EXIT
            ELSE IF ( inform%scu_status < 0 ) THEN
              IF ( printi ) WRITE( out, "( A, ' Error: SCU_solve status ',     &
             &  I0 )" ) prefix, inform%scu_status
              inform%status = GALAHAD_error_solve ; GO TO 900
            END IF

!  SCU requires a further solve

            CALL CRO_block_solve( n, m, n_free, m_fixed, basic,                &
                                  A_val, A_col, A_ptr,                         &
!                                 data%VECTOR, b_fx, b_fr, c_fx, c_fr,         &
                                  data%VECTOR, b_fr, c_fx, c_fr, len_sls_sol,  &
                                  data%SLS_SOL, data%K_r, data%SLS_data,       &
                                  data%control%SLS_control, inform%SLS_inform, &
                                  data%X_free, data%C_basic,                   &
                                  data%X_inorder, data%C_inorder,              &
                                  inform%status,                               &
                                  H_val = H_val, H_col = H_col,                &
                                  H_ptr = H_ptr, H_lm = H_lm )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
!           b_fx = .TRUE.
            b_fr = .TRUE. ; c_fr = .TRUE. ; c_fx = .TRUE.
          END DO

!  spread into dx, dy and dz

          data%DX( : n ) = data%SOL( : n )
          data%DY = zero ; data%DZ = zero
          DO l = 1, all_basic
            i = data%BASIS( l )
            IF ( i > 0 ) THEN  ! basic constraint
              data%DY( i ) = data%SOL( n + l )
            ELSE IF ( i < 0 ) THEN  ! basic variable
              data%DZ( - i ) = data%SOL( n + l )
            END IF
          END DO
          data%DY( - incoming ) = dy_i

!  no Schur complement so no need for SCU

        ELSE
          CALL CRO_block_solve( n, m, n_free, m_fixed, basic,                  &
                                A_val, A_col, A_ptr,                           &
!                               data%RHS, b_fx, b_fr, c_fx, c_fr,              &
                                data%RHS, b_fr, c_fx, c_fr, len_sls_sol,       &
                                data%SLS_SOL, data%K_r, data%SLS_data,         &
                                data%control%SLS_control, inform%SLS_inform,   &
                                data%X_free, data%C_basic,                     &
                                data%X_inorder, data%C_inorder,                &
                                inform%status,                                 &
                                H_val = H_val, H_col = H_col,                  &
                                H_ptr = H_ptr, H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  spread into dx, dy and dz

          data%DX( : n ) = data%RHS( : n )
          DO i = 1, m
            IF ( data%C_inorder( i ) > 0 ) THEN
              data%DY( i ) = data%RHS( n + data%C_inorder( i ) )
            ELSE
              data%DY( i ) = zero
            END IF
          END DO
          data%DY( - incoming ) = dy_i
          DO j = 1, n
            IF ( data%X_inorder( j ) > 0 ) THEN
              data%DZ( j ) = zero
            ELSE
              data%DZ( j ) = data%RHS( n + basic - data%X_inorder( j ) )
            END IF
          END DO
        END IF
!write(6,"( ' DX ', /, (5ES12.4 ) )" ) data%DX
!write(6,"( ' DY ', /, (5ES12.4 ) )" ) data%DY
!write(6,"( ' DZ ', /, (5ES12.4 ) )" ) data%DZ

!  print residuals for debugging

        IF ( printa ) THEN
          data%RES_p = zero ; data%RES_d = zero
          CALL CRO_KKT_residual( n, m, A_val, A_col, A_ptr, data%DX,           &
                                 - data%DY, - data%DZ, data%RES_p, data%RES_d, &
                                 inform%status, H_val = H_val, H_col = H_col,  &
                                 H_ptr = H_ptr, H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          write(6,"( '     j st    RES_d        DX' )" )
          DO j = 1, n
            write(6,"( I6, I3, 2ES12.4 )" )                                    &
              j, data%X_inorder( j ), data%RES_d( j ), data%DX( j )
          END DO
          write(6,"( '     i st    RES_p        DY' )" )
          DO i = 1, m
            write(6,"( I6, I3, 2ES12.4 )" )                                    &
              i, data%C_inorder( i ), data%RES_p( i ), data%DY( i )
          END DO
        END IF

!  write(6,*) 'dx', data%DX
!  write(6,*) 'y', Y
!  write(6,*) 'dy', data%DY
!  write(6,*) 'dz', data%DZ

!  record the largest step that still satisfies the primal bounds and
!  Lagrange multipliers, and the variable/constraint that achieves this step

!       IF ( n_free > 0 ) THEN
          DO l = 1, m_fixed
            i = data%C_fixed( l )
            IF ( i > m_equal ) THEN
              IF ( C_stat( i ) < 0 ) THEN             ! active at lower bound
                IF ( data%DY( i ) > zero ) THEN
                  IF ( Y( i ) / data%DY( i ) < step ) THEN
                    step = Y( i ) / data%DY( i )
                    outgoing = - i
                  END IF
                END IF
              ELSE IF ( C_stat( i ) > 0 ) THEN        ! active at upper bound
                IF ( data%DY( i ) < zero ) THEN
                  IF ( Y( i ) / data%DY( i ) < step ) THEN
                    step = Y( i ) / data%DY( i )
                    outgoing = - i
                  END IF
                END IF
              END IF
            END IF
          END DO
!       END IF
!write(6,*) 'outgoing y ', outgoing, step
!  record the largest step that still satisfies the dual bounds and the dual
!  variable that achieves this step

        DO i = 1, n
          IF ( X_stat( i ) < 0 ) THEN                 ! active at lower bound
            IF ( data%DZ( i ) > zero ) THEN
              IF ( Z( i ) / data%DZ( i ) < step ) THEN
                step = Z( i ) / data%DZ( i )
                outgoing = i
              END IF
            END IF
          ELSE IF ( X_stat( i ) > 0 ) THEN            ! active at upper bound
            IF ( data%DZ( i ) < zero ) THEN
              IF ( Z( i ) / data%DZ( i ) < step ) THEN
                step = Z( i ) / data%DZ( i )
                outgoing = i
              END IF
            END IF
          END IF
        END DO
!write(6,*) 'outgoing z', outgoing, step

!  compute the changes to the constraints dc = A dx_fr

        data%DC = zero
        DO i = 1, m
          IF ( data%C_inorder( i ) == 0 ) THEN
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              j = A_col( l )
              IF ( data%X_inorder( j ) > 0 )                                   &
                data%DC( i ) = data%DC( i ) + A_val( l ) * data%DX( j )
            END DO
          END IF
        END DO
!write(6,*) ' dc(12) ', data%DC( 12 )
!stop

!  record the largest step that still satisfies the constraint bounds and the
!  constraint that achieves this step

!write(6,*) ' m_equal ', m_equal
        DO i = m_equal + 1, m                         ! can this happen?
!         write(6,*) i, data%C_inorder( i )
          IF ( data%C_inorder( i ) <= 0 ) THEN        ! inactive constraint
            IF ( data%DC( i ) > zero ) THEN           ! check lower bound
              IF ( ( C_u( i ) - C( i ) ) / data%DC( i ) < step ) THEN
                step = ( C_u( i ) - C( i ) ) / data%DC( i )
                outgoing = - i
              END IF
            ELSE IF ( data%DC( i ) < zero ) THEN      ! check upper bound
              IF ( ( C_l( i ) - C( i ) ) / data%DC( i ) < step ) THEN
                step = ( C_l( i ) - C( i ) ) / data%DC( i )
                outgoing = - i
              END IF
            END IF
          END IF
        END DO
!write(6,*) 'outgoing c', outgoing, step

!  if the step is infinite and either the incoming constraint is a general
!  equality or a free variable, reverse the step and try again

!write(6,*) ' tryboth ', tryboth, step,  step == infinity
        IF ( step == infinity .AND. tryboth ) THEN
!         IF ( n_free > 0 ) THEN
            IF ( b_fr_neq_0 ) THEN
              DO l = 1, n_free
                i = data%X_free( l )
                data%DX( i ) = - data%DX( i )
              END DO

              DO l = 1, m_fixed
                i = data%C_fixed( l )
                data%DY( i ) = - data%DY( i )
              END DO

!  record the largest step that still satisfies the primal bounds and
!  Lagrange multipliers, and the variable/constraint that achieves this step

              DO l = 1, m_fixed
                i = data%C_fixed( l )
                IF ( i > m_equal ) THEN
                  IF ( C_stat( i ) < 0 ) THEN           ! active at lower bound
                    IF ( data%DY( i ) > zero ) THEN
                      IF ( Y( i ) / data%DY( i ) < step ) THEN
                        step = Y( i ) / data%DY( i )
                        outgoing = - i
                      END IF
                    END IF
                  ELSE IF ( C_stat( i ) > 0 ) THEN      ! active at upper bound
                    IF ( data%DY( i ) < zero ) THEN
                      IF ( Y( i ) / data%DY( i ) < step ) THEN
                        step = Y( i ) / data%DY( i )
                        outgoing = - i
                      END IF
                    END IF
                  END IF
                END IF
              END DO
            END IF
!         END IF

!  record the largest step that still satisfies the dual bounds and the dual
!  variable that achieves this step

          data%DY( - incoming ) = - dy_i
          data%DZ = - data%DZ
          DO i = 1, n
            IF ( X_stat( i ) < 0 ) THEN                 ! active at lower bound
              IF ( data%DZ( i ) > zero ) THEN
                IF ( Z( i ) / data%DZ( i ) < step ) THEN
                  step = Z( i ) / data%DZ( i )
                  outgoing = i
                END IF
              END IF
            ELSE IF ( X_stat( i ) > 0 ) THEN            ! active at upper bound
              IF ( data%DZ( i ) < zero ) THEN
                IF ( Z( i ) / data%DZ( i ) < step ) THEN
                  step = Z( i ) / data%DZ( i )
                  outgoing = i
                END IF
              END IF
            END IF
          END DO
!write(6,"( ' DX ', /, (5ES12.4 ) )" ) data%DX
!write(6,"( ' DY ', /, (5ES12.4 ) )" ) data%DY
!write(6,"( ' DZ ', /, (5ES12.4 ) )" ) data%DZ

!  record the largest step that still satisfies the constraint bounds and the
!  constraint that achieves this step

          data%DC = - data%DC
          DO i = m_equal + 1, m                       ! can this happen?
            IF ( data%C_inorder( i ) <= 0 ) THEN      ! inactive constraint
              IF ( data%DC( i ) > zero ) THEN         ! check lower bound
                IF ( ( C_u( i ) - C( i ) ) / data%DC( i ) < step ) THEN
                  step = ( C_u( i ) - C( i ) ) / data%DC( i )
                  outgoing = - i
                END IF
              ELSE IF ( data%DC( i ) < zero ) THEN    ! check upper bound
                IF ( ( C_l( i ) - C( i ) ) / data%DC( i ) < step ) THEN
                  step = ( C_l( i ) - C( i ) ) / data%DC( i )
                  outgoing = - i
                END IF
              END IF
            END IF
          END DO
        END IF

!  record which variable/constraint/multiplier has hit it bound

        IF ( printm ) THEN
!         IF ( outgoing > 0 ) THEN
!           IF ( X_stat( outgoing ) == 0 ) THEN            ! can this happen?
!             WRITE( out, "( ' variable ', I0, ' fixed, step ', ES12.4 )" )    &
!               outgoing, step
!           ELSE
!             WRITE( out, "( ' variable ', I0, ' freed, step ', ES12.4 )" )    &
!               outgoing, step
!           END IF
!         ELSE
!           IF ( C_stat( - outgoing ) == 0 ) THEN          ! can this happen?
!             WRITE( out, "( ' constraint ', I0, ' fixed, step ', ES12.4 )")   &
!               - outgoing, step
!           ELSE
!             WRITE( out, "( ' constraint ', I0, ' freed, step ', ES12.4 )")   &
!               - outgoing, step
!           END IF
!         END IF

          WRITE( out, "( A, ' step taken =', ES9.2 )" )  prefix, step
          IF ( incoming > 0 ) THEN
            IF ( outgoing == incoming ) THEN
              WRITE( out, "( A, ' remove non-basic variable ', I0 )" )         &
                prefix, outgoing
            ELSE IF ( outgoing > 0 ) THEN
              WRITE( out, "( A, ' exchange basic variable ', I0,               &
             &     ' with non-basic variable ', I0 )" )                        &
                prefix, outgoing, incoming
            ELSE
              WRITE( out, "( A, ' exchange basic constraint ', I0,             &
             &     ' with non-basic variable ', I0 )" )                        &
                prefix, - outgoing, incoming
            END IF
          ELSE
            IF ( outgoing == incoming ) THEN
              WRITE( out, "( A, ' remove non-basic constraint ', I0 )" )       &
                prefix, - outgoing
            ELSE IF ( outgoing > 0 ) THEN
              WRITE( out, "( A, ' exchange basic variable ', I0,               &
             &     ' with non-basic constraint ', I0 )" )                      &
                prefix, outgoing, - incoming
            ELSE
              WRITE( out, "( A, ' exchange basic constraint ', I0,             &
             &   ' with non-basic constraint ', I0 )" )                        &
                prefix, - outgoing, - incoming
            END IF
          END IF
        END IF

!  record the change of status

        IF ( outgoing > 0 ) THEN
          X_stat( outgoing ) = SIGN( 2, X_stat( outgoing ) )
        ELSE
          C_stat( - outgoing ) = SIGN( 2, C_stat( - outgoing ) )
        END IF

!  step to the new point

!write(6,*) ' X(22) ', X(22)
!write(6,*) ' X(26) ', X(26)
        DO i = 1, n
          IF ( X_stat( i ) == 0 ) THEN
            X( i ) = X( i ) + step * data%DX( i )
          ELSE
            Z( i ) = Z( i ) - step * data%DZ( i )
          END IF
        END DO
!write(6,*) ' X(22) ', X(22)
!write(6,*) ' X(26) ', X(26)

!write(6,*) 'c, dc(12)', C(12), data%DC( 12 )
        DO i = 1, m
          IF ( C_stat( i ) == 0 ) THEN
            C( i ) = C( i ) + step * data%DC( i )
          ELSE
            Y( i ) = Y( i ) - step * data%DY( i )
          END IF
        END DO

        IF ( data%control%check_io ) THEN
          CALL CRO_check_status( n, m, m_equal, A_val, A_col, A_ptr, G, C_l,   &
                                 C_u, X_l, X_u, C, X, Y, Z, C_stat, X_stat,    &
                                 control, inform, data%DX, data%DY, prefix,    &
                                 H_val = H_val, H_col = H_col, H_ptr = H_ptr,  &
                                 H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  main point: we need to move to another basis in which an incoming
!  (non-basic variable) dy_fxn in the current system

!    ( H_fx    H_od^T  A_fxxb^T   I ) ( dx_fx  )     ( A_fxxn^T )
!    ( H_od     H_fr   A_frxb^T   0 ) ( dx_fr  ) = - ( A_frxn^T ) dy_fxn
!    ( A_fxxb  A_frxb       0     0 ) ( dy_fxb )     (    0     )
!    (  I        0          0     0 ) ( dz_fx  )     (    0     )

!  replaces an outgoing one of the those from dy_fxb or dz_fx; this is
!  a ** pivot **. Generically, we may view this as wanting to solve a system

!     (  K   d ) ( u ) = ( b )
!     ( d^T  0 ) ( v )   ( 0 )

!  when we have a factorization of

!     K_0 = (  K   c )
!           ( c^T  0 )

!  This is equivalent to solving the expanded system

!     (  K   c  0  d ) ( u )    ( b )
!     ( c^T  0  1  0 ) ( s )  = ( 0 )
!     (  0   1  0  0 ) ( t )    ( 0 )
!     ( d^T  0  0  0 ) ( v )    ( 0 )

!  for which K_0 is the leading sub-matrix, and rather than simply
!  refactorizing the new matrix, we may exploit this using a Schur-complement
!  updating method such as GALAHAD_SCU in which factors of K_0 and a small
!  Schur complement are used instead; the components s and t are subsequently
!  discarded

!  update the factoriztion

!  remove non-basic constraint -incoming

        IF ( outgoing == incoming ) CYCLE

!  update the list of basic constraints and variables

!       write(6,*) ' C_inorder ',  data%C_inorder
        all_basic = all_basic + 1
        data%BASIS( all_basic ) = 0
        IF ( outgoing < 0 ) THEN
          IF ( data%C_inorder( - outgoing ) > 0 ) THEN
            data%BASIS( data%C_inorder( - outgoing ) )  = 0
            row_out = n + data%C_inorder( - outgoing )
          ELSE
            DO j = basic + n_fixed + 1, all_basic
              IF ( data%BASIS( j ) == - outgoing ) THEN
                data%BASIS( j ) = 0
                row_out = n + j
                EXIT
              END IF
            END DO
          END IF
        ELSE IF ( outgoing > 0 ) THEN
          data%BASIS( basic - data%X_inorder( outgoing ) ) = 0
        END IF
        all_basic = all_basic + 1
        data%BASIS( all_basic ) = - incoming

 !      write(6,*) ' BASIS ',  data%BASIS( : all_basic )

!  if there is not enough space to expand the Schur complement, prepare
!  for a refactorization by re-defining the basis

        IF ( data%SCU_matrix%m + 2 > data%control%max_schur_complement ) THEN
!write(6,*) ' length of old basis list ', all_basic
!     write(6,*) ' BASIS ',  data%BASIS( : all_basic )

!         sofar = nb
          nb_start = nb + 1

!  count the number of basic constraints and the overall number of basics.
!  Discover which constraints are basic, and flag them by negating C_fixed

          n_free = n
!         basic_old = basic
          basic = 0
          all_basic_old = all_basic ; all_basic = 0
          data%X_inorder = 0
          data%C_inorder( data%C_fixed( : m_fixed ) ) = 0
          DO i = 1, all_basic_old
            j = data%BASIS( i )
            IF ( j /= 0 ) THEN
              all_basic = all_basic + 1
              data%BASIS( all_basic ) = j
              IF ( j > 0 ) THEN
                basic = basic + 1
                data%C_inorder( j ) = 1
              ELSE
                data%X_inorder( - j ) = n_free
                n_free = n_free - 1
              END IF
            END IF
          END DO

          DO i = 1, m_fixed
            IF ( data%C_inorder( data%C_fixed( i ) ) == 1 ) THEN
              data%C_fixed( i ) = - data%C_fixed( i )
            END IF
          END DO

!  reorder the list of free variables ...

          i = n_free ; n_free = 0 ; n_fixed = 0
          DO j = 1, n
            IF ( data%X_inorder( j ) == 0 ) THEN
              n_free = n_free + 1
              data%X_inorder( j ) = n_free
              data%X_free( n_free ) = j
            ELSE
              n_fixed = n_fixed + 1 ; i = i + 1
              data%X_inorder( j ) = - n_fixed
              data%X_free( i ) = j
            END IF
          END DO

!  ... and fixed constraints

          nonbasic = basic ; basic = 0
!         DO j = 1, sofar
          DO j = 1, m_fixed
            i = data%C_fixed( j )
            IF ( i > 0 ) THEN
              nonbasic = nonbasic + 1
              data%C_basic( nonbasic ) = i
              data%C_inorder( i ) = - nonbasic
            ELSE
              basic = basic + 1
              data%C_basic( basic ) = - i
              data%C_inorder( - i ) = basic
              data%C_fixed( j ) = - i
            END IF
          END DO

!write(6,*) ' length of new basis list ', all_basic
!     write(6,*) ' BASIS ',  data%BASIS( : all_basic )
!write(6,*) ' now free variables = ', n_free, ' basic constraints ', basic
!write(6,*) ' nonbasic ', data%C_basic( basic + 1 : m_fixed )

          data%SCU_matrix%n = n + basic + n_fixed
          data%SCU_matrix%m = 0
          CALL SCU_restart_m_eq_0( data%SCU_data, inform%SCU_inform )
          IF ( printi ) WRITE ( out,                                           &
            "( A, ' not enough space for Schur complement ... restarting' )" ) &
            prefix
          GO TO 100
        END IF

!  exchange basic constraint -outgoing with non-basic constraint -incoming;
!  this is basic row row_out in K_b

        IF ( outgoing < 0 ) THEN
!         IF ( data%C_inorder( - outgoing ) > 0 ) THEN
!           row_out = n + data%C_inorder( - outgoing )
!         ELSE
!           DO j = basic + n_fixed + 1, all_basic
!             IF ( data%BASIS( j ) == - outgoing ) THEN
!               row_out = n + j
!               EXIT
!             END IF
!           END DO
!         END IF
          c_fr = .TRUE. ; c_fx = .FALSE.
          IF ( printt ) WRITE ( out,                                           &
            "( A, '  deleting basic constraint ', I0, ' with SCU_append' )" )  &
            prefix,  - outgoing

!  exchange basic variable outgoing with non-basic constraint -incoming;
!  this is basic row row_out in K_b

        ELSE IF ( outgoing > 0 ) THEN
          row_out = n + basic - data%X_inorder( outgoing )
          c_fr = .FALSE. ; c_fx = .TRUE.
          IF ( printt ) WRITE ( out,                                           &
            "( A, '  deleting basic variable ', I0, ' with SCU_append' )" )    &
            prefix, outgoing
        END IF
!       b_fx = .FALSE.
        b_fr = .FALSE.

!  update the factorization of the schur complement of K_0 in

!     (  K   c  0  d )
!     ( c^T  0  1  0 )
!     (  0   1  0  0 )
!     ( d^T  0  0  0 )

!  corresponding to the outgoing basic in row row_out. Set up the column ( 0 )
!                                                                        ( 1 )

        j = data%SCU_matrix%BD_col_start( data%SCU_matrix%m + 1 )
        data%SCU_matrix%BD_row( j ) = row_out
!write(6,*) ' row_out ', row_out, data%SCU_matrix%n + data%SCU_matrix%m
        data%SCU_matrix%BD_val( j ) = one
        data%SCU_matrix%BD_col_start( data%SCU_matrix%m + 2 ) = j + 1

!  update the factorization of the Schur complement using SCU

        inform%scu_status = 1
        DO  ! append loop
          CALL SCU_append( data%SCU_matrix, data%SCU_data,                     &
                           data%VECTOR, inform%scu_status, inform%SCU_inform )

!  check return status from SCU

          IF ( inform%scu_status == 0 ) THEN
            EXIT
          ELSE IF ( inform%scu_status < 0 ) THEN
            IF ( printi ) WRITE( out, "( A, ' Error: SCU_append status ',      &
           &  I0 )" ) prefix, inform%scu_status
            inform%status = GALAHAD_error_factorization ; GO TO 900
          END IF

!  solve the system

!    ( H_fx    H_od^T  A_fxxb^T  I ) ( dx_fx  )
!    ( H_od     H_fr   A_frxb^T  0 ) ( dx_fr  ) = VECTOR
!    ( A_fxxb  A_frxb       0    0 ) ( dy_fxb )
!    (  I        0          0    0 ) ( dz_fx  )

          CALL CRO_block_solve( n, m, n_free, m_fixed, basic,                  &
                                A_val, A_col, A_ptr,                           &
!                               data%VECTOR, b_fx, b_fr, c_fx, c_fr,           &
                                data%VECTOR, b_fr, c_fx, c_fr, len_sls_sol,    &
                                data%SLS_SOL, data%K_r, data%SLS_data,         &
                                data%control%SLS_control, inform%SLS_inform,   &
                                data%X_free, data%C_basic,                     &
                                data%X_inorder, data%C_inorder,                &
                                inform%status,                                 &
                                H_val = H_val, H_col = H_col,                  &
                                H_ptr = H_ptr, H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END DO  ! end of append loop

!  update the factorization of the schur complement of K_0 in

!     (  K   c  0 )
!     ( c^T  0  1 )
!     (  0   1  0 )

!  corresponding to the incoming non-basic. Set up the column ( d )
!                                                             ( 0 )

        IF ( printt ) WRITE ( out, "( A, '  adding non-basic constraint ',     &
       &                      I0, ' with SCU_append ' )" ) prefix, - incoming

        j = data%SCU_matrix%BD_col_start( data%SCU_matrix%m + 1 )
!       b_fx = .FALSE.
        b_fr = .FALSE. ; c_fr = .FALSE. ; c_fx = .FALSE.
        DO l = A_ptr( - incoming ), A_ptr( - incoming + 1 ) - 1
          data%SCU_matrix%BD_row( j ) = A_col( l )
          data%SCU_matrix%BD_val( j ) = A_val( l )
          IF ( data%X_inorder( A_col( l ) ) >= 0 ) THEN
            b_fr = .TRUE.
!         ELSE
!           b_fx = .TRUE.
          END IF
          j = j + 1
        END DO
        data%SCU_matrix%BD_col_start( data%SCU_matrix%m + 2 ) = j

!  update the factorization of the Schur complement using SCU

        inform%scu_status = 1
        DO  ! append loop
          CALL SCU_append( data%SCU_matrix, data%SCU_data,                     &
                           data%VECTOR, inform%scu_status, inform%SCU_inform )

!  check return status from SCU

          IF ( inform%scu_status == 0 ) THEN
            EXIT
          ELSE IF ( inform%scu_status < 0 ) THEN
            IF ( printi ) WRITE( out, "( A, ' Error: SCU_append status ',      &
           &  I0 )" ) prefix, inform%scu_status
            inform%status = GALAHAD_error_factorization ; GO TO 900
          END IF

!  solve the system

!    ( H_fx    H_od^T  A_fxxb^T  I ) ( dx_fx  )
!    ( H_od     H_fr   A_frxb^T  0 ) ( dx_fr  ) = VECTOR
!    ( A_fxxb  A_frxb       0    0 ) ( dy_fxb )
!    (  I        0          0    0 ) ( dz_fx  )

          CALL CRO_block_solve( n, m, n_free, m_fixed, basic,                  &
                                A_val, A_col, A_ptr,                           &
!                               data%VECTOR, b_fx, b_fr, c_fx, c_fr,           &
                                data%VECTOR, b_fr, c_fx, c_fr, len_sls_sol,    &
                                data%SLS_SOL, data%K_r, data%SLS_data,         &
                                data%control%SLS_control, inform%SLS_inform,   &
                                data%X_free, data%C_basic,                     &
                                data%X_inorder, data%C_inorder,                &
                                inform%status,                                 &
                                H_val = H_val, H_col = H_col,                  &
                                H_ptr = H_ptr, H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END DO  ! end of append loop

!  -------------------------- end of main loop ------------------------------

      END DO

!     IF ( printd ) THEN
!       IF ( nonbasic > 0 )                                                    &
!         WRITE( out, "( A, ' C_nonbasis ', /, ( 10I8 ) )" )                   &
!           prefix, data%C_basic( basic + 1 : basic + nonbasic )
!     END IF

!write(6,*) prefix, ' -- stopping in CRO_crossover'
!stop

!  if required, check the computed primal-dual point is a KKT point

      IF ( data%control%check_io ) THEN
        CALL CRO_check_status( n, m, m_equal, A_val, A_col, A_ptr, G, C_l,     &
                               C_u, X_l, X_u, C, X, Y, Z, C_stat, X_stat,      &
                               control, inform, data%DX, data%DY, prefix,      &
                               H_val = H_val, H_col = H_col, H_ptr = H_ptr,    &
                               H_lm = H_lm )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
      END IF

 800  CONTINUE

!  if required, refine the solution so that it satisfies the KKT conditions
!  more accurately. We do this by solving the reduced optimality system

!     (  H_fr   A_frab^T ) (  x_fr  ) = rhs = ( - g_fr - H_od x_fx ),  (rKKT2)
!     ( A_frab     0     ) ( - y_ab )         ( c_ab - A_fxab x_fx )

!  where x_fx and c_ab are the values of the bounds for the fixed variables
!  and basic active constraints, using the factors of the matrix K_r in
!  (rKKT2), and then recovering

!    z_fx = g_fx + H_fx x_fx + H_od^T x_fr - A_fxab^T y_ab.

!  For the limited memory Hessian case, H = delta I - V W^-1 V^T, where
!  V = R [ Y : delta S ], and thus

!     K_r = (  delta I - V_fr W^-1 V_fr^T  A_frxb^T ).
!           (             A_frxb               0    )

!  Since any system K_r v = rhs we may be required to solve can be expanded to

!     ( delta I   A_frab^T  V_fr ) (  v  )   ( rhs )
!     (  A_frab      0       0   ) (     ) = (     ),
!     (  V_fr^T      0       W   ) ( - z )   (  0  )

!  we instead factorize this expanded matrix, K_re ; both V and W are dense

      IF ( control%refine_solution ) THEN

        IF ( m_fixed > 0 ) THEN
          m_fixed = 0
          DO i = 1, m
            IF ( ABS( C_stat( i ) ) == 1 ) THEN
              m_fixed = m_fixed + 1
              data%C_inorder( i ) = m_fixed
              data%C_fixed( m_fixed ) = i
            ELSE
              data%C_inorder( i ) = 0
            END IF
          END DO
        ELSE
          array_name = 'cro: data%X_inorder'
          CALL SPACE_resize_array( n, data%X_inorder,                          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%X_free'
          CALL SPACE_resize_array( n, data%X_free,                             &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF

!  count the number of basic active constraints and free variables

        n_free = 0
        DO j = 1, n
          IF ( X_stat( j ) == 0 ) THEN
            n_free = n_free + 1
            data%X_inorder( j ) = n_free
            data%X_free( n_free ) = j
          ELSE
            data%X_inorder( j ) = 0
          END IF
        END DO

!  count the number of nonzeros in the components of K_r or K_re

        IF ( n_free > 0 ) THEN

!  firstly, find the number of nonzeros for free columns in the Jacobian of
!  active constraints

          data%A_r%ne = 0
          IF ( m_fixed > 0 ) THEN
            DO i = 1, m
              IF ( data%C_inorder( i ) > 0 ) THEN
                IF (  A_ptr( i + 1 ) > A_ptr( i ) ) data%A_r%ne = data%A_r%ne  &
                  + COUNT( data%X_inorder( A_col( A_ptr( i ) :                 &
                                                  A_ptr( i + 1 ) - 1 ) ) > 0 )
              END IF
            END DO
          END IF

!  now add the free submatrix for the limited memory Hessian case ...

          IF ( lbfgs ) THEN
            dim_w = 2 * H_lm%length
            data%H_r%ne = n_free
            data%A_r%ne = data%A_r%ne + dim_w * n_free
            data%C_r%ne = ( dim_w * ( dim_w + 1 ) ) / 2

          ELSE
            dim_w = 0 ; data%H_r%ne = 0 ; data%C_r%ne = 0

!  ... or for the Hessian stored by rows

            IF ( is_h ) THEN
              DO i = 1, n
                IF ( data%X_inorder( i ) > 0 ) THEN
                  IF (  H_ptr( i + 1 ) > H_ptr( i ) )                          &
                    data%H_r%ne = data%H_r%ne  +                               &
                      COUNT( data%X_inorder( H_col( H_ptr( i ) :               &
                                                    H_ptr( i + 1 ) - 1 ) ) > 0 )
                END IF
              END DO
            END IF
          END IF

!  allocate space for the components of  K_r or K_re as appropriate

          CALL SMT_put( data%H_r%type, 'COORDINATE', i )
          data%H_r%n = n_free ; data%H_r%m = n_free
          array_name = 'cro: data%H_r%row'
          CALL SPACE_resize_array( data%H_r%ne, data%H_r%row,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%H_r%col'
          CALL SPACE_resize_array( data%H_r%ne, data%H_r%col,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%H_r%val'
          CALL SPACE_resize_array( data%H_r%ne, data%H_r%val,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          CALL SMT_put( data%A_r%type, 'COORDINATE', i )
          data%A_r%n = n_free ; data%A_r%m = m_fixed + dim_w
          array_name = 'cro: data%A_r%row'
          CALL SPACE_resize_array( data%A_r%ne, data%A_r%row,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%A_r%col'
          CALL SPACE_resize_array( data%A_r%ne, data%A_r%col,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%A_r%val'
          CALL SPACE_resize_array( data%A_r%ne, data%A_r%val,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          CALL SMT_put( data%C_r%type, 'COORDINATE', i )
          data%C_r%n = m_fixed  + dim_w ; data%C_r%m = m_fixed + dim_w
          array_name = 'cro: data%C_r%row'
          CALL SPACE_resize_array( data%C_r%ne, data%C_r%row,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%C_r%col'
          CALL SPACE_resize_array( data%C_r%ne, data%C_r%col,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          array_name = 'cro: data%C_r%val'
          CALL SPACE_resize_array( data%C_r%ne, data%C_r%val,                  &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          K_r_n = n_free + m_fixed + dim_w

!  form K_re

          IF ( lbfgs ) THEN

!  record the components of H_fr in K_re, ...

            data%H_r%ne = 0
            IF ( H_lm%restricted == 0 ) THEN
              DO j = 1, n_free
                data%H_r%ne = data%H_r%ne + 1
                data%H_r%row( data%H_r%ne ) = j
                data%H_r%col( data%H_r%ne ) = j
                data%H_r%val( data%H_r%ne ) = H_lm%delta
              END DO
            ELSE
              DO j = 1, n
                l = data%X_inorder( j )
                IF ( l > 0 .AND. l <= H_lm%n ) THEN
                  data%H_r%ne = data%H_r%ne + 1
                  data%H_r%row( data%H_r%ne ) = l
                  data%H_r%col( data%H_r%ne ) = l
                  data%H_r%val( data%H_r%ne ) = H_lm%delta
                END IF
              END DO
            END IF

!  ... those of A_frab, ...

            data%A_r%ne = 0
            DO i = 1, m
              IF ( data%C_inorder( i ) > 0 ) THEN
                DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                  j = A_col( l )
                  IF ( data%X_inorder( j ) > 0 ) THEN
                    data%A_r%ne = data%A_r%ne + 1
                    data%A_r%row( data%A_r%ne ) = data%C_inorder( i ) + n_free
                    data%A_r%col( data%A_r%ne ) = data%X_inorder( j )
                    data%A_r%val( data%A_r%ne ) = A_val( l )
                  END IF
                END DO
              END IF
            END DO

!  ... those of W ...

            data%C_r%ne = 0
            DO j = 1, dim_w
              DO i = j, dim_w
                data%C_r%ne = data%C_r%ne + 1
                data%C_r%row( data%C_r%ne ) = i
                data%C_r%col( data%C_r%ne ) = j
                data%C_r%val( data%C_r%ne ) = data%W( i, j )
              END DO
            END DO

!  ... and those of V_fr^T where V = R [ Y : delta S ]

            DO i = 1, H_lm%length
              oi = H_lm%ORDER( i )
              DO j = 1, n
                IF ( data%X_inorder( j ) > 0 ) THEN
                  data%A_r%ne = data%A_r%ne + 1
                  data%A_r%row( data%A_r%ne ) = m_fixed + i
                  data%A_r%col( data%A_r%ne ) = data%X_inorder( j )
                  IF ( H_lm%restricted == 0 ) THEN
                    data%A_r%val( data%A_r%ne ) = H_lm%Y( j, oi )
                  ELSE
                    l = H_lm%RESTRICTION( j )
                    IF ( l <= H_lm%n ) THEN
                      data%A_r%val( data%A_r%ne ) = H_lm%Y( l, oi )
                    ELSE
                      data%A_r%ne = data%A_r%ne - 1
                    END IF
                  END IF
                END IF
              END DO
            END DO

            DO i = 1, H_lm%length
              oi = H_lm%ORDER( i )
              DO j = 1, n
                IF ( data%X_inorder( j ) > 0 ) THEN
                  data%A_r%ne = data%A_r%ne + 1
                  data%A_r%row( data%A_r%ne ) = m_fixed + i
                  data%A_r%col( data%A_r%ne ) = data%X_inorder( j )
                  IF ( H_lm%restricted == 0 ) THEN
                    data%A_r%val( data%A_r%ne ) = H_lm%delta * H_lm%S( j, oi )
                  ELSE
                    l = H_lm%RESTRICTION( j )
                    IF ( l <= H_lm%n ) THEN
                      data%A_r%val( data%A_r%ne ) = H_lm%delta * H_lm%S( l, oi )
                    ELSE
                      data%A_r%ne = data%A_r%ne - 1
                    END IF
                  END IF
                END IF
              END DO
            END DO

!  form K_r

!  record the components of H_fr in K_r ...

          ELSE
            data%H_r%ne = 0
            IF ( is_h ) THEN
              DO i = 1, n
                IF ( data%X_inorder( i ) > 0 ) THEN
                  DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                    j = H_col( l )
                    IF ( data%X_inorder( j ) > 0 ) THEN
                      data%H_r%ne = data%H_r%ne + 1
                      data%H_r%row( data%H_r%ne ) = data%X_inorder( i )
                      data%H_r%col( data%H_r%ne ) = data%X_inorder( j )
                      data%H_r%val( data%H_r%ne ) = H_val( l )
                    END IF
                  END DO
                END IF
              END DO
            END IF

!  ... and those of A_frab

            data%A_r%ne = 0
            IF ( m_fixed > 0 ) THEN
              DO i = 1, m
                IF ( data%C_inorder( i ) > 0 ) THEN
                  DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
                    j = A_col( l )
                    IF ( data%X_inorder( j ) > 0 ) THEN
                      data%A_r%ne = data%A_r%ne + 1
                      data%A_r%row( data%A_r%ne ) = data%C_inorder( i )
                      data%A_r%col( data%A_r%ne ) = data%X_inorder( j )
                      data%A_r%val( data%A_r%ne ) = A_val( l )
                    END IF
                  END DO
                END IF
              END DO
            END IF
          END IF
          data%K_r%n = K_r_n
!         CALL SMT_put( data%K_r%type, 'COORDINATE', i )

!       WRITE( 6, "( ' K_sub: n, nnz ', 2I4 )" ) data%K_r%n, data%K_r%ne
!       WRITE( 6, "( A, /, ( 10I7) )" ) ' rows =', data%K_r%row( : data%K_r%ne )
!       WRITE( 6, "( A, /, ( 10I7) )" ) ' cols =', data%K_r%col( : data%K_r%ne )
!       WRITE( 6, "( A, /, ( F7.2) )" ) ' vals =', data%K_r%val( : data%K_r%ne )

!  order the rows/columns of K_r prior to factorization

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SBLS_initialize( data%SBLS_data, data%control%SBLS_control,     &
                                inform%SBLS_inform )

!  factorize the basic sub-block K_r

          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SBLS_form_and_factorize( data%A_r%n, data%A_r%m, data%H_r,      &
                                        data%A_r, data%C_r, data%SBLS_data,    &
                                        data%control%SBLS_control,             &
                                        inform%SBLS_inform )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%factorize = inform%time%factorize + time_now - time_start
          inform%time%clock_factorize =                                        &
            inform%time%clock_factorize + clock_now - clock_record

          IF ( printi ) WRITE( out,                                            &
               "( A, ' K nnz(matrix,factors) = ', I0, ', ', I0,                &
            &  /, A, ' SBLS: factorization of K complete: status = ', I0 )" )  &
              prefix, data%K_r%ne,                                             &
              inform%SBLS_inform%SLS_inform%entries_in_factors,                &
              prefix, inform%SBLS_inform%status
          IF ( inform%SBLS_inform%status < 0 ) THEN
             inform%status = GALAHAD_error_factorization ; GO TO 900
          END IF

          IF ( printm ) WRITE( out, "( A, ' K%n ', I0, ' rank ', I0 )" )       &
            prefix, K_r_n, inform%SBLS_inform%SLS_inform%rank
        ELSE
          data%K_r%n = 0
        END IF

!   now solve the required optimality system

!     (  H_fr   A_frab^T ) (  x_fr  ) = ( - g_fr - H_od x_fx )
!     ( A_frab     0     ) ( - y_ab )   ( c_ab - A_fxab x_fx )

!  and find  z_fx = g_fx + H_fx x_fx + H_od^T x_fr - A_fxab^T y_ab,

!  where x_fx and c_ab are the values of the bounds for the fixed variables
!  and basic active constraints

!  provide space for the right-hand side

        array_name = 'cro: data%SOL'
        CALL SPACE_resize_array( data%K_r%n, data%SOL,                         &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set up the right-hand side, starting with ( - g_fr ) ...
!                                            (  c_ab  )

        DO j = 1, n
          IF ( X_stat( j ) == 0 ) THEN
            data%SOL( data%X_inorder( j ) ) = - G( j )
            V_new( data%X_inorder( j ) ) = X( j )

!  ... record the free dual variables ...

            Z_new( j ) = zero

!  ... the fixed primal variables (and initialize z_fx = g_fx) ...

          ELSE IF ( X_stat( j ) < 0 ) THEN
            X_new( j ) = X_l( j )
            Z_new( j ) = G( j )
          ELSE
            X_new( j ) = X_u( j )
            Z_new( j ) = G( j )
          END IF
        END DO

        DO i = 1, m
          IF ( C_stat( i ) == - 1 ) THEN
            data%SOL( n_free + data%C_inorder( i ) ) = C_l( i )
            V_new(  n_free + data%C_inorder( i ) ) = - Y( i )
          ELSE IF ( C_stat( i ) == 1 ) THEN
            data%SOL( n_free + data%C_inorder( i ) ) = C_u( i )
            V_new(  n_free + data%C_inorder( i ) ) = - Y( i )

!  ... and the multipliers for the inactive constraints ...

          ELSE
            Y_new( i ) = zero
          END IF
        END DO

!  ... then include the component - H_od x_fx ...

        IF ( lbfgs ) THEN
          data%SOL( n_free + m_fixed + 1 : data%K_r%n ) = zero
          write(6,*) ' *** lbfgs refinement to be written'
          inform%status = GALAHAD_not_yet_implemented ; GO TO 900
        ELSE IF ( is_h ) THEN
          DO i = 1, n
            ii = data%X_inorder( i )
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l ) ; jj = data%X_inorder( j )
              IF ( ii > 0 .AND. jj == 0 ) THEN
                IF ( X_stat( j ) < 0 ) THEN
                  data%SOL( ii ) = data%SOL( ii ) - H_val( l ) * X_l( j )
                ELSE
                  data%SOL( ii ) = data%SOL( ii ) - H_val( l ) * X_u( j )
                END IF
              ELSE IF ( jj > 0 .AND. ii == 0 ) THEN
                IF ( X_stat( i ) < 0 ) THEN
                  data%SOL( jj ) = data%SOL( jj ) - H_val( l ) * X_l( i )
                ELSE
                  data%SOL( jj ) = data%SOL( jj ) - H_val( l ) * X_u( i )
                END IF

! ... as well as z_fx -> z_fx + H_fx x_fx ...

              ELSE IF ( ii == 0 .AND. jj == 0 ) THEN
                IF ( X_stat( j ) < 0 ) THEN
                  Z_new( i ) = Z_new( i ) + H_val( l ) * X_l( j )
                ELSE
                  Z_new( i ) = Z_new( i ) + H_val( l ) * X_u( j )
                END IF
                IF ( i /= j ) THEN
                  IF ( X_stat( i ) < 0 ) THEN
                    Z_new( j ) = Z_new( j ) + H_val( l ) * X_l( i )
                  ELSE
                    Z_new( j ) = Z_new( j ) + H_val( l ) * X_u( i )
                  END IF
                END IF
              END IF
            END DO
          END DO
        END IF

!  ... and finally - A_fxab x_fx

        DO ii = 1, m_fixed
          i = data%C_fixed( ii )
          ip =  n_free + ii
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            j = A_col( l )
            IF ( data%X_inorder( j ) == 0 ) THEN
              IF ( X_stat( j ) < 0 ) THEN
                data%SOL( ip ) = data%SOL( ip ) - A_val( l ) * X_l( j )
              ELSE
                data%SOL( ip ) = data%SOL( ip ) - A_val( l ) * X_u( j )
              END IF
            END IF
          END DO
        END DO

!       CALL mop_Ax( one, data%K_r, V_new, zero, R_new, out = 6, error = 6,    &
!                    print_level = 1, symmetric = .TRUE. )
!       i = 1 ; j = data%K_r%n
!       write(6, "( /, A, ' largest residual', ES11.4, ' in position ', I0 )" )&
!         prefix, MAXVAL( ABS( R_new( i : j ) - data%SOL( i : j  ) ) ),        &
!         i - 1 + MAXLOC( ABS( R_new( i : j ) - data%SOL( i : j ) ) )
!        DO i = 1, data%K_r%n
!        write(95,"(I6, 2ES22.14)") i, R_new( i ), data%SOL( i )
!        END DO

!  solve the system

        data%control%IR_control%print_level = 2
        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SBLS_solve( data%A_r%n, data%A_r%m, data%A_r, data%C_r,           &
                         data%SBLS_data, data%control%SBLS_control,            &
                         inform%SBLS_inform, data%SOL )
!       CALL SLS_fredholm_alternative( data%K_r, data%SOL, data%SLS_data,      &
!                      data%control%SLS_control, inform%SLS_inform )
!write(6,*) ' alternative = ', inform%SLS_inform%alternative
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_factorize + clock_now - clock_record

!  record x_fr and y_ab

        X_new( data%X_free( 1 : n_free ) ) = data%SOL( 1 : n_free )
        IF ( m_fixed > 0 ) Y_new( data%C_fixed( 1 : m_fixed ) )                &
                             = - data%SOL( n_free + 1 : n_free + m_fixed )

!  recover z_fx -> z_fx + H_od^T x_fr

        IF ( lbfgs ) THEN
          write(6,*) ' *** lbfgs refinement to be written'
          inform%status = GALAHAD_not_yet_implemented ; GO TO 900
        ELSE IF ( is_h ) THEN
          DO i = 1, n
            ii = data%X_inorder( i )
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l ) ; jj = data%X_inorder( j )
              IF ( ii > 0 .AND. jj == 0 ) THEN
                Z_new( j ) = Z_new( j ) + H_val( l ) * X_new( i )
              ELSE IF ( jj > 0 .AND. ii == 0 ) THEN
                Z_new( i ) = Z_new( i ) + H_val( l ) * X_new( j )
              END IF
            END DO
          END DO
        END IF

!  and then z_fx -> z_fx - A_fxab^T y_ab

        DO ii = 1, m_fixed
          i = data%C_fixed( ii )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            j = A_col( l )
            IF ( data%X_inorder( j ) == 0 )                                    &
              Z_new( j ) = Z_new( j ) - A_val( l ) * Y_new( i )
          END DO
        END DO

        DO i = 1, m
          C_new( i ) = DOT_PRODUCT( A_val( A_ptr( i ) : A_ptr( i + 1 ) - 1 ),  &
                             X_new( A_col( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) ) )
        END DO

!  debug printing

        IF ( printd ) THEN
          WRITE( out, "( A, ' X(stat,current,new) ' )" ) prefix
          DO i = 1, n
            WRITE( out ,"( A, 2I8, 2ES22.14 )" )                               &
              prefix, i, X_stat( i ), X( i ), X_new( i )
          END DO

          WRITE( out, "( A, ' Z(stat,current,new) ' )" ) prefix
          DO i = 1, n
            WRITE( out ,"( A, 2I8, 2ES22.14 )" )                               &
              prefix, i, X_stat( i ), Z( i ), Z_new( i )
          END DO

          WRITE( out, "( A, ' Y(stat,current,new) ' )" ) prefix
          DO i = 1, m
            WRITE( out ,"( A, 2I8, 2ES22.14 )" )                               &
              prefix, i, C_stat( i ), Y( i ), Y_new( i )
          END DO

          WRITE( out, "( A, ' C(stat,current,new) ' )" ) prefix
          DO i = 1, m
            WRITE( out ,"( A, 2I8, 2ES22.14 )" )                               &
              prefix, i, C_stat( i ), C( i ), Y_new( i )
          END DO
        END IF

        X( : n ) = X_new( : n )
        Z( : n ) = Z_new( : n )
        Y( : m ) = Y_new( : m )
        C( : m ) = C_new( : m )

!  recheck

        IF ( data%control%check_io ) THEN
          CALL CRO_check_status( n, m, m_equal, A_val, A_col, A_ptr, G, C_l,   &
                                 C_u, X_l, X_u, C, X, Y, Z, C_stat, X_stat,    &
                                 control, inform, data%DX, data%DY, prefix,    &
                                 H_val = H_val, H_col = H_col, H_ptr = H_ptr,  &
                                 H_lm = H_lm )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900
        END IF
      END IF

!  prepare to exit

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printt ) WRITE( out, "( A, ' -- leaving CRO_crossover ' )" ) prefix

      RETURN

!  End of CRO_crossover_main

      END SUBROUTINE CRO_crossover_main

!-*-*-*-*-*-*-   C R O _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE CRO_terminate( data, control, inform )

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
!   data    see Subroutine CRO_initialize
!   control see Subroutine CRO_initialize
!   inform  see Subroutine CRO_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( CRO_data_type ), INTENT( INOUT ) :: data
      TYPE ( CRO_control_type ), INTENT( IN ) :: control
      TYPE ( CRO_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SLS and ULS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL ULS_terminate( data%ULS_data, control%ULS_control,                  &
                          inform%ULS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%ULS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'cro: data%X_basic'
      CALL SPACE_dealloc_array( data%X_basic,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%C_basic'
      CALL SPACE_dealloc_array( data%C_basic,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%X_inorder'
      CALL SPACE_dealloc_array( data%X_inorder,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%C_inorder'
      CALL SPACE_dealloc_array( data%C_inorder,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%X_free'
      CALL SPACE_dealloc_array( data%X_free,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%C_fixed'
      CALL SPACE_dealloc_array( data%C_fixed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%BASIS'
      CALL SPACE_dealloc_array( data%BASIS,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%AT%row'
      CALL SPACE_dealloc_array( data%AT%row,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%AT%col'
      CALL SPACE_dealloc_array( data%AT%col,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%AT%val'
      CALL SPACE_dealloc_array( data%AT%val,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%K_r%row'
      CALL SPACE_dealloc_array( data%K_r%row,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%K_r%col'
      CALL SPACE_dealloc_array( data%K_r%col,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%K_r%val'
      CALL SPACE_dealloc_array( data%K_r%val,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%SOL'
      CALL SPACE_dealloc_array( data%SOL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%SLS_SOL'
      CALL SPACE_dealloc_array( data%SLS_SOL,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%DX'
      CALL SPACE_dealloc_array( data%DX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%DY'
      CALL SPACE_dealloc_array( data%DY,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%DZ'
      CALL SPACE_dealloc_array( data%DZ,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%DC'
      CALL SPACE_dealloc_array( data%DC,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%RES_p'
      CALL SPACE_dealloc_array( data%RES_p,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%RES_d'
      CALL SPACE_dealloc_array( data%RES_d,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated for and within SCU

      array_name = 'cro: data%SCU_matrix%BD_col_start'
      CALL SPACE_dealloc_array( data%SCU_matrix%BD_col_start,                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%SCU_matrix%BD_val'
      CALL SPACE_dealloc_array( data%SCU_matrix%BD_val,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%SCU_matrix%BD_row'
      CALL SPACE_dealloc_array( data%SCU_matrix%BD_row,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL SCU_terminate( data%SCU_data, inform%scu_status, inform%SCU_inform )
      CALL IR_terminate( data%IR_data, control%IR_control, inform%IR_inform )

!  End of subroutine CRO_terminate

      END SUBROUTINE CRO_terminate

!-*-*-*-*-*-   C R O _ B L O C K _ S O L V E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE CRO_block_solve( n, m, n_free, m_fixed, basic,                &
                                  A_val, A_col, A_ptr, SOL,                    &
!                                 b_fx, b_fr, c_fx, c_fr, SLS_SOL,             &
                                  b_fr, c_fx, c_fr, len_sls_sol, SLS_SOL,      &
                                  K, SLS_data, SLS_control, SLS_inform,        &
                                  X_free, C_basic, X_inorder, C_inorder,       &
                                  status, H_val, H_col, H_ptr, H_lm )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  solve the linear system

!    ( H_fx    H_od^T  A_fxxb^T  I ) ( x_fx  )   ( b_fx )
!    ( H_od     H_fr   A_frxb^T  0 ) ( x_fr  ) = ( b_fr )
!    ( A_fxxb  A_frxb       0    0 ) ( y_fxb )   ( c_fr )
!    (  I        0          0    0 ) ( z_fx  )   ( c_fx )

!  where ( b_fx, b_fr, c_fr, c_fx ) is input compactly in SOL and
!  ( x_fx, x_fr, y_fxb, z_fx ) is output in SOL. The logicals b_fx, b_fr,
!  c_fr and c_fx should be set TRUE if the corresponding vectors are nonzero
!  and FALSE otherwise

!  Details: solve the system sequentially via

!  (a)  x_fx = c_fx
!  (b)  (  H_fr   A_frxb^T ) ( x_fr  ) = ( b_fr -  H_od x_fx  )
!       ( A_frxb     0     ) ( y_fxb )   ( c_fr - A_fxxb x_fx )
!  (c) z_fx = b_fx - H_fx x_fx - H_od^T x_fr - A_fxxb^T y_fxb

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, n_free, m_fixed, basic, len_sls_sol
      INTEGER, INTENT( OUT ) :: status
      LOGICAL, INTENT( IN ) :: b_fr, c_fx, c_fr
!     LOGICAL, INTENT( IN ) :: b_fx
      INTEGER, INTENT( IN ), DIMENSION( n ) :: X_free
      INTEGER, INTENT( IN ), DIMENSION( m_fixed ) :: C_basic
      INTEGER, INTENT( IN ), DIMENSION( n ) :: X_inorder
      INTEGER, INTENT( IN ), DIMENSION( m ) :: C_inorder
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      TYPE ( SMT_type ), INTENT( IN ) :: K
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: SLS_data
      TYPE ( SLS_control_type ), INTENT( IN ) :: SLS_control
      TYPE ( SLS_inform_type ), INTENT( INOUT ) :: SLS_inform
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( 2 * n + basic - n_free ) :: SOL
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( len_sls_sol ) :: SLS_SOL
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_col
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_val
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( INOUT ) :: H_lm

!  Local variables

      INTEGER :: i, j, l, ll, y, z
      REAL ( KIND = wp ) :: val
      LOGICAL :: lbfgs, is_h

      lbfgs = PRESENT( H_lm )
      IF ( lbfgs ) THEN
        is_h = .FALSE.
      ELSE
        is_h = PRESENT( H_val ) .AND. PRESENT( H_col ) .AND.                   &
               PRESENT( H_ptr )
      END IF
      status = GALAHAD_ok

!  starting addresses for y_fxb and z_fx

      y = n
      z = n + basic

!  (a) set x_fx = c_fx

!  if c_fx is nonzero, set x_fx = c_fx, x_fr = b_fr and z_fx = b_fx ...

      IF ( c_fx ) THEN
        DO j = 1, n
          IF ( X_inorder( j ) <= 0 ) THEN   ! fixed variables
            val = SOL( z - X_inorder( j ) )
            SOL( z - X_inorder( j ) ) = SOL( j )
            SOL( j ) = val
          END IF
        END DO

!  ... either compute the product ( H_fx    H_od^T ) ( x_fx ) = (     *     )
!                                 ( H_od     H_fr  ) (  0   )   ( H_od x_fx )

!  to obtain H_od x_fx and then reset x_fr <- x_fr - H_od x_fx ...

        IF ( lbfgs ) THEN

!  gather x_fx and 0 into SLS_SOL

          SLS_SOL( : n ) = zero
          DO i = 1, n
            IF ( X_inorder( i ) < 0 ) SLS_SOL( - X_inorder( i ) ) = SOL( i )
          END DO

!  form the product of H with SLS_SOL and return the product in SLS_SOL

          CALL LMS_apply_lbfgs( SLS_SOL, H_lm, status )
          IF ( status /= 0 ) THEN
            status = GALAHAD_error_factorization ; RETURN
          END IF

!  update z_fx <- z_fx - SLS_SOL_fx

          DO i = 1, n
            IF ( X_inorder( i ) > 0 )                                          &
              SOL( X_inorder( i ) ) = SOL( X_inorder( i ) ) - SLS_SOL( i )
          END DO

!  ... or directly reset x_fr <- x_fr - H_od x_fx ...


        ELSE IF ( is_h ) THEN
          DO i = 1, n
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l )
              IF ( X_inorder( i ) > 0 .AND. X_inorder( j ) <= 0 ) THEN
                SOL( i ) = SOL( i ) - H_val( l ) * SOL( j )
              ELSE IF ( X_inorder( i ) <= 0 .AND. X_inorder( j ) > 0 ) THEN
                SOL( j ) = SOL( j ) - H_val( l ) * SOL( i )
              END IF
            END DO
          END DO
        END IF

!  ... and y_fxb = c_fr - A_frxb x_fx

        DO ll = 1, basic
          i = C_basic( ll )
!         Y( ll ) = SOL( y + ll )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            j = A_col( l )
            IF ( X_inorder( j ) <= 0 )                                         &
              SOL( y + ll ) = SOL( y + ll ) - A_val( l ) * SOL( j )
          END DO
        END DO

!  if c_fx is zero, set x_fx = 0, x_fr = b_fr, z_fx = - b_fx and y_fxb = c_fr

      ELSE
        DO j = 1, n
          IF ( X_inorder( j ) <= 0 ) THEN  ! fixed variables
            SOL( z - X_inorder( j ) ) = SOL( j )
            SOL( j ) = zero
          END IF
        END DO
      END IF

!  (b) if A_frxn^T y_fxn is nonzero, solve the block system

!    (  H_fr   A_frxb^T ) ( x_fr  ) = sol <- ( x_fr  )
!    ( A_frxb     0     ) ( y_fxb )          ( y_fxb )

!  to find x_fr and y_fxb

      IF ( K%n > 0 .AND. ( c_fx .OR. b_fr .OR. c_fr ) ) THEN

!  gather x_fr and y_fxb into SLS_SOL

        DO l = 1, n_free
          SLS_SOL( l ) = SOL( X_free( l ) )
        END DO

        DO l = 1, basic
          SLS_SOL( n_free + l ) = SOL( y + l )
        END DO
        IF ( lbfgs ) SLS_SOL( n_free + basic + 1 : ) = zero

!  solve the system

!write(6,*) ' rhs ', SLS_SOL
        CALL SLS_solve( K, SLS_SOL, SLS_data, SLS_control, SLS_inform )
        IF ( SLS_inform%status /= GALAHAD_ok ) THEN
          status = GALAHAD_error_solve ; RETURN
        END IF
!write(6,*) ' sol ', SLS_SOL

!  scatter the changes back to x_fr and y_fxb

        DO l = 1, n_free
          SOL( X_free( l ) ) = SLS_SOL( l )
        END DO

        DO l = 1, basic
          SOL( y + l ) = SLS_SOL( n_free + l )
        END DO
      END IF

!  (c) set z_fx = b_fx - H_fx x_fx - H_od^T x_fr - A_fxxb^T y_fxb

      IF ( lbfgs ) THEN

!  ... either compute ( H_fx    H_od^T ) ( x_fx ) = ( H_fx x_fx + H_od^T x_fr )
!                     ( H_od     H_fr  ) ( x_fr )   (           *             )

!  to obtain H_fx x_fx + H_od^T x_fr, and thus update
!  z_fx <- z_fx - ( H_fx x_fx + H_od^T x_fr ) ...

!  gather x_fx and x_fr into SLS_SOL

        SLS_SOL( : n ) = SOL( : n )

!  form the product of H with SLS_SOL and return the product in SLS_SOL

        CALL LMS_apply_lbfgs( SLS_SOL, H_lm, status )
        IF ( status /= 0 ) THEN
          status = GALAHAD_error_factorization ; RETURN
        END IF

!  update z_fx <- z_fx - SLS_SOL_fx

        DO i = 1, n
          IF ( X_inorder( i ) < 0 )                                            &
            SOL( z - X_inorder( i ) ) = SOL( z - X_inorder( i ) ) - SLS_SOL( i )
        END DO

!  ... or directly reset z_fx <- z_fx - H_fx x_fx - H_od^T x_fr ...

      ELSE IF ( is_h ) THEN
        DO i = 1, n
          DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
            j = H_col( l )
            IF ( X_inorder( i ) < 0 .AND. X_inorder( j ) < 0 ) THEN
              SOL( z - X_inorder( i ) ) =                                      &
                SOL( z - X_inorder( i ) ) - H_val( l ) * SOL( j )
              IF ( i /= j ) SOL( z - X_inorder( j ) ) =                        &
                SOL( z - X_inorder( j ) ) - H_val( l ) * SOL( i )
            ELSE IF ( X_inorder( i ) > 0 .AND. X_inorder( j ) <= 0 ) THEN
              SOL( z - X_inorder( j ) ) =                                      &
                SOL( z - X_inorder( j ) ) - H_val( l ) * SOL( i )
            ELSE IF ( X_inorder( i ) <= 0 .AND. X_inorder( j ) > 0 ) THEN
              SOL( z - X_inorder( i ) ) =                                      &
                SOL( z - X_inorder( i ) ) - H_val( l ) * SOL( j )
            END IF
          END DO
        END DO
      END IF

!  ... and reset z_fx <- z_fx - A_fxxb^T y_fxb

      DO i = 1, m
        IF ( C_inorder( i ) > 0 ) THEN
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            j = A_col( l )
            IF ( X_inorder( j ) < 0 ) SOL( z - X_inorder( j ) ) =              &
              SOL( z - X_inorder( j ) ) - A_val( l ) * SOL( y + C_inorder( i ) )
          END DO
        END IF
      END DO

      RETURN

!  End of subroutine CRO_block_solve

      END SUBROUTINE CRO_block_solve

!-*-*-*-*-*-   C R O _ C H E C K _ S T A T U S   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE CRO_check_status( n, m, m_equal, A_val, A_col, A_ptr, G, C_l, &
                                   C_u, X_l, X_u, C, X, Y, Z, C_stat, X_stat,  &
                                   control, inform, RES_D, RES_P, prefix,      &
                                   H_val, H_col, H_ptr, H_lm )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Given values x, y and z, check that they satisfy

!  H x + g - A^T y - z = 0, c_l <= A x <= c_u, x_l <= x <= x_u
!    { >= 0 if A x = c_l             { >= 0 if x = x_l
!  y {  = 0 if c_l < A x < c_u and z {  = 0 if x_l < x < x_u
!    { <= 0 if A x = c_u             { <= 0 if x = x_u

! and that the corresponding status arrays X_stat and C_stat are consistent

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, m_equal
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C, Y
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, Z
      INTEGER, INTENT( IN ), DIMENSION( m ) :: C_stat
      INTEGER, INTENT( IN ), DIMENSION( n ) :: X_stat
      TYPE ( CRO_control_type ), INTENT( IN ) :: control
      TYPE ( CRO_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( OUT), DIMENSION( m ) :: RES_p
      REAL ( KIND = wp ), INTENT( OUT), DIMENSION( n ) :: RES_d
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_col
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_val
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( INOUT ) :: H_lm

!  Local variables

      INTEGER :: i, l, nviol8_xs, nviol8_cs, error, out
      INTEGER :: nviol8_p, nviol8_d, nviol8_x, nviol8_y, nviol8_z, nviol8_c
      LOGICAL :: printd, printe
      REAL ( KIND = wp ) :: tol
      REAL ( KIND = wp ) :: viol8_p, viol8_d, viol8_x, viol8_y, viol8_z, viol8_c

!WRITE(6,"( ' X ', /, (5ES12.4 ) )" ) X
!WRITE(6,"( ' Y ', /, (5ES12.4 ) )" ) Y
!WRITE(6,"( ' Z ', /, (5ES12.4 ) )" ) Z
!WRITE(6,"( ' C ', /, (5ES12.4 ) )" ) C

      out = control%out ; error = control%error
      printd = out > 0 .AND. control%print_level >= 5
      printe = error > 0 .AND. control%print_level >= 1

!  check for obvious errors

      IF ( n <= 0 .OR. m_equal < 0 .OR. m_equal > m ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( printe ) THEN
          IF ( n <= 0 ) WRITE( error, "( A, ' Error - n = ', I0,               &
         &  ' <= 0 ' )" ) prefix, n
          IF ( m_equal < 0 ) WRITE( error, "( A, ' Error - m_equal = ', I0,    &
         &  ' < 0 ' )" ) prefix, m_equal
          IF ( m_equal > m ) WRITE( error, "( A, ' Error - m = ', I0,          &
         &  ' < m_equal = ', I0 )" ) prefix, m, m_equal
        END IF
        RETURN
      END IF

      IF ( m_equal > 0 ) THEN
        l = COUNT( C_l( : m_equal ) /= C_u( : m_equal ) )
        IF ( l > 0 ) THEN
          inform%status = GALAHAD_error_restrictions
          IF ( printe ) THEN
            IF ( l > 1 ) THEN
              WRITE( error, "( A, ' Error - ', I0, ' constraints before',      &
             &   'm_equal = ', I0, ' are inequalities' )" ) prefix, l, m_equal
            ELSE
              WRITE( error, "( A, ' Error - ', I0, ' constraint before',      &
             &   'm_equal = ', I0, ' is an inequality' )" ) prefix, l, m_equal
            END IF
          END IF
          RETURN
        END IF
      END IF

      IF ( m_equal < m ) THEN
        l = COUNT( C_l( m_equal + 1 : m ) == C_u( m_equal + 1 : m ) )
        IF ( l > 0 ) THEN
          inform%status = GALAHAD_error_restrictions
          IF ( printe ) THEN
            IF ( l > 1 ) THEN
              WRITE( error, "( A, ' Error - ', I0, ' constraints beyond ',     &
             &   'm_equal = ', I0, ' are equations' )" ) prefix, l, m_equal
            ELSE
              WRITE( error, "( A, ' Error - ', I0, ' constraint beyond ',      &
             &   'm_equal = ', I0, ' is an equation' )" ) prefix, l, m_equal
            END IF
          END IF
          RETURN
        END IF
      END IF

!  store g + H x - A^T y - z in res_d and c - Ax in res_p

      RES_p = C ; RES_d = G
      CALL CRO_KKT_residual( n, m, A_val, A_col, A_ptr, X, Y, Z, RES_p, RES_d, &
                             inform%status, H_val = H_val, H_col = H_col,      &
                             H_ptr = H_ptr, H_lm = H_lm )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!     write(6,*) '     x_l          x            x_u         KKTx'
!     do i = 1, n
!     write(6,"(4ES12.4)" )                                                    &
!       X_l( i ), X( i ), X_u( i ), RES_d( i )
!     end do

!     write(6,*) '     c_l          c            c_u         KKTy'
!     do i = 1, m
!     write(6,"(4ES12.4)" )                                                    &
!       C_l( i ), C( i ), C_u( i ), RES_p( i )
!     end do

!  Check to see if we are optimal

      nviol8_p = 0 ; nviol8_d = 0 ; nviol8_x = 0
      nviol8_y = 0 ; nviol8_z = 0 ; nviol8_c = 0
      nviol8_xs = 0 ; nviol8_cs = 0
      viol8_p = zero ; viol8_d = zero ; viol8_x = zero
      viol8_y = zero ; viol8_z = zero ; viol8_c = zero

      tol = control%feasibility_tolerance
      IF ( printd ) WRITE( out, "( A, ' feasibility tolerance = ', ES8.2 )" )  &
        prefix, tol

!  check primal lower bound feasibility

      DO i = 1, n
        IF ( X( i ) < X_l( i ) - tol ) THEN
          nviol8_x = nviol8_x + 1
          viol8_x = MAX( viol8_x, ABS( X( i ) - X_l( i ) ) )
          IF ( printd ) WRITE( out, "( A, ' variable(', I0, ') = ', ES12.4,    &
         &  ' smaller than its lower bound = ', ES12.4 )" )                    &
            prefix, i, X( i ), X_l( i )

!  check dual lower bound feasibility

        ELSE IF ( X( i ) < X_l( i ) + tol ) THEN
          IF ( Z( i ) < - tol ) THEN
            nviol8_z = nviol8_z + 1
            viol8_z = MAX( viol8_z, ABS( Z( i ) ) )
            IF ( printd ) WRITE( out, "( A, ' dual variable(', I0, ') = ',     &
           &  ES12.4, ' on lower bound should be positive' )" )                &
             prefix, i, Z( i )
          END IF
        END IF

!  check primal upper bound feasibility

        IF ( X( i ) > X_u( i ) + tol ) THEN
          nviol8_x = nviol8_x + 1
          viol8_x = MAX( viol8_x, ABS( X( i ) - X_u( i ) ) )
          IF ( printd ) WRITE( out, "( A, ' variable(', I0, ') = ', ES12.4,    &
         &  ' larger than its upper bound = ', ES12.4 )" )                     &
            prefix, i, X( i ), X_u( i )

!  check dual upper bound feasibility

        ELSE IF ( X( i ) > X_u( i ) - tol ) THEN
          IF ( Z( i ) > tol ) THEN
            nviol8_z = nviol8_z + 1
            viol8_z = MAX( viol8_z, ABS( Z( i ) ) )
            IF ( printd ) WRITE( out, "( A, ' dual variable(', I0, ') = ',     &
           &  ES12.4, ' on upper bound should be negative' )" )                &
             prefix, i, Z( i )
          END IF
        END IF

!  check dual feasibility

        IF ( ABS( RES_d( i ) ) > tol ) THEN
          nviol8_d = nviol8_d + 1
          viol8_d = MAX( viol8_d, ABS( RES_d( i ) ) )
          IF ( printd ) WRITE( out, "( A, ' dual infeasibility (', I0, ') = ', &
         &  ES12.4, ' is nonzero' )" ) prefix, i, RES_d( i )
        END IF

!  check status agrees with primal bounds

        IF ( X_l( i ) == X_u( i ) ) THEN
          IF ( X_stat( i ) == 0 ) THEN
            nviol8_xs = nviol8_xs + 1
            IF ( printd ) WRITE( out, "( A, ' fixed variable(', I0, ') has',   &
           &  ' X_stat = ', I0 )" ) prefix, i, X_stat( i )
          END IF
        ELSE IF ( X( i ) <= X_l( i ) + tol ) THEN
          IF ( X_stat( i ) >= 0 ) THEN
            nviol8_xs = nviol8_xs + 1
            IF ( printd ) WRITE( out, "( A, ' variable(', I0, ') =', ES9.2,    &
           &  ' = lower bound', ES9.2, ' but X_stat = ', I0 )" )               &
              prefix, i, X( i ), X_l( i ), X_stat( i )
          END IF
        ELSE IF ( X( i ) >= X_u( i ) - tol ) THEN
          IF ( X_stat( i ) <= 0 ) THEN
            nviol8_xs = nviol8_xs + 1
            IF ( printd ) WRITE( out, "( A, ' variable(', I0, ') =', ES9.2,    &
           &  ' = upper bound', ES9.2, ' but X_stat = ', I0 )" )               &
              prefix, i, X( i ), X_u( i ), X_stat( i )
          END IF
        ELSE
          IF ( X_stat( i ) /= 0 ) THEN
            nviol8_xs = nviol8_xs + 1
            IF ( printd ) WRITE( out, "( A, ' variable(', I0, ') =', ES9.2,    &
           &  ' in (', ES9.2, ',', ES9.2, ') but X_stat = ', I0 )" )           &
              prefix, i, X( i ), X_l( i ), X_u( i ), X_stat( i )
          END IF
        END IF
      END DO

!  check constraint lower bound feasibility

      DO i = 1, m
        IF ( C( i ) < C_l( i ) - tol ) THEN
          nviol8_c = nviol8_c + 1
          viol8_c = MAX( viol8_c, ABS( C( i ) - C_l( i ) ) )
          IF ( printd ) WRITE( out, "( A, ' constraint(', I0, ') = ', ES12.4,  &
         &  ' smaller than its lower bound = ', ES12.4 )" )                    &
            prefix, i, C( i ), C_l( i )

!  check Lagrange multiplier feasibility

        ELSE IF ( C( i ) < C_l( i ) + tol ) THEN
          IF ( Y( i ) < - tol .AND. i > m_equal ) THEN
            nviol8_y = nviol8_y + 1
            viol8_y = MAX( viol8_y, ABS( Y( i ) ) )
            IF ( printd ) WRITE( out, "( A, ' Lagrange multiplier(', I0,       &
           &  ') = ',  ES12.4, ' on lower bound should be positive' )" )       &
             prefix, i, Y( i )
          END IF
        END IF

!  check constraint upper bound feasibility

        IF ( C( i ) > C_u( i ) + tol ) THEN
          nviol8_c = nviol8_c + 1
          viol8_c = MAX( viol8_c, ABS( C( i ) - C_u( i ) ) )
          IF ( printd ) WRITE( out, "( A, ' constraint(', I0, ') = ', ES12.4,  &
         &  ' larger than its upper bound = ', ES12.4 )" )                     &
            prefix, i, C( i ), C_u( i )

!  check Lagrange multiplier feasibility

        ELSE IF ( C( i ) > C_u( i ) - tol ) THEN
          IF ( Y( i ) > tol .AND. i > m_equal ) THEN
            nviol8_y = nviol8_y + 1
            viol8_y = MAX( viol8_y, ABS( Y( i ) ) )
            IF ( printd ) WRITE( out, "( A, ' Lagrange multiplier(', I0,       &
           &  ') = ',  ES12.4, ' on upper bound should be negative' )" )       &
             prefix, i, Y( i )
          END IF
        END IF

!  check primal constraint feasibility

        IF ( ABS( RES_p( i ) ) > tol ) THEN
          nviol8_p = nviol8_p + 1
          viol8_p = MAX( viol8_p, ABS( RES_p( i ) ) )
          IF ( printd ) WRITE( out, "( A, ' primal infeasibility (', I0,       &
         &  ') = ',  ES12.4, ' is nonzero' )" ) prefix, i, RES_p( i )
        END IF

!  check status agrees with constraint bounds

        IF ( C_l( i ) == C_u( i ) ) THEN
          IF ( C_stat( i ) == 0 ) THEN
            nviol8_cs = nviol8_cs + 1
            IF ( printd ) WRITE( out, "( A, ' equality constraint(', I0, ')',  &
           &  ' has C_stat = ', I0 )" ) prefix, i, C_stat( i )
          END IF
        ELSE IF ( C( i ) <= C_l( i ) + tol ) THEN
          IF ( C_stat( i ) >= 0 ) THEN
            nviol8_cs = nviol8_cs + 1
            IF ( printd ) WRITE( out, "( A, ' constraint(', I0, ') =', ES9.2,  &
           &  ' = lower bound', ES9.2, ' but C_stat = ', I0 )" )               &
              prefix, i, C( i ), C_l( i ), C_stat( i )
          END IF
        ELSE IF ( C( i ) >= C_u( i ) - tol ) THEN
          IF ( C_stat( i ) <= 0 ) THEN
            nviol8_cs = nviol8_cs + 1
            IF ( printd ) WRITE( out, "( A, ' constraint(', I0, ') =', ES9.2,  &
           &  ' = upper bound', ES9.2, ' but C_stat = ', I0 )" )               &
              prefix, i, C( i ), C_U( i ), C_stat( i )
          END IF
        ELSE
          IF ( C_stat( i ) /= 0 ) THEN
            nviol8_cs = nviol8_cs + 1
            IF ( printd ) WRITE( out, "( A, ' constraint(', I0, ') =', ES9.2,  &
           &  ' in (', ES9.2, ',', ES9.2, ') but C_stat = ', I0 )" )           &
              prefix, i, C( i ), C_l( i ), C_u( i ), C_stat( i )
          END IF
        END IF
      END DO

!  record any anomolies

      IF ( nviol8_p + nviol8_d + nviol8_x + nviol8_y + nviol8_z + nviol8_c +   &
           nviol8_xs + nviol8_cs > 0 ) THEN
        IF ( nviol8_x + nviol8_c > 0 ) THEN
          inform%status = GALAHAD_error_bad_bounds
        ELSE IF ( nviol8_p > 0 ) THEN
          inform%status = GALAHAD_error_primal_infeasible
        ELSE IF ( nviol8_d + nviol8_y + nviol8_z > 0 ) THEN
          inform%status = GALAHAD_error_dual_infeasible
        ELSE
          inform%status = GALAHAD_error_restrictions
        END IF
        IF ( printe ) THEN
          WRITE( error, "( A, ' Error - KKT violations encountered:' )" ) prefix
          IF ( nviol8_p > 0 ) WRITE( error, "( A, '  #, max primal KKT',       &
         &    ' violations = ', I0, ', ', ES8.2 )" ) prefix, nviol8_p, viol8_p
          IF ( nviol8_d > 0 ) WRITE( error, "( A, '  #, max dual KKT',         &
         &    ' violations = ', I0, ', ', ES8.2 )" ) prefix, nviol8_d, viol8_d
          IF ( nviol8_x > 0 ) WRITE( error, "( A, '  #, max x KKT violations', &
         &    ' = ', I0, ', ', ES8.2 )" ) prefix, nviol8_x, viol8_x
          IF ( nviol8_y > 0 ) WRITE( error, "( A, '  #, max y KKT violations', &
         &    ' = ', I0, ', ', ES8.2 )" ) prefix, nviol8_y, viol8_y
          IF ( nviol8_z > 0 ) WRITE( error, "( A, '  #, max z KKT violations', &
         &    ' = ', I0, ', ', ES8.2 )" ) prefix, nviol8_z, viol8_z
          IF ( nviol8_c > 0 ) WRITE( error, "( A, '  #, max c KKT violations', &
         &    ' = ', I0, ', ', ES8.2 )" ) prefix, nviol8_c, viol8_c
          IF ( nviol8_xs > 0 ) WRITE( error, "( A, '  # x_stat violations',    &
         &    ' = ', I0 )" ) prefix, nviol8_xs
          IF ( nviol8_cs > 0 ) WRITE( error, "( A, '  # c_stat violations',    &
         &    ' = ', I0 )" ) prefix, nviol8_cs
        END IF
      END IF

      RETURN

!  End of subroutine CRO_check_status

      END SUBROUTINE CRO_check_status

!-*-*-*-*-*-   C R O _ C H E C K _ S T A T U S   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE CRO_KKT_residual( n, m, A_val, A_col, A_ptr, X, Y, Z, Res_p,  &
                                   RES_d, status, H_val, H_col, H_ptr, H_lm )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

! Given values x, y, z, res_x and res_y compute the primal-dual residuals

!   res_p <- res_p - A and res_d <- res_d + H x - A^T y - z

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: status
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, Z
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: RES_p
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: RES_d
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_col
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( * ) :: H_val
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( INOUT ) :: H_lm

!  Local variables

      INTEGER :: i, j, l

      status = GALAHAD_ok

!  set res_p <- res_p - A and res_d <- res_d - A^T y

      DO i = 1, m
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          j = A_col( l )
          RES_p( i ) = RES_p( i ) - A_val( l ) * X( j )
          RES_d( j ) = RES_d( j ) - A_val( l ) * Y( i )
        END DO
      END DO

!  set res_d <- res_d - z + H x

      IF ( PRESENT( H_lm ) ) THEN
        RES_d = RES_d - Z
        CALL LMS_apply_lbfgs( X, H_lm, status, ADD_TO_RESULT = RES_d )
        IF ( status /= 0 ) status = GALAHAD_error_factorization
      ELSE IF ( PRESENT( H_val ) .AND. PRESENT( H_col ) .AND.                  &
                PRESENT( H_ptr ) ) THEN
        DO i = 1, n
          RES_d( i ) = RES_d( i ) - Z( i )
          DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
            j = H_col( l )
            RES_d( i ) = RES_d( i ) + H_val( l ) * X( j )
            IF ( i /= j ) RES_d( j ) = RES_d( j ) + H_val( l ) * X( i )
          END DO
        END DO
      ELSE
        RES_d = RES_d - Z
      END IF

      RETURN

!  End of subroutine CRO_KKT_residual

      END SUBROUTINE CRO_KKT_residual

!  End of module CRO

   END MODULE GALAHAD_CRO_double
