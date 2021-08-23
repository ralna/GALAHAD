! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ E Q P   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started March 25th 2004
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_EQP_double

!      ---------------------------------------------------
!     |                                                   |
!     | Solve the equaity-constrained quadratic program   |
!     |                                                   |
!     |   minimize     1/2 x(T) H x + g(T) x + f          |
!     |   or   1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f  |
!     |   subject to     A x + c = 0                      |
!     |   and            ||x|| <= Delta                   |
!     |                                                   |
!     | using a projected preconditined CG method         |
!     |                                                   |
!      ---------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING, ONLY: STRING_pleural, STRING_verb_pleural
      USE GALAHAD_SPACE_double
      USE GALAHAD_QPD_double, ONLY : QPD_SIF, EQP_data_type => QPD_data_type
      USE GALAHAD_QPT_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_GLTR_double
      USE GALAHAD_FDC_double
      USE GALAHAD_SPECFILE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: EQP_initialize, EQP_read_specfile, EQP_solve, EQP_terminate,   &
                EQP_solve_main, EQP_resolve, EQP_resolve_main, EQP_data_type,  &
                QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: relative_pivot_default = 0.01_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: EQP_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   the factorization to be used. Possible values are

!      0  automatic
!      1  Schur-complement factorization
!      2  augmented-system factorization                              (OBSOLETE)

        INTEGER :: factorization = 0

!   the maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization                          (OBSOLETE)

        INTEGER :: max_col = 35

!   an initial guess as to the integer workspace required by SBLS     (OBSOLETE)

        INTEGER :: indmin = 1000

!   an initial guess as to the real workspace required by SBLS        (OBSOLETE)

        INTEGER :: valmin = 1000

!   an initial guess as to the workspace required by ULS              (OBSOLETE)

        INTEGER :: len_ulsmin = 1000

!   the maximum number of iterative refinements allowed               (OBSOLETE)

        INTEGER :: itref_max = 1

!   the maximum number of CG iterations allowed. If cg_maxit < 0,
!     this number will be reset to the dimension of the system + 1
!
        INTEGER :: cg_maxit = 200

!   the preconditioner to be used for the CG is defined by precon.
!    Possible values are

!      0  automatic
!      1  no preconditioner, i.e, the identity within full factorization
!      2  full factorization
!      3  band within full factorization
!      4  diagonal using the barrier terms within full factorization  (OBSOLETE)
!      5  optionally supplied diagonal, G = D

        INTEGER :: preconditioner = 0

!   the semi-bandwidth of a band preconditioner, if appropriate       (OBSOLETE)

        INTEGER :: semi_bandwidth = 5

!   how much has A changed since last problem solved:
!    0 = not changed, 1 = values changed, 2 = structure changed

        INTEGER :: new_a = 2

!   how much has H changed since last problem solved:
!    0 = not changed, 1 = values changed, 2 = structure changed

        INTEGER :: new_h = 2

!   specifies the unit number to write generated SIF file describing the
!    current problem

        INTEGER :: sif_file_device = 49

!   the threshold pivot used by the matrix factorization.
!    See the documentation for SBLS for details                       (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol = point01

!   the threshold pivot used by the matrix factorization when finding the basis.
!    See the documentation for ULS for details                        (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol_for_basis = half

!   any pivots smaller than zero_pivot in absolute value will be regarded to be
!    zero when attempting to detect linearly dependent constraints    (OBSOLETE)

        REAL ( KIND = wp ) :: zero_pivot = epsmch

!   the computed solution which gives at least inner_fraction_opt times the
!    optimal value will be found                                      (OBSOLETE)

        REAL ( KIND = wp ) :: inner_fraction_opt = point1

!   an upper bound on the permitted step (-ve will be reset to an appropriate
!    large value by eqp_solve)

        REAL ( KIND = wp ) :: radius = - one

!   diagonal preconditioners will have diagonals no smaller than min_diagonal
!                                                                     (OBSOLETE)

        REAL ( KIND = wp ) :: min_diagonal = 0.00001_wp

!   if the constraints are believed to be rank defficient and the residual
!     at a "typical" feasible point is larger than
!      max( max_infeasibility_relative * norm A, max_infeasibility_absolute )
!     the problem will be marked as infeasible

        REAL ( KIND = wp ) :: max_infeasibility_relative = epsmch
        REAL ( KIND = wp ) :: max_infeasibility_absolute = epsmch

!   the computed solution is considered as an acceptable approximation to the
!    minimizer of the problem if the gradient of the objective in the
!    preconditioning(inverse) norm is less than
!     max( inner_stop_relative * initial preconditioning(inverse)
!                                 gradient norm, inner_stop_absolute )

        REAL ( KIND = wp ) :: inner_stop_relative = point01
        REAL ( KIND = wp ) :: inner_stop_absolute = epsmch
        REAL ( KIND = wp ) :: inner_stop_inter = point01

!   if %find_basis_by_transpose is true, implicit factorization preconditioners
!    will be based on a basis of A found by examining A's transpose   (OBSOLETE)

       LOGICAL :: find_basis_by_transpose = .TRUE.

!   if %remove_dependencies is true, the equality constraints will be
!    preprocessed to remove any linear dependencies
!
        LOGICAL :: remove_dependencies = .TRUE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "EQPPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for GLTR

        TYPE ( GLTR_control_type ) :: GLTR_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: EQP_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ) :: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0
        REAL ( KIND = wp ) :: solve_inter = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent detecting linear dependencies

        REAL ( KIND = wp ) :: clock_find_dependent = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = wp ) :: clock_solve = 0.0
      END TYPE

      TYPE, PUBLIC :: EQP_inform_type

!  return status. See EQP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of conjugate gradient iterations required

        INTEGER :: cg_iter = - 1
        INTEGER :: cg_iter_inter = - 1

!  the total integer workspace required for the factorization

        INTEGER :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER :: factorization_real = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by QPB_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  timings (see above)

        TYPE ( EQP_time_type ) :: time

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  return information from GLTR

        TYPE ( GLTR_inform_type ) :: GLTR_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   E Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE EQP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for EQP. This routine should be called before
!  EQP_solve
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

      TYPE ( EQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( EQP_control_type ), INTENT( OUT ) :: control
      TYPE ( EQP_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Real parameters

      control%zero_pivot = epsmch ** 0.75
      control%inner_stop_absolute = SQRT( epsmch )
      control%max_infeasibility_relative = epsmch ** 0.75
      control%max_infeasibility_absolute = epsmch ** 0.75

!  Initalize FDC components

      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform )
      control%FDC_control%prefix = '" - FDC:"                    '

!  Set GLTR control parameters

      CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,              &
                            inform%GLTR_inform )
      control%GLTR_control%unitm = .FALSE.
      control%GLTR_control%itmax = control%cg_maxit
      control%GLTR_control%prefix = '" - GLTR:"                    '

!  Initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%new_a = control%new_a
      control%SBLS_control%new_h = control%new_h
      control%SBLS_control%prefix = '" - SBLS:"                    '

!  initialise private data

      data%new_c = .TRUE.

      RETURN

!  End of EQP_initialize

      END SUBROUTINE EQP_initialize

!-*-*-*-*-   E Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE EQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by EQP_initialize could (roughly)
!  have been set as:

!  BEGIN EQP SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   initial-workspace-for-unsymmetric-solver        1000
!   initial-integer-workspace                       1000
!   initial-real-workspace                          1000
!   preconditioner-used                             0
!   semi-bandwidth-for-band-preconditioner          5
!   factorization-used                              0
!   maximum-column-nonzeros-in-schur-complement     35
!   maximum-refinements                             1
!   maximum-number-of-cg-iterations                 200
!   sif-file-device                                 49
!   truat-region-radius                             1.0D+19
!   minimum-diagonal                                1.0D-5
!   pivot-tolerance-used                            1.0D-12
!   pivot-tolerance-used-for-basis                  0.5
!   zero-pivot-tolerance                            1.0D-12
!   inner-iteration-fraction-optimality-required    0.1
!   inner-iteration-relative-accuracy-required      0.01
!   inner-iteration-absolute-accuracy-required      1.0E-8
!   inner-iteration-intermediate-accuracy-required  1.0D-2
!   max-relative-infeasibility-allowed              1.0E-12
!   max-absolute-infeasibility-allowed              1.0E-12
!   find-basis-by-transpose                         T
!   remove-linear-dependencies                      T
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   generate-sif-file                               F
!   sif-file-name                                   EQPPROB.SIF
!   output-line-prefix                              ""
!  END EQP SPECIFICATIONS

!  Dummy arguments

      TYPE ( EQP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 37
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'EQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level'
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  4 )%keyword = 'initial-workspace-for-unsymmetric-solver'
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 13 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 14 )%keyword = 'preconditioner-used'
      spec( 15 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 22 )%keyword = 'sif-file-device'

!  Real key-words

      spec( 19 )%keyword = 'trust-region-radius'
      spec( 25 )%keyword = 'minimum-diagonal'
      spec( 23 )%keyword = 'max-relative-infeasibility-allowed'
      spec( 24 )%keyword = 'max-absolute-infeasibility-allowed'
      spec( 26 )%keyword = 'pivot-tolerance-used'
      spec( 27 )%keyword = 'pivot-tolerance-used-for-basis'
      spec( 28 )%keyword = 'zero-pivot-tolerance'
      spec( 29 )%keyword = 'inner-iteration-intermediate-accuracy-required'
      spec( 30 )%keyword = 'inner-iteration-fraction-optimality-required'
      spec( 31 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 32 )%keyword = 'inner-iteration-absolute-accuracy-required'

!  Logical key-words

      spec( 33 )%keyword = 'find-basis-by-transpose'
      spec( 34 )%keyword = 'remove-linear-dependencies'
      spec( 16 )%keyword = 'space-critical'
      spec( 35 )%keyword = 'deallocate-error-fatal'
      spec( 21 )%keyword = 'generate-sif-file'

!  Character key-words

      spec( 36 )%keyword = 'sif-file-name'
!     spec( 37 )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( 1 ), control%error,                    &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 2 ), control%out,                      &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 3 ), control%print_level,              &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 7 ), control%factorization,            &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 8 ), control%max_col,                  &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 4 ), control%len_ulsmin,               &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 9 ), control%indmin,                   &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 10 ), control%valmin,                  &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 11 ), control%itref_max,               &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 13 ), control%cg_maxit,                &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 14 ), control%preconditioner,          &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 15 ), control%semi_bandwidth,          &
                                    control%error )
      CALL SPECFILE_assign_value( spec( 22 ), control%sif_file_device,         &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( 25 ), control%min_diagonal,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 19 ), control%radius,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 23 ),                                  &
                                  control%max_infeasibility_relative,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 24 ),                                  &
                                  control%max_infeasibility_absolute,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 26 ), control%pivot_tol,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 27 ),                                  &
                                  control%pivot_tol_for_basis,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 28 ), control%zero_pivot,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 29 ), control%inner_stop_inter,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 30 ), control%inner_fraction_opt,      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 31 ), control%inner_stop_relative,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 32 ), control%inner_stop_absolute,     &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( 33 ),                                  &
                                  control%find_basis_by_transpose,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 34 ), control%remove_dependencies,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 16 ),                                  &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 35 ),                                  &
                                  control%deallocate_error_fatal,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 21 ), control%generate_sif_file,       &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_string( spec( 36 ), control%sif_file_name,          &
                                  control%error )
!     CALL SPECFILE_assign_string( spec( 37 ), control%prefix,                 &
!                                  control%error )

!  Reset GLTR and SBLS data for this package

      control%GLTR_control%stop_relative = control%inner_stop_relative
      control%GLTR_control%stop_absolute = control%inner_stop_absolute
      control%GLTR_control%itmax = control%cg_maxit

      control%SBLS_control%new_a = control%new_a
      control%SBLS_control%new_h = control%new_h

!  Read the controls for the preconditioner and iterative solver

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FDC_read_specfile( control%FDC_control, device,                   &
                                 alt_specname = TRIM( alt_specname ) // '-FDC')
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
        CALL GLTR_read_specfile( control%GLTR_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-GLTR')
      ELSE
        CALL FDC_read_specfile( control%FDC_control, device )
        CALL SBLS_read_specfile( control%SBLS_control, device )
        CALL GLTR_read_specfile( control%GLTR_control, device )
      END IF

      RETURN

      END SUBROUTINE EQP_read_specfile

!-*-*-*-*-*-*-*-*-   E Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE EQP_solve( prob, data, control, inform, D )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     or    1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!     subject to    A x = c
!
!  where x is a vector of n components ( x_1, .... , x_n ), f is a constant
!  g, w and x^0 are an n-vectors, H is a symmetric matrix and A is an m by n
!  matrix, using a projected preconditioned conjugate-gradient method.
!  The subroutine is particularly appropriate when A and H are sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to EQP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the Hessian H will be used
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H. Four storage formats are permitted:
!
!    i) sparse, coordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   %target_kind is an INTEGER variable that defines possible special
!     targets X0. Possible values are
!
!     0  X0 will be a vector of zeros.
!        %X0 (see below) need not be set
!
!     1  X0 will be a vector of ones.
!        %X0 (see below) need not be set
!
!     any other value - the values of X0 will be those given by %X0 (see below)
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is larger
!    that 0 and %target_kind /= 0,1. If this is so, it must be of length at
!    least %n, and contain the targets X^0 for the objective function.
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!    On exit, G will most likely have been reordered.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
!
!   %A is a structure of type SMT_type used to hold the matrix A.
!    Three storage formats are permitted:
!
!    i) sparse, coordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %X is a REAL array of length %n, which must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C is a REAL array of length %m, which must be set by the user
!    to the values of the array c of constant terms of the constraints
!    Ax + c = 0
!
!  data is a structure of type EQP_data_type which holds private internal data
!
!  control is a structure of type EQP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to EQP_initialize. See EQP_initialize
!   for details
!
!  inform is a structure of type EQP_inform_type that provides
!    information on exit from EQP_solve. The component status
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
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!          prob%H%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', 'DIAGONAL'}
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
!   -15 the computed precondition is insufficient. Try another
!
!   -16 the residuals are large; the factorization may be unsatisfactory
!
!  On exit from EQP_solve, other components of inform give the
!  following:
!
!     alloc_status = the status of the last attempted allocation/deallocation
!     bad_alloc = the name of the last array for which (de)allocation failed
!     cg_iter = the total number of conjugate gradient iterations required.
!     factorization_integer = the total integer workspace required for the
!       factorization.
!     factorization_real = the total real workspace required for the
!       factorization.
!     obj = the value of the objective function at the best estimate of the
!       solution determined by EQP_solve.
!     time%total = the total time spent in the package.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!     SBLS_inform = inform components from SBLS
!     GLTR_inform = inform components from GLTR
!
!   D is an optional REAL array of length prob%n, that if present must be set
!    by the user to the (nonzero) values of the diagonal matrix G = D
!    needed when control%preconditioner = 5

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( EQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( EQP_control_type ), INTENT( IN ) :: control
      TYPE ( EQP_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( prob%n ) :: D

!  Local variables

      INTEGER :: i, j, l
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now, f
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' -- entering EQP_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      ten ** 19, .TRUE., no_bounds = .TRUE.,                   &
                      just_equality = .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )
      inform%status = GALAHAD_ok
      inform%obj = zero

!  check to see if the problem is trivial

      IF ( prob%n == 0 ) THEN
        IF ( prob%m == 0 ) THEN
          RETURN
        ELSE IF ( prob%m > 0 ) THEN
          IF ( MAXVAL( ABS( prob%C( : prob%m ) ) ) <=                          &
                 control%max_infeasibility_absolute   ) THEN
            prob%Y = zero
            RETURN
          END IF
        END IF
      END IF

!  check for trivially-faulty input data

      IF ( prob%n <= 0 .OR. prob%m < 0 .OR.                                    &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
            "( ' ', /, A, ' **  Error return ', I0,' from EQP ' )" )           &
            prefix, inform%status
        RETURN
      END IF

      IF ( prob%Hessian_kind < 0 ) THEN
        IF ( .NOT. QPT_keyword_H( prob%H%type ) ) THEN
          inform%status = GALAHAD_error_restrictions
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error,                                              &
              "( ' ', /, A, ' **  Error return ', I0,' from EQP ' )" )         &
              prefix, inform%status
          RETURN
        END IF
      END IF

!  If required, write out the problem

!     IF ( control%out > 0 .AND. control%print_level >= 10 ) THEN
      IF ( control%out > 0 .AND. control%print_level >= 1 ) THEN
        WRITE( control%out, "( /, A, ' n = ', I0, ', m = ', I0, ', f =',       &
       &                       ES24.16 )" ) prefix, prob%n, prob%m, prob%f
        IF ( prob%gradient_kind == 0 ) THEN
          WRITE( control%out, "( A, ' G = zeros' )" ) prefix
        ELSE IF ( prob%gradient_kind == 1 ) THEN
          WRITE( control%out, "( A, ' G = ones' )" ) prefix
        ELSE
          WRITE( control%out, "( A, ' G =', /, ( 5X, 3ES24.16 ) )" )           &
            prefix, prob%G( : prob%n )
        END IF
        IF ( prob%Hessian_kind == 0 ) THEN
          WRITE( control%out, "( A, ' W = zeros' )" ) prefix
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          WRITE( control%out, "( A, ' W = ones ' )" ) prefix
          IF ( prob%target_kind == 0 ) THEN
            WRITE( control%out, "( A, ' X0 = zeros' )" ) prefix
          ELSE IF ( prob%target_kind == 1 ) THEN
            WRITE( control%out, "( A, ' X0 = ones ' )" ) prefix
          ELSE
            WRITE( control%out, "( A, ' X0 =', /, ( 3ES24.16 ) )" )            &
             prefix, prob%X0( : prob%n )
          END IF
        ELSE IF ( prob%Hessian_kind == 2 ) THEN
          WRITE( control%out, "( A, ' W =', /, ( 5X, 3ES24.16 ) )" )           &
            prefix, prob%WEIGHT( : prob%n )
          IF ( prob%target_kind == 0 ) THEN
            WRITE( control%out, "( A, ' X0 = zeros' )" ) prefix
          ELSE IF ( prob%target_kind == 1 ) THEN
            WRITE( control%out, "( A, ' X0 = ones ' )" ) prefix
          ELSE
            WRITE( control%out, "( A, ' X0 =', /, ( 6X, 3ES24.16 ) )" )        &
              prefix, prob%X0( : prob%n )
          END IF
        ELSE
          IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            WRITE( control%out, "( A, ' H (diagonal) = ', /, ( 3ES24.16 ) )" ) &
              prefix, prob%H%val( : prob%n )
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            WRITE( control%out, "( A, ' H (dense) = ', /, ( 3ES24.16 ) )" )    &
              prefix, prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            WRITE( control%out, "( A, ' H (row-wise) = ' )" ) prefix
            DO i = 1, prob%n
              WRITE( control%out, "( ( 2X, 2( 2I8, ES24.16 ) ) )" )            &
                ( i, prob%H%col( j ), prob%H%val( j ),                         &
                  j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
            END DO
          ELSE
            WRITE( control%out, "( A, ' H (coordinate) = ' )" ) prefix
            WRITE( control%out, "( ( 2X, 2( 2I8, ES24.16 ) ) )" )              &
              ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ),             &
              i = 1, prob%H%ne)
          END IF
        END IF
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( A, ' A (dense) = ', /, ( 10X, 3ES24.16 ) )" ) &
            prefix, prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( A, ' A (row-wise) = ' )" ) prefix
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2X, 2( 2I8, ES24.16 ) ) )" )              &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( A, ' A (coordinate) = ' )" ) prefix
          WRITE( control%out, "( ( 2X, 2( 2I8, ES24.16 ) ) )" )                &
         ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne )
        END IF
        WRITE( control%out, "( A, ' C =', /, ( 5X, 3ES24.16 ) )" )             &
          prefix, prob%C( : prob%m )
      END IF

! ------------------------------------------------------------------
!  Check to see if the equality constraints are linearly independent
! ------------------------------------------------------------------

      IF ( prob%m > 0 .AND. control%remove_dependencies ) THEN

!  convert A to sparse row-wise format. Store matrix characteristics

        data%A_eqp%n = prob%n ; data%A_eqp%m = prob%m
        CALL SMT_put( data%A_eqp%type, 'SPARSE_BY_ROWS', inform%alloc_status )

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%A_eqp%ne = prob%n * prob%m
        CASE ( 'SPARSE_BY_ROWS' )
          data%A_eqp%ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%A_eqp%ne = prob%A%ne
        END SELECT

!  assign space for A

        array_name = 'eqp: data%A_eqp%val'
        CALL SPACE_resize_array( data%A_eqp%ne, data%A_eqp%val,                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error,              &
               exact_size = .TRUE. )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'eqp: data%A_eqp%col'
        CALL SPACE_resize_array( data%A_eqp%ne, data%A_eqp%col,                &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error,              &
               exact_size = .TRUE. )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'eqp: data%A_eqp%val'
        CALL SPACE_resize_array( data%A_eqp%m + 1, data%A_eqp%ptr,             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error,              &
               exact_size = .TRUE. )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  copy A to row-wise storage

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%A_eqp%ptr( 1 ) = 1
          l = 0
          DO i = 1, prob%m
            l = data%A_eqp%ptr( i ) - 1
            DO j = 1, prob%n
              l = l + 1
              data%A_eqp%col( l ) = j
              data%A_eqp%val( l ) = prob%A%val( l )
            END DO
            data%A_eqp%ptr( i + 1 ) = data%A_eqp%ptr( i ) +  prob%n
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          data%A_eqp%val( : data%A_eqp%ne ) = prob%A%val( : data%A_eqp%ne )
          data%A_eqp%col( : data%A_eqp%ne ) = prob%A%col( : data%A_eqp%ne )
          data%A_eqp%ptr( : data%A_eqp%m + 1 )                                 &
            = prob%A%ptr( : data%A_eqp%m + 1 )
        CASE ( 'COORDINATE' )
          data%A_eqp%ptr( 2 : prob%m + 1 ) = 0
          DO l = 1, prob%A%ne
            i = prob%A%row( l ) + 1
            data%A_eqp%ptr( i ) = data%A_eqp%ptr( i ) + 1
          END DO
          data%A_eqp%ptr( 1 ) = 1
          DO i = 2, data%A_eqp%m
            data%A_eqp%ptr( i ) = data%A_eqp%ptr( i - 1 ) + data%A_eqp%ptr( i )
          END DO
          DO l = 1, prob%A%ne
            i = prob%A%row( l ) ; j = data%A_eqp%ptr( i )
            data%A_eqp%col( j ) =  prob%A%col( l )
            data%A_eqp%val( j ) =  prob%A%val( l )
            data%A_eqp%ptr( i ) = j + 1
          END DO
          DO i = data%A_eqp%m, 1, - 1
            data%A_eqp%ptr( i + 1 ) = data%A_eqp%ptr( i )
          END DO
          data%A_eqp%ptr( 1 ) = 1
        END SELECT

!  find any dependent rows

        IF ( control%out > 0 .AND. control%print_level >= 1 ) THEN
          IF ( control%FDC_control%use_sls ) THEN
            WRITE( control%out, "( /, A, ' FDC (using ', A, ') called to',     &
           &  ' remove any dependent constraints' )" )  prefix,                &
              TRIM( control%FDC_control%symmetric_linear_solver )
          ELSE
            WRITE( control%out, "( /, A, ' FDC (using ', A, ') called to',     &
           &  ' remove any dependent constraints' )" )  prefix,                &
              TRIM( control%FDC_control%unsymmetric_linear_solver )
          END IF
        END IF

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL FDC_find_dependent( data%A_eqp%n, data%A_eqp%m,                   &
                                 data%A_eqp%val( : data%A_eqp%ne ),            &
                                 data%A_eqp%col( : data%A_eqp%ne ),            &
                                 data%A_eqp%ptr( : data%A_eqp%m + 1 ),         &
                                 prob%C, data%n_depen, data%C_depen,           &
                                 data%FDC_data, control%FDC_control,           &
                                 inform%FDC_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%find_dependent =                                           &
          inform%time%find_dependent + time_now - time_record
        inform%time%clock_find_dependent =                                     &
          inform%time%clock_find_dependent + clock_now - clock_record

!  record output parameters

        inform%status = inform%FDC_inform%status
        inform%alloc_status = inform%FDC_inform%alloc_status
        inform%bad_alloc = inform%FDC_inform%bad_alloc

!  check for error exits

        IF ( inform%status /= 0 ) THEN
          IF ( control%error > 0 .AND. control%print_level >= 1 .AND.          &
               inform%status /= GALAHAD_ok ) WRITE( control%error,             &
                 "( A, '    ** Error return ', I0, ' from ', A )" )            &
               prefix, inform%status, 'FDC_dependent'
          RETURN
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 1                    &
             .AND. inform%FDC_inform%non_negligible_pivot < thousand *         &
             control%FDC_control%SLS_control%absolute_pivot_tolerance )        &
            WRITE( control%out, "(                                             &
       &  /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /, A,                 &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, A, ' ***  perhaps increase', /, A,                                &
       &     ' FDC_control%SLS_control%absolute_pivot_tolerance from',         &
       &    ES11.4,'  ***', /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" )   &
           prefix, prefix, inform%FDC_inform%non_negligible_pivot, prefix,     &
           prefix, control%FDC_control%SLS_control%absolute_pivot_tolerance,   &
           prefix

        IF ( control%out > 0 .AND. control%print_level >= 1 .AND.              &
             data%n_depen > 0 ) THEN
          WRITE( control%out, "( /, A, 1X, I0, ' dependent constraint', A,     &
         &  ' removed' )" ) prefix, data%n_depen,                              &
             TRIM( STRING_pleural( data%n_depen ) )
          IF ( control%print_level >= 2 )                                      &
            WRITE( control%out, "( A, ' The following ', I0, ' constraint',    &
           &  A, ' appear', A, ' to be dependent', /, ( 4X, 8I8 ) )" )         &
                prefix, data%n_depen, TRIM(STRING_pleural( data%n_depen ) ),   &
              TRIM( STRING_verb_pleural( data%n_depen ) ), data%C_depen
         END IF

!  record and temporarily remove dependent constraints

        IF ( data%n_depen > 0 ) THEN
          array_name = 'eqp: data%C_status'
          CALL SPACE_resize_array( data%A_eqp%m, data%C_status,                &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc, out = control%error,            &
                 exact_size = .TRUE. )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'eqp: data%C'
          CALL SPACE_resize_array( data%A_eqp%m - data%n_depen, data%C,        &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 bad_alloc = inform%bad_alloc, out = control%error,            &
                 exact_size = .TRUE. )
          IF ( inform%status /= GALAHAD_ok ) RETURN

!  mark the independendent constraints with c_status = 1

          data%C_status( : prob%m ) = 1
          data%C_status( data%C_depen( : data%n_depen ) ) = 0

!  remove any constraint whose c_status = 0

          data%A_eqp%m = 0
          l = 0
          DO i = 1, prob%m
            IF ( data%C_status( i ) == 1 ) THEN
              j = l + data%A_eqp%ptr( i + 1 ) - data%A_eqp%ptr( i )
              data%A_eqp%m = data%A_eqp%m + 1
              data%A_eqp%val( l + 1 : j ) =                                    &
                data%A_eqp%val( data%A_eqp%ptr( i ) :                          &
                                data%A_eqp%ptr( i + 1 ) - 1 )
              data%A_eqp%col( l + 1 : j ) =                                    &
                data%A_eqp%col( data%A_eqp%ptr( i ) :                          &
                                data%A_eqp%ptr( i + 1 ) - 1 )
              data%A_eqp%ptr( data%A_eqp%m ) = l + 1
              l = j
              data%C( data%A_eqp%m ) = prob%C( i )
            END IF
          END DO
          data%A_eqp%ptr( data%A_eqp%m + 1 ) = l + 1
          IF ( control%out > 0 .AND. control%print_level >= 1 )                &
            WRITE( control%out,                                                &
              "( /, A, ' problem dimensions after removal of dependencies:',   &
           &    /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )            &
               prefix, prefix, prob%n, data%A_eqp%m,                           &
               data%A_eqp%ptr( data%A_eqp%m + 1 ) - 1
        END IF
      ELSE
        data%n_depen = 0
      END IF

!  create temporary gradient and, if necessary, Hessian storage, and set
!  H, G and f according to Hessian_kind, gradient_kind and target_kind

      array_name = 'eqp: data%G_eqp'
      CALL SPACE_resize_array( prob%n, data%G_eqp, inform%status,              &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      f = prob%f
      IF ( prob%gradient_kind == 0 ) THEN
        data%G_eqp( : prob%n ) = zero
      ELSE IF ( prob%gradient_kind == 1 ) THEN
        data%G_eqp( : prob%n ) = one
      ELSE
        data%G_eqp( : prob%n ) = prob%G( : prob%n )
      END IF

      IF ( prob%Hessian_kind >= 0 ) THEN
        data%H_eqp%n = prob%n
        CALL SMT_put( data%H_eqp%type, 'DIAGONAL', inform%alloc_status )

        array_name = 'eqp: data%H_eqp%val'
        CALL SPACE_resize_array( prob%n, data%H_eqp%val, inform%status,        &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        IF ( prob%Hessian_kind == 0 ) THEN
          data%H_eqp%val( : prob%n ) = zero
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            f = f + half * REAL( prob%n, KIND = wp ) ** 2
            data%G_eqp( : prob%n )                                             &
              = data%G_eqp( : prob%n ) - one
          ELSE IF ( prob%target_kind /= 0 ) THEN
            f = f + half *                                                     &
              DOT_PRODUCT( prob%X0( : prob%n ), prob%X0( : prob%n ) )
            data%G_eqp( : prob%n )                                             &
              = data%G_eqp( : prob%n ) - prob%X0( : prob%n )
          END IF
          data%H_eqp%val( : prob%n ) = one
        ELSE IF ( prob%Hessian_kind > 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            f = f + half *                                                     &
              DOT_PRODUCT( prob%WEIGHT( : prob%n ), prob%WEIGHT( : prob%n ) )
            data%G_eqp( : prob%n )                                             &
              = data%G_eqp( : prob%n ) -  prob%WEIGHT( : prob%n ) ** 2
          ELSE IF ( prob%target_kind /= 0 ) THEN
            f = f + half *                                                     &
              DOT_PRODUCT( prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ),      &
                           prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ) )
            data%G_eqp( : prob%n ) = data%G_eqp( : prob%n )                    &
              - prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ) ** 2
          END IF
          data%H_eqp%val( : prob%n ) = prob%WEIGHT( : prob%n ) ** 2
        END IF
      END IF

! ----------------
!  Call the solver
! ----------------

      IF ( control%out > 0 .AND. control%print_level >= 1 )                    &
        WRITE( control%out, "( /, A, ' find solution using linear equation',   &
       &  ' solvers ', A, ', ', A, ', ', A )" ) prefix,                        &
          TRIM( control%SBLS_control%symmetric_linear_solver ),                &
          TRIM( control%SBLS_control%definite_linear_solver ),                 &
          TRIM( control%SBLS_control%unsymmetric_linear_solver )

      IF ( prob%Hessian_kind >= 0 ) THEN
        IF ( data%n_depen == 0 ) THEN
          CALL EQP_solve_main( prob%n, prob%m, data%H_eqp, data%G_eqp,         &
                               f, prob%A, prob%q, prob%X, prob%Y,              &
                               data, control, inform, C = prob%C, D = D )
        ELSE
          CALL EQP_solve_main( prob%n, data%A_eqp%m, data%H_eqp, data%G_eqp,   &
                               f, data%A_eqp, prob%q, prob%X, prob%Y,          &
                               data, control, inform, C = data%C, D = D )
        END IF
      ELSE
        IF ( data%n_depen == 0 ) THEN
          CALL EQP_solve_main( prob%n, prob%m, prob%H, data%G_eqp,             &
                               f, prob%A, prob%q, prob%X, prob%Y,              &
                               data, control, inform, C = prob%C, D = D )
        ELSE
          CALL EQP_solve_main( prob%n, data%A_eqp%m, prob%H, data%G_eqp,       &
                               f, data%A_eqp, prob%q, prob%X, prob%Y,          &
                               data, control, inform, C = data%C, D = D )
        END IF
      END IF

      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( inform%status /= GALAHAD_ok ) RETURN

!  assign lagrange multipliers to the dependent constraints

      IF ( data%n_depen > 0 ) THEN
        data%C( : data%A_eqp%m ) = prob%Y( : data%A_eqp%m )
        data%A_eqp%m = 0
        DO i = 1, prob%m
          IF ( data%C_status( i ) == 1 ) THEN
            data%A_eqp%m = data%A_eqp%m + 1
            prob%Y( i ) = data%C( data%A_eqp%m )
          ELSE
            prob%Y( i ) = zero
          END IF
        END DO
      END IF

!  If required, write out the solution

      IF ( control%out > 0 .AND. control%print_level >= 10 ) THEN
        WRITE( control%out, "( A,                                              &
       &  ' X = ', 3ES24.16, /, ( 5X, 3ES24.16 ) )" ) prefix, prob%X( : prob%n )
        IF ( prob%m > 0 ) WRITE( control%out, "( A,                            &
       &  ' Y = ', 3ES24.16, /, ( 5X, 3ES24.16 ) )" ) prefix, prob%Y( : prob%m )
      END IF
      RETURN

!  End of EQP_solve

      END SUBROUTINE EQP_solve

!-*-*-*-*-   E Q P _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE EQP_solve_main( n, m, H, G, f, A, q, X, Y,                    &
                                 data, control, inform, C, D )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (Ax)_i + c_i = 0 , i = 1, .... , m,
!
!  where x is a vector of n components ( x_1, .... , x_n ), f is a constant,
!  g is an n-vector, H is a symmetric matrix, A is an m by n matrix, and
!  c is an m-vector using a projected conjugate gradient method.
!  The subroutine is particularly appropriate when A and H are sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( OUT ) :: q
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( IN ) :: H
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      TYPE ( EQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( EQP_control_type ), INTENT( IN ) :: control
      TYPE ( EQP_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: C
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: D

!  Local variables

      INTEGER :: out
      LOGICAL :: printi, printt, maxpiv
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: pivot_tol, relative_pivot_tol, min_pivot_tol
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      inform%status = GALAHAD_ok
      inform%obj = zero ; q = inform%obj

!  check to see if the problem is trivial

      IF ( n == 0 ) THEN
        IF ( m == 0 ) THEN
          RETURN
        ELSE IF ( m > 0 ) THEN
          IF ( MAXVAL( ABS( C( : m ) ) ) <=                                    &
               control%max_infeasibility_absolute   ) THEN
            Y = zero
            RETURN
          END IF
        END IF
      END IF

!  ===========================
!  Control the output printing
!  ===========================

      out = control%out

!  basic output

      printi = out > 0 .AND. control%print_level >= 1

!  single line of output per iteration with timings for various operations

      printt = out > 0 .AND. control%print_level >= 2

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        data%a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        data%a_ne = A%ptr( m + 1 ) - 1
      ELSE
        data%a_ne = A%ne
      END IF

!  Set C appropriately

      data%SBLS_control = control%SBLS_control
      IF ( data%new_c ) THEN
        data%new_c = .FALSE.
        data%SBLS_control%new_c = 2
        data%C0%ne = 0

        array_name = 'eqp: data%C0%row'
        CALL SPACE_resize_array( data%C0%ne, data%C0%row, inform%status,       &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'eqp: data%C0%col'
        CALL SPACE_resize_array( data%C0%ne, data%C0%col, inform%status,       &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'eqp: data%C0%val'
        CALL SPACE_resize_array( data%C0%ne, data%C0%val, inform%status,       &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'eqp: data%C0%type'
        CALL SPACE_dealloc_array( data%C0%type, inform%status,                 &
           inform%alloc_status, array_name = array_name, out = control%error )
        CALL SMT_put( data%C0%type, 'COORDINATE', inform%alloc_status )
      ELSE
        data%SBLS_control%new_c = 0
      END IF

!  ------------------------------------------
!   1. Form and factorize the preconditioner
!  ------------------------------------------

      CALL CPU_TIME( inform%time%factorize )

!  set initial pivot control values

      pivot_tol = control%SBLS_control%SLS_control%relative_pivot_tolerance
      min_pivot_tol = control%SBLS_control%SLS_control%minimum_pivot_tolerance
      relative_pivot_tol = pivot_tol
      maxpiv = pivot_tol >= half

      data%SBLS_control%new_a = control%new_a
      data%SBLS_control%new_h = control%new_h
      data%SBLS_control%remove_dependencies = control%remove_dependencies
      data%SBLS_control%prefix = '" - SBLS:"                    '
      data%SBLS_control%preconditioner = control%preconditioner

 100  CONTINUE
      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SBLS_form_and_factorize( n, m, H, A, data%C0, data%SBLS_data,       &
                                    data%SBLS_control, inform%SBLS_inform, D )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock_record

      IF ( printt ) WRITE( out,                                                &
         "(  A, ' on exit from SBLS: status = ', I0, ', time = ', F0.2 )" )    &
           prefix, inform%SBLS_inform%status, clock_now - clock_record

!  check for success

      IF ( inform%SBLS_inform%status < 0 ) THEN
        inform%status = inform%SBLS_inform%status
        GO TO 910
      ELSE IF ( inform%SBLS_inform%rank_def ) THEN
        IF ( printi ) WRITE( out, "( A, ' ** warning ** the matrix is not of', &
       &  ' full rank, nullity = ', I0 )" ) prefix, m - inform%SBLS_inform%rank
      END IF

      data%eqp_factors = .TRUE.
      data%m = m ; data%n = n

!  ------------------
!   2. Solve the EQP
!  ------------------

      CALL EQP_resolve_main( n, m, H, G, f, A, q, X, Y,                        &
                             data, control, inform, C = C )

!  check for success

      IF ( inform%status == GALAHAD_error_preconditioner .OR.                  &
           inform%status == GALAHAD_error_ill_conditioned .OR.                 &
           inform%status == GALAHAD_error_solve .OR.                           &
           inform%status == GALAHAD_error_uls_solve ) THEN
        GO TO 910
      END IF

      IF ( printt ) THEN
        SELECT CASE( inform%SBLS_inform%preconditioner )
        CASE( 1 )
          WRITE( out, "( A, ' Preconditioner G = I' )" ) prefix
        CASE( 2 )
          WRITE( out, "( A, ' Preconditioner G = H' )" ) prefix
        CASE( 3 )
          WRITE( out, "( A, ' Preconditioner G = diag(H)' )" ) prefix
        CASE( 4 )
          WRITE( out, "( A, ' Preconditioner G = band(H)' )" ) prefix
        CASE( 5 )
          WRITE( out, "( A, ' Preconditioner G = optional D' )" ) prefix
        CASE( 11 )
          WRITE( out, "( A, ' Preconditioner G = H_22' )" ) prefix
        CASE( 12 )
          WRITE( out, "( A, ' Preconditioner G = H_22 & H_21' )" ) prefix
        CASE( - 1 )
          WRITE( out, "( A, ' Preconditioner G_22 = I' )" ) prefix
        CASE( - 2 )
          WRITE( out, "( A, ' Preconditioner G_22 = H_22' )" ) prefix
        CASE( - 3 )
          WRITE( out, "( A, ' Preconditioner G_22 = H_22 and G_21 = H_21')")   &
            prefix
        END SELECT
      END IF
      RETURN

!  if the method failed because of a poor preconditioner, try again

 910  CONTINUE

!  we might have run out of options ....

      IF ( data%SBLS_control%factorization == 2 .AND. maxpiv ) THEN
        RETURN

!  ... or we may change the method ....

      ELSE IF ( data%SBLS_control%factorization < 2 .AND. maxpiv ) THEN
        pivot_tol = relative_pivot_tol
        maxpiv = pivot_tol >= half
        data%SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
        data%SBLS_control%SLS_control%minimum_pivot_tolerance = min_pivot_tol
        data%SBLS_control%factorization = 2
        IF ( printi )  WRITE( out,                                             &
          "( A, '    ** Switching to augmented system method' )" ) prefix

!  ... or we can increase the pivot tolerance

      ELSE IF ( data%SBLS_control%SLS_control%relative_pivot_tolerance         &
                < relative_pivot_default ) THEN
        pivot_tol = relative_pivot_default
        min_pivot_tol = relative_pivot_default
        maxpiv = .FALSE.
        data%SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
        data%SBLS_control%SLS_control%minimum_pivot_tolerance = min_pivot_tol
        IF ( printi ) WRITE( out,                                              &
          "( A, '    ** Pivot tolerance increased to', ES11.4 )" )             &
          prefix, pivot_tol
      ELSE
        pivot_tol = half
        min_pivot_tol = half
        maxpiv = .TRUE.
        data%SBLS_control%SLS_control%relative_pivot_tolerance = pivot_tol
        data%SBLS_control%SLS_control%minimum_pivot_tolerance = min_pivot_tol
        IF ( printi ) WRITE( out,                                              &
          "( A, '    ** Pivot tolerance increased to', ES11.4 )" )             &
          prefix, pivot_tol
      END IF
      inform%factorization_integer = - 1
      inform%factorization_real = - 1
      GO TO 100

!  End of EQP_solve_main

      END SUBROUTINE EQP_solve_main

!-*-*-*-*-*-*-*-   E Q P _ R E S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE EQP_resolve( prob, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Resolve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    A x = c
!
!  when the data g, f and c may have changed since a previous
!  solve but where H and A are unchanged

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( EQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( EQP_control_type ), INTENT( IN ) :: control
      TYPE ( EQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now, f

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  check to see if this is really a re-solve

      IF ( .NOT. data%eqp_factors ) THEN
        inform%status = GALAHAD_error_input_status
        RETURN
      END IF

!  record and temporarily remove dependent constraints

      IF ( data%n_depen > 0 ) THEN
        data%A_eqp%m = 0
        DO i = 1, prob%m
          IF ( data%C_status( i ) == 1 ) THEN
            data%A_eqp%m = data%A_eqp%m + 1
            data%C( data%A_eqp%m ) = prob%C( i )
          END IF
        END DO
      END IF

!  create temporary gradient and, if necessary, Hessian storage, and set
!  H, G and f according to Hessian_kind, gradient_kind and target_kind

      f = prob%f
      IF ( prob%gradient_kind == 0 ) THEN
        data%G_eqp( : prob%n ) = zero
      ELSE IF ( prob%gradient_kind == 1 ) THEN
        data%G_eqp( : prob%n ) = one
      ELSE
        data%G_eqp( : prob%n ) = prob%G( : prob%n )
      END IF

      IF ( prob%Hessian_kind >= 0 ) THEN
        IF ( prob%Hessian_kind == 0 ) THEN
          data%H_eqp%val( : prob%n ) = zero
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            f = f + half * REAL( prob%n, KIND = wp ) ** 2
            data%G_eqp( : prob%n )                                             &
              = data%G_eqp( : prob%n ) - one
          ELSE IF ( prob%target_kind /= 0 ) THEN
            f = f + half *                                                     &
              DOT_PRODUCT( prob%X0( : prob%n ), prob%X0( : prob%n ) )
            data%G_eqp( : prob%n )                                             &
              = data%G_eqp( : prob%n ) - prob%X0( : prob%n )
          END IF
          data%H_eqp%val( : prob%n ) = one
        ELSE IF ( prob%Hessian_kind > 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            f = f + half *                                                     &
              DOT_PRODUCT( prob%WEIGHT( : prob%n ), prob%WEIGHT( : prob%n ) )
            data%G_eqp( : prob%n )                                             &
              = data%G_eqp( : prob%n ) -  prob%WEIGHT( : prob%n ) ** 2
          ELSE IF ( prob%target_kind /= 0 ) THEN
            f = f + half *                                                     &
              DOT_PRODUCT( prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ),      &
                           prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ) )
            data%G_eqp( : prob%n ) = data%G_eqp( : prob%n )                    &
              - prob%X0( : prob%n ) * prob%WEIGHT( : prob%n ) ** 2
          END IF
          data%H_eqp%val( : prob%n ) = prob%WEIGHT( : prob%n ) ** 2
        END IF
      END IF

!  Call the solver

      IF ( data%m == prob%m .AND. data%n == prob%n ) THEN
        IF ( prob%Hessian_kind >= 0 ) THEN
          CALL EQP_resolve_main( prob%n, prob%m, data%H_eqp, data%G_eqp, f,    &
                                 prob%A, prob%q, prob%X, prob%Y,               &
                                 data, control, inform, C = prob%C )
        ELSE
          CALL EQP_resolve_main( prob%n, prob%m, prob%H, data%G_eqp, f,        &
                                 prob%A, prob%q, prob%X, prob%Y,               &
                                 data, control, inform, C = prob%C )
        END IF
      ELSE
        IF ( prob%Hessian_kind >= 0 ) THEN
          CALL EQP_resolve_main( data%n, data%m, data%H_eqp, data%G_eqp, f,    &
                                 data%A_eqp, prob%q, prob%X, prob%Y,           &
                                 data, control, inform, C = prob%C )
        ELSE
          CALL EQP_resolve_main( data%n, data%m, prob%H, data%G_eqp, f,        &
                                 data%A_eqp, prob%q, prob%X, prob%Y,           &
                                 data, control, inform, C = prob%C )
        END IF
      END IF

      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

!  assign lagrange multipliers to the dependent constraints

      IF ( data%n_depen > 0 ) THEN
        data%C( : data%A_eqp%m ) = prob%Y( : data%A_eqp%m )
        data%A_eqp%m = 0
        DO i = 1, prob%m
          IF ( data%C_status( i ) == 1 ) THEN
            data%A_eqp%m = data%A_eqp%m + 1
            prob%Y( i ) = data%C( data%A_eqp%m )
          ELSE
            prob%Y( i ) = zero
          END IF
        END DO
      END IF

      RETURN

!  End of EQP_resolve

      END SUBROUTINE EQP_resolve

!-*-*-*-*-   E Q P _ R E S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE EQP_resolve_main( n, m, H, G, f, A, q, X, Y,                  &
                                   data, control, inform, C )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Resolve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (Ax)_i + c_i = 0 , i = 1, .... , m,
!
!  where x is a vector of n components ( x_1, .... , x_n ), f is a constant,
!  g is an n-vector, c is an m-vector, H is a symmetric matrix,
!  A is an m by n matrix, and A and H are unchanged since a previous call,
!  using a projected conjugate gradient method.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( IN ) :: f
      REAL ( KIND = wp ), INTENT( OUT ) :: q
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( IN ) :: H
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: Y
      TYPE ( EQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( EQP_control_type ), INTENT( IN ) :: control
      TYPE ( EQP_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: C

!  Local variables

      INTEGER :: out, i, j, l
      LOGICAL :: null_space_prec, explicit_prec, rank_def, solve_inter, xfeq0
      LOGICAL :: printi, printt, printw, c_ne_0
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: radius, q_save, model, val, maxres
      REAL ( KIND = wp ) :: pgnorm, stop_inter
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check that the factorization has been performed and that the new data is
!  consistent

      IF ( .NOT. data%eqp_factors ) THEN
        inform%status = GALAHAD_error_input_status
        RETURN
!      ELSE IF ( data%m /= m .OR. data%n /= n ) THEN
!write(6,*) data%m, m, data%n, n
!        inform%status = GALAHAD_error_input_status
!        RETURN
      END IF

!  ===========================
!  Control the output printing
!  ===========================

      out = control%out

!  Basic single line of output per iteration

      printi = out > 0 .AND. control%print_level >= 1

!  As per printi, but with additional timings for various operations

      printt = out > 0 .AND. control%print_level >= 2

!  As per printt, but with checking of residuals, etc, and also with an
!  indication of where in the code we are

      printw = out > 0 .AND. control%print_level >= 4

      inform%GLTR_inform%status = 1
      inform%GLTR_inform%negative_curvature = .TRUE.

      null_space_prec = inform%SBLS_inform%factorization == 3
      explicit_prec = inform%SBLS_inform%preconditioner > 0
      rank_def = inform%SBLS_inform%rank_def

!  see if c is present, and if so if it is nonzero

      IF ( PRESENT( C ) ) THEN
        c_ne_0 = MAXVAL( ABS( C( : m ) ) ) /= zero
      ELSE
        c_ne_0 = .FALSE.
      END IF

!  ------------------
!   1. Solve the EQP
!  ------------------

      array_name = 'eqp: data%VECTOR'
      CALL SPACE_resize_array( n + m, data%VECTOR, inform%status,              &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%RES'
      CALL SPACE_resize_array( m, data%RES, inform%status,                     &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%G_f'
      CALL SPACE_resize_array( n, data%G_f, inform%status,                     &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  ---------------------------
!   1a. Find a feasible point
!  ---------------------------

!   Find a suitable feasible point x_f (stored in X)

      IF ( c_ne_0 ) THEN
!       data%VECTOR( : n  ) = - G
        data%VECTOR( : n  ) = zero
        data%VECTOR( n + 1 : n + m ) = - C( : m )
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( null_space_prec ) THEN
          CALL SBLS_solve_null_space( data%SBLS_data%nfactors,                 &
                                      data%SBLS_control, inform%SBLS_inform,   &
                                      data%VECTOR )
        ELSE IF ( explicit_prec ) THEN
          CALL SBLS_solve_explicit( n, m, A, data%C0, data%SBLS_data%efactors, &
                                    data%SBLS_control, inform%SBLS_inform,     &
                                    data%VECTOR )
        ELSE
          CALL SBLS_basis_solve( data%SBLS_data%ifactors, data%SBLS_control,   &
                                 inform%SBLS_inform, data%VECTOR )
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_solve + clock_now - clock_record

        IF ( inform%SBLS_inform%status < 0 ) THEN
          inform%status = inform%SBLS_inform%status ; RETURN
        END IF

        xfeq0 = .FALSE.
        X = data%VECTOR( : n  )
!write(6,*) ' x_f ', X
!write(6,*) ' ||x_f|| ', MAXVAL( ABS( X ) )
!  Compute the constraint residuals

        data%RES( : m ) = C( : m )
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, m
            data%RES( i )                                                      &
              = data%RES( i ) + DOT_PRODUCT( A%val( l + 1 : l + n ), X )
            l = l + n
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              data%RES( i ) = data%RES( i ) + A%val( l ) * X( A%col( l ) )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l )
            data%RES( i ) = data%RES( i ) + A%val( l ) * X( A%col( l ) )
          END DO
        END SELECT

!  Check to see if the residuals of potentially inconsistemt constraints
!  are satisfactory

        maxres = MAX( MAXVAL( ABS( A%val( : data%a_ne ) ) ),                   &
                      MAXVAL( ABS( C( : m ) ) ), MAXVAL( ABS( X( : n ) ) ) )
        maxres = MAX( control%max_infeasibility_relative * maxres,             &
                      control%max_infeasibility_absolute )
        val = MAXVAL( ABS( data%RES( : m ) ) )

        IF ( rank_def ) THEN
          IF ( val <= maxres ) THEN
            IF ( printw ) WRITE( out,                                          &
              "( A, ' residual', ES13.5, ' acceptably small relative to',      &
             &  ES13.5, /, A, ' - constraints are consistent' )" )             &
               prefix, val, maxres, prefix
          ELSE
            IF ( printi ) WRITE( out,                                          &
              "( A, ' residual', ES13.5, ' too large relative to', ES13.5, /,  &
             &   A, ' - constraints likely inconsistent' )" )                  &
               prefix, val, maxres, prefix
            inform%status = GALAHAD_error_primal_infeasible ; RETURN
          END IF
        ELSE
          IF ( val > maxres ) THEN
            IF ( printi ) WRITE( out,                                          &
              "( A, ' residual ', ES13.5, ' too large relative to', ES13.5, /, &
             &   A, ' - factorization likely inaccurate' )" )                  &
               prefix, val, maxres, prefix
            inform%status = GALAHAD_error_ill_conditioned ; RETURN
          END IF
        END IF

!  Compute the function and gradient values at x_f

        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' )
          DO i = 1, n
            data%G_f( i ) = H%val( i ) * X( i )
          END DO
        CASE ( 'DENSE' )
          data%G_f = zero
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1 ; val = H%val( l )
              data%G_f( i ) = data%G_f( i ) + val * X( j )
              IF ( i /= j ) data%G_f( j ) = data%G_f( j ) + val * X( i )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          data%G_f = zero
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              j = H%col( l ) ; val = H%val( l )
              data%G_f( i ) = data%G_f( i ) + val * X( j )
              IF ( i /= j ) data%G_f( j ) = data%G_f( j ) + val * X( i )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          data%G_f = zero
          DO l = 1, H%ne
            i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
            data%G_f( i ) = data%G_f( i ) + val * X( j )
            IF ( i /= j ) data%G_f( j ) = data%G_f( j ) + val * X( i )
          END DO
        END SELECT

        q_save = f + DOT_PRODUCT( G( : n ), X( : n ) )                         &
                   + half * DOT_PRODUCT( data%G_f( : n ), X( : n ) )
        data%G_f( : n ) = G( : n ) + data%G_f( : n )

        IF ( m > 0 ) THEN
          IF ( printi ) WRITE( out,                                            &
             "( /, A, ' objective & constraints (feasibility phase) =',        &
            &  ES13.5, ',', ES11.4 )" )                                        &
               prefix, q_save, MAXVAL( ABS( data%RES( : m ) ) )
        ELSE
          IF ( printi ) WRITE( out,                                            &
             "( /, A, ' objective & constraints (feasibility phase) =',        &
            &  ES13.5, ',', ES11.4 )" ) prefix, q_save, zero
        END IF
      ELSE
        xfeq0 = .TRUE.
        X = zero
        q_save = f
        data%G_f( : n ) = G( : n )
        IF ( printi ) WRITE( out,                                              &
           "( /, A, ' objective & constraints (feasibility phase) =',          &
          &  ES13.5, ',', ES11.4 )" ) prefix, q_save, zero
      END IF

!  --------------------------------------------------
!   1b. From the feasible point, use GLTR to minimize
!       the objective in the null-space of A
!  --------------------------------------------------

!   Compute the correction s from x_f (stored in S)

      array_name = 'eqp: data%R'
      CALL SPACE_resize_array( n + m, data%R, inform%status,                   &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%S'
      CALL SPACE_resize_array( n, data%S, inform%status,                       &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%WORK'
      CALL SPACE_resize_array( n, data%WORK, inform%status,                    &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  Set initial data

      IF ( control%radius > zero ) THEN
        radius = control%radius
      ELSE
        radius = SQRT( point1 * HUGE( one ) )
      END IF
      q = q_save
      IF ( xfeq0 ) THEN
        data%R( : n ) = G( : n )
      ELSE
        data%R( : n ) = data%G_f( : n )
      END IF
      inform%GLTR_inform%status = 1
      solve_inter = .FALSE.
      inform%cg_iter = 0 ; inform%cg_iter_inter = 0

      data%GLTR_control = control%GLTR_control
      data%GLTR_control%f_0 = q_save
      data%GLTR_control%stop_relative = control%inner_stop_relative
      data%GLTR_control%stop_absolute = control%inner_stop_absolute
!     data%GLTR_control%rminvr_zero = hundred * epsmch ** 2
      data%GLTR_control%unitm = .FALSE.
      data%GLTR_control%itmax = control%cg_maxit
      data%GLTR_control%boundary = .FALSE.
      data%GLTR_control%prefix = '" - GLTR:"                    '

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      DO
        CALL GLTR_solve( n, radius, model, data%S, data%R( : n ),              &
                         data%VECTOR( : n ), data%GLTR_data,                   &
                         data%GLTR_control, inform%GLTR_inform )

!  Check for error returns

!       WRITE(6,"( ' case ', i3  )" ) inform%GLTR_inform%status
        SELECT CASE( inform%GLTR_inform%status )

!  Successful return

        CASE ( GALAHAD_ok )
          EXIT

!  Warnings

        CASE ( GALAHAD_warning_on_boundary, GALAHAD_error_max_iterations )
          IF ( printt ) WRITE( out, "( /,                                      &
         &  A, ' Warning return from GLTR, status = ', I0 )" ) prefix,         &
              inform%GLTR_inform%status
          EXIT

!  Allocation errors

           CASE ( GALAHAD_error_allocate )
             inform%status = GALAHAD_error_allocate
             inform%alloc_status = inform%gltr_inform%alloc_status
             inform%bad_alloc = inform%gltr_inform%bad_alloc
             RETURN

!  Deallocation errors

           CASE ( GALAHAD_error_deallocate )
             inform%status = GALAHAD_error_deallocate
             inform%alloc_status = inform%gltr_inform%alloc_status
             inform%bad_alloc = inform%gltr_inform%bad_alloc
             RETURN

!  Error return

        CASE DEFAULT
          inform%status = inform%gltr_inform%status
          IF ( printt ) WRITE( out, "( /,                                      &
         &  A, ' Error return from GLTR, status = ', I0 )" ) prefix,           &
              inform%GLTR_inform%status
          EXIT

!  Find the preconditioned gradient

        CASE ( 2, 6 )
          IF ( printw ) WRITE( out,                                            &
             "( A, ' ............... precondition  ............... ' )" ) prefix
          data%VECTOR( n + 1 : n + m ) = zero

!     data%SBLS_control%out = 6
!     data%SBLS_control%print_level = 2

          data%SBLS_control%affine = .TRUE.
          CALL SBLS_solve( n, m, A, data%C0, data%SBLS_data,                   &
             data%SBLS_control, inform%SBLS_inform, data%VECTOR )

          IF ( inform%SBLS_inform%status < 0 ) THEN
            inform%status = inform%SBLS_inform%status
            RETURN
          END IF

          IF ( inform%GLTR_inform%status == 2 ) THEN
            pgnorm = DOT_PRODUCT( data%R( : n ), data%VECTOR( : n ) )
            IF ( ABS( pgnorm ) < ten * EPSILON( one ) ) pgnorm = zero
            pgnorm = SIGN( SQRT( ABS( pgnorm ) ), pgnorm )
            IF ( inform%cg_iter == 0 )                                         &
              stop_inter = pgnorm * control%inner_stop_inter
          END IF

!  Compute the residuals

          IF ( printw ) THEN
            data%RES = zero
            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                data%RES( i ) = data%RES( i )                                  &
                  + DOT_PRODUCT( A%val( l + 1 : l + n ), data%VECTOR )
                l = l + n
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  data%RES( i )                                                &
                    = data%RES( i ) + A%val( l ) * data%VECTOR( A%col( l ) )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                i = A%row( l )
                data%RES( i )                                                  &
                  = data%RES( i ) + A%val( l ) * data%VECTOR( A%col( l ) )
              END DO
            END SELECT
            WRITE( out,                                                        &
              "( A, ' Constraint residual (optimality phase) ', ES10.4 )" )    &
                prefix, MAXVAL( ABS( data%RES( : m ) ) )
          END IF

!  Form the product of VECTOR with H

        CASE ( 3, 7 )

          IF ( inform%GLTR_inform%status == 3 ) THEN
            inform%cg_iter = inform%cg_iter + 1
            IF ( .NOT. solve_inter .AND. pgnorm <= stop_inter ) THEN
              inform%cg_iter_inter = inform%cg_iter
              solve_inter = .TRUE.
            END IF
          END IF

          IF ( printw ) WRITE( out,                                            &
            "( A, ' ............ matrix-vector product ..........' )" ) prefix

          data%WORK( : n ) = data%VECTOR( : n )

          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              data%VECTOR( i ) = H%val( i ) * data%WORK( i )
            END DO
          CASE ( 'DENSE' )
            data%VECTOR( : n ) = zero
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1 ; val = H%val( l )
                data%VECTOR( i ) = data%VECTOR( i ) + val * data%WORK( j )
                IF ( i /= j ) data%VECTOR( j )                                 &
                  = data%VECTOR( j ) + val * data%WORK( i )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            data%VECTOR( : n ) = zero
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                j = H%col( l ) ; val = H%val( l )
                data%VECTOR( i ) = data%VECTOR( i ) + val * data%WORK( j )
                IF ( i /= j ) data%VECTOR( j )                                 &
                  = data%VECTOR( j ) + val * data%WORK( i )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            data%VECTOR( : n ) = zero
            DO l = 1, H%ne
              i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
              data%VECTOR( i ) = data%VECTOR( i ) + val * data%WORK( j )
              IF ( i /= j ) data%VECTOR( j )                                   &
                = data%VECTOR( j ) + val * data%WORK( i )
            END DO
          END SELECT

!  Reform the initial residual

        CASE ( 5 )

          IF ( printw ) WRITE( out,                                            &
            "( A, ' ................. restarting ................ ' )" ) prefix

          q = q_save
          IF ( xfeq0 ) THEN
            data%R( : n ) = G( : n )
          ELSE
            data%R( : n ) = data%G_f( : n )
          END IF

        END SELECT

      END DO
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_solve + clock_now - clock_record

!  Update the solution

      X( : n ) = X( : n ) + data%S( : n )
      inform%obj = model ; q = inform%obj

      IF ( .NOT. solve_inter ) THEN
        inform%cg_iter_inter = inform%cg_iter
        inform%time%solve_inter = time_now - inform%time%solve
      END IF

!  Compute the residuals

      IF ( c_ne_0 ) THEN
        data%RES( : m ) = C( : m )
      ELSE
        data%RES( : m ) = zero
      END IF
      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        l = 0
        DO i = 1, m
          data%RES( i ) =                                                      &
            data%RES( i ) + DOT_PRODUCT( A%val( l + 1 : l + n ), X )
          l = l + n
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            data%RES( i ) = data%RES( i ) + A%val( l ) * X( A%col( l ) )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          i = A%row( l )
          data%RES( i ) = data%RES( i ) + A%val( l ) * X( A%col( l ) )
        END DO
      END SELECT

!  If required, compute the function value at x

      IF ( printt ) THEN
        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' )
          DO i = 1, n
            data%G_f( i ) = H%val( i ) * X( i )
          END DO
        CASE ( 'DENSE' )
          data%G_f = zero
          l = 0
          DO i = 1, n
            DO j = 1, i
              l = l + 1 ; val = H%val( l )
              data%G_f( i ) = data%G_f( i ) + val * X( j )
              IF ( i /= j ) data%G_f( j ) = data%G_f( j ) + val * X( i )
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          data%G_f = zero
          DO i = 1, n
            DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
              j = H%col( l ) ; val = H%val( l )
              data%G_f( i ) = data%G_f( i ) + val * X( j )
              IF ( i /= j ) data%G_f( j ) = data%G_f( j ) + val * X( i )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          data%G_f = zero
          DO l = 1, H%ne
            i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
            data%G_f( i ) = data%G_f( i ) + val * X( j )
            IF ( i /= j ) data%G_f( j ) = data%G_f( j ) + val * X( i )
          END DO
        END SELECT
        q_save = f + DOT_PRODUCT( G( : n ), X( : n ) )                         &
                   + half * DOT_PRODUCT( data%G_f( : n ), X( : n ) )

        WRITE( out, "( A, ' recurred and calculated objective values ',        &
       &    2ES14.6 )" ) prefix, inform%obj, q_save
      END IF

      IF ( printt ) WRITE( out,                                                &
         "(  A, ' on exit from GLTR: status = ', I0, ', CG iterations = ', I0, &
        &   ', time = ', F0.2 )" ) prefix,                                     &
            inform%GLTR_inform%status, inform%cg_iter, inform%time%clock_solve
      IF ( m > 0 ) THEN
        IF ( printi ) WRITE( out,                                              &
           "(  A, ' objective & constraints (optimality',                      &
          &   '  phase) =', ES13.5, ',', ES11.4 )" )                           &
              prefix, inform%obj, MAXVAL( ABS( data%RES( : m ) ) )
      ELSE
        IF ( printi ) WRITE( out,                                              &
           "(  A, ' objective & constraints (optimality',                      &
          &   '  phase) =', ES13.5, ',', ES11.4 )" )                           &
              prefix, inform%obj, zero
      END IF

!  Compute the Lagrange multiplier estimates

      Y( : m ) = data%VECTOR( n + 1 : n + m )

      RETURN

!  End of EQP_resolve_main

      END SUBROUTINE EQP_resolve_main

!-*-*-*-*-*-*-   E Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE EQP_terminate( data, control, inform )

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
!   data    see Subroutine EQP_initialize
!   control see Subroutine EQP_initialize
!   inform  see Subroutine EQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( EQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( EQP_control_type ), INTENT( IN ) :: control
      TYPE ( EQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      data%eqp_factors = .FALSE.

!  Deallocate all arrays allocated by SBLS and GLTR

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
      CALL GLTR_terminate( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )

!  Deallocate all remaining allocated arrays

      array_name = 'eqp: data%C0%row'
      CALL SPACE_dealloc_array( data%C0%row,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%C0%col'
      CALL SPACE_dealloc_array( data%C0%col,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%C0%val'
      CALL SPACE_dealloc_array( data%C0%val,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%C0%type'
      CALL SPACE_dealloc_array( data%C0%type,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%G_eqp'
      CALL SPACE_dealloc_array( data%G_eqp,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%G_f'
      CALL SPACE_dealloc_array( data%G_f,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%WORK'
      CALL SPACE_dealloc_array( data%WORK,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%RES'
      CALL SPACE_dealloc_array( data%RES,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%A_eqp%val'
      CALL SPACE_dealloc_array( data%A_eqp%val,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%A_eqp%col'
      CALL SPACE_dealloc_array( data%A_eqp%col,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%A_eqp%ptr'
      CALL SPACE_dealloc_array( data%A_eqp%ptr,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%H_eqp%val'
      CALL SPACE_dealloc_array( data%H_eqp%val,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%C_depen'
      CALL SPACE_dealloc_array( data%C_depen,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'eqp: data%C_status'
      CALL SPACE_dealloc_array( data%C_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  End of subroutine EQP_terminate

      END SUBROUTINE EQP_terminate

!  End of module EQP

   END MODULE GALAHAD_EQP_double
