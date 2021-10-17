! THIS VERSION: GALAHAD 3.3 - 07/07/2021 AT 09:45 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ B G O   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.8. June 20th 2016

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BGO_double

!     ------------------------------------------------------------------
!    |                                                                  |
!    | BGO, an algorithm for bound-constrained global optimization      |
!    |                                                                  |
!    |   Aim: find a global minimizer of the objective f(x)             |
!    |        subject to x_l <= x <= x_u                                |
!    |                                                                  |
!     ------------------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_TRB_double
     USE GALAHAD_UGO_double
     USE GALAHAD_LHS_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_RAND_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: BGO_initialize, BGO_read_specfile, BGO_solve,                   &
               BGO_terminate, NLPT_problem_type,                               &
               NLPT_userdata_type, SMT_type, SMT_put,                          &
               BGO_import, BGO_solve_with_mat, BGO_solve_without_mat,          &
               BGO_solve_reverse_with_mat, BGO_solve_reverse_without_mat,      &
               BGO_full_initialize, BGO_full_terminate, BGO_reset_control,     &
               BGO_information

!----------------------
!   I n t e r f a c e s
!----------------------

      INTERFACE BGO_initialize
        MODULE PROCEDURE BGO_initialize, BGO_full_initialize
      END INTERFACE BGO_initialize

      INTERFACE BGO_terminate
        MODULE PROCEDURE BGO_terminate, BGO_full_terminate
      END INTERFACE BGO_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: sixteenth = 0.0625_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: sixteen = 16.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
     REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 9 )
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BGO_control_type

!   error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!   the maximum number of random searches from the best point found so far

       INTEGER :: attempts_max = 10

!   the maximum number of function evaluations made

       INTEGER :: max_evals = 10000

!   sampling strategy used, 1=uniform,2=Latin hyper-cube,3=2+1

       INTEGER :: sampling_strategy = 1

!   hyper-cube discretization (for sampling stategies 2 and 3)

       INTEGER :: hypercube_discretization = 2

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   the smallest value the objective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = wp ) :: obj_unbounded = - epsmch ** ( - 2 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   perform random-multistart as opposed to local minimize and probe

       LOGICAL :: random_multistart = .FALSE.

!   is the Hessian matrix of second derivatives available or is access only
!    via matrix-vector products?

       LOGICAL :: hessian_available = .TRUE.

!   if %space_critical true, every effort will be made to use as little
!    space as possible. This may result in longer computation time

       LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for TRB

       TYPE ( TRB_control_type ) :: TRB_control

!  control parameters for UGO

       TYPE ( UGO_control_type ) :: UGO_control

!  control parameters for LHS

       TYPE ( LHS_control_type ) :: LHS_control

     END TYPE BGO_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BGO_time_type

!  the total CPU time spent in the package

       REAL :: total = 0.0

!  the CPU time spent performing univariate global optimization

       REAL :: univariate_global = 0.0

!  the CPU time spent performing multivariate local optimization

       REAL :: multivariate_local = 0.0

!  the total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent performing univariate global optimization

       REAL ( KIND = wp ) :: clock_univariate_global = 0.0

!  the clock time spent performing multivariate local optimization

       REAL ( KIND = wp ) :: clock_multivariate_local = 0.0

     END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: BGO_inform_type

!  return status. See BGO_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of evaluations of the objection function

       INTEGER :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER :: g_eval = 0

!  the total number of evaluations of the Hessian of the objection function

       INTEGER :: h_eval = 0

!  the value of the objective function at the best estimate of the solution
!   determined by BGO_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm of the projected gradient of the objective function at the best
!   estimate of the solution determined by BGO_solve

       REAL ( KIND = wp ) :: norm_pg = HUGE( one )

!  timings (see above)

       TYPE ( BGO_time_type ) :: time

!  inform parameters for TRB

       TYPE ( TRB_inform_type ) :: TRB_inform

!  inform parameters for UGO

       TYPE ( UGO_inform_type ) :: UGO_inform

!  inform parameters for LHS

       TYPE ( LHS_inform_type ) :: LHS_inform

     END TYPE BGO_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: BGO_data_type
       INTEGER :: branch = 1
       INTEGER :: eval_status, out, error, start_print, stop_print
       INTEGER :: print_level, print_level_ugo, print_level_trb
       INTEGER :: jumpto, pass, attempts, lhs_count, trb_maxit, trb_maxit_large

       REAL :: time_start, time_record, time_now
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
       REAL ( KIND = wp ) :: alpha_l, alpha_u, alpha, phi, phi1, phi2, f_best
       REAL ( KIND = wp ) :: rhcd

       LOGICAL :: printi, printt, printm, printw, printd, printe
       LOGICAL :: print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printm, set_printw, set_printd
       LOGICAL :: present_eval_f, present_eval_g, present_eval_h, accurate
       LOGICAL :: present_eval_hprod, present_eval_shprod, present_eval_prec

       CHARACTER ( LEN = 1 ) :: negcur, bndry, perturb, hard
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_best
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_start
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_best
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HS

!      INTEGER, POINTER :: nnz_p_l, nnz_p_u, nnz_hp
!      LOGICAL, POINTER :: got_h
       INTEGER :: nnz_p_l, nnz_p_u, nnz_hp
       LOGICAL :: got_h
       INTEGER, POINTER, DIMENSION( : ) :: INDEX_nz_p
       INTEGER, POINTER, DIMENSION( : ) :: INDEX_nz_hp
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: P => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: HP => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: S => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: U => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: V => NULL( )
!      REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: P1 => NULL( )
!      REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: U1 => NULL( )
!      REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: V1 => NULL( )

!  copy of controls

       TYPE ( BGO_control_type ) :: control

!  random seed for RAND

       TYPE ( RAND_seed ) :: seed

!  data for TRB

       TYPE ( TRB_data_type ) :: TRB_data

!  data for UGO

       TYPE ( UGO_data_type ) :: UGO_data

!  data for LHS

       INTEGER :: lhs_seed = 1
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: X_lhs
       TYPE ( LHS_data_type ) :: LHS_data

     END TYPE BGO_data_type

     TYPE, PUBLIC :: BGO_full_data_type
       LOGICAL :: f_indexing
       TYPE ( BGO_data_type ) :: BGO_data
       TYPE ( BGO_control_type ) :: BGO_control
       TYPE ( BGO_inform_type ) :: BGO_inform
       TYPE ( NLPT_problem_type ) :: nlp
       TYPE ( NLPT_userdata_type ) :: userdata
     END TYPE BGO_full_data_type

   CONTAINS

!-*-*-  G A L A H A D -  B G O _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE BGO_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for BGO controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( BGO_control_type ), INTENT( OUT ) :: control
     TYPE ( BGO_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  Initalize random number seed

      CALL RAND_initialize( data%seed )

!  initalize TRB components

     CALL TRB_initialize( data%TRB_data, control%TRB_control,                  &
                          inform%TRB_inform )
     control%TRB_control%prefix = '" - TRB:"                     '

!  initalize UGO components

     CALL UGO_initialize( data%UGO_data, control%UGO_control,                  &
                           inform%UGO_inform )
     control%UGO_control%prefix = '" - UGO:"                     '

!  initalize LHS components

     CALL LHS_initialize( data%LHS_data, control%LHS_control,                  &
                           inform%LHS_inform )
     control%LHS_control%prefix = '" - LHS:"                     '

!  initial private data. Set branch for initial entry

     data%branch = 10

     RETURN

!  End of subroutine BGO_initialize

     END SUBROUTINE BGO_initialize

!- G A L A H A D -  B G O _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE BGO_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for BGO controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BGO_control_type ), INTENT( OUT ) :: control
     TYPE ( BGO_inform_type ), INTENT( OUT ) :: inform

     CALL BGO_initialize( data%bgo_data, control, inform )

     RETURN

!  End of subroutine BGO_full_initialize

     END SUBROUTINE BGO_full_initialize

!-*-*-*-*-   B G O _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE BGO_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by BGO_initialize could (roughly)
!  have been set as:

! BEGIN BGO SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     1
!  maximum-current-random-searches                 10
!  maximum-number-of-evaluations                   10000
!  sampling-strategy                               1
!  hypercube-discretization                        2
!  alive-device                                    40
!  infinity-value                                  1.0D+19
!  minimum-objective-before-unbounded              -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  random-multistart                               no
!  hessian-available                               yes
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END BGO SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: attempts_max = print_level + 1
     INTEGER, PARAMETER :: max_evals = attempts_max + 1
     INTEGER, PARAMETER :: sampling_strategy = max_evals + 1
     INTEGER, PARAMETER :: hypercube_discretization = sampling_strategy + 1
     INTEGER, PARAMETER :: alive_unit = hypercube_discretization + 1
     INTEGER, PARAMETER :: infinity = alive_unit + 1
     INTEGER, PARAMETER :: obj_unbounded = infinity + 1
     INTEGER, PARAMETER :: cpu_time_limit = obj_unbounded + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: random_multistart = clock_time_limit + 1
     INTEGER, PARAMETER :: hessian_available = random_multistart + 1
     INTEGER, PARAMETER :: space_critical = hessian_available + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'BGO '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( attempts_max )%keyword = 'maximum-current-random-searches'
     spec( max_evals )%keyword = 'maximum-number-of-evaluations'
     spec( sampling_strategy )%keyword = 'sampling-strategy'
     spec( hypercube_discretization )%keyword = 'hypercube-discretization'
     spec( alive_unit )%keyword = 'alive-device'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words


     spec( random_multistart )%keyword = 'random-multistart'
     spec( hessian_available )%keyword = 'hessian-available'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( alive_file )%keyword = 'alive-filename'
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
     CALL SPECFILE_assign_value( spec( attempts_max ),                         &
                                 control%attempts_max,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_evals ),                            &
                                 control%max_evals,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sampling_strategy ),                    &
                                 control%sampling_strategy,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( hypercube_discretization ),             &
                                 control%hypercube_discretization,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( obj_unbounded ),                        &
                                 control%obj_unbounded,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( random_multistart ),                    &
                                 control%random_multistart,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( hessian_available ),                    &
                                 control%hessian_available,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( alive_file ),                           &
                                 control%alive_file,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  read the controls for the sub-problem solvers and preconditioner

     IF ( PRESENT( alt_specname ) ) THEN
       CALL TRB_read_specfile( control%TRB_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-TRB' )
       CALL UGO_read_specfile( control%UGO_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-UGO' )
       CALL LHS_read_specfile( control%LHS_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-LHS' )
     ELSE
       CALL TRB_read_specfile( control%TRB_control, device )
       CALL UGO_read_specfile( control%UGO_control, device )
       CALL LHS_read_specfile( control%LHS_control, device )
     END IF

     RETURN

!  End of subroutine BGO_read_specfile

     END SUBROUTINE BGO_read_specfile

!-*-*-*-  G A L A H A D -  B G O _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE BGO_solve( nlp, control, inform, data, userdata, eval_F,       &
                           eval_G, eval_H, eval_HPROD, eval_SHPROD, eval_PREC )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  BGO_solve, a method for finding a global minimizer of a given
!    function where the variables are constrained to lie in a "box"

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_BGO.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_BGO_double
!
! nlp is a scalar variable of type NLPT_problem_type that is used to
!  hold data about the objective function. Relevant components are
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  H is scalar variable of type SMT_TYPE that holds the Hessian matrix H. The
!   following components are used here:
!
!   H%type is an allocatable array of rank one and type default character, that
!    is used to indicate the storage scheme used. If the dense storage scheme
!    is used, the first five components of H%type must contain the string DENSE.
!    For the sparse co-ordinate scheme, the first ten components of H%type must
!    contain the string COORDINATE, for the sparse row-wise storage scheme, the
!    first fourteen components of H%type must contain the string SPARSE_BY_ROWS,
!    and for the diagonal storage scheme, the first eight components of H%type
!    must contain the string DIAGONAL.
!
!    For convenience, the procedure SMT_put may be used to allocate sufficient
!    space and insert the required keyword into H%type. For example, if nlp is
!    of derived type packagename_problem_type and involves a Hessian we wish to
!    store using the co-ordinate scheme, we may simply
!
!         CALL SMT_put( nlp%H%type, 'COORDINATE', stat )
!
!    See the documentation for the galahad package SMT for further details on
!    the use of SMT_put.

!   H%ne is a scalar variable of type default integer, that holds the number of
!    entries in the  lower triangular part of H in the sparse co-ordinate
!    storage scheme. It need not be set for any of the other three schemes.
!
!   H%val is a rank-one allocatable array of type default real, that holds
!    the values of the entries of the  lower triangular part of the Hessian
!    matrix H in any of the available storage schemes.
!
!   H%row is a rank-one allocatable array of type default integer, that holds
!    the row indices of the  lower triangular part of H in the sparse
!    co-ordinate storage scheme. It need not be allocated for any of the other
!    three schemes.
!
!   H%col is a rank-one allocatable array variable of type default integer,
!    that holds the column indices of the  lower triangular part of H in either
!    the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!    be allocated when the dense or diagonal storage schemes are used.
!
!   H%ptr is a rank-one allocatable array of dimension n+1 and type default
!    integer, that holds the starting position of  each row of the  lower
!    triangular part of H, as well as the total number of entries plus one,
!    in the sparse row-wise storage scheme. It need not be allocated when the
!    other schemes are used.
!
!  G is a rank-one allocatable array of dimension n and type default real,
!   that holds the gradient g of the objective function. The j-th component of
!   G, j = 1,  ... ,  n, contains g_j.
!
!  f is a scalar variable of type default real, that holds the value of
!   the objective function.
!
!  X is a rank-one allocatable array of dimension n and type default real, that
!   holds the values x of the optimization variables. The j-th component of
!   X, j = 1, ... , n, contains x_j.
!
!  X_l is a rank-one allocatable array of dimension n and type default real,
!   that holds the values x_l of the lower bounds on the optimization
!   variables x. The j-th component of X_l, j = 1, ... , n, contains (x_l)j.
!
!  X_u is a rank-one allocatable array of dimension n and type default real,
!   that holds the values x_u of the upper bounds on the optimization
!   variables x. The j-th component of X_u, j = 1, ... , n, contains (x_u)j.
!
!  pname is a scalar variable of type default character and length 10, which
!   contains the ``name'' of the problem for printing. The default ``empty''
!   string is provided.
!
!  VNAMES is a rank-one allocatable array of dimension n and type default
!   character and length 10, whose j-th entry contains the ``name'' of the j-th
!   variable for printing. This is only used  if ``debug''printing
!   control%print_level > 4) is requested, and will be ignored if the array is
!   not allocated.
!
! control is a scalar variable of type BGO_control_type. See BGO_initialize
!  for details
!
! inform is a scalar variable of type BGO_inform_type. On initial entry,
!  inform%status should be set to 1. On exit, the following components will
!  have been set:
!
!  status is a scalar variable of type default integer, that gives
!   the exit status from the package. Possible values are:
!
!     0. The run was succesful
!
!    -1. An allocation error occurred. A message indicating the offending
!        array is written on unit control%error, and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -2. A deallocation error occurred.  A message indicating the offending
!        array is written on unit control%error and the returned allocation
!        status and a string containing the name of the offending array
!        are held in inform%alloc_status and inform%bad_alloc respectively.
!    -3. The restriction nlp%n > 0 or requirement that prob%H_type contains
!        its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS'
!          or 'DIAGONAL' has been violated.
!    -4. One or more of the simple bound restrictions (x_l)_i <= (x_u)_i
!        is violated.
!    -7. The objective function appears to be unbounded from below
!    -9. The analysis phase of the factorization failed; the return status
!        from the factorization package is given in the component
!        inform%factor_status
!   -10. The factorization failed; the return status from the factorization
!        package is given in the component inform%factor_status.
!   -11. The solution of a set of linear equations using factors from the
!        factorization package failed; the return status from the factorization
!        package is given in the component inform%factor_status.
!   -16. The problem is so ill-conditioned that further progress is impossible.
!   -18. Too many iterations have been performed. This may happen if
!        control%maxit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -19. The CPU time limit has been reached. This may happen if
!        control%cpu_time_limit is too small, but may also be symptomatic of
!        a badly scaled problem.
!   -40. The user has forced termination of solver by removing the file named
!        control%alive_file from unit unit control%alive_unit.
!
!     2. The user should compute the objective function value f(x) at the point
!        x indicated in nlp%X and then re-enter the subroutine. The required
!        value should be set in nlp%f, and data%eval_status should be set to 0.
!        If the user is unable to evaluate f(x) - for instance, if the function
!        is undefined at x - the user need not set nlp%f, but should then set
!        data%eval_status to a non-zero value.
!     3. The user should compute the gradient of the objective function
!        nabla_x f(x) at the point x indicated in nlp%X  and then re-enter the
!        subroutine. The value of the i-th component of the gradient should be
!        set in nlp%G(i), for i = 1, ..., n and data%eval_status should be set
!        to 0. If the user is unable to evaluate a component of nabla_x f(x)
!        - for instance if a component of the gradient is undefined at x - the
!        user need not set nlp%G, but should then set data%eval_status to a
!        non-zero value.
!     4. The user should compute the Hessian of the objective function
!        nabla_xx f(x) at the point x indicated in nlp%X and then re-enter the
!        subroutine. The value l-th component of the Hessian stored according to
!        the scheme input in the remainder of nlp%H should be set in
!        nlp%H%val(l), for l = 1, ..., nlp%H%ne and data%eval_status should be
!        set to 0. If the user is unable to evaluate a component of
!        nabla_xx f(x) - for instance, if a component of the Hessian is
!        undefined at x - the user need not set nlp%H%val, but should then set
!        data%eval_status to a non-zero value.
!     5. The user should compute the product nabla_xx f(x)v of the Hessian
!        of the objective function nabla_xx f(x) at the point x indicated in
!        nlp%X with the vector v and add the result to the vector u and then
!        re-enter the subroutine. The vectors u and v are given in data%U and
!        data%V respectively, the resulting vector u + nabla_xx f(x)v should be
!        set in data%U and  data%eval_status should be set to 0. If the user is
!        unable to evaluate the product - for instance, if a component of the
!        Hessian is undefined at x - the user need not alter data%U, but
!        should then set data%eval_status to a non-zero value.
!     6. The user should compute the product u = P(x)v of their preconditioner
!        P(x) at the point x indicated in nlp%X with the vector v and then
!        re-enter the subroutine. The vectors v is given in data%V, the
!        resulting vector u = P(x)v should be set in data%U and
!        data%eval_status should be set to 0. If the user is unable to evaluate
!        the product - for instance, if a component of the preconditioner is
!        undefined at x - the user need not set data%U, but should then set
!        data%eval_status to a non-zero value.
!     7. The user should compute the product hp = nabla_xx f(x)p of the Hessian
!        of the objective function nabla_xx f(x) at the point x indicated in
!        nlp%X with the *sparse* vector p and then re-enter the subroutine.
!        The nonzeros of p are stored in
!          data%P(data%INDEX_nz_p(data%nnz_p_l:data%nnz_p_u))
!        while the nonzeros of hp should be returned in
!          data%HP(data%INDEX_nz_hp(1:data%nnz_hp));
!        the user must set data%nnz_hp and data%INDEX_nz_hp accordingly,
!        and set data%eval_status to 0. If the user is unable to evaluate the
!        product - for instance, if a component of the Hessian is undefined
!        at x - the user need not alter data%HP, but should then set
!        data%eval_status to a non-zero value.
!    23. The user should follow the instructions for 2 AND 3 above before
!        returning.
!    25. The user should follow the instructions for 2 AND 5 above before
!        returning.
!    35. The user should follow the instructions for 3 AND 5 above before
!        returning.
!   235. The user should follow the instructions for 2, 3 AND 5 above before
!        returning.
!
!  alloc_status is a scalar variable of type default integer, that gives
!   the status of the last attempted array allocation or deallocation.
!   This will be 0 if status = 0.
!
!  bad_alloc is a scalar variable of type default character
!   and length 80, that  gives the name of the last internal array
!   for which there were allocation or deallocation errors.
!   This will be the null string if status = 0.
!
!  iter is a scalar variable of type default integer, that holds the
!   number of iterations performed.
!
!  factorization_status is a scalar variable of type default integer, that
!   gives the return status from the matrix factorization.
!
!  factorization_integer is a scalar variable of type default integer,
!   that gives the amount of integer storage used for the matrix factorization.
!
!  factorization_real is a scalar variable of type default integer,
!   that gives the amount of real storage used for the matrix factorization.
!
!  f_eval is a scalar variable of type default integer, that gives the
!   total number of objective function evaluations performed.
!
!  g_eval is a scalar variable of type default integer, that gives the
!   total number of objective function gradient evaluations performed.
!
!  h_eval is a scalar variable of type default integer, that gives the
!   total number of objective function Hessian evaluations performed.
!
!  obj is a scalar variable of type default real, that holds the
!   value of the objective function at the best estimate of the solution found.
!
!  norm_pg is a scalar variable of type default real, that holds the value of
!   the norm of the projected gradient of the objective function at the best
!   estimate of the solution found.
!
!  time is a scalar variable of type BGO_time_type whose components are used to
!   hold elapsed CPU and clock times for the various parts of the calculation.
!   Components are:
!
!    total is a scalar variable of type default real, that gives
!     the total CPU time spent in the package.
!
!    preprocess is a scalar variable of type default real, that gives the
!      CPU time spent reordering the problem to standard form prior to solution.
!
!    analyse is a scalar variable of type default real, that gives
!      the CPU time spent analysing required matrices prior to factorization.
!
!    factorize is a scalar variable of type default real, that gives
!      the CPU time spent factorizing the required matrices.
!
!    solve is a scalar variable of type default real, that gives
!     the CPU time spent using the factors to solve relevant linear equations.
!
!    clock_total is a scalar variable of type default real, that gives
!     the total clock time spent in the package.
!
!    clock_preprocess is a scalar variable of type default real, that gives
!      the clock time spent reordering the problem to standard form prior
!      to solution.
!
!    clock_analyse is a scalar variable of type default real, that gives
!      the clock time spent analysing required matrices prior to factorization.
!
!    clock_factorize is a scalar variable of type default real, that gives
!      the clock time spent factorizing the required matrices.
!
!    clock_solve is a scalar variable of type default real, that gives
!     the clock time spent using the factors to solve relevant linear equations.
!
!  data is a scalar variable of type BGO_data_type used for internal data.
!
!  userdata is a scalar variable of type NLPT_userdata_type which may be used
!   to pass user data to and from the eval_* subroutines (see below)
!   Available coomponents which may be allocated as required are:
!
!    integer is a rank-one allocatable array of type default integer.
!    real is a rank-one allocatable array of type default real
!    complex is a rank-one allocatable array of type default comple.
!    character is a rank-one allocatable array of type default character.
!    logical is a rank-one allocatable array of type default logical.
!    integer_pointer is a rank-one pointer array of type default integer.
!    real_pointer is a rank-one pointer array of type default  real
!    complex_pointer is a rank-one pointer array of type default complex.
!    character_pointer is a rank-one pointer array of type default character.
!    logical_pointer is a rank-one pointer array of type default logical.
!
!  eval_F is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The value of the objective
!   function f(x) evaluated at x=X must be returned in f, and the status
!   variable set to 0. If the evaluation is impossible at X, status should
!   be set to a nonzero value. If eval_F is not present, BGO_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in
!   G, and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_G is not present,
!   BGO_solve will return to the user with inform%status = 3 each time an
!   evaluation is required.
!
!  eval_H is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) of the objective function evaluated at x=X must be returned in
!   H in the same order as presented in nlp%H, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_H is not present, BGO_solve will return to the user with
!   inform%status = 4 each time an evaluation is required.
!
!  eval_HPROD is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The sum u + nabla_xx f(x) v of the
!   product of the Hessian nabla_xx f(x) of the objective function evaluated
!   at x=X with the vector v=V and the vector u=U must be returned in U, and the
!   status variable set to 0. If the evaluation is impossible at X, status
!   should be set to a nonzero value. If eval_HPROD is not present, BGO_solve
!   will return to the user with inform%status = 5 each time an evaluation is
!   required. The Hessian has already been evaluated or used at x=X if got_h
!   is .TRUE.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC
!   is not present, BGO_solve will return to the user with inform%status = 6
!   each time an evaluation is required.
!
!  eval_SHPROD is an optional subroutine which if present must have the
!   arguments given below (see the interface blocks). The product
!   u = nabla_xx f(x) v of the Hessian nabla_xx f(x) of the objective function
!   evaluated at x=X with the sparse vector v=V must be returned in U, and the
!   status variable set to 0. Only the components INDEX_nz_v(1:nnz_v) of
!   V are nonzero, and the remaining components may not have been be set. On
!   exit, the user must indicate the nnz_u indices of u that are nonzero in
!   INDEX_nz_u(1:nnz_u), and only these components of U need be set. If the
!   evaluation is impossible at X, status should be set to a nonzero value.
!   If eval_SHPROD is not present, BGO_solve will return to the user with
!   inform%status = 7 each time a sparse product is required. The Hessian has
!   already been evaluated or used at x=X if got_h is .TRUE.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( BGO_control_type ), INTENT( IN ) :: control
     TYPE ( BGO_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( BGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     OPTIONAL :: eval_F, eval_G, eval_H, eval_HPROD, eval_SHPROD, eval_PREC

!----------------------------------
!   I n t e r f a c e   B l o c k s
!----------------------------------

     INTERFACE
       SUBROUTINE eval_F( status, X, userdata, f )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), INTENT( OUT ) :: f
       REAL ( KIND = wp ), DIMENSION( : ),INTENT( IN ) :: X
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_F
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_G( status, X, userdata, G )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_G
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_H( status, X, userdata, Hval )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: Hval
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_H
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_HPROD( status, X, userdata, U, V, got_h )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_HPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_SHPROD( status, X, userdata, nnz_v, INDEX_nz_v, V,      &
                               nnz_u, INDEX_nz_u, U, got_h )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( IN ) :: nnz_v
       INTEGER, INTENT( OUT ) :: nnz_u
       INTEGER, INTENT( OUT ) :: status
       INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
       INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       LOGICAL, OPTIONAL, INTENT( IN ) :: got_h
       END SUBROUTINE eval_SHPROD
     END INTERFACE

     INTERFACE
       SUBROUTINE eval_PREC( status, X, userdata, U, V )
       USE GALAHAD_NLPT_double, ONLY: NLPT_userdata_type
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( OUT ) :: status
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
       REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V, X
       TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
       END SUBROUTINE eval_PREC
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, ic, ir, l, st
     REAL ( KIND = wp ) :: p, x, x_l, x_u, d, f_best
     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name
     TYPE ( BGO_inform_type ) :: inform_initialize

!  prefix for all output

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  branch to different sections of the code depending on input status

     IF ( inform%status < 1 ) THEN
       CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
       GO TO 990
     END IF
     IF ( inform%status == 1 ) THEN
       data%branch = 10
     ELSE IF ( inform%status == 11 ) THEN
       data%branch = 20
     END IF

     SELECT CASE ( data%branch )
     CASE ( 10 )  ! initialization
       GO TO 10
     CASE ( 20 )  ! re-entry without initialization
       GO TO 20
     CASE ( 120 )  ! function and derivative evaluations
       GO TO 120
     CASE ( 230 )  ! function and derivative evaluations
       GO TO 230
     CASE ( 420 )  ! function and derivative evaluations
       GO TO 420
     CASE ( 550 )  ! function and derivative evaluations
       GO TO 550
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )
     data%out = control%out

!  initialize components of inform 

     inform = inform_initialize

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions
       GO TO 990
     END IF

!  check that the simple bounds are consistent

     DO i = 1, nlp%n
       IF ( nlp%X_l( i ) > nlp%X_u( i ) ) THEN
         inform%status = GALAHAD_error_bad_bounds
         GO TO 990
       END IF
     END DO

!  record whether external evaluation procudures are presnt

     data%present_eval_f = PRESENT( eval_F )
     data%present_eval_g = PRESENT( eval_G )
     data%present_eval_h = PRESENT( eval_H )
     data%present_eval_hprod = PRESENT( eval_HPROD )
     data%present_eval_shprod = PRESENT( eval_SHPROD )
     data%present_eval_prec = PRESENT( eval_PREC )

!  make a local copy of the control parameters

     data%control = control

!  adjust parameters to make sure that they lie within permitted bounds

     IF ( nlp%n > 1 ) THEN
       data%control%sampling_strategy                                          &
         = MIN( 3, MAX( control%sampling_strategy, 1 ) )
       data%control%hypercube_discretization                                   &
         = MAX( control%hypercube_discretization, 1 )

!  allocate sufficient space for the problem

       array_name = 'bgo: data%X_start'
       CALL SPACE_resize_array( nlp%n, data%X_start, inform%status,            &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'bgo: data%X_best'
       CALL SPACE_resize_array( nlp%n, data%X_best, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'bgo: data%G_best'
       CALL SPACE_resize_array( nlp%n, data%G_best, inform%status,             &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'bgo: data%D'
       CALL SPACE_resize_array( nlp%n, data%D, inform%status,                  &
              inform%alloc_status, array_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       IF ( data%control%sampling_strategy > 1 ) THEN
         data%rhcd = REAL( data%control%hypercube_discretization, KIND = wp )
         array_name = 'bgo: data%X_lhs'
         CALL SPACE_resize_array( nlp%n, data%control%hypercube_discretization,&
                data%X_lhs, inform%status,                                     &
                inform%alloc_status, array_name = array_name,                  &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) GO TO 980
       END IF
     ELSE
       array_name = 'bgo: data%P'
       CALL SPACE_resize_pointer( 1, data%P, inform%status,                    &
              inform%alloc_status, point_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'bgo: data%U'
       CALL SPACE_resize_pointer( 1, data%U, inform%status,                    &
              inform%alloc_status, point_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980

       array_name = 'bgo: data%V'
       CALL SPACE_resize_pointer( 1, data%V, inform%status,                    &
              inform%alloc_status, point_name = array_name,                    &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) GO TO 980
     END IF

!  control the output printing

     data%out = data%control%out
     data%error = data%control%error
     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printw with a few vectors

     data%printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debug printing

     data%printd = data%out > 0 .AND. data%control%print_level > 10

!  error printing

     data%printe = data%error > 0

!  create a file which the user may subsequently remove to cause
!  immediate termination of a run

     IF ( control%alive_unit > 0 ) THEN
      INQUIRE( FILE = control%alive_file, EXIST = alive )
      IF ( .NOT. alive ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'NEW' )
         REWIND control%alive_unit
         WRITE( control%alive_unit, "( ' GALAHAD rampages onwards ' )" )
         CLOSE( control%alive_unit )
       END IF
     END IF

!  initial re-entry

  20 CONTINUE
     data%pass = 1

!  check whether multistart is preferred

     IF ( data%control%random_multistart ) GO TO 500

!  check for special case for which n = 1

     IF ( nlp%n == 1 ) GO TO 400

!  ============================================================================
!  Start of main local minimization and probe iteration (n > 1)
!  ============================================================================

 100 CONTINUE

!  ============================================================================
!  1. Find a local minimizer x_k of f(x) from the current starting point xs_k
!  ============================================================================

!CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
!write(6,*) ' f ', nlp%f
!write(6,*) ' x ', nlp%X

!  -----------------------------------------------------------------
!  implicit loop to perform the local bound-constrained minimization
!  -----------------------------------------------------------------

       inform%TRB_inform%status = 1
       inform%TRB_inform%iter = 0
       inform%TRB_inform%cg_iter = 0
       inform%TRB_inform%f_eval = 0
       inform%TRB_inform%g_eval = 0
       inform%TRB_inform%h_eval = 0
       data%control%TRB_control%error = 0
       data%control%TRB_control%hessian_available = control%hessian_available
       data%control%TRB_control%maxit = MIN( control%TRB_control%maxit,        &
           control%max_evals - inform%f_eval )

       CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
 110   CONTINUE

!  call the bound-constrained local minimizer

!write(6,*) 'in status ', inform%TRB_inform%status,  inform%TRB_inform%iter
         CALL TRB_solve( nlp, data%control%TRB_control, inform%TRB_inform,     &
                         data%TRB_data, userdata, eval_F = eval_F,             &
                         eval_G = eval_G, eval_H = eval_H,                     &
                         eval_HPROD = eval_HPROD, eval_SHPROD = eval_SHPROD,   &
                         eval_PREC = eval_PREC )
!write(6,*) 'out status ', inform%TRB_inform%status

!  obtain further function information if required

         SELECT CASE ( inform%TRB_inform%status )

!  obtain the objective function value

         CASE ( 2 )
           IF ( data%present_eval_f ) THEN
             CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          inform%TRB_inform%obj )
!write(6,*) ' f ', nlp%f
           ELSE
             data%branch = 120 ; inform%status = 2 ; RETURN
           END IF

!  obtain the gradient value

         CASE ( 3 )
           IF ( data%present_eval_g ) THEN
             CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%G( : nlp%n ) )
           ELSE
             data%branch = 120 ; inform%status = 3 ; RETURN
           END IF

!  obtain the Hessian value

         CASE ( 4 )
           IF ( data%present_eval_h ) THEN
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          userdata, nlp%H%val( : nlp%H%ne ) )
           ELSE
             data%branch = 120 ; inform%status = 4 ; RETURN
           END IF

!  obtain a Hessian-vector product

         CASE ( 5 )
           data%got_h = data%TRB_data%got_h
           data%U => data%TRB_data%U
           IF ( data%present_eval_hprod ) THEN
             CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),              &
                              userdata, data%U( : nlp%n ),                     &
                              data%TRB_data%S( : nlp%n ), got_h = data%got_h )
           ELSE
             data%V => data%TRB_data%S
             data%branch = 120 ; inform%status = 5 ; RETURN
           END IF

!  obtain a preconditioned vector product

         CASE ( 6 )
           data%U => data%TRB_data%U
           data%V => data%TRB_data%V
           IF ( data%present_eval_prec ) THEN
             CALL eval_PREC( data%eval_status, nlp%X( : nlp%n ), userdata,     &
                             data%U( : nlp%n ), data%V( : nlp%n ) )
           ELSE
             data%branch = 120 ; inform%status = 6 ; RETURN
           END IF

!  obtain a Hessian-sparse-vector product

         CASE ( 7 )
           data%got_h = data%TRB_data%got_h
           data%nnz_p_u = data%TRB_data%nnz_p_u
           data%nnz_p_l = data%TRB_data%nnz_p_l
           data%INDEX_nz_p => data%TRB_data%INDEX_nz_p
           data%P => data%TRB_data%P
           data%INDEX_nz_hp => data%TRB_data%INDEX_nz_hp
           data%HP => data%TRB_data%HP
           IF ( data%present_eval_shprod ) THEN
             CALL eval_SHPROD( data%eval_status, nlp%X( : nlp%n ),             &
                         userdata, data%nnz_p_u - data%nnz_p_l + 1,            &
                         data%INDEX_nz_p( data%nnz_p_l : data%nnz_p_u ),       &
                         data%P, data%nnz_hp, data%INDEX_nz_hp, data%HP,       &
                         got_h = data%got_h )
           ELSE
             data%branch = 120 ; inform%status = 7 ; RETURN
           END IF

!  terminal exits

         CASE DEFAULT
           GO TO 190
         END SELECT

!  return from reverse communication

 120     CONTINUE
         data%TRB_data%eval_status = data%eval_status

         SELECT CASE ( inform%TRB_inform%status )
         CASE ( 2 )
         CASE ( 3 )
         CASE ( 4 )
         CASE ( 5 )
         CASE ( 6 )
         CASE ( 7 )
           data%TRB_data%nnz_hp = data%nnz_hp
         END SELECT

!  ------------------------------------------------
!  end of local bound-constrained minimization loop
!  ------------------------------------------------

         GO TO 110

 190   CONTINUE

!  record the time taken in the local minimization

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       inform%time%multivariate_local =                                        &
         inform%time%multivariate_local +                                      &
           data%time_now - data%time_record
       inform%time%clock_multivariate_local =                                  &
         inform%time%clock_multivariate_local +                                &
           data%clock_now - data%clock_record

!  record details about the critical point found

       inform%f_eval = inform%f_eval + inform%TRB_inform%f_eval
       inform%g_eval = inform%g_eval + inform%TRB_inform%g_eval
       inform%h_eval = inform%h_eval + inform%TRB_inform%h_eval
       inform%obj = inform%TRB_inform%obj
       inform%norm_pg = inform%TRB_inform%norm_pg
       data%f_best = inform%obj
       data%X_best = nlp%X
       data%G_best = nlp%G
       data%P => data%TRB_data%P
!write(6,*) ' data%P associated '
       IF ( inform%TRB_inform%status < 0 .AND. data%printe ) THEN
         IF ( inform%TRB_inform%status /= GALAHAD_error_max_iterations )       &
           WRITE( data%error, "( ' Help! exit from TRB status = ', I0 )" )     &
             inform%TRB_inform%status
       END IF

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

!  control printing

       data%print_iteration_header = data%printt .OR. data%TRB_data%printi

!  print one-line summary

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2010 )
           data%print_1st_header = .FALSE.
         END IF
         WRITE( data%out, 2020 ) prefix, data%pass, 'T', 0, inform%obj,        &
           inform%norm_pg, inform%f_eval, inform%g_eval, data%clock_now
       END IF
!write(6,"( ' x ', /, ( 5ES12.4 ) )") data%X_best( : nlp%n )

!  ============================================================================
!  2. Find a global minimizer of f(x_k + alpha p) along random directions p
!  ============================================================================

       data%X_start = data%X_best
!      data%X_start = half * ( nlp%X_l + nlp%X_u )

!  --------------------------------------------------
!  implicit loop to generate random search directions
!  --------------------------------------------------

       data%attempts = 1 ; data%lhs_count = 0

!  loop over at most attempts_max random search vectors

 210   CONTINUE

!  compute a random vector p_i in [-1,1] when x_l_i < x_i < x_u_i,
!                              in [0,1] when x_i = x_l_i and
!                              in [-1,0] when x_i = x_u_i

         IF ( data%control%sampling_strategy == 1 ) THEN
           DO i = 1, nlp%n
             x = data%X_start( i ) ; x_l = nlp%X_l( i ) ; x_u = nlp%X_u( i )
             st = 0
             IF ( x <= x_l * ( one + SIGN( epsmch, x_l ) ) ) st = 1
             IF ( x >= x_u * ( one - SIGN( epsmch, x_u ) ) ) st = 2
             IF ( x_u * ( one - SIGN( epsmch, x_u ) ) <=                       &
                  x_l * ( one + SIGN( epsmch, x_l ) ) ) st = 3
             IF ( st == 0 ) THEN
               CALL RAND_random_real( data%seed, .FALSE., p )
             ELSE IF ( st == 3 ) THEN
               p = zero
             ELSE
  !            CALL RAND_random_real( data%seed, .TRUE., p )
  !            IF ( st == 2 ) p = - p
               CALL RAND_random_real( data%seed, .FALSE., p )
               IF ( st == 1 .AND. p < zero ) p = zero
               IF ( st == 2 .AND. p > zero ) p = zero
             END IF
             data%P( i ) = p * ( x_u - x_l )
           END DO

!  pick the points by partitioning each dimension of the box into
!  hypercube_discretization segments, and then selecting subboxes
!  randomly within this hypercube using Latin hypercube sampling

         ELSE
           data%lhs_count = data%lhs_count + 1
           IF ( data%lhs_count > data%control%hypercube_discretization )       &
             data%lhs_count = 1
           IF ( data%lhs_count == 1 ) THEN
             CALL LHS_ihs( nlp%n, data%control%hypercube_discretization,       &
                           data%lhs_seed, data%X_lhs,                          &
                           data%control%LHS_control,                           &
                           inform%LHS_inform, data%LHS_data )
             IF ( inform%LHS_inform%status /= 0 ) THEN
               inform%status = inform%LHS_inform%status
               GO TO 980
             END IF
           END IF

!  pick p to point to the middle of the assigned sub-hypercube

           DO i = 1, nlp%n
             x = data%X_start( i ) ; x_l = nlp%X_l( i ) ; x_u = nlp%X_u( i )
             d = ( x_u - x_l ) / data%rhcd
             p = ( REAL( data%X_lhs( i, data%lhs_count ), KIND = wp ) - half ) &
                   * d + x_l - x
             data%P( i ) = p

! if required, randonly perturb the point within this sub-hypercube

             IF ( data%control%sampling_strategy == 3 ) THEN
               CALL RAND_random_real( data%seed, .FALSE., p )
               data%P( i ) = data%P( i ) + half * d * p
             END IF
           END DO
         END IF

!  find the smallest and largest values alpha_l, alpha_u of alpha for which
!    x_l <= x_k + alpha p <= x_u

!        WRITE( data%out,                                   &
         IF ( data%printw ) WRITE( data%out,                                   &
           "( '    x_l            x         x_u            p' )" )
         data%alpha_l = - infinity ; data%alpha_u = infinity
         DO i = 1, nlp%n
           x = data%X_start( i ) ; x_l = nlp%X_l( i ) ; x_u = nlp%X_u( i )
           p = data%P( i )
           IF ( p > zero ) THEN
             data%alpha_l = MAX( data%alpha_l, ( x_l - x ) / p )
             data%alpha_u = MIN( data%alpha_u, ( x_u - x ) / p )
           ELSE IF ( p < zero ) THEN
             data%alpha_l = MAX( data%alpha_l, ( x_u - x ) / p )
             data%alpha_u = MIN( data%alpha_u, ( x_l - x ) / p )
           END IF
           IF ( data%printw ) WRITE( data%out,"( 4ES12.4 )" ) x_l, x, x_u, p
!          WRITE( data%out,"( 4ES12.4 )" ) x_l, x, x_u, p
         END DO
!        WRITE( data%out,"( ' alpha_l, alpha_u ',           &
         IF ( data%printw ) WRITE( data%out,"( ' alpha_l, alpha_u ',           &
        &                          2ES12.4 )" )  data%alpha_l,  data%alpha_u

!        write(63,"(2I6)" ) data%X_lhs( : , data%lhs_count )
!        write(64,"(2F10.6)" ) data%P

!  check that the random vector is not zero

         IF ( data%alpha_u == infinity ) GO TO 210

!  -----------------------------------------------------------
!  implicit loop to perform the global univariate minimization
!  -----------------------------------------------------------

         inform%UGO_inform%status = 1
         data%control%UGO_control%obj_sufficient = data%f_best - 0.0001_wp
         data%control%UGO_control%maxit = MIN( control%UGO_control%maxit,      &
           control%max_evals - inform%f_eval )

!write(6,"( ' suff ', ES22.14 )" ) data%control%UGO_control%obj_sufficient
         CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
 220     CONTINUE

!  find the global minimizer of phi(alpha) = f(x_k + alpha p ) within
!  the interval [alpha_l,alpha_u]

           IF ( data%alpha_l < zero .AND. data%alpha_u > zero ) THEN
             CALL UGO_solve( data%alpha_l, data%alpha_u, data%alpha,           &
                             data%phi, data%phi1, data%phi2,                   &
                             data%control%UGO_control, inform%UGO_inform,      &
                             data%UGO_data, userdata, x_extra = zero )
           ELSE
             CALL UGO_solve( data%alpha_l, data%alpha_u, data%alpha,           &
                             data%phi, data%phi1, data%phi2,                   &
                             data%control%UGO_control, inform%UGO_inform,      &
                             data%UGO_data, userdata )
           END IF
!write(6,*) ' ugo status ', inform%UGO_inform%status
!  evaluate phi(alpha) = f(x_k + alpha p) and its derivatives as required

           IF ( inform%UGO_inform%status >= 2 ) THEN
             nlp%X = data%X_start + data%alpha * data%P
             data%branch = 0

!  obtain the objective function value

             IF ( data%present_eval_f ) THEN
               CALL eval_F( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, nlp%f )
             ELSE
               data%branch = data%branch + 1
             END IF

!  obtain the gradient value

             IF ( inform%UGO_inform%status >= 3 ) THEN
               IF ( data%present_eval_g ) THEN
                 CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,    &
                              nlp%G( : nlp%n ) )
               ELSE
                 data%branch = data%branch + 2
               END IF

!  obtain a Hessian-vector product

               IF ( inform%UGO_inform%status >= 4 ) THEN
                 data%U => data%TRB_data%U
                 data%U = zero
                 IF ( data%present_eval_hprod ) THEN
                   CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),        &
                                    userdata, data%U( : nlp%n ),               &
                                    data%P( : nlp%n ) )
                 ELSE
                   data%got_h = .FALSE.
                   data%V => data%P
                   data%branch = data%branch + 5
                 END IF
               END IF
             END IF

!  reverse communication is required for at least f or one of its derivatives

             IF ( data%branch > 0 ) THEN
               SELECT CASE( data%branch )
               CASE ( 1 )
                 inform%status = 2
               CASE ( 2 )
                 inform%status = 3
               CASE ( 3 )
                 inform%status = 23
               CASE ( 5 )
                 inform%status = 5
               CASE ( 6 )
                 inform%status = 25
               CASE ( 7 )
                 inform%status = 35
               CASE ( 8 )
                 inform%status = 235
               END SELECT
               data%branch = 230 ; RETURN
             END IF
           END IF

!  return from reverse communication

 230       CONTINUE
           IF ( inform%UGO_inform%status >= 2 ) THEN
             data%phi = nlp%f
             IF ( inform%UGO_inform%status >= 3 ) THEN
               data%phi1 = DOT_PRODUCT( data%P, nlp%G )
               IF ( inform%UGO_inform%status >= 4 )                            &
                 data%phi2 = DOT_PRODUCT( data%P, data%U )
             END IF
             GO TO 220
           END IF

           IF ( data%printm )                                                  &
             WRITE( data%out, "( A, ' minimizer', ES12.4, ' in [', ES11.4,     &
          &     ',', ES10.4, '] has f =', ES12.4, ', st = ', I0 )" )           &
               prefix, data%alpha, data%alpha_l, data%alpha_u, data%phi,       &
               inform%UGO_inform%status

           IF ( inform%UGO_inform%status < 0 .AND. data%printe ) THEN
             IF ( inform%UGO_inform%status /= GALAHAD_error_max_iterations )   &
               WRITE( data%error, "( ' Help! exit from UGO status = ', I0 )" ) &
                 inform%UGO_inform%status
           END IF

!  record the time taken in the univariate global minimization

           CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
           inform%time%univariate_global =                                     &
             inform%time%univariate_global + data%time_now - data%time_record
           inform%time%clock_univariate_global =                               &
             inform%time%clock_univariate_global +                             &
               data%clock_now - data%clock_record

           inform%f_eval = inform%f_eval + inform%UGO_inform%f_eval
           inform%g_eval = inform%g_eval + inform%UGO_inform%g_eval
           inform%h_eval = inform%h_eval + inform%UGO_inform%h_eval

!   test to see whether a lower point has been found

!          IF ( data%phi < data%f_best ) THEN
           IF ( data%phi <= data%control%UGO_control%obj_sufficient ) THEN

!            WRITE( data%out, "( A, ' minimizer ', ES12.4, ' in [', ES12.4,    &
!         &     ',', ES12.4, '] has f = ', ES24.16, ', status = ', I0 )" )     &
!              prefix, data%alpha, data%alpha_l, data%alpha_u, data%phi,       &
!              inform%UGO_inform%status

!write(6,*) ' better!, alpha = ', data%alpha
             nlp%X = data%X_start + data%alpha * data%P

!CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
!write(6,*) ' better!', data%phi, nlp%f

             inform%obj = data%phi
!            inform%norm_pg = inform%TRB_inform%norm_pg
             data%f_best = inform%obj
             data%X_best = nlp%X
             inform%norm_pg = TWO_NORM( nlp%X -                                &
                TRB_projection( nlp%n, nlp%X - nlp%G, nlp%X_l, nlp%X_u ) )
             GO TO 290
           END IF

!  special exit case for univariate functions

           IF ( nlp%n == 1 ) THEN
             inform%obj = data%f_best
             nlp%X = data%X_best
             nlp%G = data%G_best
             inform%norm_pg = TWO_NORM( nlp%X -                                &
                TRB_projection( nlp%n, nlp%X - nlp%G, nlp%X_l, nlp%X_u ) )
             inform%status = GALAHAD_ok
             GO TO 900
           END IF

!  -------------------------------------------------
!  end of global univariate global minimization loop
!  -------------------------------------------------

         data%attempts = data%attempts + 1
         IF ( inform%f_eval > control%max_evals ) GO TO 290
         IF ( data%attempts <= data%control%attempts_max ) GO TO 210

!  ------------------------------------------------
!  end of loop to generate random search directions
!  ------------------------------------------------

 290   CONTINUE

!  ============================================================================
!  3. Record progress and test for convergence
!  ============================================================================

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

!  control printing

       data%print_iteration_header = data%printt .OR. data%UGO_data%printi

!  print one-line summary

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2010 )
           data%print_1st_header = .FALSE.
         END IF
         WRITE( data%out, 2020 ) prefix, data%pass, 'U', data%attempts,        &
           inform%obj, inform%norm_pg, inform%f_eval, inform%g_eval,           &
           data%clock_now
       END IF

!  record the best value found

       inform%obj = data%f_best
       nlp%X = data%X_best
       nlp%G = data%G_best
!write(6,*) ' x_best = ', nlp%X
!  debug printing for X and G

       IF ( data%printw ) THEN
         WRITE ( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2200 ) prefix, inform%f_eval, prefix,               &
           inform%g_eval, prefix, inform%h_eval, prefix,                       &
           inform%obj, prefix, inform%norm_pg
         WRITE ( data%out, 2210 ) prefix
         l = 2
         DO j = 1, 2
            IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, nlp%n )
            ELSE
               IF ( ic < nlp%n - l ) WRITE( data%out, 2240 ) prefix
               ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
            END IF
            IF ( ALLOCATED( nlp%vnames ) ) THEN
              DO i = ir, ic
                 WRITE( data%out, 2220 ) prefix, nlp%vnames( i ),              &
                   nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
              END DO
            ELSE
              DO i = ir, ic
                 WRITE( data%out, 2230 ) prefix, i,                            &
                   nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
              END DO
            END IF
         END DO
       END IF

       IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,     &
      &    F12.2,  ' seconds' )" ) prefix, data%clock_now
       IF ( ( data%control%cpu_time_limit >= zero .AND.                        &
              data%time_now > data%control%cpu_time_limit ) .OR.               &
            ( data%control%clock_time_limit >= zero .AND.                      &
              data%clock_now > data%control%clock_time_limit ) ) THEN
         inform%status = GALAHAD_error_cpu_limit ; GO TO 900
       END IF

!  check to see if the evaluation limit has been exceeded

       IF ( data%control%max_evals >= 0 .AND.                                  &
            inform%f_eval > data%control%max_evals ) THEN
         inform%status = GALAHAD_error_max_evaluations ; GO TO 900
       END IF

       IF ( data%attempts > data%control%attempts_max ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 900
       END IF

!  check to see if we are still "alive"

       IF ( data%control%alive_unit > 0 ) THEN
         INQUIRE( FILE = data%control%alive_file, EXIST = alive )
         IF ( .NOT. alive ) THEN
           inform%status = GALAHAD_error_alive ; GO TO 900
         END IF
       END IF

       data%pass = data%pass + 1
     GO TO 100

!  ============================================================================
!  Special case of main local minimization and probe iteration (n = 1)
!  ============================================================================

 400 CONTINUE

!  -----------------------------------------------------------
!  implicit loop to perform the global univariate minimization
!  -----------------------------------------------------------

     inform%UGO_inform%status = 1
     CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
 410 CONTINUE

!  find the global minimizer of the univariate function f(x) in the interval
!  [x_l,x_u]

       CALL UGO_solve( nlp%X_l( 1 ), nlp%X_u( 1 ), nlp%X( 1 ),                 &
                       data%phi, data%phi1, data%phi2,                         &
                       data%control%UGO_control, inform%UGO_inform,            &
                       data%UGO_data, userdata )

!  evaluate f and its derivatives as required

       IF ( inform%UGO_inform%status >= 2 ) THEN
         data%branch = 0

!  obtain the objective function value

         IF ( data%present_eval_f ) THEN
           CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
         ELSE
           data%branch = data%branch + 1
         END IF

!  obtain the gradient value

         IF ( inform%UGO_inform%status >= 3 ) THEN
           IF ( data%present_eval_g ) THEN
             CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%G( : nlp%n ) )
           ELSE
             data%branch = data%branch + 2
           END IF

!  obtain a Hessian-vector product

           IF ( inform%UGO_inform%status >= 4 ) THEN
!            data%U => data%U1 ; data%P => data%P1
             data%U = zero ; data%P = one
             IF ( data%present_eval_hprod ) THEN
               CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),            &
                                userdata, data%U( : nlp%n ),                   &
                                data%P( : nlp%n ) )
             ELSE
               data%got_h = .FALSE.
!              data%V => data%V1
               data%V = one
               data%branch = data%branch + 5
             END IF
           END IF
         END IF

!  reverse communication is required for at least f or one of its derivatives

         IF ( data%branch > 0 ) THEN
           SELECT CASE( data%branch )
           CASE ( 1 )
             inform%status = 2
           CASE ( 2 )
             inform%status = 3
           CASE ( 3 )
             inform%status = 23
           CASE ( 5 )
             inform%status = 5
           CASE ( 6 )
             inform%status = 25
           CASE ( 7 )
             inform%status = 35
           CASE ( 8 )
             inform%status = 235
           END SELECT
           data%branch = 420 ; RETURN
         END IF
       END IF

!  return from reverse communication

 420 CONTINUE
     IF ( inform%UGO_inform%status >= 2 ) THEN
       data%phi = nlp%f
       IF ( inform%UGO_inform%status >= 3 ) THEN
         data%phi1 = DOT_PRODUCT( data%P, nlp%G )
         IF ( inform%UGO_inform%status >= 4 )                                  &
           data%phi2 = DOT_PRODUCT( data%P, data%U )
       END IF
       GO TO 410
     END IF

     IF ( data%printm )                                                        &
       WRITE( data%out, "( A, ' minimizer', ES12.4, ' in [', ES11.4,           &
    &     ',', ES10.4, '] has f =', ES12.4, ', st = ', I0 )" )                 &
         prefix, nlp%X( 1 ), nlp%X_l( 1 ), nlp%X_u( 1 ), data%phi,             &
         inform%UGO_inform%status

     IF ( inform%UGO_inform%status < 0 .AND. data%printe ) THEN
       IF ( inform%UGO_inform%status /= GALAHAD_error_max_iterations )         &
         WRITE( data%error, "( ' Help! exit from UGO status = ', I0 )" )       &
           inform%UGO_inform%status
     END IF

!  record the time taken in the univariate global minimization

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     inform%time%univariate_global =                                           &
       inform%time%univariate_global + data%time_now - data%time_record
     inform%time%clock_univariate_global =                                     &
       inform%time%clock_univariate_global + data%clock_now - data%clock_record

!  record the global minimizer

     inform%f_eval = inform%f_eval + inform%UGO_inform%f_eval
     inform%g_eval = inform%h_eval + inform%UGO_inform%g_eval
     inform%h_eval = inform%h_eval + inform%UGO_inform%h_eval
     inform%obj = data%phi
     inform%norm_pg = TWO_NORM( nlp%X -                                        &
          TRB_projection( nlp%n, nlp%X - nlp%G, nlp%X_l, nlp%X_u ) )

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

!  control printing

     data%print_iteration_header = data%printt .OR. data%TRB_data%printi

!  print one-line summary

     IF ( data%printi ) THEN
       IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
         WRITE( data%out, 2010 )
         data%print_1st_header = .FALSE.
       END IF
       WRITE( data%out, 2020 ) prefix, data%pass, 'U', 0, inform%obj,          &
         inform%norm_pg, inform%f_eval, inform%g_eval, data%clock_now
     END IF
     GO TO 900

!  ============================================================================
!  Multistart iteration
!  ============================================================================

 500 CONTINUE
     data%f_best = HUGE( one )
     data%accurate = .FALSE.
     data%trb_maxit = data%control%TRB_control%maxit
     data%trb_maxit_large = 100 * data%trb_maxit

!  ============================================================================
!  Find a sequence of local minimizers of f(x) from random starting points
!  ============================================================================

 510 CONTINUE

!  ------------------------------------------------
!  implicit loop to generate random starting points
!  ------------------------------------------------

       data%attempts = 1 ; data%lhs_count = 0

!  loop over at most attempts_max random starting vectors

 520   CONTINUE

!  compute a random vector x in [x_l,x_u] (except on the first pass when the
!  input initial guess is used)

         IF ( data%pass > 1 ) THEN

!  pick the point uniformly in [x_l,x_u]

           IF ( data%control%sampling_strategy == 1 ) THEN
             DO i = 1, nlp%n
               x_l = nlp%X_l( i ) ; x_u = nlp%X_u( i )
               CALL RAND_random_real( data%seed, .TRUE., p )
               nlp%X( i ) = x_l + p * ( x_u - x_l )
             END DO

!  pick the points by partitioning each dimension of the box [x_l,x_u] into
!  hypercube_discretization segments, and then selecting subboxes
!  randomly within this hypercube using Latin hypercube sampling

           ELSE
             data%lhs_count = data%lhs_count + 1
             IF ( data%lhs_count > data%control%hypercube_discretization )     &
               data%lhs_count = 1
             IF ( data%lhs_count == 1 ) THEN
               CALL LHS_ihs( nlp%n, data%control%hypercube_discretization,     &
                             data%lhs_seed, data%X_lhs,                        &
                             data%control%LHS_control,                         &
                             inform%LHS_inform, data%LHS_data )
               IF ( inform%LHS_inform%status /= 0 ) THEN
                 inform%status = inform%LHS_inform%status
                 GO TO 980
               END IF
             END IF

!  pick x to point to the middle of the assigned sub-hypercube

             DO i = 1, nlp%n
               x_l = nlp%X_l( i ) ; x_u = nlp%X_u( i )
               d = ( x_u - x_l ) / data%rhcd
               nlp%X( i ) = ( REAL( data%X_lhs( i, data%lhs_count ),           &
                                    KIND = wp ) - half ) * d + x_l


! if required, randonly perturb the point within this sub-hypercube

               IF ( data%control%sampling_strategy == 3 ) THEN
                 CALL RAND_random_real( data%seed, .FALSE., p )
                 nlp%X( i ) = nlp%X( i ) + half * d * p
               END IF
             END DO
           END IF
         END IF


!CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata, nlp%f )
!write(6,*) ' f ', nlp%f
!write(6,*) ' x ', nlp%X

!  -----------------------------------------------------------------
!  implicit loop to perform the local bound-constrained minimization
!  -----------------------------------------------------------------

 530     CONTINUE
         inform%TRB_inform%status = 1
         inform%TRB_inform%iter = 0
         inform%TRB_inform%cg_iter = 0
         inform%TRB_inform%f_eval = 0
         inform%TRB_inform%g_eval = 0
         inform%TRB_inform%h_eval = 0
         data%control%TRB_control%error = 0
         data%control%TRB_control%hessian_available = control%hessian_available
         data%control%TRB_control%maxit = MIN( control%TRB_control%maxit,      &
             control%max_evals - inform%f_eval )

         CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
 540     CONTINUE

!  call the bound-constrained local minimizer

!write(6,*) 'in status ', inform%TRB_inform%status,  inform%TRB_inform%iter
           CALL TRB_solve( nlp, data%control%TRB_control, inform%TRB_inform,   &
                           data%TRB_data, userdata, eval_F = eval_F,           &
                           eval_G = eval_G, eval_H = eval_H,                   &
                           eval_HPROD = eval_HPROD, eval_SHPROD = eval_SHPROD, &
                           eval_PREC = eval_PREC )
!write(6,*) 'out status ', inform%TRB_inform%status

!  obtain further function information if required

           SELECT CASE ( inform%TRB_inform%status )

!  obtain the objective function value

           CASE ( 2 )
             IF ( data%present_eval_f ) THEN
               CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,      &
                            inform%TRB_inform%obj )
             ELSE
               data%branch = 550 ; inform%status = 2 ; RETURN
             END IF

!  obtain the gradient value

           CASE ( 3 )
             IF ( data%present_eval_g ) THEN
               CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,      &
                            nlp%G( : nlp%n ) )
             ELSE
               data%branch = 550 ; inform%status = 3 ; RETURN
             END IF

!  obtain the Hessian value

           CASE ( 4 )
             IF ( data%present_eval_h ) THEN
               CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                &
                            userdata, nlp%H%val( : nlp%H%ne ) )
             ELSE
               data%branch = 550 ; inform%status = 4 ; RETURN
             END IF

!  obtain a Hessian-vector product

           CASE ( 5 )
             data%got_h = data%TRB_data%got_h
             data%U => data%TRB_data%U
             IF ( data%present_eval_hprod ) THEN
               CALL eval_HPROD( data%eval_status, nlp%X( : nlp%n ),            &
                                userdata, data%U( : nlp%n ),                   &
                                data%TRB_data%S( : nlp%n ), got_h = data%got_h )
             ELSE
               data%V => data%TRB_data%S
               data%branch = 550 ; inform%status = 5 ; RETURN
             END IF

!  obtain a preconditioned vector product

           CASE ( 6 )
             data%U => data%TRB_data%U
             data%V => data%TRB_data%V
             IF ( data%present_eval_prec ) THEN
               CALL eval_PREC( data%eval_status, nlp%X( : nlp%n ), userdata,   &
                               data%U( : nlp%n ), data%V( : nlp%n ) )
             ELSE
               data%branch = 550 ; inform%status = 6 ; RETURN
             END IF

!  obtain a Hessian-sparse-vector product

           CASE ( 7 )
             data%got_h = data%TRB_data%got_h
             data%nnz_p_u = data%TRB_data%nnz_p_u
             data%nnz_p_l = data%TRB_data%nnz_p_l
             data%INDEX_nz_p => data%TRB_data%INDEX_nz_p
             data%P => data%TRB_data%P
             data%INDEX_nz_hp => data%TRB_data%INDEX_nz_hp
             data%HP => data%TRB_data%HP
             IF ( data%present_eval_shprod ) THEN
               CALL eval_SHPROD( data%eval_status, nlp%X( : nlp%n ),           &
                           userdata, data%nnz_p_u - data%nnz_p_l + 1,          &
                           data%INDEX_nz_p( data%nnz_p_l : data%nnz_p_u ),     &
                           data%P, data%nnz_hp, data%INDEX_nz_hp, data%HP,     &
                           got_h = data%got_h )
             ELSE
               data%branch = 550 ; inform%status = 7 ; RETURN
             END IF

!  terminal exits

           CASE DEFAULT
             GO TO 590
           END SELECT

!  return from reverse communication

 550       CONTINUE
           data%TRB_data%eval_status = data%eval_status

           SELECT CASE ( inform%TRB_inform%status )
           CASE ( 2 )
           CASE ( 3 )
           CASE ( 4 )
           CASE ( 5 )
           CASE ( 6 )
           CASE ( 7 )
             data%TRB_data%nnz_hp = data%nnz_hp
           END SELECT

!  ------------------------------------------------
!  end of local bound-constrained minimization loop
!  ------------------------------------------------

           GO TO 540
 590     CONTINUE

!  record the time taken in the local minimization

         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         inform%time%multivariate_local =                                      &
           inform%time%multivariate_local +                                    &
             data%time_now - data%time_record
         inform%time%clock_multivariate_local =                                &
           inform%time%clock_multivariate_local +                              &
             data%clock_now - data%clock_record

!  record details about the critical point found

         inform%f_eval = inform%f_eval + inform%TRB_inform%f_eval
         inform%g_eval = inform%h_eval + inform%TRB_inform%g_eval
         inform%h_eval = inform%h_eval + inform%TRB_inform%h_eval

!  record the clock time

         CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
         data%time_now = data%time_now - data%time_start
         data%clock_now = data%clock_now - data%clock_start

!  control printing

         data%print_iteration_header = data%printt .OR. data%TRB_data%printi

!   test to see whether a lower point has been found

         f_best = data%f_best
         IF ( inform%TRB_inform%obj < f_best ) THEN
!write(6,*) ' better!'
           inform%obj = inform%TRB_inform%obj
           inform%norm_pg = inform%TRB_inform%norm_pg
           data%f_best = inform%obj
           data%X_best = nlp%X
           data%G_best = nlp%G
         END IF

!  if a significantly-lower point has been found, start another cycle

!write(6,*) inform%TRB_inform%obj, f_best, &
!                f_best - MAX( ABS( f_best ), one ) * teneps
         IF ( data%accurate ) THEN
           data%accurate = .FALSE.
           IF ( data%printi ) THEN
             IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               WRITE( data%out, 2010 )
               data%print_1st_header = .FALSE.
             END IF
             WRITE( data%out, 2020 ) prefix, data%pass, 'T', data%attempts,    &
               inform%TRB_inform%obj, inform%TRB_inform%norm_pg,               &
               inform%f_eval, inform%g_eval, data%clock_now
           END IF
           IF ( data%printm ) WRITE( data%out, "( A, 78( '-' ) )" ) prefix
           data%control%TRB_control%maxit = data%trb_maxit
           data%pass = data%pass + 1
           GO TO 510
         ELSE IF ( inform%TRB_inform%obj <                                     &
                f_best - MAX( ABS( f_best ), one ) * teneps ) THEN
           data%accurate = .TRUE.
           data%control%TRB_control%maxit = data%trb_maxit_large
           GO TO 530

!  print one-line summary

         ELSE
           IF ( data%printm ) THEN
             IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
               WRITE( data%out, 2010 )
               data%print_1st_header = .FALSE.
             END IF
             WRITE( data%out, 2020 ) prefix, data%pass, 'T', data%attempts,    &
               inform%TRB_inform%obj, inform%TRB_inform%norm_pg,               &
               inform%f_eval, inform%g_eval, data%clock_now
           END IF

         END IF

!        IF ( inform%TRB_inform%status < 0 .AND. data%printe ) THEN
!          IF ( inform%TRB_inform%status /= GALAHAD_error_max_iterations )     &
!            WRITE( data%error, "( ' Help! exit from TRB status = ', I0 )" )   &
!              inform%TRB_inform%status
!        END IF

         data%attempts = data%attempts + 1
         IF ( inform%f_eval <= control%max_evals ) THEN
           IF ( data%attempts <= data%control%attempts_max ) GO TO 520
         END IF

!  check to see if the evaluation limit has been exceeded

         IF ( data%control%max_evals >= 0 .AND.                                &
              inform%f_eval > data%control%max_evals ) THEN
           inform%status = GALAHAD_error_max_evaluations ; GO TO 900
         END IF

         IF ( data%printt ) WRITE( data%out, "( /, A, ' Time so far = ', 0P,   &
        &    F12.2,  ' seconds' )" ) prefix, data%clock_now
         IF ( ( data%control%cpu_time_limit >= zero .AND.                      &
                data%time_now > data%control%cpu_time_limit ) .OR.             &
              ( data%control%clock_time_limit >= zero .AND.                    &
                data%clock_now > data%control%clock_time_limit ) ) THEN
           inform%status = GALAHAD_error_cpu_limit ; GO TO 900
         END IF

!  ----------------------------------------------
!  end of loop to generate random starting points
!  ----------------------------------------------

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

!  control printing

       data%print_iteration_header = data%printt .OR. data%UGO_data%printi

!  print one-line summary

       IF ( data%printi ) THEN
         IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
           WRITE( data%out, 2010 )
           data%print_1st_header = .FALSE.
         END IF
         WRITE( data%out, 2020 ) prefix, data%pass, 'M', data%attempts,        &
           inform%obj, inform%norm_pg, inform%f_eval, inform%g_eval,           &
           data%clock_now
       END IF

!  record the best value found

       inform%obj = data%f_best
       nlp%X = data%X_best
       nlp%G = data%G_best

!  debug printing for X and G

       IF ( data%printw ) THEN
         WRITE ( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n
         WRITE ( data%out, 2200 ) prefix, inform%f_eval, prefix,               &
           inform%g_eval, prefix, inform%h_eval, prefix,                       &
           inform%obj, prefix, inform%norm_pg
         WRITE ( data%out, 2210 ) prefix
         l = 2
         DO j = 1, 2
            IF ( j == 1 ) THEN
               ir = 1 ; ic = MIN( l, nlp%n )
            ELSE
               IF ( ic < nlp%n - l ) WRITE( data%out, 2240 ) prefix
               ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
            END IF
            IF ( ALLOCATED( nlp%vnames ) ) THEN
              DO i = ir, ic
                 WRITE( data%out, 2220 ) prefix, nlp%vnames( i ),              &
                   nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
              END DO
            ELSE
              DO i = ir, ic
                 WRITE( data%out, 2230 ) prefix, i,                            &
                   nlp%X_l( i ), nlp%X( i ), nlp%X_u( i ), nlp%G( i )
              END DO
            END IF
         END DO
       END IF

!  ============================================================================
!  End of the main iteration
!  ============================================================================

 900 CONTINUE

!  since the algorithm is only designed to stop when it reaches its 
!  evaluation limit, flag this as a successful conclusion

     IF ( inform%status == GALAHAD_error_max_evaluations .OR.                  &
          inform%status == GALAHAD_error_max_iterations ) THEN
       inform%status = GALAHAD_ok
     END IF

!  print details of solution

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start

     IF ( data%printi ) THEN
!      WRITE ( data%out, 2000 ) nlp%pname, nlp%n
!      WRITE ( data%out, 2200 ) inform%f_eval, inform%g_eval, inform%h_eval,   &
!         inform%obj, inform%norm_pg
!      WRITE ( data%out, 2210 )
!      IF ( data%print_level > 3 ) THEN
!         l = nlp%n
!      ELSE
!         l = 2
!      END IF
!      DO j = 1, 2
!         IF ( j == 1 ) THEN
!            ir = 1 ; ic = MIN( l, nlp%n )
!         ELSE
!            IF ( ic < nlp%n - l ) WRITE( data%out, 2240 )
!            ir = MAX( ic + 1, nlp%n - ic + 1 ) ; ic = nlp%n
!         END IF
!         DO i = ir, ic
!            WRITE ( data%out, 2220 ) nlp%vnames( i ), nlp%X_l( i ),
!              nlp%X( i ), nlp%X_u( i ), nlp%G( i )
!         END DO
!      END DO
       IF ( data%printm ) WRITE ( data%out, "( A, '  Total time = ', 0P, F0.2, &
      & ' seconds' )" ) prefix, inform%time%clock_total
     END IF
     IF ( inform%status /= GALAHAD_OK ) GO TO 990
     RETURN

!  -------------
!  Error returns
!  -------------

 980 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'BGO_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' Problem: ', A, ' n = ', I8 )
 2010 FORMAT( '  pass        #d             f                g        ',       &
              '  #f      #g        time' )
 2020 FORMAT( A, I6, A1, I9, ES24.16, ES11.4, 2I8, F12.2 )
 2200 FORMAT( /, A, ' # function evaluations  = ', I10,                        &
              /, A, ' # gradient evaluations  = ', I10,                        &
              /, A, ' # Hessian evaluations   = ', I10,                        &
             //, A, ' Current objective value = ', ES22.14,                    &
              /, A, ' Current gradient norm   = ', ES12.4 )
 2210 FORMAT( /, A, ' name             X_l        X         X_u         G ' )
 2220 FORMAT(  A, 1X, A10, 4ES12.4 )
 2230 FORMAT(  A, 1X, I10, 4ES12.4 )
 2240 FORMAT( A, ' .          ........... ...........' )

 !  End of subroutine BGO_solve

     END SUBROUTINE BGO_solve

!-*-*-  G A L A H A D -  B G O _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE BGO_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( BGO_control_type ), INTENT( IN ) :: control
     TYPE ( BGO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     array_name = 'bgo: data%X_start'
     CALL SPACE_dealloc_array( data%X_start,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%X_best'
     CALL SPACE_dealloc_array( data%X_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%G_best'
     CALL SPACE_dealloc_array( data%G_best,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%D'
     CALL SPACE_dealloc_array( data%D,                                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     data%P => NULL( )
     array_name = 'bgo: data%P'
     CALL SPACE_dealloc_pointer( data%P,                                       &
        inform%status, inform%alloc_status, point_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     data%U => NULL( )
     array_name = 'bgo: data%U'
     CALL SPACE_dealloc_pointer( data%U,                                       &
        inform%status, inform%alloc_status, point_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     data%V => NULL( )
     array_name = 'bgo: data%V'
     CALL SPACE_dealloc_pointer( data%V,                                       &
        inform%status, inform%alloc_status, point_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within UGO

     CALL UGO_terminate( data%UGO_data, data%control%UGO_control,              &
                         inform%UGO_inform )
     inform%status = inform%UGO_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%UGO_inform%alloc_status
       inform%bad_alloc = inform%UGO_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Deallocate all arraysn allocated within TRB

     CALL TRB_terminate( data%TRB_data, data%control%TRB_control,              &
                          inform%TRB_inform )
     inform%status = inform%TRB_inform%status
     IF ( inform%status /= 0 ) THEN
       inform%alloc_status = inform%TRB_inform%alloc_status
       inform%bad_alloc = inform%TRB_inform%bad_alloc
       IF ( control%deallocate_error_fatal ) RETURN
     END IF

!  Close and delete 'alive' file

     IF ( control%alive_unit > 0 ) THEN
       INQUIRE( FILE = control%alive_file, EXIST = alive )
       IF ( alive .AND. control%alive_unit > 0 ) THEN
         OPEN( control%alive_unit, FILE = control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
         REWIND control%alive_unit
         CLOSE( control%alive_unit, STATUS = 'DELETE' )
       END IF
     END IF

     RETURN

!  End of subroutine BGO_terminate

     END SUBROUTINE BGO_terminate

! -  G A L A H A D -  B G O _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE BGO_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BGO_control_type ), INTENT( IN ) :: control
     TYPE ( BGO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL BGO_terminate( data%bgo_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'bgo: data%nlp%X'
     CALL SPACE_dealloc_array( data%nlp%X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%G'
     CALL SPACE_dealloc_array( data%nlp%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%X_l'
     CALL SPACE_dealloc_array( data%nlp%X_l,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%X_u'
     CALL SPACE_dealloc_array( data%nlp%X_u,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%H%row'
     CALL SPACE_dealloc_array( data%nlp%H%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%H%col'
     CALL SPACE_dealloc_array( data%nlp%H%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%H%ptr'
     CALL SPACE_dealloc_array( data%nlp%H%ptr,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%H%val'
     CALL SPACE_dealloc_array( data%nlp%H%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'bgo: data%nlp%H%type'
     CALL SPACE_dealloc_array( data%nlp%H%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine BGO_full_terminate

     END SUBROUTINE BGO_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  B G O _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE BGO_import( control, data, status, n, X_l, X_u,                &
                            H_type, ne, H_row, H_col, H_ptr )

!  import problem data into internal storage prior to solution. 
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading 
!   comments to BGO_solve
!
!  data is a scalar variable of type BGO_full_data_type used for internal data
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
!   -3. The restriction n > 0 or requirement that type contains
!       its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'ABSENT' has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  X_l is a rank-one array of dimension n and type default real,
!   that holds the values x_l of the lower bounds on the optimization
!   variables x. The j-th component of X_l, j = 1, ... , n, contains (x_l)j.
!
!  X_u is a rank-one array of dimension n and type default real,
!   that holds the values x_u of the upper bounds on the optimization
!   variables x. The j-th component of X_u, j = 1, ... , n, contains (x_u)j.
!
!  H_type is a character string that specifies the Hessian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   'diagonal' or 'absent', the latter if access to the Hessian is via
!   matrix-vector products; lower or upper case variants are allowed
!
!  ne is a scalar variable of type default integer, that holds the number of
!   entries in the  lower triangular part of H in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other three schemes.
!
!  H_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of H in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  H_col is a rank-one array variable of type default integer,
!   that holds the column indices of the  lower triangular part of H in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense or diagonal storage schemes are used, and in this 
!   case can be of length 0
!
!  H_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the  lower
!   triangular part of H, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_control_type ), INTENT( INOUT ) :: control
     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( IN ) :: n, ne
     INTEGER, INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: H_type
     INTEGER, DIMENSION( : ), INTENT( IN ) :: H_row, H_col, H_ptr
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: X_l, X_u

!  local variables

     INTEGER :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     data%bgo_control = control

     error = data%bgo_control%error
     space_critical = data%bgo_control%space_critical
     deallocate_error_fatal = data%bgo_control%deallocate_error_fatal

!  allocate space if required

     array_name = 'bgo: data%nlp%X'
     CALL SPACE_resize_array( n, data%nlp%X,                                   &
            data%bgo_inform%status, data%bgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bgo_inform%bad_alloc, out = error )
     IF ( data%bgo_inform%status /= 0 ) GO TO 900

     array_name = 'bgo: data%nlp%G'
     CALL SPACE_resize_array( n, data%nlp%G,                                   &
            data%bgo_inform%status, data%bgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bgo_inform%bad_alloc, out = error )
     IF ( data%bgo_inform%status /= 0 ) GO TO 900

     array_name = 'bgo: data%nlp%X_l'
     CALL SPACE_resize_array( n, data%nlp%X_l,                                 &
            data%bgo_inform%status, data%bgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bgo_inform%bad_alloc, out = error )
     IF ( data%bgo_inform%status /= 0 ) GO TO 900

     array_name = 'bgo: data%nlp%X_u'
     CALL SPACE_resize_array( n, data%nlp%X_u,                                 &
            data%bgo_inform%status, data%bgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%bgo_inform%bad_alloc, out = error )
     IF ( data%bgo_inform%status /= 0 ) GO TO 900

!  put data into the required components of the nlpt storage type

     data%nlp%n = n
     data%nlp%X_l( : n ) = X_l( : n )
     data%nlp%X_u( : n ) = X_u( : n )

!  set H appropriately in the nlpt storage type

     SELECT CASE ( H_type )
     CASE ( 'coordinate', 'COORDINATE' )
       CALL SMT_put( data%nlp%H%type, 'COORDINATE',                            &
                     data%bgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = ne

       array_name = 'bgo: data%nlp%H%row'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%row,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900

       array_name = 'bgo: data%nlp%H%col'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%col,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900

       array_name = 'bgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900

       data%nlp%H%row( : data%nlp%H%ne ) = H_row( : data%nlp%H%ne )
       data%nlp%H%col( : data%nlp%H%ne ) = H_col( : data%nlp%H%ne )
     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       CALL SMT_put( data%nlp%H%type, 'SPARSE_BY_ROWS',                        &
                     data%bgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = H_ptr( n + 1 ) - 1

       array_name = 'bgo: data%nlp%H%ptr'
       CALL SPACE_resize_array( n + 1, data%nlp%H%ptr,                         &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900

       array_name = 'bgo: data%nlp%H%col'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%col,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900

       array_name = 'bgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900

       data%nlp%H%ptr( : n + 1 ) = H_ptr( : n + 1 )
       data%nlp%H%col( : data%nlp%H%ne ) = H_col( : data%nlp%H%ne )
     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%nlp%H%type, 'DENSE',                                 &
                     data%bgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'bgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900
     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%nlp%H%type, 'DIAGONAL',                              &
                     data%bgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = n

       array_name = 'bgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%bgo_inform%status, data%bgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%bgo_inform%bad_alloc, out = error )
       IF ( data%bgo_inform%status /= 0 ) GO TO 900
     CASE ( 'absent', 'ABSENT' )
       data%bgo_control%hessian_available = .FALSE.
     CASE DEFAULT
       data%bgo_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT       

     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%bgo_inform%status
     RETURN

!  End of subroutine BGO_import

     END SUBROUTINE BGO_import

!-  G A L A H A D -  B G O _ r e s e t _ c o n t r o l   S U B R O U T I N E  -

     SUBROUTINE BGO_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See BGO_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_control_type ), INTENT( IN ) :: control
     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: status

!  set control in internal data

     data%bgo_control = control
     
!  flag a successful call

     status = GALAHAD_ready_to_solve
     RETURN

!  end of subroutine BGO_reset_control

     END SUBROUTINE BGO_reset_control

!-  G A L A H A D -  B G O _ s o l v e _ w i t h _ m a t   S U B R O U T I N E 

     SUBROUTINE BGO_solve_with_mat( data, userdata, status, X, G, eval_F,      &
                                    eval_G, eval_H, eval_HPROD, eval_PREC )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian and preconditioning operations are
!  available via subroutine calls. See BGO_solve for a description of 
!  the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_F, eval_G, eval_H, eval_HPROD, eval_PREC

     data%bgo_inform%status = status
     IF ( data%bgo_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     CALL BGO_solve( data%nlp, data%bgo_control, data%bgo_inform,              &
                     data%bgo_data, userdata, eval_F = eval_F,                 &
                     eval_G = eval_G, eval_H = eval_H,                         &
                     eval_HPROD = eval_HPROD, eval_PREC = eval_PREC )

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     IF ( data%bgo_inform%status == GALAHAD_ok )                               &
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )

     status = data%bgo_inform%status
     RETURN

     END SUBROUTINE BGO_solve_with_mat

!  G A L A H A D -  B G O _ s o l v e _ w i t h o u t _mat  S U B R O U T I N E 

     SUBROUTINE BGO_solve_without_mat( data, userdata, status, X, G,           &
                                       eval_F, eval_G, eval_HPROD,             &
                                       eval_SHPROD, eval_PREC )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian-vector and preconditioning operations 
!  are available via subroutine calls. See BGO_solve for a description 
!  of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_F, eval_G, eval_HPROD, eval_SHPROD, eval_PREC

     data%bgo_inform%status = status
     IF ( data%bgo_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     CALL BGO_solve( data%nlp, data%bgo_control, data%bgo_inform,              &
                     data%bgo_data, userdata, eval_F = eval_F,                 &
                     eval_G = eval_G, eval_HPROD = eval_HPROD,                 &
                     eval_SHPROD = eval_SHPROD, eval_PREC = eval_PREC )

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     IF ( data%bgo_inform%status == GALAHAD_ok )                               &
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )

     status = data%bgo_inform%status
     RETURN

     END SUBROUTINE BGO_solve_without_mat

!-  G A L A H A D -  B G O _ s o l v e _ reverse _ m a t  S U B R O U T I N E  -

     SUBROUTINE BGO_solve_reverse_with_mat( data, status, eval_status,         &
                                            X, f, G, H_val, U, V )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian and preconditioning operations are
!  available via reverse communication. See BGO_solve for a description 
!  of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( INOUT ) :: status, eval_status
     REAL ( KIND = wp ), INTENT( IN ) :: f
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: G
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: H_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: V

!  recover data from reverse communication

     data%bgo_inform%status = status
     data%bgo_data%eval_status = eval_status

     SELECT CASE ( data%bgo_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%f = f
     CASE( 3 ) 
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
     CASE( 4 ) 
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%H%val( : data%bgo_data%trb_data%h_ne ) =                     &
           H_val( : data%bgo_data%trb_data%h_ne )
     CASE( 5, 6 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
     CASE ( 23 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
       END IF
     CASE ( 25 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 35 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 235 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     END SELECT

!  call the solver

     CALL BGO_solve( data%nlp, data%bgo_control, data%bgo_inform,              &
                     data%bgo_data, data%userdata )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%bgo_inform%status )
     CASE( 0 )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 5, 25, 35, 235 )
       U( : data%nlp%n ) = data%bgo_data%U( : data%nlp%n )
       V( : data%nlp%n ) = data%bgo_data%V( : data%nlp%n )
    CASE( 6 )
       V( : data%nlp%n ) = data%bgo_data%V( : data%nlp%n )
     CASE( 7 ) 
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%bgo_inform%status
     END SELECT

     status = data%bgo_inform%status
     RETURN

     END SUBROUTINE BGO_solve_reverse_with_mat

!  G A L A H A D -  B G O _ s o l v e _ reverse _ no _mat  S U B R O U T I N E  

     SUBROUTINE BGO_solve_reverse_without_mat( data, status, eval_status,      &
                                               X, f, G, U, V,                  &
                                               INDEX_nz_v, nnz_v,              &
                                               INDEX_nz_u, nnz_u )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian-vector and preconditioning operations 
!  are available via reverse communication. See BGO_solve for a description 
!  of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status, eval_status
     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: nnz_v
     INTEGER, INTENT( IN ) :: nnz_u
     REAL ( KIND = wp ), INTENT( IN ) :: f
     INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_v
     INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_u 
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: G
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: V

!  recover data from reverse communication

     data%bgo_inform%status = status
     data%bgo_data%eval_status = eval_status

     SELECT CASE ( data%bgo_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%f = f
     CASE( 3 ) 
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
     CASE( 5, 6 ) 
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
          data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
     CASE( 7 ) 
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%bgo_data%nnz_hp = nnz_u
         data%bgo_data%INDEX_nz_hp( : nnz_u ) = INDEX_nz_u( : nnz_u )
         data%bgo_data%HP( INDEX_nz_u( 1 : nnz_u ) )                           &
            = U( INDEX_nz_u( 1 : nnz_u ) ) 
       END IF
     CASE ( 23 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
       END IF
     CASE ( 25 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 35 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 235 )
       data%bgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%bgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     END SELECT

!  call the solver

     CALL BGO_solve( data%nlp, data%bgo_control, data%bgo_inform,              &
                     data%bgo_data, data%userdata )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE (  data%bgo_inform%status )
     CASE( 0 )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 2, 3 ) 
     CASE( 4 ) 
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%bgo_inform%status
     CASE( 5, 25, 35, 235 )
       U( : data%nlp%n ) = data%bgo_data%U( : data%nlp%n )
       V( : data%nlp%n ) = data%bgo_data%V( : data%nlp%n )
     CASE( 6 )
       V( : data%nlp%n ) = data%bgo_data%V( : data%nlp%n )
     CASE( 7 )
       nnz_v = data%bgo_data%nnz_p_u - data%bgo_data%nnz_p_l + 1
       INDEX_nz_v( : nnz_v ) =                                                 &
          data%bgo_data%INDEX_nz_p( data%bgo_data%nnz_p_l :                    &
                                    data%bgo_data%nnz_p_u )
       V( INDEX_nz_v( 1 : nnz_v ) )                                            &
          = data%bgo_data%P( INDEX_nz_v( 1 : nnz_v ) )
     END SELECT

     status = data%bgo_inform%status
     RETURN

     END SUBROUTINE BGO_solve_reverse_without_mat

!-  G A L A H A D -  B G O _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE BGO_information( data, inform, status )

!  return solver information during or after solution by BGO
!  See BGO_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BGO_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%bgo_inform
     
!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine BGO_information

     END SUBROUTINE BGO_information

!  End of module GALAHAD_BGO

   END MODULE GALAHAD_BGO_double


