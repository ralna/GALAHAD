! THIS VERSION: GALAHAD 3.3 - 03/07/2021 AT 15:15 GMT

!-*-*-*-*-*-*-*-*-  G A L A H A D _ D G O   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes and Nick Gould

!  History -
!   originally released GALAHAD Version 3.3. July 3rd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_DGO_double

!     ------------------------------------------------------
!    |                                                      |
!    | DGO, a determinitic algorithm for bound-constrained  |
!    |      global optimization                             |
!    |                                                      |
!    |   Aim: find a global minimizer of the objective f(x) |
!    |        subject to x_l <= x <= x_u                    |
!    |                                                      |
!     ------------------------------------------------------

     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_HASH
     USE GALAHAD_SMT_double
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_UGO_double
     USE GALAHAD_TRB_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_NORMS_double, ONLY: TWO_NORM

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: DGO_initialize, DGO_read_specfile, DGO_solve,                   &
               DGO_terminate, NLPT_problem_type,                               &
               NLPT_userdata_type, SMT_type, SMT_put,                          &
               DGO_import, DGO_solve_with_mat, DGO_solve_without_mat,          &
               DGO_solve_reverse_with_mat, DGO_solve_reverse_without_mat,      &
               DGO_full_initialize, DGO_full_terminate, DGO_reset_control,     &
               DGO_information

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE DGO_initialize
       MODULE PROCEDURE DGO_initialize, DGO_full_initialize
     END INTERFACE DGO_initialize

     INTERFACE DGO_terminate
       MODULE PROCEDURE DGO_terminate, DGO_full_terminate
     END INTERFACE DGO_terminate

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: sp = KIND( 1.0 )
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

     INTEGER, PARAMETER :: rwidth = 24 ! space required for a double-precision #
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: twothirds = two / three
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: quarter = 0.25_wp
     REAL ( KIND = wp ), PARAMETER :: tenth = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = ten ** ( - 1 )
     REAL ( KIND = wp ), PARAMETER :: point01 = ten ** ( - 2 )
     REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
     REAL ( KIND = wp ), PARAMETER :: tenm8 = ten ** ( - 9 )
     REAL ( KIND = wp ), PARAMETER :: infinity = ten ** 19
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
     REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: DGO_control_type

!   error and warning diagnostics occur on stream error

       INTEGER :: error = 6

!   general output occurs on stream out

       INTEGER :: out = 6

!   the level of output required. <= 0 gives no output, = 1 gives a one-line
!    summary for every iteration, = 2 gives a summary of the inner iteration
!    for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!   any printing will start on this iteration

       INTEGER :: start_print = - 1

!   any printing will stop on this iteration

       INTEGER :: stop_print = - 1

!   the number of iterations between printing

       INTEGER :: print_gap = 10000

!   the maximum number of iterations performed

       INTEGER :: maxit = 1000

!   the maximum number of function evaluations made

       INTEGER :: max_evals = 10000

!  the size of the initial hash dictionary

       INTEGER :: dictionary_size = 100000

!   removal of the file alive_file from unit alive_unit terminates execution

       INTEGER :: alive_unit = 40
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d'

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!    a small positive constant (<= 1e-6) that ensure that the estimted
!     gradient Lipschitz constant is not too small

       REAL ( KIND = wp ) :: lipschitz_lower_bound = ten ** ( - 6 )

!   the Lipschitz reliability parameter, the Lipschiz constant used will
!    be a factor lipschitz_reliability times the largest value observed

       REAL ( KIND = wp ) :: lipschitz_reliability = 2.0_wp

!    the reliablity control parameter, the actual reliability parameter used 
!     will be 
!      lipschitz_reliability + MAX( 1, n - 1 ) * lipschitz_control / iteration

       REAL ( KIND = wp ) :: lipschitz_control = 50.0_wp

!   the iteration will stop if the length, delta, of the diagonal in the box
!    with the smallest-found objective function is smaller than %stop_length
!    times that of the original bound box, delta_0

       REAL ( KIND = wp ) :: stop_length = ten ** ( - 4 )

!   the iteration will stop if the gap between the best objective value
!    found and the smallest lower bound is smaller than %stop_f

       REAL ( KIND = wp ) :: stop_f = ten ** ( - 4 )

!   the smallest value the objective function may take before the problem
!    is marked as unbounded

       REAL ( KIND = wp ) :: obj_unbounded = - epsmch ** ( - 2 )

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   is the Hessian matrix of second derivatives available or is access only
!    via matrix-vector products?

       LOGICAL :: hessian_available = .TRUE.

!   should boxes that cannot contain the global minimizer be pruned (i.e.,
!    removed from further consideration)?

       LOGICAL :: prune = .TRUE.

!   should approximate minimizers be impoved by judicious local minimization?

       LOGICAL :: perform_local_optimization = .TRUE.

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

!  control parameters for HASH

       TYPE ( HASH_control_type ) :: HASH_control

!  control parameters for UGO

       TYPE ( UGO_control_type ) :: UGO_control

!  control parameters for TRB

       TYPE ( TRB_control_type ) :: TRB_control

     END TYPE DGO_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: DGO_time_type

!  the total CPU time spent in the package

       REAL ( KIND = sp ) :: total = 0.0

!  the CPU time spent performing univariate global optimization

       REAL ( KIND = sp ) :: univariate_global = 0.0

!  the CPU time spent performing multivariate local optimization

       REAL ( KIND = sp ) :: multivariate_local = 0.0

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

     TYPE, PUBLIC :: DGO_inform_type

!  return status. See DGO_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations performed

       INTEGER :: iter = 0

!  the total number of evaluations of the objection function

       INTEGER :: f_eval = 0

!  the total number of evaluations of the gradient of the objection function

       INTEGER :: g_eval = 0

!  the total number of evaluations of the Hessian of the objection function

       INTEGER :: h_eval = 0

!  the value of the objective function at the best estimate of the solution
!   determined by DGO_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm of the projected gradient of the objective function at the best
!   estimate of the solution determined by DGO_solve

       REAL ( KIND = wp ) :: norm_pg = HUGE( one )

!  the ratio of the final to the initial box lengths

       REAL ( KIND = wp ) :: length_ratio = HUGE( one )

!  the gap between the best objective value found and the lowest bound

       REAL ( KIND = wp ) :: f_gap = HUGE( one )

!  why did the iteration stop? This wil be 'D' if the box length is small 
!   enough, 'F' if the objective gap is small enough, and ' ' otherwise

       CHARACTER ( LEN = 1 ) :: why_stop = ' '

!  timings (see above)

       TYPE ( DGO_time_type ) :: time

!  inform parameters for HASH

       TYPE ( HASH_inform_type ) :: HASH_inform

!  inform parameters for UGO

       TYPE ( UGO_inform_type ) :: UGO_inform

!  inform parameters for UGO

       TYPE ( TRB_inform_type ) :: TRB_inform

     END TYPE DGO_inform_type

!  - - - - - - - - - -
!   box derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: DGO_box_type

!  has the box been pruned (i.e, excluded from further consideration)

       LOGICAL :: pruned

!  positions in vertex array of lower and upper box bounds, l and u

       INTEGER :: index_l, index_u

!  length of the box diagonal

       REAL ( KIND = wp ) :: delta

!  objective values at box bounds (for efficiency) 

       REAL ( KIND = wp ) :: f_l, f_u

!  directional derivative at box bounds

       REAL ( KIND = wp ) :: df_l, df_u

!  estimate of gradient Lipschitz constant in the box

       REAL ( KIND = wp ) :: gradient_lipschitz_estimate

!  lower and upper objective bounds in the box

       REAL ( KIND = wp ) :: f_upper, f_lower

     END TYPE DGO_box_type

!  - - - - - - - - - - -
!   vertex derived type
!  - - - - - - - - - - -

     TYPE, PUBLIC :: DGO_vertex_type

!  vertex vector

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X

!  objective value at vertex

       REAL ( KIND = wp ) :: f

!  gradient at vertex

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G

     END TYPE DGO_vertex_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

     TYPE, PUBLIC :: DGO_data_type
       INTEGER :: branch = 1
       INTEGER :: eval_status, out, error, start_print, stop_print, print_gap
       INTEGER :: print_level, print_level_ugo
       INTEGER :: jumpto, pass, length, attempts, boxes

       REAL :: time_start, time_record, time_now
       REAL ( KIND = wp ) :: clock_start, clock_record, clock_now

!      REAL ( KIND = wp ) :: alpha_l, alpha_u, alpha, phi, phi1, phi2, f_best
!      REAL ( KIND = wp ) :: rhcd

       LOGICAL :: printi, printt, printm, printw, printd, printe
       LOGICAL :: print_iteration, print_iteration_header, print_1st_header
       LOGICAL :: set_printi, set_printt, set_printm, set_printw, set_printd
       LOGICAL :: present_eval_f, present_eval_g, present_eval_h, accurate
       LOGICAL :: present_eval_hprod, present_eval_shprod, present_eval_prec

!      CHARACTER ( LEN = 1 ) :: negcur, bndry, perturb, hard
!      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_start
!      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_best
!      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: D
!      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
!      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HS

!!     INTEGER, POINTER :: nnz_p_l, nnz_p_u, nnz_hp
       INTEGER :: nnz_p_l, nnz_p_u, nnz_hp
!      LOGICAL :: got_h
       INTEGER, POINTER, DIMENSION( : ) :: INDEX_nz_p
       INTEGER, POINTER, DIMENSION( : ) :: INDEX_nz_hp
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: P => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: HP => NULL( )
!      REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: S => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: U => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: V => NULL( )

       INTEGER :: nchar
       REAL ( KIND = wp ) :: delta_0, f_best, f_upper, delta_best
       REAL ( KIND = wp ) :: phi, phi1, phi2
       LOGICAL :: got_h
       CHARACTER ( LEN = rwidth ) :: rstring
       CHARACTER ( LEN = : ), ALLOCATABLE :: string
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_l, X_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_best, G_best
       TYPE ( DGO_box_type ), ALLOCATABLE, DIMENSION( : ) :: BOX
       TYPE ( DGO_vertex_type ), ALLOCATABLE, DIMENSION( : ) :: VERTEX

!  copy of controls

       TYPE ( DGO_control_type ) :: control

!  data for HASH

       TYPE ( HASH_data_type ) :: HASH_data

!   data for UGO

       TYPE ( UGO_data_type ) :: UGO_data

!  data for TRB

       TYPE ( TRB_data_type ) :: TRB_data

     END TYPE DGO_data_type

     TYPE, PUBLIC :: DGO_full_data_type
       LOGICAL :: f_indexing
       TYPE ( DGO_data_type ) :: DGO_data
       TYPE ( DGO_control_type ) :: DGO_control
       TYPE ( DGO_inform_type ) :: DGO_inform
       TYPE ( NLPT_problem_type ) :: nlp
       TYPE ( NLPT_userdata_type ) :: userdata
     END TYPE DGO_full_data_type

   CONTAINS

!-*-*-  G A L A H A D -  D G O _ I N I T I A L I Z E  S U B R O U T I N E  -*-

     SUBROUTINE DGO_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for DGO controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( DGO_control_type ), INTENT( OUT ) :: control
     TYPE ( DGO_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     inform%status = GALAHAD_ok

!  initalize TRB components

     CALL TRB_initialize( data%TRB_data, control%TRB_control,                  &
                          inform%TRB_inform )
     control%TRB_control%prefix = '" - TRB:"                     '

!  initalize UGO components

     CALL UGO_initialize( data%UGO_data, control%UGO_control,                  &
                           inform%UGO_inform )
     control%UGO_control%prefix = '" - UGO:"                     '

!  initial private data. Set branch for initial entry

     data%branch = 10

     RETURN

!  End of subroutine DGO_initialize

     END SUBROUTINE DGO_initialize

!- G A L A H A D -  D G O _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE DGO_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for DGO controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( DGO_control_type ), INTENT( OUT ) :: control
     TYPE ( DGO_inform_type ), INTENT( OUT ) :: inform

     CALL DGO_initialize( data%dgo_data, control, inform )

     RETURN

!  End of subroutine DGO_full_initialize

     END SUBROUTINE DGO_full_initialize

!-*-*-*-*-   D G O _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE DGO_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by DGO_initialize could (roughly)
!  have been set as:

! BEGIN DGO SPECIFICATIONS (DEFAULT)
!  error-printout-device                           6
!  printout-device                                 6
!  print-level                                     1
!  start-print                                     -1
!  stop-print                                      -1
!  iterations-between-printing                     1
!  maximum-number-of-iterations                    10000
!  maximum-number-of-evaluations                   10000
!  initial-dictionary-size                         100000
!  alive-device                                    40
!  infinity-value                                  1.0D+19
!  max-box-length-required                         0.0001
!  lipschitz-lower-bound                           0.000001
!  lipschitz-reliability-parameter                 2.0
!  lipschitz-control-parameter                     50.0
!  maximum-box-length-required                     0.0001
!  maximum-objective-gap-required                  0.0001
!  minimum-objective-before-unbounded              -1.0D+32
!  maximum-cpu-time-limit                          -1.0
!  maximum-clock-time-limit                        -1.0
!  hessian-available                               yes
!  prune-boxes                                     yes
!  perform-local-optimization                      yes
!  space-critical                                  no
!  deallocate-error-fatal                          no
!  alive-filename                                  ALIVE.d
! END DGO SPECIFICATIONS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: start_print = print_level + 1
     INTEGER, PARAMETER :: stop_print = start_print + 1
     INTEGER, PARAMETER :: print_gap = stop_print + 1
     INTEGER, PARAMETER :: maxit = print_gap + 1
     INTEGER, PARAMETER :: max_evals = maxit + 1
     INTEGER, PARAMETER :: dictionary_size = max_evals + 1
     INTEGER, PARAMETER :: alive_unit = dictionary_size + 1
     INTEGER, PARAMETER :: infinity = alive_unit + 1
     INTEGER, PARAMETER :: lipschitz_lower_bound = infinity + 1
     INTEGER, PARAMETER :: lipschitz_reliability = lipschitz_lower_bound + 1
     INTEGER, PARAMETER :: lipschitz_control = lipschitz_reliability + 1
     INTEGER, PARAMETER :: stop_length = lipschitz_control + 1
     INTEGER, PARAMETER :: stop_f = stop_length + 1
     INTEGER, PARAMETER :: obj_unbounded = stop_f + 1
     INTEGER, PARAMETER :: cpu_time_limit = obj_unbounded + 1
     INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
     INTEGER, PARAMETER :: hessian_available = clock_time_limit + 1
     INTEGER, PARAMETER :: prune = hessian_available + 1
     INTEGER, PARAMETER :: perform_local_optimization = prune + 1
     INTEGER, PARAMETER :: space_critical = perform_local_optimization + 1
     INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
     INTEGER, PARAMETER :: alive_file = deallocate_error_fatal + 1
     INTEGER, PARAMETER :: prefix = alive_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'DGO '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( start_print )%keyword = 'start-print'
     spec( stop_print )%keyword = 'stop-print'
     spec( print_gap )%keyword = 'iterations-between-printing'
     spec( maxit )%keyword = 'maximum-number-of-iterations'
     spec( max_evals )%keyword = 'maximum-number-of-evaluations'
     spec( dictionary_size )%keyword = 'initial-dictionary-size'
     spec( alive_unit )%keyword = 'alive-device'

!  Real key-words

     spec( infinity )%keyword = 'infinity-value'
     spec( lipschitz_lower_bound )%keyword = 'lipschitz-lower-bound'
     spec( lipschitz_reliability )%keyword = 'lipschitz-reliability-parameter'
     spec( lipschitz_control )%keyword = 'lipschitz-control-parameter'
     spec( stop_length )%keyword = 'maximum-box-length-required'
     spec( stop_f )%keyword = 'maximum-objective-gap-required'
     spec( obj_unbounded )%keyword = 'minimum-objective-before-unbounded'
     spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
     spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

     spec( hessian_available )%keyword = 'hessian-available'
     spec( prune )%keyword = 'prune-boxes'
     spec( perform_local_optimization )%keyword = 'perform-local-optimization'
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
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print ),                           &
                                 control%stop_print,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_gap ),                            &
                                 control%print_gap,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_evals ),                            &
                                 control%max_evals,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dictionary_size ),                      &
                                 control%dictionary_size,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( alive_unit ),                           &
                                 control%alive_unit,                           &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lipschitz_lower_bound ),                &
                                 control%lipschitz_lower_bound,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lipschitz_reliability ),                &
                                 control%lipschitz_reliability,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lipschitz_control ),                    &
                                 control%lipschitz_control,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_length ),                          &
                                 control%stop_length,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_f ),                               &
                                 control%stop_f,                               &
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

     CALL SPECFILE_assign_value( spec( hessian_available ),                    &
                                 control%hessian_available,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prune ),                                &
                                 control%prune,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( perform_local_optimization ),           &
                                 control%perform_local_optimization,           &
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

!  read the controls for the hashing procedure

     IF ( PRESENT( alt_specname ) ) THEN
       CALL HASH_read_specfile( control%HASH_control, device,                  &
                                alt_specname = TRIM( alt_specname ) // '-HASH' )
       CALL TRB_read_specfile( control%TRB_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-TRB' )
       CALL UGO_read_specfile( control%UGO_control, device,                    &
                               alt_specname = TRIM( alt_specname ) // '-UGO' )
     ELSE
       CALL HASH_read_specfile( control%HASH_control, device )
       CALL TRB_read_specfile( control%TRB_control, device )
       CALL UGO_read_specfile( control%UGO_control, device )
     END IF

     RETURN

!  End of subroutine DGO_read_specfile

     END SUBROUTINE DGO_read_specfile

!-*-*-*-  G A L A H A D -  D G O _ s o l v e  S U B R O U T I N E  -*-*-*-

     SUBROUTINE DGO_solve( nlp, control, inform, data, userdata, eval_F,       &
                           eval_G, eval_H, eval_HPROD, eval_SHPROD, eval_PREC )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  DGO_solve, a deterministic method for finding a global minimizer of a 
!    given function where the variables are constrained to lie in a "box"

!  Many ingredients are based on the SmoothD algorithm from the paper

!   Yaroslav D. Sergeyev and Dmitri E. Kasov,
!   "A deterministic global optimization using smooth diagonal 
!    auxiliary functions. "
!   Comm. Nonlinear Science & Numerical Simulation, 
!   Vol 21, Nos 1-3, pp. 99-111 (2015)

!  but adapted to use 2nd derivatives

!  *-*-*-*-*-*-*-*-*-*-*-*-  A R G U M E N T S  -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  For full details see the specification sheet for GALAHAD_DGO.
!
!  ** NB. default real/complex means double precision real/complex in
!  ** GALAHAD_DGO_double
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
! control is a scalar variable of type DGO_control_type. See DGO_initialize
!  for details
!
! inform is a scalar variable of type DGO_inform_type. On initial entry,
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
!    -3. The restriction nlp%n > 0 or requirement that nlp%H_type contains
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
!  time is a scalar variable of type DGO_time_type whose components are used to
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
!  data is a scalar variable of type DGO_data_type used for internal data.
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
!   be set to a nonzero value. If eval_F is not present, DGO_solve will
!   return to the user with inform%status = 2 each time an evaluation is
!   required.
!
!  eval_G is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The components of the gradient
!   nabla_x f(x) of the objective function evaluated at x=X must be returned in
!   G, and the status variable set to 0. If the evaluation is impossible at X,
!   status should be set to a nonzero value. If eval_G is not present,
!   DGO_solve will return to the user with inform%status = 3 each time an
!   evaluation is required.
!
!  eval_H is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The nonzeros of the Hessian
!   nabla_xx f(x) of the objective function evaluated at x=X must be returned in
!   H in the same order as presented in nlp%H, and the status variable set to 0.
!   If the evaluation is impossible at X, status should be set to a nonzero
!   value. If eval_H is not present, DGO_solve will return to the user with
!   inform%status = 4 each time an evaluation is required.
!
!  eval_HPROD is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The sum u + nabla_xx f(x) v of the
!   product of the Hessian nabla_xx f(x) of the objective function evaluated
!   at x=X with the vector v=V and the vector u=U must be returned in U, and the
!   status variable set to 0. If the evaluation is impossible at X, status
!   should be set to a nonzero value. If eval_HPROD is not present, DGO_solve
!   will return to the user with inform%status = 5 each time an evaluation is
!   required. The Hessian has already been evaluated or used at x=X if got_h
!   is .TRUE.
!
!  eval_PREC is an optional subroutine which if present must have the arguments
!   given below (see the interface blocks). The product u = P(x) v of the
!   user's preconditioner P(x) evaluated at x=X with the vector v=V, the result
!   u must be retured in U, and the status variable set to 0. If the evaluation
!   is impossible at X, status should be set to a nonzero value. If eval_PREC
!   is not present, DGO_solve will return to the user with inform%status = 6
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
!   If eval_SHPROD is not present, DGO_solve will return to the user with
!   inform%status = 7 each time a sparse product is required. The Hessian has
!   already been evaluated or used at x=X if got_h is .TRUE.
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( DGO_control_type ), INTENT( IN ) :: control
     TYPE ( DGO_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( DGO_data_type ), INTENT( INOUT ) :: data
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

     INTEGER :: i, j, l, st, best, index_l, index_u, index_l_best, index_u_best
     INTEGER, DIMENSION( 1 ) :: loc
     REAL ( KIND = wp ) :: b, d, m, phix, term1, term2, term3
     REAL ( KIND = wp ) :: lipschitz_estimate_max, x, y, yp, delta_upper
     LOGICAL :: alive
     CHARACTER ( LEN = 1 ) :: it_type
     CHARACTER ( LEN = 80 ) :: array_name
     TYPE ( DGO_inform_type ) :: inform_initialize

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
     CASE ( 60 )  ! function and derivatives evaluations
       GO TO 60
     CASE ( 110 )  ! function and derivative evaluations
       GO TO 110
     CASE ( 120 )  ! function and derivative evaluations
       GO TO 120
     CASE ( 250 )  ! function and derivative evaluations
       GO TO 250
     CASE ( 510 )  ! function and derivative evaluations
       GO TO 510
     CASE ( 520 )  ! function and derivative evaluations
       GO TO 520
     END SELECT

!  ============================================================================
!  0. Initialization
!  ============================================================================

  10 CONTINUE
     CALL CPU_time( data%time_start ) ; CALL CLOCK_time( data%clock_start )

!  record input control components

     data%control = control

!  initialize components of inform 

     inform = inform_initialize

!  basic single line of output per iteration

     data%out = data%control%out
     data%set_printi = data%out > 0 .AND. data%control%print_level >= 1

!  ensure that input parameters are within allowed ranges

     IF ( nlp%n <= 0 ) THEN
       inform%status = GALAHAD_error_restrictions ; GO TO 990
     END IF

!  check that the simple bounds are consistent and finite

     DO i = 1, nlp%n
       IF ( nlp%X_l( i ) > nlp%X_u( i ) .OR.                                   &
            nlp%X_l( i ) <= - data%control%infinity .OR.                       &
            nlp%X_u( i ) >= data%control%infinity ) THEN
         inform%status = GALAHAD_error_bad_bounds ; GO TO 990
       END IF       
     END DO

!  record whether external evaluation procudures are presnt

     data%present_eval_f = PRESENT( eval_F )
     data%present_eval_g = PRESENT( eval_G )
     data%present_eval_h = PRESENT( eval_H )
     data%present_eval_hprod = PRESENT( eval_HPROD )
     data%present_eval_shprod = PRESENT( eval_SHPROD )
     data%present_eval_prec = PRESENT( eval_PREC )

!  control the output printing

     IF ( data%control%start_print < 0 ) THEN
       data%start_print = - 1
     ELSE
       data%start_print = data%control%start_print
     END IF

     IF ( data%control%stop_print < 0 ) THEN
       data%stop_print = data%control%maxit + 1
     ELSE
       data%stop_print = data%control%stop_print
     END IF

     IF ( data%control%print_gap < 2 ) THEN
       data%print_gap = 1
     ELSE
       data%print_gap = data%control%print_gap
     END IF

     data%print_1st_header = .TRUE.

!  basic single line of output per iteration

     data%printi = data%out > 0 .AND. data%control%print_level >= 1

!  as per printi, but with additional timings for various operations

     data%set_printt = data%out > 0 .AND. data%control%print_level >= 2

!  as per printt with a few more scalars

     data%set_printm = data%out > 0 .AND. data%control%print_level >= 3

!  as per printw with a few vectors

     data%set_printw = data%out > 0 .AND. data%control%print_level >= 4

!  full debug printing

     data%set_printd = data%out > 0 .AND. data%control%print_level > 10

!  set iteration-specific print controls

     IF ( inform%iter >= data%start_print .AND.                                &
          inform%iter < data%stop_print ) THEN
!         inform%iter < data%stop_print .AND.                                  &
!         MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 ) THEN
       data%printi = data%set_printi ; data%printt = data%set_printt
       data%printm = data%set_printm ; data%printw = data%set_printw
       data%printd = data%set_printd
       data%print_level = data%control%print_level
     ELSE
       data%printi = .FALSE. ; data%printt = .FALSE.
       data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
       data%print_level = 0
     END IF

!  create a file which the user may subsequently remove to cause
!  immediate termination of a run

     IF ( data%control%alive_unit > 0 ) THEN
      INQUIRE( FILE = data%control%alive_file, EXIST = alive )
      IF ( .NOT. alive ) THEN
         OPEN( data%control%alive_unit, FILE = data%control%alive_file,                  &
               FORM = 'FORMATTED', STATUS = 'NEW' )
         REWIND data%control%alive_unit
         WRITE( data%control%alive_unit, "( ' GALAHAD rampages onwards ' )" )
         CLOSE( data%control%alive_unit )
       END IF
     END IF

!  re-entry

  20 CONTINUE
     IF ( data%printi ) WRITE( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n

!  initialize iteration counter

     inform%iter = 0

!  initialize the stop indicator

     inform%why_stop  = ' '

!  check for special case for which n = 1

     IF ( nlp%n > 1 ) GO TO 100

!  ===============================================================
!  Special case when n = 1: perform univariate global optimization
!  ===============================================================

!  -----------------------------------------------------------
!  implicit loop to perform the global univariate minimization
!  -----------------------------------------------------------

     inform%UGO_inform%status = 1
     CALL CPU_time( data%time_record ); CALL CLOCK_time( data%clock_record )
  50 CONTINUE

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
           data%branch = 60 ; RETURN
         END IF
       END IF

!  return from reverse communication

  60 CONTINUE
     IF ( inform%UGO_inform%status >= 2 ) THEN
       data%phi = nlp%f
       IF ( inform%UGO_inform%status >= 3 ) THEN
         data%phi1 = DOT_PRODUCT( data%P, nlp%G )
         IF ( inform%UGO_inform%status >= 4 )                                  &
           data%phi2 = DOT_PRODUCT( data%P, data%U )
       END IF
       GO TO 50
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
     inform%iter = inform%UGO_inform%iter
     IF ( inform%status == GALAHAD_ok ) THEN
       inform%why_stop  = 'D'
       inform%length_ratio = inform%UGO_inform%dx_best
     END IF

!  record the clock time

     CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
     data%time_now = data%time_now - data%time_start
     data%clock_now = data%clock_now - data%clock_start

!  control printing

     data%print_iteration_header = data%printt .OR. data%TRB_data%printi

!  print one-line summary

     IF ( data%printi ) THEN
       WRITE( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n
       IF ( data%print_iteration_header .OR. data%print_1st_header )           &
         WRITE( data%out, "( '             f                g         ',       &
        &                    ' #f      #g        time' )" )
       WRITE( data%out, "( A, ES24.16, ES11.4, 2I8, F12.2 )" ) prefix,         &
         inform%obj, inform%norm_pg, inform%f_eval, inform%g_eval,             &
         data%clock_now
     END IF
     GO TO 900

!  ========================================================
!  end of special univariate global optimization when n = 1
!  ========================================================

!  generic case of global optimization when n > 1

 100 CONTINUE

!  set space required to hold n double-precision numbers

     data%nchar = rwidth * nlp%n 

!  allocate a workspace string of this length

     ALLOCATE( CHARACTER( LEN = data%nchar ) :: data%string,                   &
               STAT = inform%alloc_status )
     IF ( inform%alloc_status /= GALAHAD_ok ) THEN
       inform%bad_alloc = 'dgo: data%string'
       inform%status = GALAHAD_error_allocate ; GO TO 910
     END IF

!  initialiize the hash table 

     data%length = data%control%dictionary_size
     CALL HASH_initialize( data%nchar, data%length, data%HASH_data,            &
                           data%control%HASH_control, inform%HASH_inform )
     IF ( inform%HASH_inform%status /= GALAHAD_ok ) THEN
       inform%status = inform%HASH_inform%status ; GO TO 910
     END IF

!  provide storage for the vertices and boxes

     ALLOCATE( data%VERTEX( data%length ), STAT = inform%alloc_status )
     IF ( inform%alloc_status /= GALAHAD_ok ) THEN
       inform%bad_alloc = 'dgo: data%VERTEX'
       inform%status = GALAHAD_error_allocate ; GO TO 910
     END IF

     ALLOCATE( data%BOX( 2 * ( data%control%maxit ) + 1 ),                     &
               STAT = inform%alloc_status )
     IF ( inform%alloc_status /= GALAHAD_ok ) THEN
       inform%bad_alloc = 'dgo: data%BOX'
       inform%status = GALAHAD_error_allocate ; GO TO 910
     END IF

!  provide storage for new verties as they arise

     array_name = 'dgo: data%X_l'
     CALL SPACE_resize_array( nlp%n, data%X_l, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

     array_name = 'dgo: data%X_u'
     CALL SPACE_resize_array( nlp%n, data%X_u, inform%status,                  &
            inform%alloc_status, array_name = array_name,                      &
            deallocate_error_fatal = data%control%deallocate_error_fatal,      &
            exact_size = data%control%space_critical,                          &
            bad_alloc = inform%bad_alloc, out = data%control%error )
     IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  initialisation

     data%f_upper = HUGE( one )
     data%f_best = HUGE( one )
     data%control%lipschitz_control = data%control%lipschitz_control *         &
       REAL( MAX( 1, nlp%n - 1 ), KIND = wp )
       
!  consider vertex x_l, and find its position, index_l, in the dictionary

     CALL DGO_vertex( nlp%n, nlp%X_l, data%string, data%rstring, index_l,      &
                      data%HASH_data, data%control%HASH_control,               &
                      inform%HASH_inform )

!  the vertex is new

     IF ( index_l > 0 ) THEN

!  initialize the vertex data

       CALL DGO_allocate_vertex_arrays( nlp%n, data%VERTEX( index_l ),         &
                                        data%control, inform )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  record the vertex

       data%VERTEX( index_l )%X( : nlp%n ) = nlp%X_l( : nlp%n )

!  compute the objective function value if explicitly available ...

       inform%f_eval = inform%f_eval + 1
       IF ( data%present_eval_f ) THEN
         CALL eval_F( inform%status, data%VERTEX( index_l )%X, userdata,       &
                      data%VERTEX( index_l )%f )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910
         data%f_upper = MIN( data%f_upper, data%VERTEX( index_l )%f )
       ELSE
         inform%status = 2
       END IF

!  ... and its gradient

       inform%g_eval = inform%g_eval + 1
       IF ( data%present_eval_g ) THEN
         CALL eval_G( inform%status, data%VERTEX( index_l )%X, userdata,       &
                      data%VERTEX( index_l )%G )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910
       ELSE
         IF ( inform%status == 2 ) THEN
           inform%status = 23
         ELSE
           inform%status = 3
         END IF
       END IF 
     ELSE
       inform%status = GALAHAD_ok
     END IF

!  if necessary, obtain f and/or g via reverse communication

     IF ( inform%status /= GALAHAD_ok ) THEN
       nlp%X( : nlp%n ) = data%VERTEX( index_l )%X( : nlp%n )
       data%branch = 110 ; RETURN
     END IF

!  return from reverse communication with the objective and gradient values

 110 CONTINUE

!  the vertex is new, record any reverse-communication f and g

     IF ( index_l > 0 ) THEN
       IF ( .NOT. data%present_eval_f ) THEN
         data%VERTEX( index_l )%f = nlp%f
         data%f_upper = MIN( data%f_upper, data%VERTEX( index_l )%f )
       END IF
       IF ( .NOT. data%present_eval_g ) THEN
         data%VERTEX( index_l )%G( : nlp%n ) = nlp%G( : nlp%n )
       END IF

!  this is an existing vertex, reuse the objective value and gradient

     ELSE IF ( index_l < 0 ) THEN
       index_l = - index_l

!  the dictionary is full

     ELSE
       IF ( data%printi ) WRITE( data%out, "( ' dictionary full' )" ) 
       inform%status = GALAHAD_error_max_storage ; GO TO 990
     END IF

!  do the same to vertex x_u

     CALL DGO_vertex( nlp%n, nlp%X_u, data%string, data%rstring, index_u,      &
                      data%HASH_data, data%control%HASH_control,               &
                      inform%HASH_inform )

!  the vertex is new

     IF ( index_u > 0 ) THEN

!  initialize the vertex data

       CALL DGO_allocate_vertex_arrays( nlp%n, data%VERTEX( index_u ),         &
                                        data%control, inform )
       IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  record the vertex

       data%VERTEX( index_u )%X( : nlp%n ) = nlp%X_u( : nlp%n )

!  compute the objective function value if explicitly available ...

       inform%f_eval = inform%f_eval + 1
       IF ( data%present_eval_f ) THEN
         CALL eval_F( inform%status, data%VERTEX( index_u )%X, userdata,       &
                      data%VERTEX( index_u )%f )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910
         data%f_upper = MIN( data%f_upper, data%VERTEX( index_u )%f )
       ELSE
         inform%status = 2
       END IF

!  ... and its gradient

       inform%g_eval = inform%g_eval + 1
       IF ( data%present_eval_g ) THEN
         CALL eval_G( inform%status, data%VERTEX( index_u )%X, userdata,       &
                      data%VERTEX( index_u )%G )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910
       ELSE
         IF ( inform%status == 2 ) THEN
           inform%status = 23
         ELSE
           inform%status = 3
         END IF
       END IF 
     ELSE
       inform%status = GALAHAD_ok
     END IF

!  if necessary, obtain f and/or g via reverse communication

     IF ( inform%status /= GALAHAD_ok ) THEN
       nlp%X( : nlp%n ) = data%VERTEX( index_u )%X( : nlp%n )
       data%branch = 120 ; RETURN
     END IF

!  return from reverse communication with the objective and gradient values

 120 CONTINUE

!  the vertex is new, record any reverse-communication f and g

     IF ( index_u > 0 ) THEN
       IF ( .NOT. data%present_eval_f ) THEN
         data%VERTEX( index_u )%f = nlp%f
         data%f_upper = MIN( data%f_upper, data%VERTEX( index_u )%f )
       END IF
       IF ( .NOT. data%present_eval_g ) THEN
         data%VERTEX( index_u )%G( : nlp%n ) = nlp%G( : nlp%n )
       END IF

!  this is an existing vertex, reuse the objective value and gradient

     ELSE IF ( index_u < 0 ) THEN
       index_u = - index_u

!  the dictionary is full

     ELSE
       IF ( data%printi ) WRITE( data%out, "( ' dictionary full' )" ) 
       inform%status = GALAHAD_error_max_storage ; GO TO 990
     END IF

!  set up the first box with vertices x_l and x_u

     data%boxes = 1
     CALL DGO_initialize_box( index_l, data%VERTEX( index_l ),                 &
                              index_u, data%VERTEX( index_u ),                 &
                              data%BOX( data%boxes ) )

!  record the initial diagonal

     data%delta_0 = data%BOX( data%boxes )%delta
     data%delta_best = data%delta_0

!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!   main iteration (steps correspond to Sergeyev & Kvasov's SmoothD algorithm)
!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

 200 CONTINUE

!  control printing

       IF ( inform%iter >= data%start_print .AND.                              &
            inform%iter < data%stop_print ) THEN
!           inform%iter < data%stop_print .AND.                                &
!           MOD( inform%iter + 1 - data%start_print, data%print_gap ) == 0 )   &
!          THEN
         data%printi = data%set_printi ; data%printt = data%set_printt
         data%printm = data%set_printm ; data%printw = data%set_printw
         data%printd = data%set_printd
         data%print_level = data%control%print_level
       ELSE
         data%printi = .FALSE. ; data%printt = .FALSE.
         data%printm = .FALSE. ; data%printw = .FALSE. ; data%printd = .FALSE.
         data%print_level = 0
       END IF
       data%print_iteration_header = data%print_level > 1
       data%print_iteration = .FALSE.

!  if desired, improve the best estimate found using local optimization

       IF ( data%f_upper >= data%f_best .OR.                                   &
            .NOT. data%control%perform_local_optimization ) GO TO 300

!  =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
!                         Local optimization 
!  =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

       data%print_iteration = .TRUE.

!  start from the current box vertex with the smallest value

       loc = MINLOC( data%BOX( : data%boxes )%f_upper )
       best = loc( 1 )

!  return the best value of the objective found

       data%f_upper = data%BOX( best )%f_upper
       nlp%f = data%BOX( best )%f_l
       IF ( data%f_upper == nlp%f ) THEN
         nlp%X( : nlp%n ) = data%VERTEX( data%BOX( best )%index_l )%X( : nlp%n )
!        nlp%G( : nlp%n ) = data%VERTEX( data%BOX( best )%index_l )%G( : nlp%n )
       ELSE
         nlp%f = data%BOX( best )%f_u
         nlp%X( : nlp%n ) = data%VERTEX( data%BOX( best )%index_u )%X( : nlp%n )
!        nlp%G( : nlp%n ) = data%VERTEX( data%BOX( best )%index_u )%G( : nlp%n )
       END IF
!      inform%norm_pg = TWO_NORM( nlp%X -                                      &
!            TRB_projection( nlp%n, nlp%X - nlp%G, nlp%X_l, nlp%X_u ) )

!  ============================================================================
!  Find a local minimizer x_k of f(x) from the current starting point xs_k
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
       data%control%TRB_control%hessian_available                              &
         = data%control%hessian_available
       data%control%TRB_control%maxit = MIN( data%control%TRB_control%maxit,   &
           data%control%max_evals - inform%f_eval )

       CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
 210   CONTINUE

!  call the bound-constrained local minimizer

         CALL TRB_solve( nlp, data%control%TRB_control, inform%TRB_inform,     &
                         data%TRB_data, userdata, eval_F = eval_F,             &
                         eval_G = eval_G, eval_H = eval_H,                     &
                         eval_HPROD = eval_HPROD, eval_SHPROD = eval_SHPROD,   &
                         eval_PREC = eval_PREC )

!  obtain further function information if required

         SELECT CASE ( inform%TRB_inform%status )

!  obtain the objective function value

         CASE ( 2 )
           IF ( data%present_eval_f ) THEN
             CALL eval_F( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          inform%TRB_inform%obj )
           ELSE
             data%branch = 250 ; inform%status = 2 ; RETURN
           END IF

!  obtain the gradient value

         CASE ( 3 )
           IF ( data%present_eval_g ) THEN
             CALL eval_G( data%eval_status, nlp%X( : nlp%n ), userdata,        &
                          nlp%G( : nlp%n ) )
           ELSE
             data%branch = 250 ; inform%status = 3 ; RETURN
           END IF

!  obtain the Hessian value

         CASE ( 4 )
           IF ( data%present_eval_h ) THEN
             CALL eval_H( data%eval_status, nlp%X( : nlp%n ),                  &
                          userdata, nlp%H%val( : nlp%H%ne ) )
           ELSE
             data%branch = 250 ; inform%status = 4 ; RETURN
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
             data%branch = 250 ; inform%status = 5 ; RETURN
           END IF

!  obtain a preconditioned vector product

         CASE ( 6 )
           data%U => data%TRB_data%U
           data%V => data%TRB_data%V
           IF ( data%present_eval_prec ) THEN
             CALL eval_PREC( data%eval_status, nlp%X( : nlp%n ), userdata,     &
                             data%U( : nlp%n ), data%V( : nlp%n ) )
           ELSE
             data%branch = 250 ; inform%status = 6 ; RETURN
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
             data%branch = 250 ; inform%status = 7 ; RETURN
           END IF

!  terminal exits

         CASE DEFAULT
           GO TO 290
         END SELECT

!  return from reverse communication

 250     CONTINUE
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

         GO TO 210

!  ------------------------------------------------
!  end of local bound-constrained minimization loop
!  ------------------------------------------------

 290   CONTINUE

!  record the time taken in the local minimization

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       inform%time%multivariate_local =                                        &
         inform%time%multivariate_local +                                      &
           data%time_now - data%time_record
       inform%time%clock_multivariate_local =                                  &
         inform%time%clock_multivariate_local +                                &
           data%clock_now - data%clock_record

!  record details about the critical point found

!write(6,*) ' # f, g, h ', inform%TRB_inform%f_eval, inform%TRB_inform%g_eval, &
!inform%TRB_inform%h_eval
       inform%f_eval = inform%f_eval + inform%TRB_inform%f_eval
       inform%g_eval = inform%g_eval + inform%TRB_inform%g_eval
       inform%h_eval = inform%h_eval + inform%TRB_inform%h_eval
       inform%obj = inform%TRB_inform%obj
       inform%norm_pg = inform%TRB_inform%norm_pg
       data%f_best = inform%obj
       data%f_upper = data%f_best
       data%X_best = nlp%X
       data%G_best = nlp%G
       data%P => data%TRB_data%P
!write(6,*) ' data%P associated '
       IF ( inform%TRB_inform%status < 0 .AND. data%printe ) THEN
         IF ( inform%TRB_inform%status /= GALAHAD_error_max_iterations )       &
           WRITE( data%error, "( ' Help! exit from TRB status = ', I0 )" )     &
             inform%TRB_inform%status
       END IF

!  =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=
!                        End of local optimization 
!  =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=

  300  CONTINUE

!  record the clock time

       CALL CPU_time( data%time_now ) ; CALL CLOCK_time( data%clock_now )
       data%time_now = data%time_now - data%time_start
       data%clock_now = data%clock_now - data%clock_start

!  control printing

       IF ( data%printi ) THEN
         loc = MAXLOC( data%BOX( : data%boxes )%delta,                         &
                       MASK = .NOT. DATA%BOX( : data%boxes )%pruned )
         delta_upper = data%BOX( loc( 1 ) )%delta
!        delta_upper = MAXVAL( data%BOX( : data%boxes )%delta )
         IF ( data%print_iteration ) THEN
           it_type = 'F'
         ELSE IF ( delta_upper < data%delta_best ) THEN
           data%delta_best = delta_upper
           data%print_iteration = .TRUE.
           it_type = 'D'
         ELSE IF ( MOD( inform%iter, data%print_gap ) == 0 ) THEN
           data%print_iteration = .TRUE.
           it_type = ' '
         END IF

!  print one-line summary

         IF ( data%print_iteration ) THEN
!          data%print_iteration_header = data%printt .OR.                      &
!             ( data%UGO_data%printi .AND. it_type == 'F' )
           data%print_iteration_header = data%printt
           IF ( data%print_iteration_header .OR. data%print_1st_header ) THEN
             WRITE( data%out, 2010 ) prefix
             data%print_1st_header = .FALSE.
           END IF
           WRITE( data%out, 2020 ) prefix, inform%iter, it_type, data%f_best,  &
             data%delta_best, inform%f_eval, inform%g_eval, data%boxes -       &
             COUNT( DATA%BOX( : data%boxes )%pruned ), data%clock_now
         END IF
       END IF

!  check that the iteration limit has not been reached

       inform%iter = inform%iter + 1
       IF ( inform%iter > data%control%maxit ) THEN
         inform%status = GALAHAD_error_max_iterations ; GO TO 800
       END IF

!  ============================================================================
!  1. Estimate the gradient Lipschitz constant
!  ============================================================================

       loc = MAXLOC( data%BOX( : data%boxes )%gradient_lipschitz_estimate,     &
                     MASK = .NOT. data%BOX( : data%boxes )%pruned )
       lipschitz_estimate_max = data%BOX( loc( 1 ) )%gradient_lipschitz_estimate
       m = ( data%control%lipschitz_reliability +                              &
             data%control%lipschitz_control / REAL( inform%iter, KIND = wp ) ) &
             * MAX( data%control%lipschitz_lower_bound, lipschitz_estimate_max )

!  ============================================================================
!  2. Characteristics calculation
!  ============================================================================

       DO i = 1, data%boxes
         IF ( data%BOX( i )%pruned ) CYCLE

!  2.1: compute a lower bound of the underestimating function phi

         term1 = quarter * data%BOX( i )%delta
         term2 = quarter * ( data%BOX( i )%df_u - data%BOX( i )%df_l ) / m
         term3 = ( data%BOX( i )%f_l - data%BOX( i )%f_u                       &
                   + data%BOX( i )%df_u * data%BOX( i )%delta                  &
                   + half * m * data%BOX( i )%delta ** 2 )                     &
                   / ( m * data%BOX( i )%delta + data%BOX( i )%df_u            &
                         - data%BOX( i )%df_l )
         y = term1 + term2 + term3
         yp = - term1 - term2 + term3
         b = data%BOX( i )%df_u -two * m * y + m * data%BOX( i )%delta

!  2.2: lower bound at phi(x) or endpoints

         IF ( ( m * y + b ) * ( m * yp + b ) < zero ) THEN
           x = two * y - data%BOX( i )%df_u / m - data%BOX( i )%delta
           phix = data%BOX( i )%f_u - data%BOX( i )%df_u * data%BOX( i )%delta &
                    - half * m * data%BOX( i )%delta**2 + m * y ** 2           &
                    - half * m * x ** 2
           data%BOX( i )%f_lower                                               &
             = MIN( data%BOX( i )%f_l, phix, data%BOX( i )%f_u )

!  2.3: lower bound at endpoints

         ELSE
           data%BOX( i )%f_lower = MIN( data%BOX( i )%f_l, data%BOX( i )%f_u ) 
         END IF

!  if desired, prune the current boxes to exclude those that cannot contain
!  a global minimizer

         IF ( data%control%prune ) THEN
           IF ( data%BOX( i )%f_lower > data%f_upper ) THEN
             IF ( data%printm ) WRITE( data%out, "( A, ' iteration ', I0,      &
            &  ' pruning box ', I0, ' l, u ', 2ES12.4 )" ) prefix,             &
               inform%iter, i, data%BOX( i )%f_lower, data%f_upper
             data%BOX( i )%pruned = .TRUE.
           END IF
         END IF
       END DO

!  ============================================================================
!  3. Hyperinterval selection: compute a "best" box, i.e., one whose lower 
!     objective bound is smallest
!  ============================================================================

       loc = MINLOC( data%BOX( : data%boxes )%f_lower )
       best = loc( 1 )

!  record the indices of the vertices x_l_best and x_l_best in this box

       index_l_best = data%BOX( best )%index_l
       index_u_best = data%BOX( best )%index_u 

!  describe the box if required

       IF ( data%printm ) THEN
         WRITE( data%out, "( /, A, ' iteration ', I0, ' best box ', I0 )" )    &
           prefix, inform%iter, best
         WRITE( data%out, "( A, ' l =', /, ( 5ES16.8 ) )" )                    &
           prefix, DATA%VERTEX( index_l_best )%X( : nlp%n )
         WRITE( data%out, "( A, ' u =', /, ( 5ES16.8 ) )" )                    &
           prefix, DATA%VERTEX( index_u_best )%X( : nlp%n )
         WRITE( data%out, "( A, ' f_lower, f_upper, delta', 3ES16.8 )" )       &
           prefix, data%BOX( best )%f_lower, data%BOX( best )%f_upper,         &
           data%BOX( best )%delta
       END IF

!  ============================================================================
!  4. Test for termination
!  ============================================================================

!  stop if the maximum box length is sufficiently small

       inform%length_ratio = data%BOX( best )%delta / data%delta_0
       IF ( inform%length_ratio <= data%control%stop_length ) THEN
         inform%why_stop  = 'D' ; inform%status = GALAHAD_ok ; GO TO 800
       END IF

!  stop if the objective function gap is sufficiently small

       inform%f_gap = data%f_best - data%BOX( best )%f_lower 
       IF  ( inform%f_gap <= data%control%stop_f ) THEN
         inform%why_stop  = 'F' ; inform%status = GALAHAD_ok ; GO TO 800
       END IF

!  ============================================================================
!  5. Generation of the new points x_l and x_u
!  ============================================================================

!  initialize these as the current lower and upper vertices in the best box

       data%X_l( : nlp%n ) = data%VERTEX( index_l_best )%X( : nlp%n )
       data%X_u( : nlp%n ) = data%VERTEX( index_u_best )%X( : nlp%n )

!  compute an index that corresponds to a longest edge d of the box

       loc = MAXLOC( ABS( data%X_u( : nlp%n ) - data%X_l( : nlp%n ) ) )
       i = loc( 1 )
       d = data%X_u( i ) - data%X_l( i )

!  pick x_l and x_u to trisect this edge

       data%X_l( i ) = data%X_l( i ) + twothirds * d
       data%X_u( i ) = data%X_u( i ) - twothirds * d

!  consider vertex x_l, and find its position, index_l, in the dictionary

       CALL DGO_vertex( nlp%n, data%X_l, data%string, data%rstring, index_l,   &
                        data%HASH_data, data%control%HASH_control,             &
                        inform%HASH_inform )

!  the vertex is new

       IF ( index_l > 0 ) THEN

!  initialize the vertex data

         CALL DGO_allocate_vertex_arrays( nlp%n, data%VERTEX( index_l ),       &
                                          data%control, inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  record the vertex

         data%VERTEX( index_l )%X( : nlp%n ) = data%X_l( : nlp%n )

!  compute the objective function value if explicitly available ...

         inform%f_eval = inform%f_eval + 1
         IF ( data%present_eval_f ) THEN
           CALL eval_F( inform%status, data%VERTEX( index_l )%X, userdata,     &
                        data%VERTEX( index_l )%f )
           IF ( inform%status /= GALAHAD_ok ) GO TO 910
           data%f_upper = MIN( data%f_upper, data%VERTEX( index_l )%f )
         ELSE
           inform%status = 2
         END IF

!  ... and its gradient

         inform%g_eval = inform%g_eval + 1
         IF ( data%present_eval_g ) THEN
           CALL eval_G( inform%status, data%VERTEX( index_l )%X, userdata,     &
                        data%VERTEX( index_l )%G )
           IF ( inform%status /= GALAHAD_ok ) GO TO 910
         ELSE
           IF ( inform%status == 2 ) THEN
             inform%status = 23
           ELSE
             inform%status = 3
           END IF
         END IF 
       ELSE
         inform%status = GALAHAD_ok
       END IF

!  if necessary, obtain f and/or g via reverse communication

       IF ( inform%status /= GALAHAD_ok ) THEN
         nlp%X( : nlp%n ) = data%VERTEX( index_l )%X( : nlp%n )
         data%branch = 510 ; RETURN
       END IF

!  return from reverse communication with the objective and gradient values

 510   CONTINUE

!  the vertex is new, record any reverse-communication f and g

       IF ( index_l > 0 ) THEN
         IF ( .NOT. data%present_eval_f ) THEN
           data%VERTEX( index_l )%f = nlp%f
           data%f_upper = MIN( data%f_upper, data%VERTEX( index_l )%f )
         END IF
         IF ( .NOT. data%present_eval_g ) THEN
           data%VERTEX( index_l )%G( : nlp%n ) = nlp%G( : nlp%n )
         END IF

!  this is an existing vertex, reuse the objective value and gradient

       ELSE IF ( index_l < 0 ) THEN
         index_l = - index_l

!  the dictionary is full

       ELSE
         IF ( data%printi ) WRITE( data%out, "( ' dictionary full' )" ) 
         inform%status = GALAHAD_error_max_storage ; GO TO 990
       END IF

!  do the same to vertex x_u

       CALL DGO_vertex( nlp%n, data%X_u, data%string, data%rstring, index_u,   &
                        data%HASH_data, data%control%HASH_control,             &
                        inform%HASH_inform )

!  the vertex is new

       IF ( index_u > 0 ) THEN

!  initialize the vertex data

         CALL DGO_allocate_vertex_arrays( nlp%n, data%VERTEX( index_u ),       &
                                          data%control, inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  record the vertex

         data%VERTEX( index_u )%X( : nlp%n ) = data%X_u( : nlp%n )

!  compute the objective function value if explicitly available ...

         inform%f_eval = inform%f_eval + 1
         IF ( data%present_eval_f ) THEN
           CALL eval_F( inform%status, data%VERTEX( index_u )%X, userdata,     &
                        data%VERTEX( index_u )%f )
           IF ( inform%status /= GALAHAD_ok ) GO TO 910
           data%f_upper = MIN( data%f_upper, data%VERTEX( index_u )%f )
         ELSE
           inform%status = 2
         END IF

!  ... and its gradient

         inform%g_eval = inform%g_eval + 1
         IF ( data%present_eval_g ) THEN
           CALL eval_G( inform%status, data%VERTEX( index_u )%X, userdata,     &
                        data%VERTEX( index_u )%G )
           IF ( inform%status /= GALAHAD_ok ) GO TO 910
         ELSE
           IF ( inform%status == 2 ) THEN
             inform%status = 23
           ELSE
             inform%status = 3
           END IF
         END IF 
       ELSE
         inform%status = GALAHAD_ok
       END IF

!  if necessary, obtain f and/or g via reverse communication

       IF ( inform%status /= GALAHAD_ok ) THEN
         nlp%X( : nlp%n ) = data%VERTEX( index_u )%X( : nlp%n )
         data%branch = 520 ; RETURN
       END IF

!  return from reverse communication with the objective and gradient values

 520   CONTINUE

!  the vertex is new, record any reverse-communication f and g

       IF ( index_u > 0 ) THEN
         IF ( .NOT. data%present_eval_f ) THEN
           data%VERTEX( index_u )%f = nlp%f
           data%f_upper = MIN( data%f_upper, data%VERTEX( index_u )%f )
         END IF
         IF ( .NOT. data%present_eval_g ) THEN
           data%VERTEX( index_u )%G( : nlp%n ) = nlp%G( : nlp%n )
         END IF

!  this is an existing vertex, reuse the objective value and gradient

       ELSE IF ( index_u < 0 ) THEN
         index_u = - index_u

!  the dictionary is full

       ELSE
         IF ( data%printi ) WRITE( data%out, "( ' dictionary full' )" ) 
         inform%status = GALAHAD_error_max_storage ; GO TO 990
       END IF

!  ============================================================================
!  6. Efficient diagonal partition
!  ============================================================================

!  replace the best box with one with vertices x_l and x_u

       CALL DGO_initialize_box( index_l, data%VERTEX( index_l ),               &
                                index_u, data%VERTEX( index_u ),               &
                                data%BOX( best ) )

!  add a new box with vertices x_l_best and x_u

       data%boxes = data%boxes + 1
       CALL DGO_initialize_box( index_l_best, data%VERTEX( index_l_best ),     &
                                index_u, data%VERTEX( index_u ),               &
                                data%BOX( data%boxes ) )

!  and another with vertices x_l and x_u_best

       data%boxes = data%boxes + 1
       CALL DGO_initialize_box( index_l, data%VERTEX( index_l ),               &
                                index_u_best, data%VERTEX( index_u_best ),     &
                                data%BOX( data%boxes ) )

       GO TO 200

!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!                            end of main iteration
!  =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

 800 CONTINUE

!  print one-line summary

     IF ( data%printi ) THEN
       IF ( data%printt .OR. data%print_1st_header ) THEN
         WRITE( data%out, 2010 ) prefix
         data%print_1st_header = .FALSE.
       END IF
       loc = MAXLOC( data%BOX( : data%boxes )%delta,                           &
                     MASK = .NOT. DATA%BOX( : data%boxes )%pruned )
       data%delta_best = data%BOX( loc( 1 ) )%delta
       WRITE( data%out, 2020 ) prefix, inform%iter, it_type, data%f_best,      &
         data%delta_best, inform%f_eval, inform%g_eval, data%boxes -           &
         COUNT( DATA%BOX( : data%boxes )%pruned ), data%clock_now
     END IF

!  if local optimization has been used, return the best value found as a
!  consequence as the candidate global mimimizer

     IF ( data%control%perform_local_optimization ) THEN
       inform%obj = data%f_best
       nlp%X = data%X_best
       nlp%G = data%G_best

!  otherwise, return the vertex with smallest upper objective bound as 
!  the candidate global mimimizer

     ELSE
       loc = MINLOC( data%BOX( : data%boxes )%f_upper )
       best = loc( 1 )

!  return the best value of the objective found

       data%f_upper = data%BOX( best )%f_upper
       inform%obj = data%BOX( best )%f_l
       IF ( data%f_upper == inform%obj ) THEN
         nlp%X( : nlp%n ) = data%VERTEX( data%BOX( best )%index_l )%X( : nlp%n )
         nlp%G( : nlp%n ) = data%VERTEX( data%BOX( best )%index_l )%G( : nlp%n )
       ELSE
         inform%obj = data%BOX( best )%f_u
         nlp%X( : nlp%n ) = data%VERTEX( data%BOX( best )%index_u )%X( : nlp%n )
         nlp%G( : nlp%n ) = data%VERTEX( data%BOX( best )%index_u )%G( : nlp%n )
       END IF
       inform%norm_pg = TWO_NORM( nlp%X -                                      &
             TRB_projection( nlp%n, nlp%X - nlp%G, nlp%X_l, nlp%X_u ) )
     END IF

!  output the global minimum

 900 CONTINUE
     IF ( data%printi ) THEN
       WRITE( data%out, 2000 ) prefix, TRIM( nlp%pname ), nlp%n
       IF ( inform%status == GALAHAD_ok ) THEN
         WRITE( data%out, "( A, ' Minimum value of', ES22.14 )" )              &
           prefix, inform%obj
         IF ( data%printw ) WRITE( data%out,                                   &
           "( A, ' at X = ', /, ( 5X, 5ES12.4 ) )" ) prefix, nlp%X( : nlp%n )
         IF ( inform%why_stop == 'D' ) THEN
           WRITE( data%out, "( A, ' found with guaranteed length tolerance',   &
          &   ES11.4, ' in ', I0, ' iterations', /, A, ' using ', I0,          &
          &  ' function and ', I0, ' gradient evaluations' )" )                &
            prefix, data%control%stop_length, inform%iter,                     &
            prefix, inform%f_eval, inform%g_eval
         ELSE IF ( inform%why_stop == 'F' ) THEN
           WRITE( data%out, "( A, ' found with guaranteed objective gap',      &
          &   ES11.4, ' in ', I0, ' iterations', /, A, ' using ', I0,          &
          &  ' function and ', I0, ' gradient evaluations' )" )                &
            prefix, data%control%stop_f, inform%iter,                          &
            prefix, inform%f_eval, inform%g_eval
         END IF
       ELSE
         WRITE( data%out, "( A, ' Budget limit reached' )" ) prefix
         WRITE( data%out, "( A, ' Smallest value of', ES22.14 )" )             &
           prefix, inform%obj
         IF ( data%printw ) WRITE( data%out,                                   &
           "( A, ' at X = ', /, ( 5X, 5ES12.4 ) )" ) prefix, nlp%X( : nlp%n )
         WRITE( data%out, "( A, ' found in ', I0, ' iterations', /, A,         &
        & ' using ', I0, ' function and ', I0, ' gradient evaluations' )" )    &
           prefix, inform%iter, prefix, inform%f_eval, inform%g_eval
         WRITE( data%out, "( A, ' maximum unpruned box length and gap are',    &
        &  ' currently ', 2ES11.4 )" ) prefix, inform%length_ratio, inform%f_gap
       END IF
       WRITE( data%out, "( A, 1X, I0, ' from ', I0, ' boxes have been ',       &
      &  'pruned' )" ) prefix, COUNT( DATA%BOX( : data%boxes )%pruned ),       &
        data%boxes
     END IF

!  prepare for a successful exit

     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

!  -------------
!  Error returns
!  -------------

 910 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     RETURN

 990 CONTINUE
     CALL CPU_time( data%time_record ) ; CALL CLOCK_time( data%clock_record )
     inform%time%total = data%time_record - data%time_start
     inform%time%clock_total = data%clock_record - data%clock_start
     IF ( data%printi ) THEN
       CALL SYMBOLS_status( inform%status, data%out, prefix, 'DGO_solve' )
       WRITE( data%out, "( ' ' )" )
     END IF
     RETURN

!  Non-executable statements

 2000 FORMAT( /, A, ' DGO solver, problem: ', A, ' (n = ', I0, ')', / )
 2010 FORMAT( A, '    iter              f              delta        #f   ',    &
              '   #g   boxes        time' )
 2020 FORMAT( A, I8, A1, ES24.16, ES11.4, 3I8, F12.2 )

!  End of subroutine DGO_solve

     END SUBROUTINE DGO_solve

!-*-*-  G A L A H A D -  D G O _ t e r m i n a t e  S U B R O U T I N E -*-*-

     SUBROUTINE DGO_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_data_type ), INTENT( INOUT ) :: data
     TYPE ( DGO_control_type ), INTENT( IN ) :: control
     TYPE ( DGO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     LOGICAL :: alive
     CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaining allocated arrays

     IF ( ALLOCATED( data%VERTEX ) ) THEN
       DO i = 1, data%length
         IF ( ALLOCATED( data%VERTEX( i )%X ) )                                &
           DEALLOCATE( data%VERTEX( i )%X, STAT = inform%alloc_status )
         IF ( ALLOCATED( data%VERTEX( i )%G ) )                                &
           DEALLOCATE( data%VERTEX( i )%G, STAT = inform%alloc_status )
       END DO
       DEALLOCATE( data%VERTEX, STAT = inform%alloc_status )
     END IF
     IF ( ALLOCATED( data%BOX ) )                                              &
       DEALLOCATE( data%BOX, STAT = inform%alloc_status )
     IF ( ALLOCATED( data%string ) )                                           &
       DEALLOCATE( data%string, STAT = inform%alloc_status )

     data%P => NULL( )
     array_name = 'dgo: data%P'
     CALL SPACE_dealloc_pointer( data%P,                                       &
        inform%status, inform%alloc_status, point_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     data%U => NULL( )
     array_name = 'dgo: data%U'
     CALL SPACE_dealloc_pointer( data%U,                                       &
        inform%status, inform%alloc_status, point_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

     data%V => NULL( )
     array_name = 'dgo: data%V'
     CALL SPACE_dealloc_pointer( data%V,                                       &
        inform%status, inform%alloc_status, point_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND.                                 &
          inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated within UGO

!    CALL UGO_terminate( data%UGO_data, data%control%UGO_control,              &
!                        inform%UGO_inform )
!    inform%status = inform%UGO_inform%status
!    IF ( inform%status /= GALAHAD_ok ) THEN
!      inform%alloc_status = inform%UGO_inform%alloc_status
!      inform%bad_alloc = inform%UGO_inform%bad_alloc
!      IF ( control%deallocate_error_fatal ) RETURN
!    END IF

!  Deallocate all arrays allocated within UGO

     CALL HASH_terminate( data%HASH_data, data%control%HASH_control,           &
                         inform%HASH_inform )
     inform%status = inform%HASH_inform%status
     IF ( inform%status /= GALAHAD_ok ) THEN
       inform%alloc_status = inform%HASH_inform%alloc_status
       inform%bad_alloc = inform%HASH_inform%bad_alloc
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

!  End of subroutine DGO_terminate

     END SUBROUTINE DGO_terminate

! -  G A L A H A D -  D G O _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE DGO_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( DGO_control_type ), INTENT( IN ) :: control
     TYPE ( DGO_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL DGO_terminate( data%dgo_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'dgo: data%nlp%X'
     CALL SPACE_dealloc_array( data%nlp%X,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%G'
     CALL SPACE_dealloc_array( data%nlp%G,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%X_l'
     CALL SPACE_dealloc_array( data%nlp%X_l,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%X_u'
     CALL SPACE_dealloc_array( data%nlp%X_u,                                   &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%H%row'
     CALL SPACE_dealloc_array( data%nlp%H%row,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%H%col'
     CALL SPACE_dealloc_array( data%nlp%H%col,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%H%ptr'
     CALL SPACE_dealloc_array( data%nlp%H%ptr,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%H%val'
     CALL SPACE_dealloc_array( data%nlp%H%val,                                 &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dgo: data%nlp%H%type'
     CALL SPACE_dealloc_array( data%nlp%H%type,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine DGO_full_terminate

     END SUBROUTINE DGO_full_terminate

! -*-*-*-*-  G A L A H A D -  D G O _ v e r t e x  S U B R O U T I N E -*-*-*-*-

     SUBROUTINE DGO_vertex( n, X, string, rstring, position,                   &
                            HASH_data, HASH_control, HASH_inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  determine the position of the vertex X in the dictionary by hashing

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     INTEGER, INTENT( OUT ) :: position
     CHARACTER ( LEN = n * rwidth ), INTENT( OUT ) :: string
     CHARACTER ( LEN = rwidth ), INTENT( OUT ) :: rstring
     REAL( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     TYPE ( HASH_data_type ), INTENT( INOUT ) :: HASH_data
     TYPE ( HASH_control_type ), INTENT( IN ) :: HASH_control
     TYPE ( HASH_inform_type ), INTENT( INOUT ) :: HASH_inform

!  local variables

     INTEGER :: i, nchar
     CHARACTER ( LEN = 1 ) :: FIELD( n * rwidth )

     nchar = n * rwidth
     string = REPEAT( ' ', nchar )
     DO i = 1, n
       WRITE( rstring, "( ES24.16 )" ) X( i )
       string = TRIM( string ) // TRIM( ADJUSTL( rstring ) )
     END DO
!    write(6,*) TRIM( string )
     DO i = 1, nchar
       FIELD( i ) = string( i : i )
     END DO
     CALL HASH_insert( nchar, FIELD, position, HASH_data,                      &
                       HASH_control, HASH_inform )
     RETURN

     END SUBROUTINE DGO_vertex

! -*-  G A L A H A D -  D G O _ a l l o c a t e _ v e r t e x _ a r r a y s  -*-

    SUBROUTINE DGO_allocate_vertex_arrays( n, VERTEX, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  allocate the component arrays X and G of VERTEX

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

    INTEGER, INTENT ( IN ) :: n
    TYPE ( DGO_vertex_type ), INTENT( INOUT ) :: VERTEX
    TYPE ( DGO_control_type ), INTENT( IN ) :: control
    TYPE ( DGO_inform_type ), INTENT( INOUT ) :: inform

!  local variables

    CHARACTER ( LEN = 80 ) :: array_name

    array_name = 'dgo: data%VERTEX%X'
    CALL SPACE_resize_array( n, VERTEX%X, inform%status,                       &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
    IF ( inform%status /= GALAHAD_ok ) RETURN

    array_name = 'dgo: data%VERTEX%G'
    CALL SPACE_resize_array( n, VERTEX%G, inform%status,                       &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
    RETURN

!  end of subroutine DGO_allocate_vertex_arrays

    END SUBROUTINE DGO_allocate_vertex_arrays

! -  G A L A H A D -  D G O _ i n i t i a l i z e _ b o x  S U B R O U T I N E -

     SUBROUTINE DGO_initialize_box( index_l, VERTEX_l, index_u, VERTEX_u, BOX )
     INTEGER, INTENT( IN ) :: index_l, index_u
     TYPE( DGO_vertex_type ), INTENT( IN ) :: VERTEX_l, VERTEX_u
     TYPE( DGO_box_type ), INTENT( OUT ) :: BOX

     REAL ( KIND = wp ) :: d, t

!  initialize the box as currently under consideration

     BOX%pruned = .FALSE.

!  store the indices of the lower and upper box bounds, x_l and x_u

     BOX%index_l = index_l ; BOX%index_u = index_u

!  compute the length of the diagonal from x_l to x_u

     BOX%delta = TWO_NORM( VERTEX_l%X - VERTEX_u%X )

!   compute the objective function values at x_l and x_u

     BOX%f_l = VERTEX_l%f ; BOX%f_u = VERTEX_u%f

!   compute the directional derivatives at x_l and x_u

     BOX%df_l = DOT_PRODUCT( VERTEX_l%G, VERTEX_u%X - VERTEX_l%X ) / BOX%delta
     BOX%df_u = DOT_PRODUCT( VERTEX_u%G, VERTEX_u%X - VERTEX_l%X ) / BOX%delta

!  estimate the gradient Lipschitz constant over the box

     t = two * ( VERTEX_l%f - VERTEX_u%f ) + BOX%delta * ( BOX%df_l + BOX%df_u )
     d = SQRT( t ** 2 + ( BOX%delta * ( BOX%df_u - BOX%df_l ) ) ** 2 )
     BOX%gradient_lipschitz_estimate = ( ABS( t ) + d ) / ( BOX%delta ** 2 )

!  compute an upper bound of the objective in box

     BOX%f_upper = MIN( BOX%f_l, BOX%f_u )

!  set the lower bound of the objective in box to minus infinity

     BOX%f_lower = - infinity

     RETURN

!  end of subroutine DGO_initialize_box

     END SUBROUTINE DGO_initialize_box

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  D G O _ i m p o r t _ S U B R O U T I N E -*-*-*-*-

     SUBROUTINE DGO_import( control, data, status, n, X_l, X_u,                &
                            H_type, ne, H_row, H_col, H_ptr )

!  import problem data into internal storage prior to solution. 
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading 
!   comments to DGO_solve
!
!  data is a scalar variable of type DGO_full_data_type used for internal data
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

     TYPE ( DGO_control_type ), INTENT( INOUT ) :: control
     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( IN ) :: n, ne
     INTEGER, INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: H_type
     INTEGER, DIMENSION( : ), INTENT( IN ) :: H_row, H_col, H_ptr
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: X_l, X_u

!  local variables

     INTEGER :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

     error = data%dgo_control%error
     space_critical = data%dgo_control%space_critical
     deallocate_error_fatal = data%dgo_control%deallocate_error_fatal

!  allocate space if required

     array_name = 'dgo: data%nlp%X'
     CALL SPACE_resize_array( n, data%nlp%X,                                   &
            data%dgo_inform%status, data%dgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%dgo_inform%bad_alloc, out = error )
     IF ( data%dgo_inform%status /= 0 ) GO TO 900

     array_name = 'dgo: data%nlp%G'
     CALL SPACE_resize_array( n, data%nlp%G,                                   &
            data%dgo_inform%status, data%dgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%dgo_inform%bad_alloc, out = error )
     IF ( data%dgo_inform%status /= 0 ) GO TO 900

     array_name = 'dgo: data%nlp%X_l'
     CALL SPACE_resize_array( n, data%nlp%X_l,                                 &
            data%dgo_inform%status, data%dgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%dgo_inform%bad_alloc, out = error )
     IF ( data%dgo_inform%status /= 0 ) GO TO 900

     array_name = 'dgo: data%nlp%X_u'
     CALL SPACE_resize_array( n, data%nlp%X_u,                                 &
            data%dgo_inform%status, data%dgo_inform%alloc_status,              &
            array_name = array_name,                                           &
            deallocate_error_fatal = deallocate_error_fatal,                   &
            exact_size = space_critical,                                       &
            bad_alloc = data%dgo_inform%bad_alloc, out = error )
     IF ( data%dgo_inform%status /= 0 ) GO TO 900

!  put data into the required components of the nlpt storage type

     data%nlp%n = n
     data%nlp%X_l( : n ) = X_l( : n )
     data%nlp%X_u( : n ) = X_u( : n )

!  set H appropriately in the nlpt storage type

     SELECT CASE ( H_type )
     CASE ( 'coordinate', 'COORDINATE' )
       CALL SMT_put( data%nlp%H%type, 'COORDINATE',                            &
                     data%dgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = ne

       array_name = 'dgo: data%nlp%H%row'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%row,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900

       array_name = 'dgo: data%nlp%H%col'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%col,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900

       array_name = 'dgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900

       data%nlp%H%row( : data%nlp%H%ne ) = H_row( : data%nlp%H%ne )
       data%nlp%H%col( : data%nlp%H%ne ) = H_col( : data%nlp%H%ne )

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       CALL SMT_put( data%nlp%H%type, 'SPARSE_BY_ROWS',                        &
                     data%dgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = H_ptr( n + 1 ) - 1

       array_name = 'dgo: data%nlp%H%ptr'
       CALL SPACE_resize_array( n + 1, data%nlp%H%ptr,                         &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900

       array_name = 'dgo: data%nlp%H%col'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%col,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900

       array_name = 'dgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900

       data%nlp%H%ptr( : n + 1 ) = H_ptr( : n + 1 )
       data%nlp%H%col( : data%nlp%H%ne ) = H_col( : data%nlp%H%ne )

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%nlp%H%type, 'DENSE', data%dgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'dgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900
     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%nlp%H%type, 'DIAGONAL', data%dgo_inform%alloc_status )
       data%nlp%H%n = n
       data%nlp%H%ne = n

       array_name = 'dgo: data%nlp%H%val'
       CALL SPACE_resize_array( data%nlp%H%ne, data%nlp%H%val,                 &
              data%dgo_inform%status, data%dgo_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dgo_inform%bad_alloc, out = error )
       IF ( data%dgo_inform%status /= 0 ) GO TO 900
     CASE ( 'absent', 'ABSENT' )
       control%hessian_available = .FALSE.
       control%trb_control%hessian_available = .FALSE.
     CASE DEFAULT
       data%dgo_inform%status = GALAHAD_error_unknown_storage
     END SELECT       

!  copy control to data

     data%dgo_control = control
     status = GALAHAD_ready_to_solve
     RETURN

!  error returns

 900 CONTINUE
     status = data%dgo_inform%status
     RETURN

!  End of subroutine DGO_import

     END SUBROUTINE DGO_import

!-  G A L A H A D -  D G O _ r e s e t _ c o n t r o l   S U B R O U T I N E  -

     SUBROUTINE DGO_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See DGO_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_control_type ), INTENT( IN ) :: control
     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: status

!  set control in internal data

     data%dgo_control = control
     
!  flag a successful call

     status = GALAHAD_ready_to_solve
     RETURN

!  end of subroutine DGO_reset_control

     END SUBROUTINE DGO_reset_control

!-  G A L A H A D -  D G O _ s o l v e _ w i t h _ m a t   S U B R O U T I N E 

     SUBROUTINE DGO_solve_with_mat( data, userdata, status, X, G,              &
                                    eval_F, eval_G, eval_H, eval_HPROD,        &
                                    eval_PREC )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian and preconditioning operations are
!  available via subroutine calls. See DGO_solve for a description of 
!  the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_F, eval_G, eval_H, eval_HPROD, eval_PREC

     data%dgo_inform%status = status
     IF ( data%dgo_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     CALL DGO_solve( data%nlp, data%dgo_control, data%dgo_inform,              &
                     data%dgo_data, userdata, eval_F = eval_F,                 &
                     eval_G = eval_G, eval_H = eval_H,                         &
                     eval_HPROD = eval_HPROD, eval_PREC = eval_PREC )

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     IF ( data%dgo_inform%status == GALAHAD_ok )                               &
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )

     status = data%dgo_inform%status
     RETURN

     END SUBROUTINE DGO_solve_with_mat

! - G A L A H A D -  D G O _ s o l v e _ without _ m a t  S U B R O U T I N E -

     SUBROUTINE DGO_solve_without_mat( data, userdata, status, X, G,           &
                                       eval_F, eval_G, eval_HPROD,             &
                                       eval_SHPROD, eval_PREC )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian-vector and preconditioning operations 
!  are available via subroutine calls. See DGO_solve for a description 
!  of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( INOUT ) :: status
     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     EXTERNAL :: eval_F, eval_G, eval_HPROD, eval_SHPROD, eval_PREC

     data%dgo_inform%status = status
     IF ( data%dgo_inform%status == 1 )                                        &
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )

!  call the solver

     CALL DGO_solve( data%nlp, data%dgo_control, data%dgo_inform,              &
                     data%dgo_data, userdata, eval_F = eval_F,                 &
                     eval_G = eval_G, eval_HPROD = eval_HPROD,                 &
                     eval_SHPROD = eval_SHPROD, eval_PREC = eval_PREC )

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     IF ( data%dgo_inform%status == GALAHAD_ok )                               &
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )

     status = data%dgo_inform%status
     RETURN

     END SUBROUTINE DGO_solve_without_mat

!-  G A L A H A D -  D G O _ s o l v e _ reverse _ m a t  S U B R O U T I N E  -

     SUBROUTINE DGO_solve_reverse_with_mat( data, status, eval_status,         &
                                            X, f, G, H_val, U, V )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian and preconditioning operations are
!  available via reverse communication. See DGO_solve for a description 
!  of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( INOUT ) :: status, eval_status
     REAL ( KIND = wp ), INTENT( IN ) :: f
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: G
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: H_val
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: V

!  recover data from reverse communication

     data%dgo_inform%status = status
     data%dgo_data%eval_status = eval_status

     SELECT CASE ( data%dgo_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%f = f
     CASE( 3 ) 
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
     CASE( 4 ) 
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%H%val( : data%dgo_data%trb_data%h_ne ) =                     &
           H_val( : data%dgo_data%trb_data%h_ne )
     CASE( 5, 6 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
     CASE ( 23 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
       END IF
     CASE ( 25 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 35 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 235 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     END SELECT

!  call the solver

     CALL DGO_solve( data%nlp, data%dgo_control, data%dgo_inform,              &
                     data%dgo_data, data%userdata )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%dgo_inform%status )
     CASE( 0 )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 5, 25, 35, 235 )
       U( : data%nlp%n ) = data%dgo_data%U( : data%nlp%n )
       V( : data%nlp%n ) = data%dgo_data%V( : data%nlp%n )
    CASE( 6 )
       V( : data%nlp%n ) = data%dgo_data%V( : data%nlp%n )
     CASE( 7 ) 
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%dgo_inform%status
     END SELECT

     status = data%dgo_inform%status
     RETURN

     END SUBROUTINE DGO_solve_reverse_with_mat

!  G A L A H A D -  D G O _ s o l v e _ reverse _ no _mat  S U B R O U T I N E  

     SUBROUTINE DGO_solve_reverse_without_mat( data, status, eval_status,      &
                                               X, f, G, U, V,                  &
                                               INDEX_nz_v, nnz_v,              &
                                               INDEX_nz_u, nnz_u )

!  solve the bound-constrained problem previously imported when access
!  to function, gradient, Hessian-vector and preconditioning operations 
!  are available via reverse communication. See DGO_solve for a description 
!  of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: nnz_v
     INTEGER, INTENT( INOUT ) :: status, eval_status
     INTEGER, INTENT( IN ) :: nnz_u
     REAL ( KIND = wp ), INTENT( IN ) :: f
     INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_v
     INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_u 
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: G
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: V

!  recover data from reverse communication

     data%dgo_inform%status = status
     data%dgo_data%eval_status = eval_status

     SELECT CASE ( data%dgo_inform%status )
     CASE ( 1 )
       data%nlp%X( : data%nlp%n ) = X( : data%nlp%n )
     CASE ( 2 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%f = f
     CASE( 3 ) 
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
     CASE( 5, 6 ) 
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 )                                                 &
          data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
     CASE( 7 ) 
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%dgo_data%nnz_hp = nnz_u
         data%dgo_data%INDEX_nz_hp( : nnz_u ) = INDEX_nz_u( : nnz_u )
         data%dgo_data%HP( INDEX_nz_u( 1 : nnz_u ) )                           &
            = U( INDEX_nz_u( 1 : nnz_u ) ) 
       END IF
     CASE ( 23 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
       END IF
     CASE ( 25 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 35 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     CASE ( 235 )
       data%dgo_data%eval_status = eval_status
       IF ( eval_status == 0 ) THEN
         data%nlp%f = f
         data%nlp%G( : data%nlp%n ) = G( : data%nlp%n )
         data%dgo_data%U( : data%nlp%n ) = U( : data%nlp%n )
       END IF
     END SELECT

!  call the solver

     CALL DGO_solve( data%nlp, data%dgo_control, data%dgo_inform,              &
                     data%dgo_data, data%userdata )

!  collect data for reverse communication

     X( : data%nlp%n ) = data%nlp%X( : data%nlp%n )
     SELECT CASE ( data%dgo_inform%status )
     CASE( 0 )
       G( : data%nlp%n ) = data%nlp%G( : data%nlp%n )
     CASE( 2, 3 ) 
     CASE( 4 ) 
       WRITE( 6, "( ' there should not be a case ', I0, ' return' )" )         &
         data%dgo_inform%status
     CASE( 5, 25, 35, 235 )
       U( : data%nlp%n ) = data%dgo_data%U( : data%nlp%n )
       V( : data%nlp%n ) = data%dgo_data%V( : data%nlp%n )
     CASE( 6 )
       V( : data%nlp%n ) = data%dgo_data%V( : data%nlp%n )
     CASE( 7 )
       nnz_v = data%dgo_data%nnz_p_u - data%dgo_data%nnz_p_l + 1
       INDEX_nz_v( : nnz_v ) =                                                 &
          data%dgo_data%INDEX_nz_p( data%dgo_data%nnz_p_l :                    &
                                    data%dgo_data%nnz_p_u )
       V( INDEX_nz_v( 1 : nnz_v ) )                                            &
          = data%dgo_data%P( INDEX_nz_v( 1 : nnz_v ) )
     END SELECT

     status = data%dgo_inform%status
     RETURN

     END SUBROUTINE DGO_solve_reverse_without_mat

!-  G A L A H A D -  D G O _ i n f o r m a t i o n   S U B R O U T I N E  -

     SUBROUTINE DGO_information( data, inform, status )

!  return solver information during or after solution by DGO
!  See DGO_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DGO_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( DGO_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%dgo_inform
     
!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine DGO_information

     END SUBROUTINE DGO_information

!  End of module GALAHAD_DGO

   END MODULE GALAHAD_DGO_double


