! THIS VERSION: GALAHAD 2.6 - 12/03/2014 AT 10:30 GMT.

!-*-*-  L A N C E L O T  -B-  L A N C E L O T _ T Y P E S _  M O D U L E  -*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   extracted from LANCELOT B. March 12th 2014

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE LANCELOT_types_double

!  |-------------------------------------------------------|
!  |                                                       |
!  |  Derived types used by LANCELOT B and its associates  |
!  |                                                       |
!  |-------------------------------------------------------|

     USE LANCELOT_ASMBL_double, ONLY: ASMBL_save_type
     USE LANCELOT_EXTEND_double, ONLY: EXTEND_save_type
     USE LANCELOT_CAUCHY_double, ONLY: CAUCHY_save_type
     USE LANCELOT_CG_double, ONLY: CG_save_type
     USE LANCELOT_OTHERS_double, ONLY: OTHERS_fdgrad_save_type
     USE LANCELOT_PRECN_double, ONLY: PRECN_save_type
     USE GALAHAD_SCU_double, ONLY: SCU_matrix_type, SCU_data_type,             &
                                   SCU_info_type
     USE GALAHAD_SILS_double, ONLY: SILS_control, SILS_factors,                &
                                    SILS_ainfo, SILS_finfo, SILS_sinfo
     USE GALAHAD_SMT_double, ONLY: SMT_type
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: LANCELOT_problem_type, LANCELOT_save_type, LANCELOT_data_type,  &
               LANCELOT_control_type, LANCELOT_inform_type

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!  ======================================
!  The LANCELOT_problem_type derived type
!  ======================================

     TYPE :: LANCELOT_problem_type
       INTEGER :: n, ng, nel
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IELING
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTADG
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IELVAR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTAEV
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INTVAR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTADH
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ICNA
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTADA
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: KNDOFG
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITYPEE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTEPA
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITYPEG
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTGPA
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BU
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GSCALE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ESCALE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VSCALE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: EPVALU
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GPVALU
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: GXEQX
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: INTREP
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: VNAMES
       CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: GNAMES
     END TYPE LANCELOT_problem_type

!  ===================================
!  The LANCELOT_save_type derived type
!  ===================================

     TYPE :: LANCELOT_save_type

       LOGICAL :: full_solution

!  the relevant parts of the old INWKSP common block

       INTEGER :: igetfd
       LOGICAL :: unsucc

!  the old AUGLG saved variables

       INTEGER :: nobjgr, m, icrit, ncrit, p_type
       REAL ( KIND = wp ) :: ocnorm, cnorm_major, etak, eta0, omegak, omega0
       REAL ( KIND = wp ) :: tau, tau_steering, gamma1, alphae, betae, alphak
       REAL ( KIND = wp ) :: alphao, betao, omega_min, eta_min, epstol, epsgrd
       REAL ( KIND = wp ) :: cnorm
       CHARACTER ( LEN = 5 ), DIMENSION( 5 ) :: STATE
       LOGICAL :: itzero, reeval

!  the old SBMIN saved variables

       INTEGER :: ifactr, ldx, lfxi, lgxi, lhxi, lggfx, nvar2, first_derivatives
       INTEGER :: jumpto, nbprod, infor, number, nfixed, ibqpst, nfreef, nfree
       INTEGER :: nmhist, maxsel, ntotin, nfreec, lnguvl, lnhuvl, nnonnz, nadd
       INTEGER :: ntype, nsets, nvargp, l_suc, msweep, nbnd, mortor_its, ntotel
       INTEGER :: nvrels, nnza, error, out, print_level, second_derivatives
       INTEGER :: start_print, stop_print, print_gap, inform_status, icfact
       INTEGER :: n_steering, n_steering_this_iteration
       REAL ( KIND = wp ) :: epstlp, gmodel, vscmax, rad, maximum_radius
       REAL ( KIND = wp ) :: epsrcg, fnew, radmin, cgstop, diamin, diamax
       REAL ( KIND = wp ) :: ared, prered, rho, fmodel, curv, dxsqr, fcp, f0
       REAL ( KIND = wp ) :: stepmx, smallh, resmin, qgnorm, oldrad, epscns
       REAL ( KIND = wp ) :: radtol, fill, step, teneps, stpmin, epstln
       REAL ( KIND = wp ) :: f_min, f_r, f_c, sigma_r, sigma_c, findmx
       REAL ( KIND = wp ) :: f_min_lag, f_r_lag, f_c_lag
       REAL ( KIND = wp ) :: f_min_viol, f_r_viol, f_c_viol
       REAL ( KIND = wp ) :: violation, delta_qv, delta_qv_steering
       LOGICAL :: alllin, altriv, next, second, print_header, modchl
       LOGICAL :: iprcnd, munks , seprec, densep, calcdi, xactcp, reusec
       LOGICAL :: gmpspr, slvbqp, refact, fdgrad, centrl, dprcnd, strctr
       LOGICAL :: use_band, icfs, mortor, firsup, twonrm, direct, myprec
       LOGICAL :: prcond, firstc, nobnds, getders, save_c
       LOGICAL :: printt, printi, printm, printw, printd, printe, set_printe
       LOGICAL :: set_printt, set_printi, set_printm, set_printw, set_printd
       LOGICAL :: skipg, steering, new_major
       CHARACTER ( LEN = 6 ) :: cgend, lisend
       CHARACTER ( LEN = 1 ) :: cgend1, lisend1
       REAL ( KIND = KIND( 1.0E0 ) ) :: t, time, tmv, tca, tls, tup
       INTEGER, DIMENSION( 5 ) :: ISYS
       CHARACTER ( LEN = 6 ), DIMENSION( 5 ) :: CGENDS
       CHARACTER ( LEN = 6 ), DIMENSION( 5 ) :: LSENDS
       CHARACTER ( LEN = 1 ), DIMENSION( 5 ) :: CGENDS1
       CHARACTER ( LEN = 1 ), DIMENSION( 5 ) :: LSENDS1

!  the old CAUCH saved variables

       TYPE( CAUCHY_save_type ) :: CAUCHY

!  the old CG saved variables

       TYPE( CG_save_type ) :: CG

!  the old ASMBL saved variables

       TYPE( ASMBL_save_type ) :: ASMBL

!  the old PRECN saved variables

       TYPE( PRECN_save_type ) :: PRECN

!  the old OTHERS saved variables

       TYPE( OTHERS_fdgrad_save_type ) :: OTHERS

!  the old EXTEND saved variables

       TYPE( EXTEND_save_type ) :: EXTEND

     END TYPE LANCELOT_save_type

!  ===================================
!  The LANCELOT_data_type derived type
!  ===================================

     TYPE :: LANCELOT_data_type

       TYPE( LANCELOT_save_type ) :: S

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITRANS
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW_start
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: POS_in_H
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: USED
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: FILLED
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LINK_elem_uses_var
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WTRANS

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISYMMD
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISWKSP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTAJC
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTAGV
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISVGRP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISLGRP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IGCOLJ
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVALJR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IUSED
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITYPER
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISSWTR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISSITR
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISET
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISVSET
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INVSET
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IFREE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IFREEC
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INNONZ
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LIST_elements
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: ISYMMH
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FUVALS_temp
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: XCP
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GX0
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RADII
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DELTAX
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: QGRAD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRJAC
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: CDASH
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C2DASH
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GV_old
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND_radius

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW_asmbl
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_comp_w
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_ws
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_el
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_in
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_el
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_in

       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: IKEEP
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: IW1
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVUSE
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col_ptr
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: L_col_ptr
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIAG
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAKP
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: W1
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: OFFDIA

       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: GROUP_SCALING => NULL( )
       LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_AUG => NULL( )

       TYPE ( SCU_matrix_type ) :: SCU_matrix
       TYPE ( SCU_data_type ) :: SCU_data
       TYPE ( SMT_type ) :: matrix
       TYPE ( SILS_factors ) :: SILS_data

     END TYPE LANCELOT_data_type

!  ======================================
!  The LANCELOT_control_type derived type
!  ======================================

     TYPE :: LANCELOT_control_type

!  Error and ordinary output unit numbers

       INTEGER :: error = 6
       INTEGER :: out = 6

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

       INTEGER :: alive_unit = 60
       CHARACTER ( LEN = 30 ) :: alive_file = 'ALIVE.d                       '

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

       INTEGER :: print_level = 0

!  Maximum number of iterations

       INTEGER :: maxit = 1000

!   Any printing will start on this iteration (-1 = always print)

       INTEGER :: start_print = - 1

!   Any printing will stop on this iteration (-1 = always print)

       INTEGER :: stop_print = - 1

!   Printing will only occur every print_gap iterations

       INTEGER :: print_gap = 1

!  linear_solver gives the method to be used for solving the
!                linear system. 1=CG, 2=diagonal preconditioned CG,
!                3=user-provided preconditioned CG, 4=expanding band
!                preconditioned CG, 5=Munksgaard's preconditioned CG,
!                6=Schnabel-Eskow modified Cholesky preconditioned CG,
!                7=Gill-Murray-Ponceleon-Saunders modified Cholesky
!                preconditioned CG, 8=band matrix preconditioned CG,
!                9=Lin-More' preconditioned CG, 11=multifrontal direct
!                method, 12=direct modified multifrontal method

       INTEGER :: linear_solver = 8

!  The number of vectors allowed in Lin and More's incomplete factorization

       INTEGER :: icfact = 5

!  The semi-bandwidth of the band factorization

       INTEGER :: semibandwidth = 5

!   The maximum dimension of the Schur complement

       INTEGER :: max_sc = 100

!  Unit number of i/o buffer for writing temporary files (if needed)

       INTEGER :: io_buffer = 75

!  more_toraldo >= 1 gives the number of More'-Toraldo projected searches
!                to be used to improve upon the Cauchy point, anything
!                else is for the standard add-one-at-a-time CG search

       INTEGER :: more_toraldo = 0

!   non-monotone <= 0 monotone strategy used, anything else non-monotone
!                strategy with this history length used

       INTEGER :: non_monotone = 1

!  first_derivatives = 0 if exact first derivatives are given, = 1 if forward
!             finite difference approximations are to be calculated, and
!             = 2 if central finite difference approximations are to be used

       INTEGER :: first_derivatives = 0

!  second_derivatives specifies the approximation to the second derivatives
!                used. 0=exact, 1=BFGS, 2=DFP, 3=PSB, 4=SR1

       INTEGER :: second_derivatives = 0

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of violation of the constraints (the "primal infeasibility") is smaller than
!  control%stopc and the norm of the gradient of the Lagrangian function (the
!  "dual infeasibility") is smaller than control%stopg

       REAL ( KIND = wp ) :: stopc = ten ** ( - 5 )
       REAL ( KIND = wp ) :: stopg = ten ** ( - 5 )

!  It will also terminate if the merit function (objective or augmented
!  Lagrangian as appropriate) is smaller than control%min_aug

       REAL ( KIND = wp ) :: min_aug = - ( HUGE( one ) / 8.0_wp )

!  Require a relative reduction in the resuiduals from CG of at least acccg

       REAL ( KIND = wp ) :: acccg = 0.01_wp

!  The initial trust-region radius - a non-positive value allows the
!  package to choose its own

       REAL ( KIND = wp ) :: initial_radius = - one

!  The largest possible trust-region radius

       REAL ( KIND = wp ) :: maximum_radius = ten ** 20

!  Parameters that define when to decrease/increase the trust-region
!  (specialists only!)

       REAL ( KIND = wp ) :: eta_successful = 0.01_wp
       REAL ( KIND = wp ) :: eta_very_successful = 0.9_wp
       REAL ( KIND = wp ) :: eta_extremely_successful = 0.95_wp
       REAL ( KIND = wp ) :: gamma_smallest = 0.0625_wp
       REAL ( KIND = wp ) :: gamma_decrease = 0.25_wp
       REAL ( KIND = wp ) :: gamma_increase = 2.0_wp
       REAL ( KIND = wp ) :: mu_meaningful_model = 0.01_wp
       REAL ( KIND = wp ) :: mu_meaningful_group = 0.1_wp

!  The initial value of the penalty parameter

       REAL ( KIND = wp ) :: initial_mu = 0.1_wp

!  The penalty parameter decrease factor

       REAL ( KIND = wp ) :: mu_decrease = 0.1_wp

!  The penalty parameter decrease factor when steering

       REAL ( KIND = wp ) :: mu_steering_decrease = 0.7_wp

!  The value of the penalty parameter above which the algorithm
!  will not attempt to update the estimates of the Lagrange multipliers

       REAL ( KIND = wp ) :: mu_tol = 0.1_wp

!  The required accuracy of the norm of the projected gradient at the end
!  of the first major iteration

       REAL ( KIND = wp ) :: firstg = 0.1_wp

!  The required accuracy of the norm of the constraints at the end
!  of the first major iteration

       REAL ( KIND = wp ) :: firstc = 0.1_wp

!  control parameters from Curtis-Jiang-Robinson steering

       INTEGER :: num_mudec = HUGE( 1 )
       INTEGER :: num_mudec_per_iteration = HUGE( 1 )
       REAL ( KIND = wp ) :: kappa_3 = ten ** ( - 5 )
       REAL ( KIND = wp ) :: kappa_t = 0.9_wp
       REAL ( KIND = wp ) :: mu_min = zero

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!  Is the function quadratic ?

       LOGICAL :: quadratic_problem = .FALSE.

!  Do we want to steer the iterates towards feasibility ?

       LOGICAL :: steering = .FALSE.

!  two_norm_tr is true if a 2-norm trust-region is to be used, and false
!                for the infinity norm

       LOGICAL :: two_norm_tr = .FALSE.

!  exact_gcp is true if the exact Cauchy point is required, and false if an
!                approximation suffices

       LOGICAL :: exact_gcp = .TRUE.

!  use a Gauss-Newton model of the infeasibility rather than a second-order
!   Taylor model

       LOGICAL :: gn_model = .FALSE.

!  use a Gauss-Newton model of the infeasibility rather than a second-order
!   Taylor model once the Cauchy point has been found

       LOGICAL :: gn_model_after_cauchy = .FALSE.

!  magical_steps is true if magical steps are to be used to improve upon
!                already accepted points, and false otherwise

       LOGICAL :: magical_steps = .FALSE.

!  accurate_bqp is true if the the minimizer of the quadratic model within
!                the intersection of the trust-region and feasible box
!                is to be sought (to a prescribed accuracy), and false
!                if an approximation suffices

       LOGICAL :: accurate_bqp = .FALSE.

!  structured_tr is true if a structured trust region will be used, and false
!                if a standard trust-region suffices

       LOGICAL :: structured_tr = .FALSE.

!  For printing, if we are maximizing rather than minimizing, print_max
!  should be .TRUE.

       LOGICAL :: print_max = .FALSE.

!  .TRUE. if all components of the solution and constraints are to be printed
!  on termination, and .FALSE. if only the first and last (representative) few
!  are required

       LOGICAL :: full_solution = .TRUE.

!  space_critical. If true, every effort will be made to use as little
!  space as possible. This may result in longer computation times

       LOGICAL :: space_critical = .FALSE.

!  deallocate_error_fatal. If true, any array/pointer deallocation error
!  will terminate execution. Otherwise, computation will continue

       LOGICAL :: deallocate_error_fatal = .FALSE.

!  control parameters for SILS

       TYPE ( SILS_control ) :: SILS_cntl
     END TYPE LANCELOT_control_type

!  =====================================
!  The LANCELOT_inform_type derived type
!  =====================================

     TYPE :: LANCELOT_inform_type

!  return status. See LANCELOT_solve for details

       INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

       INTEGER :: alloc_status = 0

!  the total number of iterations required

       INTEGER :: iter = - 1

!  the total number of CG iterations required

       INTEGER :: itercg = - 1

!  the maximum number of CG iterations permitted per inner iteration

       INTEGER :: itcgmx = - 1

!  the number of element functions that must be re-evaluated when %status > 0

       INTEGER :: ncalcf = 0

!  the number of group functions that must be re-evaluated when %status > 0

       INTEGER :: ncalcg = 0

!  the current number of free variables

       INTEGER :: nvar = 0

!  the number of derivative evaluations made

       INTEGER :: ngeval = 0

!  the total number of secant updates that are skipped

       INTEGER :: iskip = 0

!  the variable that most recently encountered on of its bounds

       INTEGER :: ifixed = 0

!  the bandwidth used with the expanding-band preconditioner

       INTEGER :: nsemib = 0

!  the value of the augmented Lagrangian merit function at the best estimate
!   of the solution determined by LANCELOT_solve

       REAL ( KIND = wp ) :: aug = HUGE( one )

!  the value of the objective function at the best estimate of the solution
!   determined by LANCELOT_solve

       REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm of the projected gradient of the merit function

       REAL ( KIND = wp ) :: pjgnrm = HUGE( one )

!  the infinity norm of the equality constraints

       REAL ( KIND = wp ) :: cnorm = zero

!  the current ratio of predicted to achieved merit function reduction

       REAL ( KIND = wp ) :: ratio = zero

!  the current value of the penalty parameter

       REAL ( KIND = wp ) :: mu = zero

!  the current value of the trust-region radius

       REAL ( KIND = wp ) :: radius = zero

!  the pivot tolerance used when ICCG is used for preconditioning

       REAL ( KIND = wp ) :: ciccg = zero

!  newsol is true if a major iteration has just been completed

       LOGICAL :: newsol = .FALSE.

!  the name of the array for which an allocation error occurred

       CHARACTER ( LEN = 80 ) :: bad_alloc =  REPEAT( ' ', 80 )

!  return information from SCU

       TYPE ( SCU_info_type ) :: SCU_info

!  return information from SILS

       TYPE ( SILS_ainfo ) :: SILS_infoa
       TYPE ( SILS_finfo ) :: SILS_infof
       TYPE ( SILS_sinfo ) :: SILS_infos
     END TYPE LANCELOT_inform_type

!  End of module LANCELOT_types_double

   END MODULE LANCELOT_types_double
