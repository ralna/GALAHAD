! THIS VERSION: GALAHAD 2.4 - 09/04/2010 AT 07:45 GMT.

!  |--------------------------------------------------------------------|
!  |                                                                    |
!  |  WARNING - THIS MODULE HAS BEEN SUPERCEDED  - FOR INFORMATION ONLY |
!  |                                                                    |
!  |--------------------------------------------------------------------|

!-*-*-*-*-*-*-  L A N C E L O T  -B-  LANCELOT   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   ( based on Conn-Gould-Toint fortran 77 version LANCELOT A, ~1992 )
!   originally released pre GALAHAD Version 1.0. February 7th 1995
!   update released with GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE LANCELOT_double

!  |------------------------------------------------------------------|
!  |                                                                  |
!  |  Find a local minimizer of a smooth (group partially separable)  |
!  |  objective function subject to (partially separable) constraints |
!  |  and simple bounds                                               |
!  |                                                                  |
!  |------------------------------------------------------------------|

!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_SPECFILE_double
     USE LANCELOT_INITW_double
     USE LANCELOT_OTHERS_double
     USE LANCELOT_HSPRD_double
     USE LANCELOT_CAUCHY_double
     USE LANCELOT_CG_double
     USE LANCELOT_PRECN_double
     USE LANCELOT_FRNTL_double
     USE LANCELOT_STRUTR_double
     USE GALAHAD_SMT_double
     USE GALAHAD_SILS_double
     USE GALAHAD_SCU_double, ONLY : SCU_matrix_type, SCU_data_type,            &
       SCU_info_type, SCU_factorize, SCU_terminate
     USE LANCELOT_ASMBL_double, ONLY : ASMBL_save_type
     USE LANCELOT_EXTEND_double, ONLY : EXTEND_save_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: LANCELOT_initialize, LANCELOT_read_specfile, LANCELOT_solve,    &
               LANCELOT_terminate, LANCELOT_problem_type, LANCELOT_save_type,  &
               LANCELOT_control_type, LANCELOT_inform_type, LANCELOT_data_type,&
               LANCELOT_problem_pointer_type

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
     REAL ( KIND = wp ), PARAMETER :: point99 = 0.99_wp
     REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: hundrd = 100.0_wp
     REAL ( KIND = wp ), PARAMETER :: tenten = ten ** 10
     REAL ( KIND = wp ), PARAMETER :: tenm2 = 0.01_wp
     REAL ( KIND = wp ), PARAMETER :: tenm4 = 0.0001_wp
     REAL ( KIND = wp ), PARAMETER :: tenm5 = 0.00001_wp
     REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
                            
     REAL ( KIND = wp ), PARAMETER :: wmin = point1
     REAL ( KIND = wp ), PARAMETER :: theta = point1
     REAL ( KIND = wp ), PARAMETER :: stptol = point1

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

     TYPE :: LANCELOT_problem_pointer_type
       INTEGER :: n, ng, nel
       INTEGER, POINTER, DIMENSION( : ) :: IELING => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ISTADG => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: IELVAR => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ISTAEV => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: INTVAR => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ISTADH => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ICNA => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ISTADA => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: KNDOFG => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ITYPEE => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ISTEPA => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ITYPEG => NULL( )
       INTEGER, POINTER, DIMENSION( : ) :: ISTGPA => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: A => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: B => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: BL => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: BU => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: X => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: C => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: Y => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: GSCALE => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: ESCALE => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: VSCALE => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: EPVALU => NULL( )
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: GPVALU => NULL( )
       LOGICAL, POINTER, DIMENSION( : ) :: GXEQX => NULL( )
       LOGICAL, POINTER, DIMENSION( : ) :: INTREP => NULL( )
       CHARACTER ( LEN = 10 ), POINTER, DIMENSION( : ) :: VNAMES => NULL( )
       CHARACTER ( LEN = 10 ), POINTER, DIMENSION( : ) :: GNAMES => NULL( )
     END TYPE LANCELOT_problem_pointer_type

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
       REAL ( KIND = wp ) :: ocnorm, etak, eta0, omegak, omega0
       REAL ( KIND = wp ) :: tau, gamma1, alphae, betae, alphak, epstol
       REAL ( KIND = wp ) :: alphao, betao, omega_min, eta_min
       REAL ( KIND = wp ) :: epsgrd
       CHARACTER ( LEN = 5 ), DIMENSION( 5 ) :: STATE
       LOGICAL :: itzero, reeval

!  the old SBMIN saved variables

       INTEGER :: ifactr, ldx, lfxi, lgxi, lhxi, lggfx, nvar2, first_derivatives
       INTEGER :: jumpto, nbprod, infor, number, nfixed, ibqpst, nfreef, nfree
       INTEGER :: nmhist, maxsel, ntotin, nfreec, lnguvl, lnhuvl, nnonnz, nadd
       INTEGER :: ntype, nsets, nvargp, l_suc, msweep, nbnd, mortor_its, ntotel
       INTEGER :: nvrels, nnza, error, out, print_level, second_derivatives
       INTEGER :: start_print, stop_print, print_gap, inform_status, icfact
       REAL ( KIND = wp ) :: epstlp, gmodel, vscmax, rad, maximum_radius
       REAL ( KIND = wp ) :: epsrcg, fnew, radmin, cgstop, diamin, diamax
       REAL ( KIND = wp ) :: ared, prered, rho, fmodel, curv, dxsqr, fcp, f0
       REAL ( KIND = wp ) :: stepmx, smallh, resmin, qgnorm, oldrad, epscns
       REAL ( KIND = wp ) :: radtol, fill, step, teneps, stpmin, epstln 
       REAL ( KIND = wp ) :: f_min, f_r, f_c, sigma_r, sigma_c, findmx
       LOGICAL :: alllin, altriv, next, second, print_header, modchl
       LOGICAL :: iprcnd, munks , seprec, densep, calcdi, xactcp, reusec
       LOGICAL :: gmpspr, slvbqp, refact, fdgrad, centrl, dprcnd, strctr
       LOGICAL :: use_band, icfs, mortor, firsup, twonrm, direct, myprec
       LOGICAL :: prcond, firstc, nobnds, getders
       LOGICAL :: printt, printi, printm, printw, printd, printe, set_printe
       LOGICAL :: set_printt, set_printi, set_printm, set_printw, set_printd
       LOGICAL :: skipg
       CHARACTER ( LEN = 6 ) :: cgend, lisend
       REAL ( KIND = KIND( 1.0E0 ) ) :: t, time, tmv, tca, tls, tup
       INTEGER, DIMENSION( 5 ) :: ISYS
       CHARACTER ( LEN = 6 ), DIMENSION( 5 ) :: CGENDS
       CHARACTER ( LEN = 6 ), DIMENSION( 5 ) :: LSENDS

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
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: LINK_col
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: POS_in_H
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
     
       REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: GROUP_SCALING  => NULL( )
       LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_AUG  => NULL( )

       TYPE ( SCU_matrix_type ) :: SCU_matrix
       TYPE ( SCU_data_type ) :: SCU_data
       TYPE ( SMT_type ) :: matrix
       TYPE ( SILS_factors ) :: SILS_data

     END TYPE LANCELOT_data_type

!  ======================================
!  The LANCELOT_control_type derived type
!  ======================================

     TYPE :: LANCELOT_control_type
       INTEGER :: error, out, alive_unit, print_level, maxit
       INTEGER :: start_print, stop_print, print_gap, linear_solver
       INTEGER :: icfact, semibandwidth, max_sc, io_buffer, more_toraldo
       INTEGER :: non_monotone, first_derivatives, second_derivatives
       REAL ( KIND = wp ) :: stopc, stopg, min_aug, acccg
       REAL ( KIND = wp ) :: initial_radius, maximum_radius
       REAL ( KIND = wp ) :: eta_successful, eta_very_successful
       REAL ( KIND = wp ) :: eta_extremely_successful
       REAL ( KIND = wp ) :: gamma_smallest, gamma_decrease, gamma_increase
       REAL ( KIND = wp ) :: mu_meaningful_model, mu_meaningful_group
       REAL ( KIND = wp ) :: initial_mu, mu_tol, firstg, firstc
       LOGICAL :: quadratic_problem, two_norm_tr, exact_gcp, magical_steps
       LOGICAL :: accurate_bqp, structured_tr, print_max, full_solution
       CHARACTER ( LEN = 30 ) :: alive_file
       TYPE ( SILS_control ) :: SILS_cntl
     END TYPE LANCELOT_control_type

!  =====================================
!  The LANCELOT_inform_type derived type
!  =====================================

     TYPE :: LANCELOT_inform_type
       INTEGER :: status, alloc_status, iter, itercg, itcgmx
       INTEGER :: ncalcf, ncalcg, nvar, ngeval, iskip, ifixed, nsemib
       REAL ( KIND = wp ) :: aug, obj, pjgnrm, cnorm
       REAL ( KIND = wp ) :: ratio, mu, radius, ciccg
       LOGICAL :: newsol
       CHARACTER ( LEN = 24 ) :: bad_alloc
       TYPE ( SCU_info_type ) :: SCU_info
       TYPE ( SILS_ainfo ) :: SILS_infoa
       TYPE ( SILS_finfo ) :: SILS_infof
       TYPE ( SILS_sinfo ) :: SILS_infos
     END TYPE LANCELOT_inform_type

!-------------------------------
!   I n t e r f a c e  B l o c k
!-------------------------------

     INTERFACE LANCELOT_solve
       MODULE PROCEDURE LANCELOT_solve, LANCELOT_pointer_solve
     END INTERFACE LANCELOT_solve

   CONTAINS

!-*-*-*-*  L A N C E L O T -B- LANCELOT_initialize  S U B R O U T I N E -*-*-*-*

     SUBROUTINE LANCELOT_initialize( data, control )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LANCELOT controls

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_data_type ), INTENT( OUT ) :: data
     TYPE ( LANCELOT_control_type ), INTENT( OUT ) :: control
 
!    INTEGER, PARAMETER :: lmin = 1
     INTEGER, PARAMETER :: lmin = 10000

!  Error and ordinary output unit numbers

     control%error = 6 ; data%S%error = control%error
     control%out = 6 ; data%S%out = control%out

!  Removal of the file alive_file from unit alive_unit causes execution
!  to cease

     control%alive_unit = 60
     control%alive_file = 'ALIVE.d'

!  Level of output required. <= 0 gives no output, = 1 gives a one-line
!  summary for every iteration, = 2 gives a summary of the inner iteration
!  for each iteration, >= 3 gives increasingly verbose (debugging) output

     control%print_level = 0

!  Maximum number of iterations

     control%maxit = 1000

!   Any printing will start on this iteration

     control%start_print = - 1

!   Any printing will stop on this iteration

     control%stop_print = - 1

!   Printing will only occur every print_gap iterations

     control%print_gap = 1

!  linear_solver gives the method to be used for solving the
!                linear system. 1=CG, 2=diagonal preconditioned CG,
!                3=user-provided preconditioned CG, 4=expanding band
!                preconditioned CG, 5=Munksgaard's preconditioned CG,
!                6=Schnabel-Eskow modified Cholesky preconditioned CG,
!                7=Gill-Murray-Ponceleon-Saunders modified Cholesky
!                preconditioned CG, 8=band matrix preconditioned CG, 
!                9=Lin-More' preconditioned CG, 11=multifrontal direct
!                method, 12=direct modified multifrontal method

     control%linear_solver = 8

!  The number of vectors allowed in Lin and More's incomplete factorization

     control%icfact = 5

!  The semi-bandwidth of the band factorization

     control%semibandwidth = 5

!   The maximum dimension of the Schur complement 

     control%max_sc = 100

!  Unit number of i/o buffer for writing temporary files (if needed)

     control%io_buffer = 75

!  more_toraldo >= 1 gives the number of More'-Toraldo projected searches 
!                to be used to improve upon the Cauchy point, anything
!                else is for the standard add-one-at-a-time CG search

     control%more_toraldo = 0

!  non-monotone <= 0 monotone strategy used, anything else non-monotone
!                strategy with this history length used.

     control%non_monotone = 1

!  first_derivatives = 0 if exact first derivatives are given, = 1 if forward
!             finite difference approximations are to be calculated, and 
!             = 2 if central finite difference approximations are to be used

     control%first_derivatives = 0

!  second_derivatives specifies the approximation to the second derivatives
!                used. 0=exact, 1=BFGS, 2=DFP, 3=PSB, 4=SR1

     control%second_derivatives = 0

!  Overall convergence tolerances. The iteration will terminate when the norm
!  of violation of the constraints (the "primal infeasibility") is smaller than 
!  control%stopc and the norm of the gradient of the Lagrangian function (the
!  "dual infeasibility") is smaller than control%stopg

     control%stopc = tenm5
     control%stopg = tenm5

!  It will also terminate if the merit function (objective or augmented
!  Lagrangian as appropriate) is smaller than control%min_aug

     control%min_aug = - ( HUGE( one ) / 8.0_wp )
     
!  Require a relative reduction in the resuiduals from CG of at least acccg

     control%acccg = 0.01_wp

!  The initial trust-region radius - a non-positive value allows the
!  package to choose its own

     control%initial_radius = - one

!  The largest possible trust-region radius

     control%maximum_radius = ten ** 20

!  Parameters that define when to decrease/increase the trust-region 
!  (specialists only!)

     control%eta_successful = 0.01_wp
     control%eta_very_successful = 0.9_wp
     control%eta_extremely_successful = 0.95_wp
     
     control%gamma_smallest = 0.0625_wp
     control%gamma_decrease = 0.25_wp
     control%gamma_increase = 2.0_wp
     
     control%mu_meaningful_model = 0.01_wp
     control%mu_meaningful_group = 0.1_wp

!  The initial value of the penalty parameter

     control%initial_mu = point1

!  The value of the penalty parameter above which the algorithm
!  will not attempt to update the estimates of the Lagrange multipliers

     control%mu_tol = point1

!  The required accuracy of the norm of the projected gradient at the end
!  of the first major iteration

     control%firstg = point1

!  The required accuracy of the norm of the constraints at the end
!  of the first major iteration

     control%firstc = point1

!  Is the function quadratic ? 

     control%quadratic_problem = .FALSE.

!  two_norm_tr is true if a 2-norm trust-region is to be used, and false 
!                for the infinity norm

     control%two_norm_tr = .FALSE.

!  exact_gcp is true if the exact Cauchy point is required, and false if an
!                approximation suffices

     control%exact_gcp = .TRUE.

!  magical_steps is true if magical steps are to be used to improve upon
!                already accepted points, and false otherwise

     control%magical_steps = .FALSE.

!  accurate_bqp is true if the the minimizer of the quadratic model within
!                the intersection of the trust-region and feasible box
!                is to be sought (to a prescribed accuracy), and false 
!                if an approximation suffices

     control%accurate_bqp = .FALSE.

!  structured_tr is true if a structured trust region will be used, and false
!                if a standard trust-region suffices

     control%structured_tr = .FALSE.

!  For printing, if we are maximizing rather than minimizing, print_max
!  should be .TRUE.

     control%print_max = .FALSE.

!  .TRUE. if all components of the solution and constraints are to be printed 
!  on termination, and .FALSE. if only the first and last (representative) few 
!  are required

     control%full_solution = .TRUE.

!  Set initial array lengths for EXTEND arrays

     data%S%EXTEND%lirnh = lmin
     data%S%EXTEND%ljcnh = lmin
     data%S%EXTEND%llink_min = lmin
     data%S%EXTEND%lirnh_min = lmin
     data%S%EXTEND%ljcnh_min = lmin
     data%S%EXTEND%lh_min = lmin
     data%S%EXTEND%lwtran_min = lmin
     data%S%EXTEND%litran_min = lmin
     data%S%EXTEND%lh = lmin
     data%S%ASMBL%ptr_status = .FALSE.

     CALL SILS_initialize( data%SILS_data, control%SILS_cntl )
      control%SILS_cntl%ordering = 3
!57V2 control%SILS_cntl%ordering = 2
!57V3 control%SILS_cntl%ordering = 5
!57V2 control%SILS_cntl%scaling = 0
!57V2 control%SILS_cntl%static_tolerance = zero
!57V2 control%SILS_cntl%static_level = zero

     RETURN

!  End of subroutine LANCELOT_initialize

     END SUBROUTINE LANCELOT_initialize

!-*-*-   L A N C E L O T _ R E A D _ S P E C F I L E  S U B R O U T I N E  -*-*-

     SUBROUTINE LANCELOT_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The default values as given by LANCELOT_initialize could (roughly) 
!  have been set as:

! BEGIN LANCELOT SPECIFICATIONS (DEFAULT)
!  error-printout-device                          6
!  printout-device                                6
!  alive-device                                   60
!  print-level                                    0
!  maximum-number-of-iterations                   1000
!  start-print                                    -1 
!  stop-print                                     -1
!  iterations-between-printing                    1
!  linear-solver-used                             BAND_CG
!  number-of-lin-more-vectors-used                5
!  semi-bandwidth-for-band-preconditioner         5
!  maximum-dimension-of-schur-complement          100
!  unit-number-for-temporary-io                   75
!  more-toraldo-search-length                     0
!  history-length-for-non-monotone-descent        0
!  first-derivative-approximations                EXACT
!  second-derivative-approximations               SR1
!  primal-accuracy-required                       1.0D-5
!  dual-accuracy-required                         1.0D-5
!  minimum-merit-value                            -1.0D+300
!  inner-iteration-relative-accuracy-required     0.01
!  initial-trust-region-radius                    -1.0
!  maximum-radius                                 1.0D+20
!  eta-successful                                 0.01
!  eta-very-successful                            0.9
!  eta-extremely-successful                       0.95
!  gamma-smallest                                 0.0625
!  gamma-decrease                                 0.25
!  gamma-increase                                 2.0
!  mu-meaningful-model                            0.01
!  mu-meaningful-group                            0.1
!  initial-penalty-parameter                      0.1
!  no-dual-updates-until-penalty-parameter-below  0.1
!  initial-dual-accuracy-required                 0.1
!  initial-primal-accuracy-required               0.1
!  pivot-tolerance-used                           0.1
!  quadratic-problem                              NO
!  two-norm-trust-region-used                     NO
!  exact-GCP-used                                 YES
!  magical-steps-allowed                          NO
!  subproblem-solved-accuractely                  NO
!  structured-trust-region-used                   NO
!  print-for-maximimization                       NO
!  print-full-solution                            YES
!  alive-filename                                 ALIVE.d
! END LANCELOT SPECIFICATIONS

!  Dummy arguments

     TYPE ( LANCELOT_control_type ), INTENT( INOUT ) :: control        
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: lspec = 45
     CHARACTER( LEN = 16 ), PARAMETER :: specname = 'LANCELOT        '
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

     spec(  1 )%keyword = 'error-printout-device'
     spec(  2 )%keyword = 'printout-device'
     spec(  3 )%keyword = 'alive-device'
     spec(  4 )%keyword = 'print-level' 
     spec(  5 )%keyword = 'maximum-number-of-iterations'
     spec(  6 )%keyword = 'start-print'
     spec(  7 )%keyword = 'stop-print'
     spec(  8 )%keyword = 'iterations-between-printing'
     spec(  9 )%keyword = 'linear-solver-used'
     spec( 10 )%keyword = 'number-of-lin-more-vectors-used'
     spec( 11 )%keyword = 'semi-bandwidth-for-band-preconditioner'
     spec( 12 )%keyword = 'maximum-dimension-of-schur-complement'
     spec( 13 )%keyword = 'unit-number-for-temporary-io'
     spec( 14 )%keyword = 'more-toraldo-search-length'
     spec( 15 )%keyword = 'history-length-for-non-monotone-descent'
     spec( 16 )%keyword = 'first-derivative-approximations'
     spec( 17 )%keyword = 'second-derivative-approximations'

!  Real key-words

     spec( 18 )%keyword = 'primal-accuracy-required'
     spec( 19 )%keyword = 'dual-accuracy-required'
     spec( 20 )%keyword = 'inner-iteration-relative-accuracy-required'
     spec( 21 )%keyword = 'initial-trust-region-radius'
     spec( 22 )%keyword = 'maximum-radius'
     spec( 23 )%keyword = 'eta-successful'
     spec( 24 )%keyword = 'eta-very-successful'
     spec( 25 )%keyword = 'eta-extremely-successful'
     spec( 26 )%keyword = 'gamma-smallest'
     spec( 27 )%keyword = 'gamma-decrease'
     spec( 28 )%keyword = 'gamma-increase'
     spec( 29 )%keyword = 'mu-meaningful-model'
     spec( 30 )%keyword = 'mu-meaningful-group'
     spec( 31 )%keyword = 'initial-penalty-parameter'
     spec( 32 )%keyword = 'no-dual-updates-until-penalty-parameter-below'
     spec( 33 )%keyword = 'initial-dual-accuracy-required'
     spec( 34 )%keyword = 'initial-primal-accuracy-required'
     spec( 35 )%keyword = 'pivot-tolerance-used'
     spec( 43 )%keyword = 'minimum-merit-value'

!  Logical key-words

     spec( 36 )%keyword = 'quadratic-problem'
     spec( 37 )%keyword = 'two-norm-trust-region-used'
     spec( 38 )%keyword = 'exact-GCP-used'
     spec( 39 )%keyword = 'magical-steps-allowed'
     spec( 40 )%keyword = 'subproblem-solved-accuractely'
     spec( 41 )%keyword = 'structured-trust-region-used'
     spec( 42 )%keyword = 'print-for-maximimization'
     spec( 44 )%keyword = 'print-full-solution'

!  Character key-words

     spec( 45 )%keyword = 'alive-filename'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_integer( spec( 1 ), control%error,                   &
                                   control%error )
     CALL SPECFILE_assign_integer( spec( 2 ), control%out,                     &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 3 ), control%out,                     &
                                   control%alive_unit )                         
     CALL SPECFILE_assign_integer( spec( 4 ), control%print_level,             &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 5 ), control%maxit,                   &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 6 ), control%start_print,             &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 7 ), control%stop_print,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 8 ), control%print_gap,               &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 9 ), control%linear_solver,            &
                                  control%error )                           
     CALL SPECFILE_assign_integer( spec( 10 ), control%icfact,                 &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 11 ), control%semibandwidth,          &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 12 ), control%max_sc,                 &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 13 ), control%io_buffer,              &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 14 ), control%more_toraldo,           &
                                   control%error )                           
     CALL SPECFILE_assign_integer( spec( 15 ), control%non_monotone,           &
                                   control%error )                           
     CALL SPECFILE_assign_symbol( spec( 16 ), control%first_derivatives,       &
                                  control%error )                           
     CALL SPECFILE_assign_symbol( spec( 17 ), control%second_derivatives,      &
                                  control%error )                           

!  Set real values

     CALL SPECFILE_assign_real( spec( 18 ), control%stopc,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 19 ), control%stopg,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 20 ), control%acccg,                     &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 21 ), control%initial_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 22 ), control%maximum_radius,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 23 ), control%eta_successful,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 24 ), control%eta_very_successful,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 25 ), control%eta_extremely_successful,  &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 26 ), control%gamma_smallest,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 27 ), control%gamma_decrease,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 28 ), control%gamma_increase,            &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 29 ), control%mu_meaningful_model,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 30 ), control%mu_meaningful_group,       &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 31 ), control%initial_mu,                &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 32 ), control%mu_tol,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 33 ), control%firstg,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 34 ), control%firstc,                    &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 35 ), control%SILS_cntl%u,               &
                                control%error )                           
     CALL SPECFILE_assign_real( spec( 43 ), control%min_aug,                   &
                                control%error )                           

!  Set logical values

     CALL SPECFILE_assign_logical( spec( 36 ), control%quadratic_problem,      &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 37 ), control%two_norm_tr,            &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 38 ), control%exact_gcp,              &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 39 ), control%magical_steps,          &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 40 ), control%accurate_bqp,           &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 41 ), control%structured_tr,          &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 42 ), control%print_max,              &
                                   control%error )                           
     CALL SPECFILE_assign_logical( spec( 44 ), control%full_solution,          &
                                   control%error ) 

!  Set character values

     CALL SPECFILE_assign_string( spec( 45 ), control%alive_file,              &
                                  control%error )                           

     RETURN

     END SUBROUTINE LANCELOT_read_specfile

!-*-*-*-*-*  L A N C E L O T -B- LANCELOT_solve  S U B R O U T I N E  -*-*-*-*-*

     SUBROUTINE LANCELOT_solve( prob, RANGE , GVALS, FT, XT, FUVALS, lfuval,   &
                                ICALCF, ICALCG, IVAR, Q, DGRAD, control,       &
                                inform, data, ELFUN, GROUP, ELFUN_flexible,    &
                                ELDERS )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  LANCELOT_solve, a method for finding a local minimizer of a
!  function subject to general constraints and simple bounds on the sizes of
!  the variables. The method is described in the paper 'A globally convergent
!  augmented Lagrangian algorithm for optimization with general constraints
!  and simple bounds' by A. R. Conn, N. I. M. Gould and Ph. L. Toint,
!  SIAM J. Num. Anal. 28 (1991) PP.545-572

!  See LANCELOT_solve_main for more details

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_control_type ), INTENT( INOUT ) :: control
     TYPE ( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( LANCELOT_problem_type ), INTENT( INOUT ) :: prob

     INTEGER, INTENT( IN ) :: lfuval
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n  ) :: IVAR
     INTEGER, INTENT( INOUT ), DIMENSION( prob%nel ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( prob%ng ) :: ICALCG
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( prob%ng, 3 ) :: GVALS 
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( prob%n ) :: Q, XT, DGRAD
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng ) :: FT
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( lfuval ) :: FUVALS

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN 

!  Interface block for ELFUN_flexible 

       SUBROUTINE ELFUN_flexible(                                              &
                          FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS,      &
                          ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( IN ) :: llders
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf), ELDERS(2,llders)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN_flexible

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( 2, prob%nel ) :: ELDERS
     OPTIONAL :: ELFUN, ELFUN_flexible, GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig, k1, k2, scu_status, alloc_status
     REAL ( KIND = wp ) :: epsmch
     LOGICAL :: alive, internal_el, internal_gr, use_elders
     CHARACTER ( LEN = 24 ) :: bad_alloc
     LOGICAL, ALLOCATABLE, DIMENSION( : ) :: GXEQX_used

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

     epsmch = EPSILON( one )
     internal_el = PRESENT( ELFUN ) .OR. PRESENT( ELFUN_flexible )
     internal_gr = PRESENT( GROUP )
     use_elders = PRESENT( ELDERS )

     IF ( inform%status > 0 .AND. inform%status /= 14 ) RETURN

! Initial entry: set up data

     IF ( inform%status == 0 ) THEN

!  Record time at which subroutine initially called
  
        CALL CPU_TIME( data%S%time )
  
!  Initialize integer inform parameters
  
!  iter gives the number of iterations performed
!  itercg gives the total numbr of CG iterations performed
!  itcgmx is the maximum number of CG iterations permitted per inner iteration
!  ncalcf gives the number of element functions that must be re-evaluated
!  ncalcg gives the number of group functions that must be re-evaluated
!  nvar gives the current number of free variables
!  ngeval is the number of derivative evaluations made
!  iskip gives the total number of secant updates that are skipped
!  ifixed is the variable that most ecently encountered on of its bounds
!  nsemib is the bandwidth used with the expanding-band preconditioner
  
       inform%iter = 0 ; inform%itercg = 0 ; inform%itcgmx = 0
       inform%ncalcf = 0 ; inform%ncalcg = 0 ; inform%nvar = 0
       inform%ngeval = 0 ; inform%iskip = 0 ; inform%ifixed = 0
       inform%nsemib = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
  
!  Initialize real inform parameters
  
!  aug gives the value of the augmented Lagrangian merit function
!  obj gives the value of the objective function
!  pjgnrm is the norm of the projected gradient of the merit function
!  cnorm gives the norm of the equality constraints
!  ratio gives the current ratio of predicted to achieved merit fn. reduction
!  mu is the current value of the penalty parameter
!  radius is the current value of the trust-region radius
!  ciccg gives the pivot tolerance used when ICCG is used for preconditioning
  
       inform%aug = HUGE( one ) ; inform%obj = HUGE( one ) 
       inform%pjgnrm = HUGE( one )
       inform%cnorm = zero ; inform%ratio = zero ; inform%mu = zero
       inform%radius = zero ; inform%ciccg = zero
  
!  Initialize logical inform parameter
  
!  newsol is true if a major iteration has just been completed
  
       inform%newsol = .FALSE.
  
!  Check problem dimensions

       IF ( prob%n <= 0 .OR. prob%ng <= 0 .OR. prob%nel < 0 ) THEN
         inform%status = 15 ; RETURN ; END IF

!  Set output character strings
  
       data%S%STATE = (/ ' FREE', 'LOWER', 'UPPER', 'FIXED', 'DEGEN' /)
       data%S%ISYS = (/ 0, 0, 0, 0, 0 /)
       data%S%CGENDS = (/ ' CONVR', ' MAXIT', ' BOUND', ' -CURV', ' S<EPS' /)
       data%S%LSENDS = (/ ' PSDEF', ' INDEF', ' SINGC', ' SINGI', ' PRTRB' /)

!  Initialize floating-point parameters
  
!  epstlp and epstln are tolerances on how far a variable may lie away from
!         its bound and still be considered active
!  radtol is the smallest value that the trust region radius is allowed
!  stpmin is the smallest allowable step between consecutive iterates
!  teneps is 10 times the machine precision(!)
!  epsrcg is the smallest that the CG residuals will be required to be
!  vscmax is the largest specified variable scaling
!  fill   is the amount of fill-in when a direct method is used to find an
!         approximate solution to the model problem
  
       data%S%epstlp = epsmch ; data%S%epstln = epsmch
       data%S%epsrcg = hundrd * epsmch ** 2 ; data%S%teneps = ten * epsmch
       data%S%radtol = point1 * epsmch ; data%S%smallh = epsmch ** 0.3333
       data%S%stpmin = epsmch ** 0.75 ; data%S%vscmax = zero
       data%S%fill = zero

!  Timing parameters: tca, tls, tmv and tup are, respectively, the times spent
!  in finding the Cauchy step, finding the approximate minimizer of the model,
!  forming the product of the Hessian with a specified vector and in updating
!  the second derivative approximations. time gives the clock time on initial
!  entry to the subroutine. t and time give the instantaneous clock time
  
       data%S%tmv = 0.0 ; data%S%tca = 0.0 ; data%S%tls = 0.0 ; data%S%tup = 0.0
  
!  number is used to control which negative eigenvalue is picked when the
!         negative curvature exploiting multifrontal scheme is used
  
       data%S%number = 0
  
!  Initialize logical parameters
  
!  full_solution is .TRUE. if all components of the solution and constraints 
!  are to be printed on termination, and .FALSE. if only the first and last
!  (representative) few are required

       data%S%full_solution = control%full_solution

!  S%firsup is .TRUE. if initial second derivative approximations are
!  to be rescaled using the Shanno-Phua scalings
  
       data%S%firsup = .FALSE. ; data%S%next = .FALSE.
  
!  alllin is .TRUE. if there are no nonlinear elements and .FALSE. otherwise
  
       data%S%alllin = prob%nel== 0
  
!  p_type indicates the type of problem: 
!    1  (unconstrained or bound constrained)
!    2  (feasibility, maybe bound constrained)
!    3  (generally constrained)
  
       IF ( ALLOCATED( prob%KNDOFG ) ) THEN
         IF ( SIZE( prob%KNDOFG ) < prob%ng ) THEN
           inform%status = 9 ; RETURN ; END IF
         IF ( COUNT( prob%KNDOFG( : prob%ng ) <= 1 )                           &
              == prob%ng ) THEN
           data%S%p_type = 1
         ELSE
           IF ( ALLOCATED( prob%C ) ) THEN
             IF ( SIZE( prob%C ) < prob%ng ) THEN
               inform%status = 9 ; RETURN ; END IF
           ELSE
             inform%status = 9 ; RETURN
           END IF
           IF ( ALLOCATED( prob%Y ) ) THEN
             IF ( SIZE( prob%Y ) < prob%ng ) THEN
               inform%status = 9 ; RETURN ; END IF
           ELSE
             inform%status = 9 ; RETURN
           END IF
           data%S%p_type = 2
           DO i = 1, prob%ng
             IF ( prob%KNDOFG( i ) == 1 ) THEN 
               data%S%p_type = 3 ; EXIT ; END IF
           END DO
         END IF

!  See if any of the groups are to be skipped

         data%S%skipg = COUNT( prob%KNDOFG == 0 ) > 0
       ELSE
         data%S%p_type = 1 
         data%S%skipg = .FALSE.
       END IF
  
     END IF
  
     IF ( inform%status == 0 .OR. inform%status == 14 ) THEN
  
!  Record the print level and output channel
  
       data%S%out = control%out
  
!  Only print between iterations start_print and stop_print

       IF ( control%start_print < inform%iter ) THEN
         data%S%start_print = 0
       ELSE
         data%S%start_print = control%start_print
       END IF
 
       IF ( control%stop_print < inform%iter ) THEN
         data%S%stop_print = control%maxit
       ELSE
         data%S%stop_print = control%stop_print
       END IF
 
       IF ( control%print_gap < 2 ) THEN
         data%S%print_gap = 1
       ELSE
         data%S%print_gap = control%print_gap
       END IF
 
!  Print warning and error messages
  
       data%S%set_printe = data%S%out > 0 .AND. control%print_level >= 0
  
       IF ( data%S%start_print <= 0 .AND. data%S%stop_print > 0 ) THEN
         data%S%printe = data%S%set_printe
         data%S%print_level = control%print_level
       ELSE
         data%S%printe = .FALSE.
         data%S%print_level = 0
       END IF

!  Test whether the maximum allowed number of iterations has been reached
  
       IF ( control%maxit < 0 ) THEN
         IF ( data%S%printe ) WRITE( data%S%out,                               &
           "( /, ' LANCELOT_solve : maximum number of iterations reached ' )" )
         inform%status = 1 ; RETURN
       END IF
  
!  Basic single line of output per iteration
  
       data%S%set_printi = data%S%out > 0 .AND. control%print_level >= 1 
  
!  As per printi, but with additional timings for various operations
  
       data%S%set_printt = data%S%out > 0 .AND. control%print_level >= 2 
  
!  As per printm, but with checking of residuals, etc
  
       data%S%set_printm = data%S%out > 0 .AND. control%print_level >= 3 
  
!  As per printm but also with an indication of where in the code we are
  
       data%S%set_printw = data%S%out > 0 .AND. control%print_level >= 4
  
!  Full debugging printing with significant arrays printed
  
       data%S%set_printd = data%S%out > 0 .AND. control%print_level >= 10
  
       IF ( data%S%start_print <= 0 .AND. data%S%stop_print > 0 ) THEN
         data%S%printi = data%S%set_printi
         data%S%printt = data%S%set_printt
         data%S%printm = data%S%set_printm
         data%S%printw = data%S%set_printw
         data%S%printd = data%S%set_printd
       ELSE
         data%S%printi = .FALSE.
         data%S%printt = .FALSE.
         data%S%printm = .FALSE.
         data%S%printw = .FALSE.
         data%S%printd = .FALSE.
       END IF

!  Create a file which the user may subsequently remove to cause
!  immediate termination of a run

       IF ( control%alive_unit > 0 ) THEN
         INQUIRE( FILE = control%alive_file, EXIST = alive )
        IF ( .NOT. alive ) THEN
           OPEN( control%alive_unit, FILE = control%alive_file,                &
                 FORM = 'FORMATTED', STATUS = 'NEW' )
           REWIND control%alive_unit
           WRITE( control%alive_unit, "( ' LANCELOT rampages onwards ' )" )
           CLOSE( control%alive_unit )
         END IF
       END IF

       IF ( control%print_max ) THEN
         data%S%findmx = - one ; ELSE ; data%S%findmx = one ; END IF
  
!  twonrm is .TRUE. if the two-norm trust region is to be used, and is .FALSE.
!  if the infinity-norm trust region is required
  
       data%S%twonrm = control%two_norm_tr
       data%S%maximum_radius = MAX( one, control%maximum_radius )
  
!  direct is .TRUE. if the linear system is to be solved using a direct method
!  (MA27). Otherwise, the linear system will be solved using conjugate gradients
  
       data%S%direct = control%linear_solver >= 11
  
!  modchl is .TRUE. if the Hessian is to be forced to be positive definite
!  prcond is .TRUE. if preconditioning is to be used in the conjugate
!         gradient iteration
  
       data%S%modchl = control%linear_solver == 12
       data%S%prcond =  .NOT. data%S%direct .AND. control%linear_solver >= 2
  
!  dprcnd is .TRUE. if the user wishes to use a diagonal preconditioner
  
       data%S%dprcnd = control%linear_solver == 2 
       data%S%calcdi = data%S%dprcnd
  
!  myprec is .TRUE. if the user is to take responsibility for providing the
!  preconditioner
  
       data%S%myprec = control%linear_solver == 3
  
!  iprcnd is .TRUE. if the user wishes to use a positive definite
!  perturbation of the inner band of the true matrix as a preconditioner
  
       data%S%iprcnd = control%linear_solver == 4
  
!  munks is .TRUE. if the Munksgaard preconditioner is to be used
  
       data%S%munks = control%linear_solver == 5
  
!  seprec is .TRUE. if the user wishes to use the Schnabel-Eskow positive
!  definite perturbation of the complete matrix as a preconditioner
  
       data%S%seprec = control%linear_solver == 6
  
!  gmpspr is .TRUE. if the user wishes to use the Gill-Murray-Ponceleon-
!  Saunders positive definite perturbation of the complete matrix as a
!  preconditioner
  
       data%S%gmpspr = control%linear_solver == 7
  
!  use_band is .TRUE. if the user wishes to use a bandsolver as a
!    preconditioner
  
       data%S%use_band = control%linear_solver == 8
  
!  icfs is .TRUE. if the user wishes to use Lin and More's incomplete Cholesky
!  factorization as a preconditioner
  
       data%S%icfs = control%linear_solver == 9
       data%S%icfact = MAX( control%icfact, 0 )
  
!  fdgrad is .FALSE. if the user provides exact first derivatives of the
!  nonlinear element functions and .TRUE. otherwise
  
       IF ( use_elders ) THEN
         ELDERS( 1 , : ) = MAX( MIN( ELDERS( 1 , : ), 2 ), 0 )
         data%S%first_derivatives = MAXVAL( ELDERS( 1 , : ) )
         i = COUNT( ELDERS( 1 , : ) <= 0 )
         data%S%fdgrad = i /= prob%nel
         data%S%getders = i /= 0
       ELSE
         data%S%first_derivatives = MIN( control%first_derivatives, 2 )
         data%S%fdgrad = data%S%first_derivatives >= 1
         data%S%getders = .NOT. data%S%fdgrad
       END IF
  
!  second is .TRUE. if the user provides exact second derivatives
!  of the nonlinear element functions and .FALSE. otherwise
  
       IF ( use_elders ) THEN
         DO i = 1, prob%nel
           ELDERS( 2 , i ) = MAX( MIN( ELDERS( 2 , i ), 4 ), 0 )
           IF ( ELDERS( 1 , i ) > 0 ) ELDERS( 2 , i ) = 4
         END DO
         data%S%second = COUNT( ELDERS( 2 , : ) <= 0 ) == prob%nel
       ELSE
         data%S%second_derivatives = MIN( control%second_derivatives, 4 )
         data%S%second = data%S%second_derivatives <= 0 
         IF ( data%S%fdgrad .AND. data%S%second ) THEN
           data%S%second_derivatives = 4 ; data%S%second = .FALSE.
         END IF
       END IF
  
!  xactcp is .TRUE, if the user wishes to calculate the exact Cauchy
!  point in the fashion of Conn, Gould and Toint ( 1988 ). If an
!  approximation suffices, xactcp will be .FALSE.
  
       data%S%xactcp = control%exact_gcp
  
!  slvbqp is .TRUE. if a good approximation to the minimum of the quadratic
!  model is to be sought at each iteration, while slvbqp is .FALSE. if a less
!  accurate solution is desired
  
       data%S%slvbqp = control%accurate_bqp
  
!  strctr is .TRUE. if a structured trust-region is to be used
           
       data%S%strctr = control%structured_tr
  
!  S%mortor is .TRUE. if the More-Toraldo projected search is to be used
           
       data%S%msweep = control%more_toraldo ; data%S%mortor = data%S%msweep /= 0
  
!  unsucc is .TRUE. if the last attempted step proved unsuccessful
  
       data%S%unsucc = .FALSE.
  
!  nmhist is the length of the history if a non-monotone strategy is to be used
  
       data%S%nmhist = control%non_monotone

!  The problem is generally constrained
  
       IF ( data%S%p_type == 3 ) THEN
  
!  Set initial real values
  
         data%S%tau = point1
         data%S%gamma1 = point1
         data%S%alphae = point1 ; data%S%alphao = one
         data%S%betae = point9 ; data%S%betao = one
         data%S%epstol = epsmch ** 0.75
         inform%mu = MAX( epsmch, control%initial_mu )
         inform%cnorm = HUGE( one )
         data%S%omega_min = control%stopg ; data%S%eta_min = control%stopc
         data%S%epsgrd = control%stopg
         data%S%omega0 = control%firstg                                        &
           / MIN( inform%mu, data%S%gamma1 ) ** data%S%alphao
         data%S%eta0 = control%firstc                                          &
           / MIN( inform%mu, data%S%gamma1 ) ** data%S%alphae
         data%S%icrit = 0 ; data%S%ncrit = 9
         data%S%itzero = .TRUE.
  
!  Set the convergence tolerances
  
         data%S%alphak = MIN( inform%mu, data%S%gamma1 )
         data%S%etak   = MAX( data%S%eta_min,                                  &
                              data%S%eta0 * data%S%alphak ** data%S%alphae )
         data%S%omegak = MAX( data%S%omega_min,                                &
                              data%S%omega0 * data%S%alphak ** data%S%alphao )
         IF ( data%S%printi )                                                  &
           WRITE( data%S%out, 2010 ) inform%mu, data%S%omegak, data%S%etak
       ELSE
         data%S%omegak = control%stopg
       END IF
     END IF
  
     IF ( inform%status == 0 ) THEN

!  Check that ELFUN has not been provided when ELDERS is present

       IF ( use_elders .AND. PRESENT( ELFUN ) ) THEN
         inform%status = 16 ; RETURN ; END IF

!  Check that if ELFUN_flexible is present, then so is ELDERS

       IF ( PRESENT( ELFUN_flexible ) .AND. .NOT. use_elders ) THEN
         inform%status = 17 ; RETURN ; END IF

!  If the element functions are to be evaluated internally, check that
!  the user has supplied appropriate information

       IF ( internal_el ) THEN
         IF ( ALLOCATED( prob%ISTEPA ) .AND. ALLOCATED( prob%EPVALU ) ) THEN
           IF ( SIZE( prob%ISTEPA ) < prob%nel + 1 ) THEN
             inform%status = 10 ; RETURN ; END IF
           IF ( SIZE( prob%EPVALU ) < prob%ISTEPA( prob%nel + 1 ) - 1 ) THEN
             inform%status = 10 ; RETURN ; END IF
         ELSE
           inform%status = 10 ; RETURN
         END IF
       END IF

!  Do the same if the group functions are to be evaluated internally.

       IF ( internal_gr ) THEN
         IF ( ALLOCATED( prob%ISTGPA ) .AND. ALLOCATED( prob%ITYPEG ) .AND.  &
              ALLOCATED( prob%GPVALU ) ) THEN
           IF ( SIZE( prob%ISTGPA ) < prob%ng + 1 .OR.                         &
                SIZE( prob%ITYPEG ) < prob%ng ) THEN
             inform%status = 11 ; RETURN ; END IF
           IF ( SIZE( prob%GPVALU ) < prob%ISTGPA( prob%ng + 1 ) - 1 ) THEN
             inform%status = 11 ; RETURN ; END IF
         ELSE
           inform%status = 11 ; RETURN
         END IF
       END IF

!  Allocate extra local workspace when there are constraints

       ALLOCATE( GXEQX_USED( prob%ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         bad_alloc = 'GXEQX_USED' ; GO TO 980 ; END IF
       IF ( data%S%p_type == 2 .OR. data%S%p_type == 3 ) THEN
         ALLOCATE( data%GROUP_SCALING( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GROUP_SCALING' ; GO TO 980 ; END IF
         ALLOCATE( data%GXEQX_AUG( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           bad_alloc = 'data%GXEQX_AUG' ; GO TO 980 ; END IF
       END IF
  
!  The problem is generally constrained
  
       IF ( data%S%p_type == 3 ) THEN

!  Set initial integer values
  
         data%S%m = 0 ; data%S%nobjgr = 0
         DO ig = 1, prob%ng

!  KNDOFG = 1 correspomd to objective groups, while KNDOFG = 0
!  are groups which are to be excluded from the problem solved.
!  KNDOFG > 1 corresponds to constraint groups. More specifically,
!  KNDOFG = 2 is a general equality constraint, while KNDOFG = 3,4
!  are general equalities resulting after appending a slack variable to 
!  less-than-or-equal or greater-than-or-equal inequalities respectively


           IF ( prob%KNDOFG( ig ) >= 2 ) THEN
             IF ( prob%KNDOFG( ig ) > 4 ) THEN
               inform%status = 7
               RETURN
             ELSE
               data%S%m = data%S%m + 1
             END IF
           ELSE
             data%S%nobjgr = data%S%nobjgr + 1
           END IF
         END DO
  
!  Set initial values for the internal group scalings, GROUP_scaling, 
!  and the array, GXEQX_aug, which tells if each group is trivial
  
         IF ( prob%ng > 0 ) THEN
           WHERE ( prob%KNDOFG > 1 )
             data%GROUP_SCALING = one ; data%GXEQX_AUG = .FALSE.
           ELSEWHERE
             data%GROUP_SCALING = prob%GSCALE ; data%GXEQX_AUG = prob%GXEQX
           END WHERE
         END IF
         GXEQX_used = data%GXEQX_AUG
  
!  The problem is un-/bound-constrained
  
       ELSE IF ( data%S%p_type == 2 ) THEN
         data%S%m = prob%ng ; data%S%nobjgr = 0
         data%GROUP_SCALING = one ; data%GXEQX_AUG = .FALSE.
         GXEQX_used = data%GXEQX_AUG
  
!  The problem is un-/bound-constrained
  
       ELSE
         data%S%m = 0 ; data%S%nobjgr = prob%ng
         GXEQX_used = prob%GXEQX
       END IF
  
       IF ( data%S%printi ) WRITE( data%S%out, 2000 )

!  Print details of the objective function characteristics
  
       IF ( data%S%printi ) WRITE( data%S%out,                                 &
         "( /, ' There are ', I8, ' variables', /,                             &
      &        ' There are ', I8, ' groups', /,                                &
      &        ' There are ', I8, ' nonlinear elements ' )" )                  &
               prob%n, prob%ng, prob%nel
  
       IF ( data%S%printm ) THEN
         WRITE( data%S%out, "( /, ' ------- Group information ------ ' )" )
         IF ( data%S%printd .OR. prob%ng <= 100 ) THEN
           DO ig = 1, prob%ng
             k1 = prob%ISTADG( ig ) ; k2 = prob%ISTADG( ig + 1 ) - 1
  
!  Print details of the groups
  
             IF ( k1 <= k2 ) THEN
               IF ( k1 == k2 ) THEN
                 WRITE( data%S%out, "( /, ' Group ', I5, ' contains ', I5,     &
                &  ' nonlinear element.  This  is  element ', I5 )" )          &
                   ig, 1, prob%IELING( k1 )
               ELSE
                 WRITE( data%S%out, "( /, ' Group ', I5, ' contains ', I5,     &
                &  ' nonlinear element( s ). These are element( s )', 2I5,     &
                &  /, ( 16I5 ) )" ) ig, k2 - k1 + 1, prob%IELING( k1 : k2 )
               END IF
             ELSE
               WRITE( data%S%out, "( /, ' Group ', I5,                         &
              &  ' contains     no nonlinear', ' elements. ')" ) ig
             END IF
             IF ( .NOT. prob%GXEQX( ig ) )                                     &
               WRITE( data%S%out, "( '  * The group function is non-trivial')" )
  
!  Print details of the nonlinear elements
  
             WRITE( data%S%out,                                                &
               "( :, '  * The group has a linear element with variable( s )',  &
            &       ' X( i ), i =', 3I5, /, ( 3X, 19I5 ) )" )                  &
               prob%ICNA( prob%ISTADA( ig ) : prob%ISTADA( ig + 1 ) - 1 )
           END DO
         END IF
         IF ( .NOT. data%S%alllin ) THEN
           WRITE( data%S%out, "( /, ' ------ Element information ----- ' )" )
           IF ( data%S%printd .OR. prob%nel<= 100 ) THEN
             DO iel = 1, prob%nel
               k1 = prob%ISTAEV( iel ) ; k2 = prob%ISTAEV( iel + 1 ) - 1
  
!  Print details of the nonlinear elements
  
               IF ( k1 <= k2 ) THEN
                 WRITE( data%S%out, "( /, ' Nonlinear element', I5, ' has ',   &
                &  I4, ' internal and ', I4, ' elemental variable( s ), ', /,  &
                &  ' X( i ), i =   ', 13I5, /, ( 16I5 ) )" )                   &
                  iel, prob%INTVAR( iel ), k2 - k1 + 1, prob%IELVAR( k1 : k2 )
               ELSE
                 WRITE( data%S%out, "( /, ' Nonlinear element', I5,            &
                &  ' has   no internal',                                       &
                &  ' or       elemental variables.' )" ) iel
               END IF
             END DO
           END IF
         END IF
       END IF
  
!  Partition the workspace array FUVALS and initialize other workspace
!  arrays

       data%S%ntotel = prob%ISTADG( prob%ng  + 1 ) - 1
       data%S%nvrels = prob%ISTAEV( prob%nel + 1 ) - 1
       data%S%nnza   = prob%ISTADA( prob%ng  + 1 ) - 1

       IF ( ALLOCATED( prob%KNDOFG ) ) THEN
       CALL INITW_initialize_workspace(                                        &
             prob%n, prob%ng, prob%nel,                                        &
             data%S%ntotel, data%S%nvrels, data%S%nnza, prob%n,                &
             data%S%nvargp, prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,&
             prob%INTVAR, prob%ISTADH, prob%ICNA, prob%ISTADA, prob%ITYPEE,    &
             GXEQX_used, prob%INTREP, data%S%altriv, data%S%direct,            &
             data%S%fdgrad, data%S%lfxi, data%S%lgxi, data%S%lhxi,             &
             data%S%lggfx, data%S%ldx, data%S%lnguvl, data%S%lnhuvl,           &
             data%S%ntotin, data%S%ntype, data%S%nsets , data%S%maxsel,        &
             RANGE, data%S%print_level, data%S%out, control%io_buffer,         &
!  workspace
             data%S%EXTEND%lwtran, data%S%EXTEND%litran,                       &
             data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,               &
             data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,              &
             data%ITRANS, data%LINK_elem_uses_var, data%WTRANS,                &
             data%ISYMMD, data%ISWKSP, data%ISTAJC, data%ISTAGV,               &
             data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR,               &
             data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,               &
             data%ISET  , data%ISVSET, data%INVSET, data%LIST_elements,        &
             data%ISYMMH, data%IW_asmbl, data%NZ_comp_w, data%W_ws,            &
             data%W_el, data%W_in, data%H_el, data%H_in,                       &
             inform%status, alloc_status, bad_alloc,                           &
             data%S%skipg, KNDOFG = prob%KNDOFG )
       ELSE
       CALL INITW_initialize_workspace(                                        &
             prob%n, prob%ng, prob%nel,                                        &
             data%S%ntotel, data%S%nvrels, data%S%nnza, prob%n,                &
             data%S%nvargp, prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,&
             prob%INTVAR, prob%ISTADH, prob%ICNA, prob%ISTADA, prob%ITYPEE,    &
             GXEQX_used, prob%INTREP, data%S%altriv, data%S%direct,            &
             data%S%fdgrad, data%S%lfxi, data%S%lgxi, data%S%lhxi,             &
             data%S%lggfx, data%S%ldx, data%S%lnguvl, data%S%lnhuvl,           &
             data%S%ntotin, data%S%ntype, data%S%nsets , data%S%maxsel,        &
             RANGE, data%S%print_level, data%S%out, control%io_buffer,         &
!  workspace
             data%S%EXTEND%lwtran, data%S%EXTEND%litran,                       &
             data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,               &
             data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,              &
             data%ITRANS, data%LINK_elem_uses_var, data%WTRANS,                &
             data%ISYMMD, data%ISWKSP, data%ISTAJC, data%ISTAGV,               &
             data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR,               &
             data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,               &
             data%ISET  , data%ISVSET, data%INVSET, data%LIST_elements,        &
             data%ISYMMH, data%IW_asmbl, data%NZ_comp_w, data%W_ws,            &
             data%W_el, data%W_in, data%H_el, data%H_in,                       &
             inform%status, alloc_status, bad_alloc,                           &
             data%S%skipg )
       END IF

       IF ( ALLOCATED( GXEQX_USED ) ) THEN
         DEALLOCATE( GXEQX_USED, STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           inform%status = 12
           inform%alloc_status = alloc_status
           inform%bad_alloc = 'GXEQX_USED'
           WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
         END IF
       END IF

       IF ( inform%status == 12 ) THEN
         inform%alloc_status = alloc_status
         inform%bad_alloc = bad_alloc
       END IF
       IF ( inform%status /= 0 ) RETURN                              
                                                                              
!  Allocate arrays                                                          
                                                                              
       FUVALS = - HUGE( one )  ! needed for epcf90 debugging compiler
       
       ALLOCATE( data%P( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%P' ; GO TO 980 ; END IF
       
       ALLOCATE( data%XCP( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%XCP' ; GO TO 980 ; END IF
       
       ALLOCATE( data%X0( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%X0' ; GO TO 980 ; END IF
       
       ALLOCATE( data%GX0( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%GX0' ; GO TO 980 ; END IF
       
       ALLOCATE( data%DELTAX( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%DELTAX' ; GO TO 980 ; END IF
       
       ALLOCATE( data%QGRAD( MAX( prob%n, data%S%ntotin ) ),                   &
         STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%QGRAD' ; GO TO 980 ; END IF
       
       ALLOCATE( data%GRJAC( data%S%nvargp ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
          bad_alloc = 'data%GRJAC' ; GO TO 980 ; END IF
       
       ALLOCATE( data%BND( prob%n, 2 ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BND' ; GO TO 980 ; END IF
       
       ALLOCATE( data%BREAKP( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BREAKP' ; GO TO 980 ; END IF

       IF ( data%S%xactcp ) THEN
         ALLOCATE( data%GRAD( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GRAD' ; GO TO 980 ; END IF
       ELSE       
         ALLOCATE( data%GRAD( prob%n ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GRAD' ; GO TO 980 ; END IF
       END IF

       data%S%nbnd = prob%n
       IF ( data%S%mortor .AND. .NOT. data%S%twonrm ) THEN
         ALLOCATE( data%BND_radius( data%S%nbnd, 2 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%BND_radius' ; GO TO 980 ; END IF
       ELSE IF ( data%S%strctr ) THEN
         ALLOCATE( data%BND_radius( data%S%nbnd, 1 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%BND_radius' ; GO TO 980 ; END IF
       ELSE
         data%S%nbnd = 0
         ALLOCATE( data%BND_radius( data%S%nbnd, 2 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%BND_radius' ; GO TO 980 ; END IF
       END IF
       
       IF ( data%S%strctr ) THEN
!        ALLOCATE( data%D_model( prob%ng ), STAT = alloc_status )
!        IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%D_model' ; GO TO 980
!        END IF
         
!        ALLOCATE( data%D_function( prob%ng ), STAT = alloc_status )
!        IF ( alloc_status /= 0 ) THEN 
!          bad_alloc = 'data%D_function' ; GO TO 980
!        END IF
         
         ALLOCATE( data%RADII( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%RADII' ; GO TO 980
         END IF
         
         ALLOCATE( data%GV_old( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%GV_old' ; GO TO 980
         END IF
       
       ELSE

         ALLOCATE( data%RADII( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%RADII' ; GO TO 980
         END IF
         
         ALLOCATE( data%GV_old( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%GV_old' ; GO TO 980
         END IF
       
       END IF
  
!  Store the free variables as the the first nfree components of IFREE
  
       ALLOCATE( data%IFREE( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%IFREE' ; GO TO 980 ; END IF
  
!  INDEX( j ), j = 1, ..., n, will contain the status of the
!  j-th variable as the current iteration progresses. Possible values
!  are 0 if the variable lies away from its bounds, 1 and 2 if it lies
!  on its lower or upper bounds (respectively) - these may be problem
!  bounds or trust-region bounds, and 3 if the variable is fixed
  
       ALLOCATE( data%INDEX( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%INDEX' ; GO TO 980 ; END IF
  
!  IFREEC( j ), j = 1, ..., n will give the indices of the
!  variables which are considered to be free from their bounds at the
!  current generalized cauchy point
  
       ALLOCATE( data%IFREEC( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%IFREEC' ; GO TO 980 ; END IF
  
!  INNONZ( j ), j = 1, ..., nnnonz will give the indices of the nonzeros
!  in the vector obtained as a result of the matrix-vector product from
!  subroutine HSPRD
  
       ALLOCATE( data%INNONZ( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%INNONZ' ; GO TO 980 ; END IF
  
!  Make space for finite-difference values if required
  
       IF ( data%S%fdgrad .AND. .NOT. data%S%alllin ) THEN
         ALLOCATE( data%FUVALS_temp( prob%nel ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%FUVA_t' ; GO TO 980 ; END IF
       ELSE
         ALLOCATE( data%FUVALS_temp( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%FUVA_t' ; GO TO 980 ; END IF
       END IF

!  Space required for the Schur complement

       data%SCU_matrix%m = 0
       data%SCU_matrix%n = 1
       data%SCU_matrix%m_max = MAX( control%max_sc, 1 )
       data%SCU_matrix%class = 4
     
       ALLOCATE( data%SCU_matrix%BD_col_start( data%SCU_matrix%m_max + 1 ),    &
                 STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BD_col_st' ; GO TO 980 ; END IF
       
       ALLOCATE( data%SCU_matrix%BD_row( data%SCU_matrix%m_max ),              &
                 STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BD_row' ; GO TO 980 ; END IF
       
       ALLOCATE( data%SCU_matrix%BD_val( data%SCU_matrix%m_max ),              &
                 STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BD_val' ; GO TO 980 ; END IF
      
!  Space required for the factors of the Schur complement

       data%SCU_matrix%BD_col_start( 1 ) = 1
       scu_status = 1
       CALL SCU_factorize( data%SCU_matrix, data%SCU_data, data%P, scu_status, &
                           inform%SCU_info )
       IF ( scu_status /= 0 ) THEN
         WRITE( data%S%out, "( ' SCU_factorize: status = ', I2 )" ) scu_status
         inform%status = 12
         inform%alloc_status = inform%SCU_info%alloc_status
         inform%bad_alloc = 'SCU_factorize array'
         RETURN
       END IF

     END IF

!  ===============================================
!  Call the solver to perform the bulk of the work
!  ===============================================

!  Both internal element and group evaluations will be performed
!  -------------------------------------------------------------

     IF ( internal_el .AND. internal_gr ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

        IF ( data%S%skipg ) THEN
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          END IF

!  Use all groups

        ELSE
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          END IF
        END IF
 
!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
        IF ( use_elders ) THEN
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ENDIF
       END IF

!  Just internal element evaluations will be performed
!  ---------------------------------------------------

     ELSE IF ( internal_el ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

        IF ( data%S%skipg ) THEN

          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          END IF

!  Use all groups

        ELSE
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          END IF
        END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
        IF ( use_elders ) THEN
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        END IF
       END IF

!  Just internal group evaluations will be performed
!  -------------------------------------------------

     ELSE IF ( internal_gr ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

         IF ( data%S%skipg ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )

!  Use all groups

         ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
         END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
       END IF

!  Element and group evaluations will be performed via reverse communication
!  -------------------------------------------------------------------------

     ELSE

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

         IF ( data%S%skipg ) THEN

           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional argument
             ELDERS = ELDERS )

!  Use all groups

         ELSE

           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional argument
             ELDERS = ELDERS )
         END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y, ELDERS = ELDERS,    &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )

       END IF
     END IF

     RETURN

!  Unsuccessful returns

 980 CONTINUE
     inform%status = 12
     inform%alloc_status = alloc_status
     inform%bad_alloc = bad_alloc
     WRITE( data%S%error, 2990 ) alloc_status, bad_alloc

     IF ( ALLOCATED( GXEQX_USED ) ) THEN
       DEALLOCATE( GXEQX_USED, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'GXEQX_USED'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ASSOCIATED( data%GXEQX_AUG ) ) THEN
       DEALLOCATE( data%GXEQX_AUG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GXEQX_AUG'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ASSOCIATED( data%GROUP_SCALING ) ) THEN
       DEALLOCATE( data%GROUP_SCALING, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GROUP_SCALING'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     RETURN

!  Non-executable statements

 2000  FORMAT( /, ' *********  Starting optimization  ************** ' )
 2010  FORMAT( /, ' Penalty parameter ', ES12.4,                               &
                  ' Required projected gradient norm = ', ES12.4, /,           &
                  '                   ', 12X,                                  &
                  ' Required constraint         norm = ', ES12.4 )             
 2990  FORMAT( ' ** Message from -LANCELOT_solve-', /,                         &
               ' Allocation error (status = ', I6, ') for ', A24 )

!  End of subroutine LANCELOT_solve

     END SUBROUTINE LANCELOT_solve

!-*-*-*  L A N C E L O T -B- LANCELOT_pointer_solve  S U B R O U T I N E  -*-*-*

     SUBROUTINE LANCELOT_pointer_solve(                                        &
                                prob, RANGE , GVALS, FT, XT, FUVALS, lfuval,   &
                                ICALCF, ICALCG, IVAR, Q, DGRAD, control,       &
                                inform, data, ELFUN, GROUP, ELFUN_flexible,    &
                                ELDERS )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  LANCELOT_solve, a method for finding a local minimizer of a
!  function subject to general constraints and simple bounds on the sizes of
!  the variables. The method is described in the paper 'A globally convergent
!  augmented Lagrangian algorithm for optimization with general constraints
!  and simple bounds' by A. R. Conn, N. I. M. Gould and Ph. L. Toint,
!  SIAM J. Num. Anal. 28 (1991) PP.545-572

!  See LANCELOT_solve_main for more details

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_control_type ), INTENT( INOUT ) :: control
     TYPE ( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( LANCELOT_problem_pointer_type ), INTENT( INOUT ) :: prob

     INTEGER, INTENT( IN ) :: lfuval
     INTEGER, INTENT( INOUT ), DIMENSION( prob%n  ) :: IVAR
     INTEGER, INTENT( INOUT ), DIMENSION( prob%nel ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( prob%ng ) :: ICALCG
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( prob%ng, 3 ) :: GVALS 
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( prob%n ) :: Q, XT, DGRAD
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng ) :: FT
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
                         DIMENSION( lfuval ) :: FUVALS

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN 

!  Interface block for ELFUN_flexible 

       SUBROUTINE ELFUN_flexible(                                              &
                          FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS,      &
                          ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( IN ) :: llders
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf), ELDERS(2,llders)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN_flexible

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( 2, prob%nel ) :: ELDERS
     OPTIONAL :: ELFUN, ELFUN_flexible, GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig, k1, k2, scu_status, alloc_status
     REAL ( KIND = wp ) :: epsmch
     LOGICAL :: alive, internal_el, internal_gr, use_elders
     CHARACTER ( LEN = 24 ) :: bad_alloc
     LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_used

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

     epsmch = EPSILON( one )
     internal_el = PRESENT( ELFUN ) .OR. PRESENT( ELFUN_flexible )
     internal_gr = PRESENT( GROUP )
     use_elders = PRESENT( ELDERS )

     IF ( inform%status > 0 .AND. inform%status /= 14 ) RETURN

! Initial entry: set up data

     IF ( inform%status == 0 ) THEN

!  Record time at which subroutine initially called
  
        CALL CPU_TIME( data%S%time )
  
!  Initialize integer inform parameters
  
!  iter gives the number of iterations performed
!  itercg gives the total numbr of CG iterations performed
!  itcgmx is the maximum number of CG iterations permitted per inner iteration
!  ncalcf gives the number of element functions that must be re-evaluated
!  ncalcg gives the number of group functions that must be re-evaluated
!  nvar gives the current number of free variables
!  ngeval is the number of derivative evaluations made
!  iskip gives the total number of secant updates that are skipped
!  ifixed is the variable that most ecently encountered on of its bounds
!  nsemib is the bandwidth used with the expanding-band preconditioner
  
       inform%iter = 0 ; inform%itercg = 0 ; inform%itcgmx = 0
       inform%ncalcf = 0 ; inform%ncalcg = 0 ; inform%nvar = 0
       inform%ngeval = 0 ; inform%iskip = 0 ; inform%ifixed = 0
       inform%nsemib = 0 ; inform%alloc_status = 0 ; inform%bad_alloc = ''
  
!  Initialize real inform parameters
  
!  aug gives the value of the augmented Lagrangian merit function
!  obj gives the value of the objective function
!  pjgnrm is the norm of the projected gradient of the merit function
!  cnorm gives the norm of the equality constraints
!  ratio gives the current ratio of predicted to achieved merit fn. reduction
!  mu is the current value of the penalty parameter
!  radius is the current value of the trust-region radius
!  ciccg gives the pivot tolerance used when ICCG is used for preconditioning
  
       inform%aug = HUGE( one ) ; inform%obj = HUGE( one ) 
       inform%pjgnrm = HUGE( one )
       inform%cnorm = zero ; inform%ratio = zero ; inform%mu = zero
       inform%radius = zero ; inform%ciccg = zero
  
!  Initialize logical inform parameter
  
!  newsol is true if a major iteration has just been completed
  
       inform%newsol = .FALSE.
  
!  Check problem dimensions

       IF ( prob%n <= 0 .OR. prob%ng <= 0 .OR. prob%nel < 0 ) THEN
         inform%status = 15 ; RETURN ; END IF

!  Set output character strings
  
       data%S%STATE = (/ ' FREE', 'LOWER', 'UPPER', 'FIXED', 'DEGEN' /)
       data%S%ISYS = (/ 0, 0, 0, 0, 0 /)
       data%S%CGENDS = (/ ' CONVR', ' MAXIT', ' BOUND', ' -CURV', ' S<EPS' /)
       data%S%LSENDS = (/ ' PSDEF', ' INDEF', ' SINGC', ' SINGI', ' PRTRB' /)

!  Initialize floating-point parameters
  
!  epstlp and epstln are tolerances on how far a variable may lie away from
!         its bound and still be considered active
!  radtol is the smallest value that the trust region radius is allowed
!  stpmin is the smallest allowable step between consecutive iterates
!  teneps is 10 times the machine precision(!)
!  epsrcg is the smallest that the CG residuals will be required to be
!  vscmax is the largest specified variable scaling
!  fill   is the amount of fill-in when a direct method is used to find an
!         approximate solution to the model problem
  
       data%S%epstlp = epsmch ; data%S%epstln = epsmch
       data%S%epsrcg = hundrd * epsmch ** 2 ; data%S%teneps = ten * epsmch
       data%S%radtol = point1 * epsmch ; data%S%smallh = epsmch ** 0.3333
       data%S%stpmin = epsmch ** 0.75 ; data%S%vscmax = zero
       data%S%fill = zero

!  Timing parameters: tca, tls, tmv and tup are, respectively, the times spent
!  in finding the Cauchy step, finding the approximate minimizer of the model,
!  forming the product of the Hessian with a specified vector and in updating
!  the second derivative approximations. time gives the clock time on initial
!  entry to the subroutine. t and time give the instantaneous clock time
  
       data%S%tmv = 0.0 ; data%S%tca = 0.0 ; data%S%tls = 0.0 ; data%S%tup = 0.0
  
!  number is used to control which negative eigenvalue is picked when the
!         negative curvature exploiting multifrontal scheme is used
  
       data%S%number = 0
  
!  Initialize logical parameters
  
!  full_solution is .TRUE. if all components of the solution and constraints 
!  are to be printed on termination, and .FALSE. if only the first and last
!  (representative) few are required

       data%S%full_solution = control%full_solution

!  S%firsup is .TRUE. if initial second derivative approximations are
!  to be rescaled using the Shanno-Phua scalings
  
       data%S%firsup = .FALSE. ; data%S%next = .FALSE.
  
!  alllin is .TRUE. if there are no nonlinear elements and .FALSE. otherwise
  
       data%S%alllin = prob%nel== 0
  
!  p_type indicates the type of problem: 
!    1  (unconstrained or bound constrained)
!    2  (feasibility, maybe bound constrained)
!    3  (generally constrained)
  
       IF ( ASSOCIATED( prob%KNDOFG ) ) THEN
         IF ( SIZE( prob%KNDOFG ) < prob%ng ) THEN
           inform%status = 9 ; RETURN ; END IF
         IF ( COUNT( prob%KNDOFG( : prob%ng ) <= 1 )                           &
              == prob%ng ) THEN
           data%S%p_type = 1
         ELSE
           IF ( ASSOCIATED( prob%C ) ) THEN
             IF ( SIZE( prob%C ) < prob%ng ) THEN
               inform%status = 9 ; RETURN ; END IF
           ELSE
             inform%status = 9 ; RETURN
           END IF
           IF ( ASSOCIATED( prob%Y ) ) THEN
             IF ( SIZE( prob%Y ) < prob%ng ) THEN
               inform%status = 9 ; RETURN ; END IF
           ELSE
             inform%status = 9 ; RETURN
           END IF
           data%S%p_type = 2
           DO i = 1, prob%ng
             IF ( prob%KNDOFG( i ) == 1 ) THEN 
               data%S%p_type = 3 ; EXIT ; END IF
           END DO
         END IF

!  See if any of the groups are to be skipped

         data%S%skipg = COUNT( prob%KNDOFG == 0 ) > 0
       ELSE
         data%S%p_type = 1 
         data%S%skipg = .FALSE.
       END IF
  
     END IF
  
     IF ( inform%status == 0 .OR. inform%status == 14 ) THEN
  
!  Record the print level and output channel
  
       data%S%out = control%out
  
!  Only print between iterations start_print and stop_print

       IF ( control%start_print < inform%iter ) THEN
         data%S%start_print = 0
       ELSE
         data%S%start_print = control%start_print
       END IF
 
       IF ( control%stop_print < inform%iter ) THEN
         data%S%stop_print = control%maxit
       ELSE
         data%S%stop_print = control%stop_print
       END IF
 
       IF ( control%print_gap < 2 ) THEN
         data%S%print_gap = 1
       ELSE
         data%S%print_gap = control%print_gap
       END IF
 
!  Print warning and error messages
  
       data%S%set_printe = data%S%out > 0 .AND. control%print_level >= 0
  
       IF ( data%S%start_print <= 0 .AND. data%S%stop_print > 0 ) THEN
         data%S%printe = data%S%set_printe
         data%S%print_level = control%print_level
       ELSE
         data%S%printe = .FALSE.
         data%S%print_level = 0
       END IF

!  Test whether the maximum allowed number of iterations has been reached
  
       IF ( control%maxit < 0 ) THEN
         IF ( data%S%printe ) WRITE( data%S%out,                               &
           "( /, ' LANCELOT_solve : maximum number of iterations reached ' )" )
         inform%status = 1 ; RETURN
       END IF
  
!  Basic single line of output per iteration
  
       data%S%set_printi = data%S%out > 0 .AND. control%print_level >= 1 
  
!  As per printi, but with additional timings for various operations
  
       data%S%set_printt = data%S%out > 0 .AND. control%print_level >= 2 
  
!  As per printm, but with checking of residuals, etc
  
       data%S%set_printm = data%S%out > 0 .AND. control%print_level >= 3 
  
!  As per printm but also with an indication of where in the code we are
  
       data%S%set_printw = data%S%out > 0 .AND. control%print_level >= 4
  
!  Full debugging printing with significant arrays printed
  
       data%S%set_printd = data%S%out > 0 .AND. control%print_level >= 10
  
       IF ( data%S%start_print <= 0 .AND. data%S%stop_print > 0 ) THEN
         data%S%printi = data%S%set_printi
         data%S%printt = data%S%set_printt
         data%S%printm = data%S%set_printm
         data%S%printw = data%S%set_printw
         data%S%printd = data%S%set_printd
       ELSE
         data%S%printi = .FALSE.
         data%S%printt = .FALSE.
         data%S%printm = .FALSE.
         data%S%printw = .FALSE.
         data%S%printd = .FALSE.
       END IF

!  Create a file which the user may subsequently remove to cause
!  immediate termination of a run

       IF ( control%alive_unit > 0 ) THEN
         INQUIRE( FILE = control%alive_file, EXIST = alive )
        IF ( .NOT. alive ) THEN
           OPEN( control%alive_unit, FILE = control%alive_file,                &
                 FORM = 'FORMATTED', STATUS = 'NEW' )
           REWIND control%alive_unit
           WRITE( control%alive_unit, "( ' LANCELOT rampages onwards ' )" )
           CLOSE( control%alive_unit )
         END IF
       END IF

       IF ( control%print_max ) THEN
         data%S%findmx = - one ; ELSE ; data%S%findmx = one ; END IF
  
!  twonrm is .TRUE. if the two-norm trust region is to be used, and is .FALSE.
!  if the infinity-norm trust region is required
  
       data%S%twonrm = control%two_norm_tr
       data%S%maximum_radius = MAX( one, control%maximum_radius )
  
!  direct is .TRUE. if the linear system is to be solved using a direct method
!  (MA27). Otherwise, the linear system will be solved using conjugate gradients
  
       data%S%direct = control%linear_solver >= 11
  
!  modchl is .TRUE. if the Hessian is to be forced to be positive definite
!  prcond is .TRUE. if preconditioning is to be used in the conjugate
!         gradient iteration
  
       data%S%modchl = control%linear_solver == 12
       data%S%prcond =  .NOT. data%S%direct .AND. control%linear_solver >= 2
  
!  dprcnd is .TRUE. if the user wishes to use a diagonal preconditioner
  
       data%S%dprcnd = control%linear_solver == 2 
       data%S%calcdi = data%S%dprcnd
  
!  myprec is .TRUE. if the user is to take responsibility for providing the
!  preconditioner
  
       data%S%myprec = control%linear_solver == 3
  
!  iprcnd is .TRUE. if the user wishes to use a positive definite
!  perturbation of the inner band of the true matrix as a preconditioner
  
       data%S%iprcnd = control%linear_solver == 4
  
!  munks is .TRUE. if the Munksgaard preconditioner is to be used
  
       data%S%munks = control%linear_solver == 5
  
!  seprec is .TRUE. if the user wishes to use the Schnabel-Eskow positive
!  definite perturbation of the complete matrix as a preconditioner
  
       data%S%seprec = control%linear_solver == 6
  
!  gmpspr is .TRUE. if the user wishes to use the Gill-Murray-Ponceleon-
!  Saunders positive definite perturbation of the complete matrix as a
!  preconditioner
  
       data%S%gmpspr = control%linear_solver == 7
  
!  use_band is .TRUE. if the user wishes to use a bandsolver as a
!    preconditioner
  
       data%S%use_band = control%linear_solver == 8
  
!  icfs is .TRUE. if the user wishes to use Lin and More's incomplete Cholesky
!  factorization as a preconditioner
  
       data%S%icfs = control%linear_solver == 9
       data%S%icfact = MAX( control%icfact, 0 )
  
!  fdgrad is .FALSE. if the user provides exact first derivatives of the
!  nonlinear element functions and .TRUE. otherwise
  
       IF ( use_elders ) THEN
         ELDERS( 1 , : ) = MAX( MIN( ELDERS( 1 , : ), 2 ), 0 )
         data%S%first_derivatives = MAXVAL( ELDERS( 1 , : ) )
         i = COUNT( ELDERS( 1 , : ) <= 0 )
         data%S%fdgrad = i /= prob%nel
         data%S%getders = i /= 0
       ELSE
         data%S%first_derivatives = MIN( control%first_derivatives, 2 )
         data%S%fdgrad = data%S%first_derivatives >= 1
         data%S%getders = .NOT. data%S%fdgrad
       END IF
  
!  second is .TRUE. if the user provides exact second derivatives
!  of the nonlinear element functions and .FALSE. otherwise
  
       IF ( use_elders ) THEN
         DO i = 1, prob%nel
           ELDERS( 2 , i ) = MAX( MIN( ELDERS( 2 , i ), 4 ), 0 )
           IF ( ELDERS( 1 , i ) > 0 ) ELDERS( 2 , i ) = 4
         END DO
         data%S%second = COUNT( ELDERS( 2 , : ) <= 0 ) == prob%nel
       ELSE
         data%S%second_derivatives = MIN( control%second_derivatives, 4 )
         data%S%second = data%S%second_derivatives <= 0 
         IF ( data%S%fdgrad .AND. data%S%second ) THEN
           data%S%second_derivatives = 4 ; data%S%second = .FALSE.
         END IF
       END IF
  
!  xactcp is .TRUE, if the user wishes to calculate the exact Cauchy
!  point in the fashion of Conn, Gould and Toint ( 1988 ). If an
!  approximation suffices, xactcp will be .FALSE.
  
       data%S%xactcp = control%exact_gcp
  
!  slvbqp is .TRUE. if a good approximation to the minimum of the quadratic
!  model is to be sought at each iteration, while slvbqp is .FALSE. if a less
!  accurate solution is desired
  
       data%S%slvbqp = control%accurate_bqp
  
!  strctr is .TRUE. if a structured trust-region is to be used
           
       data%S%strctr = control%structured_tr
  
!  S%mortor is .TRUE. if the More-Toraldo projected search is to be used
           
       data%S%msweep = control%more_toraldo ; data%S%mortor = data%S%msweep /= 0
  
!  unsucc is .TRUE. if the last attempted step proved unsuccessful
  
       data%S%unsucc = .FALSE.
  
!  nmhist is the length of the history if a non-monotone strategy is to be used
  
       data%S%nmhist = control%non_monotone

!  The problem is generally constrained
  
       IF ( data%S%p_type == 3 ) THEN
  
!  Set initial real values
  
         data%S%tau = point1
         data%S%gamma1 = point1
         data%S%alphae = point1 ; data%S%alphao = one
         data%S%betae = point9 ; data%S%betao = one
         data%S%epstol = epsmch ** 0.75
         inform%mu = MAX( epsmch, control%initial_mu )
         inform%cnorm = HUGE( one )
         data%S%omega_min = control%stopg ; data%S%eta_min = control%stopc
         data%S%epsgrd = control%stopg
         data%S%omega0 = control%firstg                                        &
           / MIN( inform%mu, data%S%gamma1 ) ** data%S%alphao
         data%S%eta0 = control%firstc                                          &
           / MIN( inform%mu, data%S%gamma1 ) ** data%S%alphae
         data%S%icrit = 0 ; data%S%ncrit = 9
         data%S%itzero = .TRUE.
  
!  Set the convergence tolerances
  
         data%S%alphak = MIN( inform%mu, data%S%gamma1 )
         data%S%etak   = MAX( data%S%eta_min,                                  &
                              data%S%eta0 * data%S%alphak ** data%S%alphae )
         data%S%omegak = MAX( data%S%omega_min,                                &
                              data%S%omega0 * data%S%alphak ** data%S%alphao )
         IF ( data%S%printi )                                                  &
           WRITE( data%S%out, 2010 ) inform%mu, data%S%omegak, data%S%etak
       ELSE
         data%S%omegak = control%stopg
       END IF
     END IF
  
     IF ( inform%status == 0 ) THEN

!  Check that ELFUN has not been provided when ELDERS is present

       IF ( use_elders .AND. PRESENT( ELFUN ) ) THEN
         inform%status = 16 ; RETURN ; END IF

!  Check that if ELFUN_flexible is present, then so is ELDERS

       IF ( PRESENT( ELFUN_flexible ) .AND. .NOT. use_elders ) THEN
         inform%status = 17 ; RETURN ; END IF

!  If the element functions are to be evaluated internally, check that
!  the user has supplied appropriate information

       IF ( internal_el ) THEN
         IF ( ASSOCIATED( prob%ISTEPA ) .AND. ASSOCIATED( prob%EPVALU ) ) THEN
           IF ( SIZE( prob%ISTEPA ) < prob%nel + 1 ) THEN
             inform%status = 10 ; RETURN ; END IF
           IF ( SIZE( prob%EPVALU ) < prob%ISTEPA( prob%nel + 1 ) - 1 ) THEN
             inform%status = 10 ; RETURN ; END IF
         ELSE
           inform%status = 10 ; RETURN
         END IF
       END IF

!  Do the same if the group functions are to be evaluated internally.

       IF ( internal_gr ) THEN
         IF ( ASSOCIATED( prob%ISTGPA ) .AND. ASSOCIATED( prob%ITYPEG ) .AND.  &
              ASSOCIATED( prob%GPVALU ) ) THEN
           IF ( SIZE( prob%ISTGPA ) < prob%ng + 1 .OR.                         &
                SIZE( prob%ITYPEG ) < prob%ng ) THEN
             inform%status = 11 ; RETURN ; END IF
           IF ( SIZE( prob%GPVALU ) < prob%ISTGPA( prob%ng + 1 ) - 1 ) THEN
             inform%status = 11 ; RETURN ; END IF
         ELSE
           inform%status = 11 ; RETURN
         END IF
       END IF

!  Allocate extra local workspace when there are constraints

       IF ( data%S%p_type == 2 .OR. data%S%p_type == 3 ) THEN
         ALLOCATE( data%GROUP_SCALING( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GROUP_SCALING' ; GO TO 980 ; END IF
         ALLOCATE( data%GXEQX_AUG( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN
           bad_alloc = 'data%GXEQX_AUG' ; GO TO 980 ; END IF
       END IF
  
!  The problem is generally constrained
  
       IF ( data%S%p_type == 3 ) THEN

!  Set initial integer values
  
         data%S%m = 0 ; data%S%nobjgr = 0
         DO ig = 1, prob%ng

!  KNDOFG = 1 correspomd to objective groups, while KNDOFG = 0
!  are groups which are to be excluded from the problem solved.
!  KNDOFG > 1 corresponds to constraint groups. More specifically,
!  KNDOFG = 2 is a general equality constraint, while KNDOFG = 3,4
!  are general equalities resulting after appending a slack variable to 
!  less-than-or-equal or greater-than-or-equal inequalities respectively


           IF ( prob%KNDOFG( ig ) >= 2 ) THEN
             IF ( prob%KNDOFG( ig ) > 4 ) THEN
               inform%status = 7
               RETURN
             ELSE
               data%S%m = data%S%m + 1
             END IF
           ELSE
             data%S%nobjgr = data%S%nobjgr + 1
           END IF
         END DO
  
!  Set initial values for the internal group scalings, GROUP_scaling, 
!  and the array, GXEQX_aug, which tells if each group is trivial
  
         IF ( prob%ng > 0 ) THEN
           WHERE ( prob%KNDOFG > 1 )
             data%GROUP_SCALING = one ; data%GXEQX_AUG = .FALSE.
           ELSEWHERE
             data%GROUP_SCALING = prob%GSCALE ; data%GXEQX_AUG = prob%GXEQX
           END WHERE
         END IF
         GXEQX_used => data%GXEQX_AUG
  
!  The problem is un-/bound-constrained
  
       ELSE IF ( data%S%p_type == 2 ) THEN
         data%S%m = prob%ng ; data%S%nobjgr = 0
         data%GROUP_SCALING = one ; data%GXEQX_AUG = .FALSE.
         GXEQX_used => data%GXEQX_AUG
  
!  The problem is un-/bound-constrained
  
       ELSE
         data%S%m = 0 ; data%S%nobjgr = prob%ng
         GXEQX_used => prob%GXEQX
       END IF
  
      IF ( data%S%printi ) WRITE( data%S%out, 2000 )

!  Print details of the objective function characteristics
  
       IF ( data%S%printi ) WRITE( data%S%out,                                 &
         "( /, ' There are ', I8, ' variables', /,                             &
      &        ' There are ', I8, ' groups', /,                                &
      &        ' There are ', I8, ' nonlinear elements ' )" )                  &
               prob%n, prob%ng, prob%nel
  
       IF ( data%S%printm ) THEN
         WRITE( data%S%out, "( /, ' ------- Group information ------ ' )" )
         IF ( data%S%printd .OR. prob%ng <= 100 ) THEN
           DO ig = 1, prob%ng
             k1 = prob%ISTADG( ig ) ; k2 = prob%ISTADG( ig + 1 ) - 1
  
!  Print details of the groups
  
             IF ( k1 <= k2 ) THEN
               IF ( k1 == k2 ) THEN
                 WRITE( data%S%out, "( /, ' Group ', I5, ' contains ', I5,     &
                &  ' nonlinear element.  This  is  element ', I5 )" )          &
                   ig, 1, prob%IELING( k1 )
               ELSE
                 WRITE( data%S%out, "( /, ' Group ', I5, ' contains ', I5,     &
                &  ' nonlinear element( s ). These are element( s )', 2I5,     &
                &  /, ( 16I5 ) )" ) ig, k2 - k1 + 1, prob%IELING( k1 : k2 )
               END IF
             ELSE
               WRITE( data%S%out, "( /, ' Group ', I5,                         &
              &  ' contains     no nonlinear', ' elements. ')" ) ig
             END IF
             IF ( .NOT. prob%GXEQX( ig ) )                                     &
               WRITE( data%S%out, "( '  * The group function is non-trivial')" )
  
!  Print details of the nonlinear elements
  
             WRITE( data%S%out,                                                &
               "( :, '  * The group has a linear element with variable( s )',  &
            &       ' X( i ), i =', 3I5, /, ( 3X, 19I5 ) )" )                  &
               prob%ICNA( prob%ISTADA( ig ) : prob%ISTADA( ig + 1 ) - 1 )
           END DO
         END IF
         IF ( .NOT. data%S%alllin ) THEN
           WRITE( data%S%out, "( /, ' ------ Element information ----- ' )" )
           IF ( data%S%printd .OR. prob%nel<= 100 ) THEN
             DO iel = 1, prob%nel
               k1 = prob%ISTAEV( iel ) ; k2 = prob%ISTAEV( iel + 1 ) - 1
  
!  Print details of the nonlinear elements
  
               IF ( k1 <= k2 ) THEN
                 WRITE( data%S%out, "( /, ' Nonlinear element', I5, ' has ',   &
                &  I4, ' internal and ', I4, ' elemental variable( s ), ', /,  &
                &  ' X( i ), i =   ', 13I5, /, ( 16I5 ) )" )                   &
                  iel, prob%INTVAR( iel ), k2 - k1 + 1, prob%IELVAR( k1 : k2 )
               ELSE
                 WRITE( data%S%out, "( /, ' Nonlinear element', I5,            &
                &  ' has   no internal',                                       &
                &  ' or       elemental variables.' )" ) iel
               END IF
             END DO
           END IF
         END IF
       END IF
  
!  Partition the workspace array FUVALS and initialize other workspace
!  arrays

       data%S%ntotel = prob%ISTADG( prob%ng  + 1 ) - 1
       data%S%nvrels = prob%ISTAEV( prob%nel + 1 ) - 1
       data%S%nnza   = prob%ISTADA( prob%ng  + 1 ) - 1

       IF ( ASSOCIATED( prob%KNDOFG ) ) THEN
       CALL INITW_initialize_workspace(                                        &
             prob%n, prob%ng, prob%nel,                                        &
             data%S%ntotel, data%S%nvrels, data%S%nnza, prob%n,                &
             data%S%nvargp, prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,&
             prob%INTVAR, prob%ISTADH, prob%ICNA, prob%ISTADA, prob%ITYPEE,    &
             GXEQX_used, prob%INTREP, data%S%altriv, data%S%direct,            &
             data%S%fdgrad, data%S%lfxi, data%S%lgxi, data%S%lhxi,             &
             data%S%lggfx, data%S%ldx, data%S%lnguvl, data%S%lnhuvl,           &
             data%S%ntotin, data%S%ntype, data%S%nsets , data%S%maxsel,        &
             RANGE, data%S%print_level, data%S%out, control%io_buffer,         &
!  workspace
             data%S%EXTEND%lwtran, data%S%EXTEND%litran,                       &
             data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,               &
             data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,              &
             data%ITRANS, data%LINK_elem_uses_var, data%WTRANS,                &
             data%ISYMMD, data%ISWKSP, data%ISTAJC, data%ISTAGV,               &
             data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR,               &
             data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,               &
             data%ISET  , data%ISVSET, data%INVSET, data%LIST_elements,        &
             data%ISYMMH, data%IW_asmbl, data%NZ_comp_w, data%W_ws,            &
             data%W_el, data%W_in, data%H_el, data%H_in,                       &
             inform%status, alloc_status, bad_alloc,                           &
             data%S%skipg, KNDOFG = prob%KNDOFG )
       ELSE
       CALL INITW_initialize_workspace(                                        &
             prob%n, prob%ng, prob%nel,                                        &
             data%S%ntotel, data%S%nvrels, data%S%nnza, prob%n,                &
             data%S%nvargp, prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,&
             prob%INTVAR, prob%ISTADH, prob%ICNA, prob%ISTADA, prob%ITYPEE,    &
             GXEQX_used, prob%INTREP, data%S%altriv, data%S%direct,            &
             data%S%fdgrad, data%S%lfxi, data%S%lgxi, data%S%lhxi,             &
             data%S%lggfx, data%S%ldx, data%S%lnguvl, data%S%lnhuvl,           &
             data%S%ntotin, data%S%ntype, data%S%nsets , data%S%maxsel,        &
             RANGE, data%S%print_level, data%S%out, control%io_buffer,         &
!  workspace
             data%S%EXTEND%lwtran, data%S%EXTEND%litran,                       &
             data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,               &
             data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,              &
             data%ITRANS, data%LINK_elem_uses_var, data%WTRANS,                &
             data%ISYMMD, data%ISWKSP, data%ISTAJC, data%ISTAGV,               &
             data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR,               &
             data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,               &
             data%ISET  , data%ISVSET, data%INVSET, data%LIST_elements,        &
             data%ISYMMH, data%IW_asmbl, data%NZ_comp_w, data%W_ws,            &
             data%W_el, data%W_in, data%H_el, data%H_in,                       &
             inform%status, alloc_status, bad_alloc,                           &
             data%S%skipg )
       END IF

       IF ( inform%status == 12 ) THEN
         inform%alloc_status = alloc_status
         inform%bad_alloc = bad_alloc
       END IF
       IF ( inform%status /= 0 ) RETURN                              
                                                                              
!  Allocate arrays                                                          
                                                                              
       FUVALS = - HUGE( one )  ! needed for epcf90 debugging compiler
       
       ALLOCATE( data%P( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%P' ; GO TO 980 ; END IF
       
       ALLOCATE( data%XCP( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%XCP' ; GO TO 980 ; END IF
       
       ALLOCATE( data%X0( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%X0' ; GO TO 980 ; END IF
       
       ALLOCATE( data%GX0( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%GX0' ; GO TO 980 ; END IF
       
       ALLOCATE( data%DELTAX( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%DELTAX' ; GO TO 980 ; END IF
       
       ALLOCATE( data%QGRAD( MAX( prob%n, data%S%ntotin ) ),                   &
         STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%QGRAD' ; GO TO 980 ; END IF
       
       ALLOCATE( data%GRJAC( data%S%nvargp ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
          bad_alloc = 'data%GRJAC' ; GO TO 980 ; END IF
       
       ALLOCATE( data%BND( prob%n, 2 ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BND' ; GO TO 980 ; END IF
       
       ALLOCATE( data%BREAKP( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BREAKP' ; GO TO 980 ; END IF

       IF ( data%S%xactcp ) THEN
         ALLOCATE( data%GRAD( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GRAD' ; GO TO 980 ; END IF
       ELSE       
         ALLOCATE( data%GRAD( prob%n ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%GRAD' ; GO TO 980 ; END IF
       END IF

       data%S%nbnd = prob%n
       IF ( data%S%mortor .AND. .NOT. data%S%twonrm ) THEN
         ALLOCATE( data%BND_radius( data%S%nbnd, 2 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%BND_radius' ; GO TO 980 ; END IF
       ELSE IF ( data%S%strctr ) THEN
         ALLOCATE( data%BND_radius( data%S%nbnd, 1 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%BND_radius' ; GO TO 980 ; END IF
       ELSE
         data%S%nbnd = 0
         ALLOCATE( data%BND_radius( data%S%nbnd, 2 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%BND_radius' ; GO TO 980 ; END IF
       END IF
       
       IF ( data%S%strctr ) THEN
!        ALLOCATE( data%D_model( prob%ng ), STAT = alloc_status )
!        IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%D_model' ; GO TO 980
!        END IF
         
!        ALLOCATE( data%D_function( prob%ng ), STAT = alloc_status )
!        IF ( alloc_status /= 0 ) THEN 
!          bad_alloc = 'data%D_function' ; GO TO 980
!        END IF
         
         ALLOCATE( data%RADII( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%RADII' ; GO TO 980
         END IF
         
         ALLOCATE( data%GV_old( prob%ng ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%GV_old' ; GO TO 980
         END IF
       
       ELSE

         ALLOCATE( data%RADII( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%RADII' ; GO TO 980
         END IF
         
         ALLOCATE( data%GV_old( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'data%GV_old' ; GO TO 980
         END IF
       
       END IF
  
!  Store the free variables as the the first nfree components of IFREE
  
       ALLOCATE( data%IFREE( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%IFREE' ; GO TO 980 ; END IF
  
!  INDEX( j ), j = 1, ..., n, will contain the status of the
!  j-th variable as the current iteration progresses. Possible values
!  are 0 if the variable lies away from its bounds, 1 and 2 if it lies
!  on its lower or upper bounds (respectively) - these may be problem
!  bounds or trust-region bounds, and 3 if the variable is fixed
  
       ALLOCATE( data%INDEX( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%INDEX' ; GO TO 980 ; END IF
  
!  IFREEC( j ), j = 1, ..., n will give the indices of the
!  variables which are considered to be free from their bounds at the
!  current generalized cauchy point
  
       ALLOCATE( data%IFREEC( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%IFREEC' ; GO TO 980 ; END IF
  
!  INNONZ( j ), j = 1, ..., nnnonz will give the indices of the nonzeros
!  in the vector obtained as a result of the matrix-vector product from
!  subroutine HSPRD
  
       ALLOCATE( data%INNONZ( prob%n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%INNONZ' ; GO TO 980 ; END IF
  
!  Make space for finite-difference values if required
  
       IF ( data%S%fdgrad .AND. .NOT. data%S%alllin ) THEN
         ALLOCATE( data%FUVALS_temp( prob%nel ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%FUVA_t' ; GO TO 980 ; END IF
       ELSE
         ALLOCATE( data%FUVALS_temp( 0 ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN 
           bad_alloc = 'data%FUVA_t' ; GO TO 980 ; END IF
       END IF

!  Space required for the Schur complement

       data%SCU_matrix%m = 0
       data%SCU_matrix%n = 1
       data%SCU_matrix%m_max = MAX( control%max_sc, 1 )
       data%SCU_matrix%class = 4
     
       ALLOCATE( data%SCU_matrix%BD_col_start( data%SCU_matrix%m_max + 1 ),    &
                 STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BD_col_st' ; GO TO 980 ; END IF
       
       ALLOCATE( data%SCU_matrix%BD_row( data%SCU_matrix%m_max ),              &
                 STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BD_row' ; GO TO 980 ; END IF
       
       ALLOCATE( data%SCU_matrix%BD_val( data%SCU_matrix%m_max ),              &
                 STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN 
         bad_alloc = 'data%BD_val' ; GO TO 980 ; END IF
      
!  Space required for the factors of the Schur complement

       data%SCU_matrix%BD_col_start( 1 ) = 1
       scu_status = 1
       CALL SCU_factorize( data%SCU_matrix, data%SCU_data, data%P, scu_status, &
                           inform%SCU_info )
       IF ( scu_status /= 0 ) THEN
         WRITE( data%S%out, "( ' SCU_factorize: status = ', I2 )" ) scu_status
         inform%status = 12
         inform%alloc_status = inform%SCU_info%alloc_status
         inform%bad_alloc = 'SCU_factorize array'
         RETURN
       END IF

     END IF

!  ===============================================
!  Call the solver to perform the bulk of the work
!  ===============================================

!  Both internal element and group evaluations will be performed
!  -------------------------------------------------------------

     IF ( internal_el .AND. internal_gr ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

        IF ( data%S%skipg ) THEN
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          END IF

!  Use all groups

        ELSE
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2,                    &
             data%G, data%IW_asmbl, data%NZ_comp_w,                            &
             data%W_ws, data%W_el,                                             &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius, data%BREAKP, data%GRAD,                &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
          END IF
        END IF
 
!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
        IF ( use_elders ) THEN
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP  = GROUP , ISTGPA = prob%ISTGPA,                            &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ENDIF
       END IF

!  Just internal element evaluations will be performed
!  ---------------------------------------------------

     ELSE IF ( internal_el ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

        IF ( data%S%skipg ) THEN

          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          END IF

!  Use all groups

        ELSE
          IF ( use_elders ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
          END IF
        END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
        IF ( use_elders ) THEN
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, ELFUN_flexible = ELFUN_flexible,                 &
             ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELFUN  = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,      &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
        END IF
       END IF

!  Just internal group evaluations will be performed
!  -------------------------------------------------

     ELSE IF ( internal_gr ) THEN

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

         IF ( data%S%skipg ) THEN
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG,                                             &
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )

!  Use all groups

         ELSE
           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
         END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y,                     &
             ELDERS = ELDERS, GROUP  = GROUP , ISTGPA = prob%ISTGPA,           &
             ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU,                       &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )
       END IF

!  Element and group evaluations will be performed via reverse communication
!  -------------------------------------------------------------------------

     ELSE

!  Unconstrained or bound-constrained minimization (old SBMIN)

       IF ( data%S%p_type == 1 ) THEN

!  Skip some groups

         IF ( data%S%skipg ) THEN

           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional argument
             ELDERS = ELDERS )

!  Use all groups

         ELSE

           CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1,data%RHS, data%RHS2, data%P2, data%G,             &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional argument
             ELDERS = ELDERS )
         END IF

!  Unconstrained or bound-constrained least-squares minimization, or
!  generally constrained minimization (old AUGLG)

       ELSE
         CALL LANCELOT_solve_main(     prob%n     , prob%ng    , prob%nel   ,  &
             lfuval     , prob%IELING, prob%ISTADG, prob%IELVAR, prob%ISTAEV,  &
             prob%INTVAR, prob%ISTADH, prob%ICNA  , prob%ISTADA, prob%A     ,  &
             prob%B     , prob%BL    , prob%BU    , prob%GSCALE, prob%ESCALE,  &
             prob%VSCALE, prob%GXEQX , prob%INTREP, RANGE      , prob%X     ,  &
             GVALS , FT , XT, FUVALS , ICALCF     , ICALCG, IVAR, Q, DGRAD  ,  &
             prob%VNAMES, prob%GNAMES, prob%ITYPEE, control, inform, data%S ,  &
!  workspace
             data%S%EXTEND%lirnh, data%S%EXTEND%ljcnh,                         &
             data%S%EXTEND%lirnh_min, data%S%EXTEND%ljcnh_min,                 &
             data%S%EXTEND%lh, data%S%EXTEND%lh_min,                           &
             data%S%EXTEND%llink, data%S%EXTEND%lpos,                          &
             data%ITRANS, data%LINK_col,                                       &
             data%POS_in_H, data%LINK_elem_uses_var, data%WTRANS,              &
             data%DIAG, data%OFFDIA, data%IW, data%IKEEP, data%IW1,            &
             data%IVUSE, data%H_col_ptr, data%L_col_ptr,                       &
             data%W, data%W1, data%RHS, data%RHS2, data%P2, data%G,            &
             data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,              &
             data%W_in, data%H_el, data%H_in, data%ISYMMD, data%ISWKSP,        &
             data%ISTAJC, data%ISTAGV, data%ISVGRP, data%ISLGRP, data%IGCOLJ,  &
             data%IVALJR, data%IUSED , data%ITYPER, data%ISSWTR, data%ISSITR,  &
             data%ISET  , data%ISVSET, data%INVSET, data%IFREE , data%INDEX ,  &
             data%IFREEC, data%INNONZ, data%LIST_elements , data%ISYMMH,       &
             data%FUVALS_temp, data%P, data%X0, data%XCP, data%GX0,            &
             data%RADII, data%DELTAX, data%QGRAD, data%GRJAC , data%GV_old,    &
             data%BND, data%BND_radius,  data%BREAKP, data%GRAD,               &
             data%SCU_matrix, data%SCU_data,                                   &
             data%matrix, data%SILS_data,                                      &
!  optional arguments
             KNDOFG = prob%KNDOFG, C = prob%C, Y = prob%Y, ELDERS = ELDERS,    &
             GROUP_SCALING = data%GROUP_SCALING, GXEQX_AUG = data%GXEQX_AUG )

       END IF
     END IF

!  De-allocate workspace

     IF ( inform%status >= 0 .AND. data%S%p_type == 3 ) NULLIFY( GXEQX_used )
     RETURN

!  Unsuccessful returns

 980 CONTINUE
     inform%status = 12
     inform%alloc_status = alloc_status
     inform%bad_alloc = bad_alloc
     WRITE( data%S%error, 2990 ) alloc_status, bad_alloc

     IF ( ASSOCIATED( data%GXEQX_AUG ) ) THEN
       DEALLOCATE( data%GXEQX_AUG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GXEQX_AUG'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ASSOCIATED( data%GROUP_SCALING ) ) THEN
       DEALLOCATE( data%GROUP_SCALING, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GROUP_SCALING'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     RETURN

!  Non-executable statements

 2000  FORMAT( /, ' *********  Starting optimization  ************** ' )
 2010  FORMAT( /, ' Penalty parameter ', ES12.4,                               &
                  ' Required projected gradient norm = ', ES12.4, /,           &
                  '                   ', 12X,                                  &
                  ' Required constraint         norm = ', ES12.4 )             
 2990  FORMAT( ' ** Message from -LANCELOT_solve-', /,                         &
               ' Allocation error (status = ', I6, ') for ', A24 )

!  End of subroutine LANCELOT_pointer_solve

     END SUBROUTINE LANCELOT_pointer_solve

!-*-*-*-  L A N C E L O T -B- LANCELOT_solve_main  S U B R O U T I N E  -*-*-*-

     SUBROUTINE LANCELOT_solve_main(                                           &
                      n, ng, nel, lfuval,                                      &
                      IELING, ISTADG, IELVAR, ISTAEV, INTVAR, ISTADH,          &
                      ICNA  , ISTADA, A , B , BL, BU, GSCALE, ESCALE, VSCALE,  &
                      GXEQX , INTREP, RANGE , X ,     GVALS , FT, XT, FUVALS,  &
                      ICALCF, ICALCG, IVAR  , Q     , DGRAD , VNAMES, GNAMES,  &
                      ITYPEE, control, inform, S    ,                          &
!  workspace
                      lirnh, ljcnh, lirnh_min, ljcnh_min,                      &
                      lh, lh_min, llink, lpos,                                 &
                      ITRANS, LINK_col      , POS_in_H,                        &
                      LINK_elem_uses_var    , WTRANS,                          &
                      DIAG  , OFFDIA, IW    , IKEEP  , IW1  ,                  &
                      IVUSE , H_col_ptr, L_col_ptr,                            &
                      W , W1, RHS   , RHS2   , P2, G,                          &
                      IW_asmbl, NZ_comp_w, W_ws, W_el, W_in, H_el, H_in,       &
                      ISYMMD, ISWKSP, ISTAJC, ISTAGV, ISVGRP, ISLGRP,          &
                      IGCOLJ, IVALJR, IUSED , ITYPER, ISSWTR, ISSITR,          &
                      ISET  , ISVSET, INVSET, IFREE , INDEX , IFREEC,          &
                      INNONZ, LIST_elements , ISYMMH, FUVALS_temp   ,          &
                      P , X0, XCP   , GX0   , RADII , DELTAX, QGRAD ,          &
                      GRJAC , GV_old, BND   , BND_radius    ,                  &
                      BREAKP, GRAD, SCU_matrix, SCU_data,                      &
                      matrix, SILS_data,                                       &
!  optional arguments
                      KNDOFG, C, Y  ,                                          &
                      ELDERS, ELFUN_flexible, ELFUN , ISTEPA , EPVALU,         &
                      GROUP , ISTGPA, ITYPEG, GPVALU, GROUP_SCALING, GXEQX_AUG )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  LANCELOT_solve, a method for finding a local minimizer of a
!  function subject to general constraints and simple bounds on the sizes of
!  the variables. The method is described in the paper 'A globally convergent
!  augmented Lagrangian algorithm for optimization with general constraints
!  and simple bounds' by A. R. Conn, N. I. M. Gould and Ph. L. Toint,
!  SIAM J. Num. Anal. 28 (1991) PP.545-572
!
!  The objective function is assumed to be of the form
!
!                             ISTADG(obj+1)-1
!      F( X ) = SUM GS   * G(   SUM    ES * F( X ) + A (TRANS) X - B )
!               obj   obj   obj  j=      j   j        m+1           m+1
!            IN OBJSET        ISTADG(obj)
!
!  and the constraints (i = 1, .... , ng, i .NE. objset) of the form
!
!                        ISTADG(i+1)-1
!      CI( X ) = GS * G(   SUM     ES * F ( X ) + A (TRANS) X - B ) = 0
!                  i   i j=ISTADG(i) j   j         i             i
!
!  Where the Fj( X ) are known as nonlinear element functions, the
!  Ai(trans) X + Bi are the linear element functions, the GSi are group
!  weights and the ESi are element weights. Each Fj is expected only to
!  involve a few 'internal' variables, that is there is a linear
!  transformation from the problem variables to the variables actually needed
!  to define the element (and its derivatives) whose range space is very small

!  Contents of the array FUVALS:
!  -----------------------------

!     <-nel-><-- ntotin --> <-- nhel --> <---- n ---> <-

!    ---------------------------------------------------------
!    |  Fj(X)  |  grad Fj(X)  | Hess Fj(X) | grad F(X)) | ....
!    ---------------------------------------------------------
!   |         |              |            |            |
!  lfxi     lgxi           lhxi         lggfx         ldx
!  (=0)
!        -> <------- n ------> <-- nvargp ->

!       ------------------------
!      ... | Diag scaling F(X) |
!       ------------------------
!         |                   | 
!        ldx                lend

!  Only the upper triangular part of each element Hessian is stored;
!  the storage is by columns

!  Contents of the arrays ISTADG, ESCALE AND IELING:
!  ------------------------------------------------

!           <---------------------- S%ntotel ------------------------>

!           --------------------------------------------------------
!  ESCALE:  | el.scales | el.scales | ............... | el.scales  |
!           | group 1   | group 2   |                 | group ng   |
!           --------------------------------------------------------
!  IELING:  | elements  | elements  | ............... | elements   |
!           | group 1   | group 2   |                 | group ng   |
!           --------------------------------------------------------
!            |           |                             |            |
!            | |--- > ---|                             |            |
!            | |   |-------------------- > ------------|            |
!            | |   | |------------------------- > ------------------|
!            ---------
!  ISTADG:   | ..... |    pointer to the position of the 1st element
!            ---------    of each group within the array
!            <-ng+1->

!  Contents of the arrays IELVAR and ISTAEV:
!  ----------------------------------------

!          <--------------------- nelvar -------------------------->

!          ---------------------------------------------------------
!          | variables | variables | ............... |  variables  |
!  IELVAR: | element 1 | element 2 |                 | element nel |
!          ---------------------------------------------------------
!           |           |                             |             |
!           | |--- > ---|                             |             |
!           | |    |------------------- > ------------|             |
!           | |    | |----------------- > --------------------------|
!           ----------
!  ISTAEV:  | ...... |    pointer to the position of the 1st variable
!           ----------    in each element (including one to the end).
!          <- nel+1 ->

!  Contents of the array INTVAR:
!  -----------------------------

!  On initial entry, INTVAR( i ), i = 1, ... , nel, gives the number of
!  internal variables for element i. Thereafter, INTVAR provides pointers to
!  the start of each element gradient with respect to its internal variables
!  as follows:

!         -> <---------------------- ntotin -----------------------> <-

!         -------------------------------------------------------------
!  part of  | gradient  | gradient  | ............... |  gradient   | .
!  FUVALS   | element 1 | element 2 |                 | element nel |
!         -------------------------------------------------------------
!          | |           |                             |           | |
!       lgxi | |--- > ---|                             |         lhxi|
!            | |   |-------------------- > ------------|             |
!            | |   | |------------------------- > -------------------|
!            ---------
!  INTVAR:   | ..... |    pointer to the position of the 1st entry of
!            ---------    the gradient for each element
!            <-nel+1->

!  Contents of the array ISTADH:
!  -----------------------------

!         -> <---------------------- nhel -------------------------> <-

!         -------------------------------------------------------------
!  part of  | Hessian   | Hessian   | ............... |  Hessian    | .
!  FUVALS   | element 1 | element 2 |                 | element nel |
!         -------------------------------------------------------------
!          | |           |                             |           |
!       lhxi | |--- > ---|                             |         lggfx
!            | |    |------------------- > ------------|
!            | |    |
!            ---------
!  ISTADH:   | ..... |    pointer to the position of the 1st entry of the
!            ---------    Hessian for each element, with respect to its
!                         internal variables
!            <- nel ->

!  Contents of the arrays A, ICNA AND ISTADA:
!  ------------------------------------------

!          <--------------------- na ----------------------------->

!          ---------------------------------------------------------
!          |   values  |   values  | ............... |    values   |
!  A:      |    A(1)   |    A(2)   |                 |     A(ng)   |
!          ---------------------------------------------------------
!          | variables | variables | ............... |  variables  |
!  ICNA:   |    A(1)   |    A(2)   |                 |     A(ng)   |
!          ---------------------------------------------------------
!           |           |                             |             |
!           | |--- > ---|                             |             |
!           | |    |------------------- > ------------|             |
!           | |    | |----------------- > --------------------------|
!           ----------
!  ISTADA:  | ...... |    pointer to the position of the 1st variable in the
!           ----------    linear element for each group (including one to the
!                         end)

!           <- ng+1 ->

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  If the routine terminates with a negative value of inform%status, the user is
!  asked to re-enter the subroutine with further information

!  If status  = - 1, the user must supply the function and derivative values of
!                    each Fj at the point XT.
!  If status  = - 2, the user must supply the function and derivative of each
!                    Gi for the argument FT(i) .
!  If status  = - 3, the user must supply the value, alone, of each function
!                    Fj evaluated at the point XT.
!  If status  = - 4, the user must supply the value of each function Gi, alone,
!                    for the argument FT(i).
!  If status  = - 5, the user must supply the derivatives, alone, of the
!                    functions Fj and Gi at the point XT and argument FT(i)
!                    respectively.
!  If status  = - 6, the user must supply the derivatives, alone, of the
!                    functions Fj at the point XT.
!  If status  = - 7, the user must supply the value, alone, of each function
!                    Fj evaluated at the point XT.
!  If status  = - 8, 9, 10, the user must provide the product of the inverse
!                    of the preconditioning matrix and the vector GRAD. The
!                    nonzero components of GRAD occur in positions IVAR(i),
!                    i = 1,..,nvar and have the values DGRAD(i). The product
!                    must be returned in the vector Q. This return is only
!                    possible if ICHOSE( 2 ) is 3.
!  If status  = - 12, the user must supply the derivatives, alone, of the
!                    functions Fj at the point XT.
!  If status  = - 13, the user must supply the derivative valuse of each 
!                     function Gi, alone, for the argument FT(i).

!  If the user does not wish to compute an element or group function at
!  a particular argument returned from the inner iteration, she may
!  reset status to -11 and re-enter. The routine will treat such a re-entry as
!  if the current iteration had been unsuccessful and reduce the trust region.
!  This facility is useful when, for instance, the user is asked to evaluate
!  a function at a point outside its domain of definition.

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE( LANCELOT_control_type ), INTENT( INOUT ) :: control
     TYPE( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
     TYPE( LANCELOT_save_type ), INTENT( INOUT ) :: S
     INTEGER, INTENT( IN ) :: n, ng, nel, lfuval
     INTEGER, INTENT( IN ), DIMENSION( ng + 1 ) :: ISTADA
     INTEGER, INTENT( IN ), DIMENSION( ISTADA( ng + 1 ) - 1 ) :: ICNA
     INTEGER, INTENT( INOUT ), DIMENSION( ng + 1 ) :: ISTADG
     INTEGER, INTENT( IN ), DIMENSION( ISTADG( ng + 1 ) - 1 ) :: IELING
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( ISTAEV( nel + 1 ) - 1 ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTADH, INTVAR
     INTEGER, INTENT( INOUT ), DIMENSION( n  ) :: IVAR
     INTEGER, INTENT( INOUT ), DIMENSION( nel ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( ng ) :: ICALCG
     REAL ( KIND = wp ), INTENT( IN  ),                                        &
                               DIMENSION( ISTADA( ng + 1 ) - 1 ) :: A
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: BL, BU
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: B
     REAL ( KIND = wp ), INTENT( IN  ),                                        &
            DIMENSION( ISTADG( ng + 1 ) - 1 ) :: ESCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng, 3 ) :: GVALS 
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
            DIMENSION( n ) :: X, Q, XT, DGRAD, VSCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng ) :: FT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     REAL ( KIND = wp ), INTENT( IN  ), TARGET, DIMENSION( ng ) :: GSCALE
     LOGICAL, INTENT( IN ), TARGET, DIMENSION( ng ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     CHARACTER ( LEN = 10 ), INTENT( IN ), DIMENSION( n ) :: VNAMES
     CHARACTER ( LEN = 10 ), INTENT( IN ), DIMENSION( ng ) :: GNAMES

!--------------------------------------------------------------
!   D u m m y   A r g u m e n t s  f o r   W o r k s p a c e 
!--------------------------------------------------------------

     INTEGER, INTENT( INOUT ) :: lirnh, ljcnh, lirnh_min, ljcnh_min
     INTEGER, INTENT( INOUT ) :: lh, lh_min, llink, lpos
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITRANS
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: LINK_col, POS_in_H
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: LINK_elem_uses_var
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WTRANS
 
     INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: IKEEP, IW1
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW, IVUSE
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col_ptr, L_col_ptr
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) ::                            &
       W, RHS, RHS2, P2, G , DIAG
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: W1, OFFDIA
     
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISYMMD, ISWKSP, ISTAJC
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISTAGV, ISVGRP, ISLGRP
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IGCOLJ, IVALJR, IUSED 
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ITYPER, ISSWTR, ISSITR
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISET  , ISVSET, INVSET
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IFREE , INDEX , IFREEC
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: INNONZ, LIST_elements
     INTEGER, INTENT( INOUT ), DIMENSION( : , : ) :: ISYMMH
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: FUVALS_temp
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: P, X0
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: XCP
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: GX0, RADII
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: DELTAX
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: QGRAD, GRJAC
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: GV_old
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : , : ) :: BND, BND_radius
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: BREAKP, GRAD
     
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IW_asmbl
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: NZ_comp_w
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W_ws
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: H_el
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: H_in

     TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_matrix
     TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
     TYPE ( SMT_type ), INTENT( INOUT ) :: matrix
     TYPE ( SILS_factors ), INTENT( INOUT ) :: SILS_data

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

!  Interface block for RANGE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                          lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN 

!  Interface block for ELFUN_flexible 

       SUBROUTINE ELFUN_flexible(                                              &
                          FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, llders, ifflag, ELDERS,      &
                          ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( IN ) :: llders
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(ltypee), ISTAEV(lstaev), IELVAR(lelvar)
       INTEGER, INTENT( IN ) :: INTVAR(lntvar), ISTADH(lstadh), ISTEPA(lstepa)
       INTEGER, INTENT( IN ) :: ICALCF(lcalcf), ELDERS(2,llders)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(lxvalu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(lepvlu)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(lfuval)
       END SUBROUTINE ELFUN_flexible

!  Interface block for GROUP

       SUBROUTINE GROUP ( GVALUE, lgvalu, FVALUE, GPVALU, ncalcg,              &
                          ITYPEG, ISTGPA, ICALCG, ltypeg, lstgpa,              &
                          lcalcg, lfvalu, lgpvlu, derivs, igstat )
       INTEGER, INTENT( IN ) :: lgvalu, ncalcg
       INTEGER, INTENT( IN ) :: ltypeg, lstgpa, lcalcg, lfvalu, lgpvlu
       INTEGER, INTENT( OUT ) :: igstat
       LOGICAL, INTENT( IN ) :: derivs
       INTEGER, INTENT( IN ), DIMENSION ( ltypeg ) :: ITYPEG
       INTEGER, INTENT( IN ), DIMENSION ( lstgpa ) :: ISTGPA
       INTEGER, INTENT( IN ), DIMENSION ( lcalcg ) :: ICALCG
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lfvalu ) :: FVALUE
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ),                           &
                                       DIMENSION ( lgpvlu ) :: GPVALU
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ),                        &
                                       DIMENSION ( lgvalu, 3 ) :: GVALUE
       END SUBROUTINE GROUP

     END INTERFACE

!-----------------------------------------------------
!   O p t i o n a l   D u m m y   A r g u m e n t s
!-----------------------------------------------------

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( nel + 1 ) :: ISTEPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng + 1 ) :: ISTGPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: ITYPEG
     INTEGER, INTENT( INOUT ), OPTIONAL, DIMENSION( 2, nel ) :: ELDERS
     REAL ( KIND = wp ), INTENT( INOUT ), OPTIONAL,                            &
            DIMENSION( ng ) :: C, Y
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL,                               &
            DIMENSION( : ) :: EPVALU
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL,                               &
            DIMENSION( : ) :: GPVALU
     REAL ( KIND = wp ), INTENT( IN  ), TARGET, OPTIONAL,                      &
            DIMENSION( ng ) :: GROUP_SCALING
     LOGICAL, INTENT( IN ), TARGET, OPTIONAL, DIMENSION( ng ) :: GXEQX_AUG
     OPTIONAL :: ELFUN, ELFUN_flexible, GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, ig, ic, is, j, lgfx, ifixd, k, k1, k2, l, ifflag
     INTEGER :: ipdgen, iddgen, istate, ir, nvar1, alloc_status
     REAL ( KIND = KIND( 1.0E0 ) ) :: tim
     REAL ( KIND = wp ) :: hmuinv, yiui  , scaleg
     REAL ( KIND = wp ) :: epsmch, epslam, hdash , ctt
     REAL ( KIND = wp ) :: ftt, xi, gi, bli, bui, dltnrm
     REAL ( KIND = wp ) :: gnorm, distan, ar_h, pr_h, slope
     LOGICAL :: external_el, external_gr, start_p, alive, use_elders
     CHARACTER ( LEN = 7 ) :: atime
     CHARACTER ( LEN = 6 ) :: citer, cngevl, citcg
     CHARACTER ( LEN = 24 ) :: bad_alloc

!---------------------------------
!   L o c a l   P o i n t e r s
!---------------------------------

     REAL ( KIND = wp ), POINTER, DIMENSION( : ) :: GSCALE_used
     LOGICAL, POINTER, DIMENSION( : ) :: GXEQX_used

     epsmch = EPSILON( one )
     external_el = .NOT. ( PRESENT( ELFUN ) .OR. PRESENT( ELFUN_flexible ) )
     external_gr = .NOT. PRESENT( GROUP )
     use_elders = PRESENT( ELDERS )

!  If the run is being continued after the "alive" file has been reinstated
!  jump to the appropriate place in the code

     IF ( inform%status == 14 ) THEN
       IF ( S%inform_status < 0 ) THEN
         inform%status = S%inform_status
         GO TO 700
       ELSE
         inform%status = - S%inform_status
         GO TO 800
       END IF
     END IF

!  Branch to the interior of the code if a re-entry is being made

     IF ( inform%status < 0 ) GO TO 810

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  10 CONTINUE

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!                      I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

  20   CONTINUE

!  Start inner loop to minimize penalty function for the current values of
!  mu and Y

         IF ( S%p_type == 1 ) THEN
           GXEQX_used => GXEQX ; GSCALE_used => GSCALE
         ELSE
           GXEQX_used => GXEQX_AUG ; GSCALE_used => GROUP_SCALING
         END IF

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  The basic algorithm used for the inner iteration is described in the paper 
!  'Testing a class of methods for solving minimization problems with 
!  simple bounds on their variables' by A. R. Conn, N. I. M. Gould and 
!  Ph. L. Toint, Mathematics of Computation, 50 (1988) pp. 399-430

!  The objective function is assumed to be of the form

!               ng
!     F( X ) = sum  GS * G(   sum     ES  * F ( X ) + A (trans) X - B )
!              i=1    i   i j in J(i)   ij   j         i             i

!  Where the F(j)( X ) are known as nonlinear element functions, the
!  A(i)(trans) X - Bi are the linear element functions, the GS(i) are group
!  weights, the ES(ij) are element weights, the G(i) are called group
!  functions and each set J(i) is a subset of the set of the first nel
!  integers. Each F(j) is expected only to involve a few 'internal' variables,
!  that is there is a linear transformation from the problem variables to the
!  variables actually needed to define the element (and its derivatives) whose
!  range space is very small. Each group function is a differentiable
!  univariate function

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  Branch to different parts of the code depending on the input value of status

         SELECT CASE ( inform%status )
         CASE ( -  1 ) ; GO TO 40
         CASE ( -  2 ) ; GO TO 70
         CASE ( -  3 ) ; GO TO 430
         CASE ( -  4 ) ; GO TO 470
         CASE ( - 13 : - 12, - 7 : - 5 ) ; GO TO 540
         CASE ( -  8 ) ; GO TO 90
         CASE ( -  9 ) ; GO TO 320
         CASE ( - 10 ) ; GO TO 570
         CASE ( - 11 ) ; GO TO 590
         END SELECT

!  -----------------------------------
!  Step 0 of the algorithm (see paper)
!  -----------------------------------

         DO i = 1, n

!  Project user supplied starting point into the feasible box

           bli = BL( i )
           bui = BU( i )
           IF ( bli > bui ) THEN
             IF ( S%printe ) WRITE( S%out,                                     &
               "( /, ' Lower bound ', ES12.4, ' on variable ', I8,             &
            &        ' larger than upper bound ', ES12.4, //,                  &
            &        ' Execution terminating ' )" ) bli, i, bui
             inform%status = 8 ; GO TO 820
           END IF
           xi = MAX( bli, MIN( bui, X( i ) ) )

!  Find the maximum variable scale factor

           S%vscmax = MAX( S%vscmax, VSCALE( i ) )

!  Find initial active set

           is = 0
           IF ( xi <= bli * ( one + SIGN( S%epstlp, bli ) ) ) is = 1
           IF ( xi >= bui * ( one - SIGN( S%epstln, bui ) ) ) is = 2
           IF ( bui * ( one - SIGN( S%epstln, bui ) ) <=                       &
                bli * ( one + SIGN( S%epstlp, bli ) ) ) is = 3
           INDEX( i ) = is
           IF ( is == 3 ) xi = half * ( bli + bui )

!  Copy the initial point into XT prior to calculating function values

           X( i ) = xi ; XT( i ) = xi
         END DO

!  Ensure that all the element functions are evaluated at the initial point

         inform%ncalcf = nel
         DO i = 1, inform%ncalcf ; ICALCF( i ) = i ; END DO

!  Return to the calling program to obtain the element function
!  and, if possible, derivative values at the initial point

         IF ( S%fdgrad ) S%igetfd = 0
         inform%ngeval = inform%ngeval + 1
         inform%status = - 1
         IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF

!  If finite-difference gradients are used, compute their values

   40    CONTINUE
         IF ( S%fdgrad .AND. .NOT. S%alllin ) THEN

!  Store the values of the nonlinear elements for future use

           IF ( S%igetfd == 0 ) THEN
             FUVALS_temp( : nel ) = FUVALS( : nel )
             S%centrl = S%first_derivatives == 2
           END IF

!  Obtain a further set of differences

           IF ( use_elders ) THEN
             CALL OTHERS_fdgrad_flexible(                                      &
                                 n, nel, lfuval, S%ntotel, S%nvrels, S%nsets,  &
                                 IELVAR, ISTAEV, IELING, ICALCF, inform%ncalcf,&
                                 INTVAR, S%ntype , X , XT, FUVALS, S%centrl,   &
                                 S%igetfd, S%OTHERS, ISVSET, ISET, INVSET,     &
                                 ISSWTR, ISSITR, ITYPER, LIST_elements,        &
                                 LINK_elem_uses_var, WTRANS, ITRANS,           &
                                 ELDERS( 1, : ) )
           ELSE
             CALL OTHERS_fdgrad( n, nel, lfuval, S%ntotel, S%nvrels, S%nsets,  &
                                 IELVAR, ISTAEV, IELING, ICALCF, inform%ncalcf,&
                                 INTVAR, S%ntype , X , XT, FUVALS, S%centrl,   &
                                 S%igetfd, S%OTHERS, ISVSET, ISET, INVSET,     &
                                 ISSWTR, ISSITR, ITYPER, LIST_elements,        &
                                 LINK_elem_uses_var, WTRANS, ITRANS )
           END IF
           IF ( S%igetfd > 0 ) THEN
             IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
           END IF

!  Restore the values of the nonlinear elements at X

           S%igetfd = S%nsets + 1
           FUVALS( : nel ) = FUVALS_temp( : nel )
         END IF

!  The convergence tolerance is modified to reflect the scaling

         S%epscns = S%omegak * S%vscmax

!  Compute the norm of the residual that is to be required when obtaining the
!  approximate minimizer of the model problem

         S%resmin = MIN( tenm4, MAX( S%epsrcg, S%epscns ** 2.02 ) )

!  Compute the group argument values FT

         DO ig = 1, ng

!  Include the contribution from the linear element

!          ftt = SUM( A( ISTADA( ig ): ISTADA( ig + 1 ) - 1 ) *                &
!            X( ICNA( ISTADA( ig ) : ISTADA( ig + 1 ) - 1 ) ) ) - B( ig )
           ftt = - B( ig )
           DO i = ISTADA( ig ), ISTADA( ig + 1 ) - 1
             ftt = ftt + A( i ) * X( ICNA( i ) )
           END DO

!  Include the contributions from the nonlinear elements

!          ftt = ftt + SUM( ESCALE( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) *    &
!            FUVALS( IELING( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) ) )
           DO i = ISTADG( ig ), ISTADG( ig + 1 ) - 1
             ftt = ftt + ESCALE( i ) * FUVALS( IELING( i ) ) ; END DO
           FT( ig ) = ftt
         END DO

!  Compute the group function values

         IF ( S%altriv ) THEN
!          inform%aug = DOT_PRODUCT( GSCALE_used, FT )
           inform%aug = zero
           IF ( S%skipg ) THEN
             DO ig = 1, ng
               IF ( KNDOFG( ig ) > 0 ) THEN
                 inform%aug = inform%aug + GSCALE_used( ig ) * FT( ig )
                 GVALS( ig, 1 ) = FT( ig )
                 GVALS( ig, 2 ) = one
               END IF
             END DO
           ELSE
             DO ig = 1, ng
               inform%aug = inform%aug + GSCALE_used( ig ) * FT( ig ) ; END DO
             GVALS( : , 1 ) = FT
             GVALS( : , 2 ) = one
           END IF
         ELSE

!  If necessary, return to the calling program to obtain the group function
!  and derivative values at the initial point. Ensure that all the group
!  functions are evaluated at the initial point

           inform%ncalcg = ng
           DO ig = 1, ng
             ICALCG( ig ) = ig
             IF ( GXEQX_used( ig ) ) THEN
               GVALS( ig, 1 ) = FT( ig )
               GVALS( ig, 2 ) = one
             END IF
           END DO
           inform%status = - 2
           IF ( external_gr ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
         END IF

   70    CONTINUE
         S%reusec = .FALSE.
!        IF ( .NOT. S%altriv )                                                 &
!          inform%aug = DOT_PRODUCT( GSCALE_used, GVALS( : , 1 ) )
         IF ( .NOT. S%altriv ) THEN
           inform%aug = zero
           IF ( S%skipg ) THEN
             DO ig = 1, ng
!              write(6,*)  ' kndofg ', KNDOFG( ig ), GSCALE_used( ig ),        &
!                           GVALS( ig, 1 )
               IF ( KNDOFG( ig ) > 0 )                                         &
                inform%aug = inform%aug + GSCALE_used( ig ) * GVALS( ig, 1 )
             END DO
           ELSE
             DO ig = 1, ng 
               inform%aug = inform%aug + GSCALE_used( ig ) * GVALS( ig, 1 )
             END DO
           END IF
         END IF

!  If a structured trust-region is to be used, store the current values
!  of the group functions

         IF ( S%strctr ) THEN
           DO ig = 1, ng
             IF ( GXEQX_used( ig ) ) THEN
               GV_old( ig ) = FT( ig )
             ELSE
               GV_old( ig ) = GVALS( ig, 1 )
             END IF
           END DO
         END IF

!  If a secant method is to be used, initialize the second
!  derivatives of each element as a scaled identity matrix

         CALL CPU_TIME( S%t )
         IF ( .NOT. S%second .AND. .NOT. S%alllin ) THEN
           IF ( use_elders ) THEN
             CALL OTHERS_scaleh_flexible(                                      &
                 .TRUE., n, nel, lfuval, S%nvrels, S%ntotin,                   &
                 inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR, IELVAR,        &
                 ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el, H_in,         &
                 ELDERS( 2, : ), RANGE )
           ELSE
             CALL OTHERS_scaleh( .TRUE., n, nel, lfuval, S%nvrels, S%ntotin,   &
                 inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR, IELVAR,        &
                 ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el, H_in, RANGE )
           END IF
         END IF

!  If a two-norm trust region is to be used, initialize the vector P

         IF ( S%twonrm ) P = zero
         CALL CPU_TIME( tim )
         S%tup = S%tup + tim - S%t

!  Compute the gradient value

         CALL LANCELOT_form_gradients(                                         &
             n, ng, nel, S%ntotel, S%nvrels, S%nnza,                           &
             S%nvargp, .TRUE., ICNA, ISTADA, IELING, ISTADG, ISTAEV,           &
             IELVAR, INTVAR, A, GVALS( : , 2 ), FUVALS( : S%lnguvl ),          &
             S%lnguvl, FUVALS( S%lggfx  + 1 : S%lggfx + n ),                   &
             GSCALE_used, ESCALE, GRJAC, GXEQX_used, INTREP, ISVGRP, ISTAGV,   &
             ITYPEE, ISTAJC, W_ws, W_el, RANGE, KNDOFG )

!  Find the initial projected gradient and its norm

         CALL LANCELOT_projected_gradient(                                     &
             n, X, FUVALS( S%lggfx + 1 : S%lggfx + n ),            &
             VSCALE, BL, BU, DGRAD, IVAR, inform%nvar, inform%pjgnrm )
         S%nfree = inform%nvar

!  Find the norm of the projected gradient

         IF ( S%prcond .AND. inform%nvar > 0 .AND. S%myprec ) THEN

!  Use the users preconditioner

           inform%status = - 8 ; GO TO 800
         END IF

   90    CONTINUE

!  Find the norm of the 'preconditioned' projected gradient. Also, find the
!  diagonal elements of the assembled Hessian as scalings, if required

         CALL LANCELOT_norm_proj_grad(                                         &
             n , ng, nel, S%ntotel, S%nvrels, S%nvargp,      &
             inform%nvar, S%smallh, inform%pjgnrm, S%calcdi, S%dprcnd,         &
             S%myprec, IVAR(:inform%nvar ), ISTADH, ISTAEV, IELVAR, INTVAR,    &
             IELING, DGRAD( : inform%nvar ), Q, GVALS( : , 2 ), GVALS( : , 3 ),&
             FUVALS( S%ldx + 1 : S%ldx + n ), GSCALE_used, ESCALE,       &
             GRJAC, FUVALS( : S%lnhuvl ), S%lnhuvl, S%qgnorm, GXEQX_used,      &
             INTREP, ISYMMD, ISYMMH, ISTAGV, ISLGRP, ISVGRP, IVALJR, ITYPEE,   &
             W_el, W_in, H_in, RANGE, KNDOFG )

!  If a non-monotone method is to be used, initialize counters

         IF ( S%nmhist > 0 ) THEN
           S%l_suc = 0
           S%f_min = inform%aug ; S%f_r = S%f_min ; S%f_c = S%f_min
           S%sigma_r = zero ; S%sigma_c = zero
         END IF

!  Set initial trust-region radius

         S%print_header = .TRUE.
         inform%radius = control%initial_radius
         IF ( inform%radius > zero ) THEN
           S%oldrad = inform%radius
           IF ( S%strctr ) RADII = inform%radius
         ELSE

!  An unsophisticated method is to be used. Ensure that the initial Cauchy 
!  step is of order unity
       
!          gnorm = SQRT( SUM( FUVALS( S%lggfx + 1 : S%lggfx + n ) ** 2 ) )
           gnorm = zero
           DO i = 1, n ; gnorm = gnorm + FUVALS( S%lggfx + i ) ** 2 ; END DO
           gnorm = SQRT( gnorm )
           inform%radius = MIN( S%maximum_radius, point1 * gnorm )
           S%oldrad = inform%radius
           IF ( S%strctr ) RADII = inform%radius
         END IF 

!  ------------------------------------------------
!  Main iteration loop of the algorithm (see paper)
!  ------------------------------------------------

  120    CONTINUE

!  If required, print one line of details of the current iteration

           IF ( S%out > 0 .AND.                                                &
                ( S%print_level == 1 .OR. S%print_level == 2 ) ) THEN

!  If needed, print the iteration header

             IF ( S%print_header .OR. S%print_level == 2 ) THEN
               IF ( S%direct ) THEN
                 WRITE( S%out, "( /, '  Iter #g.ev fill-in     f    proj.g ',  &
                &  '   rho    radius   step   lsend #free   time' )" )
               ELSE
                 WRITE( S%out, "( /, '  Iter #g.ev c.g.it      f    proj.g',   &
                &  '    rho   radius   step   cgend #free   time' )" )
               END IF
             END IF

!  Print the iteration details

             CALL CPU_TIME( tim )
             atime = OTHERS_time( tim - S%time )
             citer = OTHERS_iter( inform%iter )
             cngevl = OTHERS_iter( inform%ngeval )
             IF ( S%direct ) THEN
               IF ( S%print_header ) THEN
                 WRITE( S%out, "( 2A6, 6X, ES10.2, ES8.1, '     -       -  ',  &
                &  '     -      -  ', I6, A7 )" ) citer, cngevl,               &
                     inform%aug * S%findmx, inform%pjgnrm, S%nfree, atime
               ELSE IF ( inform%status == - 11 ) THEN
                 WRITE( S%out, "( 2A6, F6.1, ES10.2, ES8.1, '     -   ',       &
                &  2ES8.1, A6, I6, A7 )" ) citer, cngevl, S%fill,              &
                     inform%aug * S%findmx, inform%pjgnrm, S%oldrad,           &
                     S%step, S%lisend, S%nfree, atime
               ELSE
                 WRITE( S%out,                                                 &
                   "( 2A6, F6.1, ES10.2, ES8.1, ES9.1, 2ES8.1, A6, I6, A7 )" ) &
                     citer, cngevl, S%fill,                                    &
                     inform%aug * S%findmx, inform%pjgnrm, S%rho,              &
                     S%oldrad, S%step, S%lisend, S%nfree, atime
               END IF
             ELSE
               citcg = OTHERS_iter( inform%itercg )
               IF ( S%print_header ) THEN
                 WRITE( S%out, "( 3A6, ES10.2, ES8.1, '     -       -  ',      &
                &  '     -      -  ', I6, A7 )" ) citer, cngevl,               &
                     citcg, inform%aug * S%findmx, inform%pjgnrm, S%nfree, atime
               ELSE IF ( inform%status == - 11 ) THEN
                 WRITE( S%out, "( 3A6, ES10.2, ES8.1, '     -   ', 2ES8.1,     &
                &  A6, I6, A7 )" ) citer, cngevl, citcg,                       &
                     inform%aug * S%findmx, inform%pjgnrm, S%oldrad,           &
                     S%step, S%cgend, S%nfree, atime
               ELSE
                 WRITE( S%out,                                                 &
                   "( 3A6, ES10.2, ES8.1, ES9.1, 2ES8.1, A6, I6, A7 )" )       &
                     citer, cngevl, citcg,                                     &
                     inform%aug * S%findmx, inform%pjgnrm, S%rho,              &
                     S%oldrad, S%step, S%cgend, S%nfree, atime
               END IF
             END IF
             S%print_header = .FALSE.
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, ' Required gradient accuracy ', ES8.1 )" ) S%epscns
           END IF

!  -----------------------------------
!  Step 1 of the algorithm (see paper)
!  -----------------------------------

!  If required, print more thorough details of the current iteration

           IF ( S%printm ) THEN
             IF ( inform%iter == 0 ) THEN
               WRITE( S%out, 2570 ) inform%iter, inform%aug *  S%findmx,       &
                 inform%ngeval, inform%pjgnrm, inform%itercg, inform%iskip
             ELSE
               WRITE( S%out, 2550 ) inform%iter, inform%aug *  S%findmx,       &
                 inform%ngeval, inform%pjgnrm, inform%itercg, S%oldrad,        &
                 inform%iskip
             END IF
             WRITE( S%out, 2500 ) X
             IF ( S%printw ) THEN
               WRITE( S%out, 2510 )                                            &
                 FUVALS( S%lggfx + 1 : S%lggfx + n ) * S%findmx
               IF ( S%print_level >= 5 ) THEN
                 WRITE( S%out, "( /, ' Element values ', / (  6ES12.4 ) )" )   &
                   FUVALS( : nel )
                 WRITE( S%out, "( /, ' Group values ', / (  6ES12.4 ) )" )     &
                   GVALS( : , 1 )
                 IF ( S%calcdi ) WRITE( S%out,                                 &
                   "( /, ' Diagonals of second derivatives', / ( 6ES12.4 ) )") &
                   FUVALS( S%ldx + 1 : S%ldx + n ) * S%findmx
                 IF ( S%print_level >= 6 ) THEN
                   WRITE( S%out, "( :, /, ' Element gradients ', /             &
                  &  ( 6ES12.4 ) )" ) FUVALS( S%lgxi + 1 : S%lhxi )
                   WRITE( S%out, "( /, ' Group derivatives ', /                &
                  &  ( 6ES12.4 ) )" ) GVALS( : , 2 )
                   WRITE( S%out,                                               &
                     "( :, /, ' Element hessians ', / (  6ES12.4 ) )" )        &
                     FUVALS( S%lhxi + 1 : S%lggfx )
                  END IF
               END IF
             END IF
           END IF

!  Test for convergence

           IF ( inform%pjgnrm <= S%epscns ) THEN
             inform%status = 0
             GO TO 600
           END IF

           IF ( inform%aug < control%min_aug ) THEN
             inform%status = 18
             GO TO 600
           END IF

!  Test whether the maximum allowed number of iterations has been reached

           IF ( inform%iter >= control%maxit ) THEN
             IF ( S%printe ) WRITE( S%out,                                     &
               "( /, ' LANCELOT_solve : maximum number of iterations reached')")

             inform%status = 1 ; GO TO 600
           END IF

!  Test whether the trust-region radius is too small for progress

           IF ( inform%radius < S%radtol ) THEN
             IF ( S%printe ) WRITE( S%out, 2540 )
             inform%status = 2 ; GO TO 600
           END IF
           inform%iter = inform%iter + 1

!  Check that the print status remains unchanged

           IF ( ( inform%iter >= S%start_print .AND.                           &
                  inform%iter < S%stop_print ) .AND.                           &
                MOD( inform%iter - S%start_print, S%print_gap ) == 0 ) THEN
             S%printe = S%set_printe
             S%printi = S%set_printi
             S%printt = S%set_printt
             S%printm = S%set_printm
             S%printw = S%set_printw
             S%printd = S%set_printd
             S%print_level = control%print_level
           ELSE
             S%printe = .FALSE.
             S%printi = .FALSE.
             S%printt = .FALSE.
             S%printm = .FALSE.
             S%printw = .FALSE.
             S%printd = .FALSE.
             S%print_level = 0
           END IF

!  -----------------------------------
!  Step 2 of the algorithm (see paper)
!  -----------------------------------

!  Use ISWKSP to indicate which elements are needed for the matrix-vector
!  product B * P. If ISWKSP( I ) = nbprod, the I-th element is used

           S%nbprod = 0
           IF ( .NOT. S%alllin ) ISWKSP( : S%ntotel ) = 0

!  Estimate the norm of the preconditioning matrix by computing its smallest
!  and largest (in magnitude) diagonals

           S%diamin = HUGE( one ) ; S%diamax = zero
           IF ( S%calcdi ) THEN
             DO i = 1, n
               IF ( INDEX( i ) == 0 ) THEN
                S%diamin = MIN( S%diamin, FUVALS( S%ldx + i ) )
                S%diamax = MAX( S%diamax, FUVALS( S%ldx + i ) )
               END IF
             END DO
           END IF

!  If all the diagonals are small, the norm will be estimated as one

           IF ( S%diamax <= epsmch ) THEN
             S%diamin = one ; S%diamax = one
           END IF

!  Initialize values for the generalized Cauchy point calculation

           S%stepmx = zero ; S%f0 = inform%aug ; S%ibqpst = 1

!  Calculate the radius bounds for the structured trust region

           IF ( S%strctr ) THEN
             BND_radius( : , 1 ) = S%maximum_radius
             DO ig = 1, ng
               k1 = ISTAGV( ig ) ; k2 = ISTAGV( ig + 1 ) - 1
               BND_radius( ISVGRP( k1 : k2 ), 1 ) =                            &
                 MIN( RADII( ig ), BND_radius( ISVGRP( k1 : k2 ), 1 ) )
             END DO
           END IF

!DIR$ IVDEP
           DO i = 1, n

!  Set the bounds on the variables for the model problem. If a two-norm
!  trust region is to be used, the bounds are just the box constraints

             IF ( S%twonrm ) THEN
               BND( i, 1 ) = BL( i ) ; BND( i, 2 ) = BU( i )
             ELSE

!  If an infinity-norm trust region is to be used, the bounds are the
!  intersection of the trust region with the box constraints

               IF ( S%strctr ) THEN
                 S%rad = BND_radius( i, 1 )
               ELSE
                 S%rad = inform%radius
               END IF
               IF ( S%calcdi ) THEN
                 distan = S%rad / SQRT( FUVALS( S%ldx + i ) )
               ELSE
                 distan = S%rad * VSCALE( i )
               END IF
               BND( i, 1 ) = MAX( X( i ) - distan, BL( i ) )
               BND( i, 2 ) = MIN( X( i ) + distan, BU( i ) )
               IF ( S%mortor ) THEN
                BND_radius( i, 1 ) = X( i ) - distan
                BND_radius( i, 2 ) = X( i ) + distan
               END IF
             END IF

!  Compute the Cauchy direction, DGRAD, as a scaled steepest-descent
!  direction. Normalize the diagonal scalings if necessary

             X0( i ) = X( i )

             DELTAX( i ) = zero
             GX0( i ) = FUVALS( S%lggfx + i )
             P( i ) = zero
             IF ( S%reusec ) CYCLE
             IF ( S%calcdi ) THEN
               j = S%ldx + i
               DGRAD( i ) = - FUVALS( S%lggfx + i ) / FUVALS( j )
               FUVALS( j ) = FUVALS( j ) / S%diamax
             ELSE
               DGRAD( i ) = - FUVALS( S%lggfx + i ) *                          &
                            ( VSCALE( i ) / S%vscmax ) ** 2
             END IF

!  If an approximation to the Cauchy point is to be used, calculate a
!  suitable initial estimate of the line minimum, stepmx

             IF ( S%xactcp ) THEN
               S%stepmx = HUGE( one )
             ELSE  
               IF ( DGRAD( i ) /= zero ) THEN
                 IF ( DGRAD( i ) > zero ) THEN
                   S%stepmx = MAX( S%stepmx, ( BU( i ) - X( i ) ) / DGRAD( i ) )
                 ELSE
                   S%stepmx = MAX( S%stepmx, ( BL( i ) - X( i ) ) / DGRAD( i ) )
                 END IF
               END IF
             END IF

!  Release any artificially fixed variables from their bounds

             IF ( INDEX( i ) == 4 ) INDEX( i ) = 0
           END DO

!  The value of the integer S%ifactr controls whether a new factorization
!  of the Hessian of the model is obtained (S%ifactr = 1) or whether a
!  Schur-complement update to an existing factorization is required
!  (S%ifactr = 2) when forming the preconditioner

           S%ifactr = 1 ; S%refact = .FALSE.

!  If a previously calculated generalized Cauchy point still lies
!  within the trust-region bounds, it will be reused

           IF ( S%reusec ) THEN

!  Retrieve the Cauchy point

             XT( : n ) = XCP( : n )

!  Retrieve the set of free variables

             inform%nvar = S%nfreec
             IVAR( : inform%nvar ) = IFREEC( : inform%nvar )
             INDEX( IVAR( : inform%nvar ) ) = 0

!  Skip the remainder of step 2

             S%reusec = .FALSE.
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, ' Reusing previous generalized Cauchy point ' )" )
             GO TO 290
           END IF

!  Evaluate the generalized Cauchy point, XT

           S%jumpto = 1
           S%firstc = .TRUE.
           S%mortor_its = 0
           S%rad = inform%radius

  240      CONTINUE
           CALL CPU_TIME( S%t )
           IF ( S%xactcp ) THEN

!  The exact generalized Cauchy point is required

             CALL CAUCHY_get_exact_gcp(                                        &
                 n, X0, XT, GX0, BND, INDEX, S%f0, S%stepmx, S%epstlp,         &
                 S%twonrm, S%dxsqr, S%rad, S%fmodel, DGRAD, Q, IVAR,           &
                 inform%nvar, nvar1, S%nvar2, S%nnonnz, INNONZ, S%out,         &
                 S%jumpto, S%print_level, S%findmx, BREAKP, S%CAUCHY )
           ELSE

!  An approximation to the Cauchy point suffices

             CALL CAUCHY_get_approx_gcp(                                       &
                 n, X0, XT, GX0, BND, INDEX, S%f0, S%epstlp, S%stepmx,         &
                 point1, S%twonrm, S%rad, S%fmodel, DGRAD, Q, IVAR,            &
                 inform%nvar, nvar1, S%nvar2, S%out, S%jumpto, S%print_level,  &
                 S%findmx, BREAKP, GRAD, S%CAUCHY )
           END IF
           CALL CPU_TIME( tim )
           S%tca = S%tca + tim - S%t

!  Scatter the nonzeros in DGRAD onto P

           P( IVAR( nvar1 : S%nvar2 ) ) = DGRAD( IVAR( nvar1 : S%nvar2 ) )

!  A further matrix-vector product is required

           IF ( S%jumpto > 0 ) THEN
             CALL CPU_TIME( S%t )
             S%nbprod = S%nbprod + 1

!  Calculate the product of the Hessian with the vector P

             S%densep = S%jumpto == 2 .OR. ( S%xactcp .AND. S%jumpto == 4 )

             CALL HSPRD_hessian_times_vector(                                  &
                 n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                    &
                 inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                    &
                 S%nbprod, S%alllin, IVAR , ISTAEV, ISTADH, INTVAR, IELING,    &
                 IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                  &
                 P , Q , GVALS( : , 2 )  , GVALS( : , 3 ),                     &
                 GRJAC, GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl,   &
                 GXEQX_used , INTREP, S%densep,                                &
                 IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,       &
                 ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,             &
                 NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )

             IF ( S%printd .AND. S%jumpto == 3 ) WRITE( S%out,                 &
               "( ' Nonzeros of Hessian * P are in positions', /, ( 24I3 ))" ) &
                 INNONZ( : S%nnonnz )
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t

!  Reset the components of P that have changed to zero

             P( IVAR( nvar1 : S%nvar2 ) ) = zero

!  If required, print a list of the nonzeros of P

             IF ( S%jumpto == 3 .AND. S%printd .AND. .NOT. S%alllin )          &
               WRITE( S%out, 2560 ) S%nbprod, ISWKSP( : S%ntotel )

!  Continue the Cauchy point calculation

             GO TO 240
           END IF

!  Check to see if there are any remaining free variables

           IF ( nvar1 > S%nvar2 ) THEN
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, '    No free variables - search direction complete ' )" )
             GO TO 400
           ELSE 
             IF ( S%printt ) WRITE( S%out,                                     &
               "( /, '    There are now ', I7, ' free variables ' )" )         &
                 S%nvar2 - nvar1 + 1
           END IF
  
!          IF ( S%mortor_its >= 1 ) GO TO 400
           IF ( S%msweep > 0 .AND. S%mortor_its > S%msweep ) GO TO 400

!  Store the Cauchy point and its gradient for future use

           XCP( : n ) = XT( : n )
           S%fcp = S%fmodel

!  Store the set of free variables at Cauchy point for future use

           S%nfreec = inform%nvar
           IFREEC( : S%nfreec ) = IVAR( : S%nfreec )

!          IF ( S%mortor ) THEN
!            WHERE( INDEX == 1 .OR. INDEX == 2 )
!              INDEX = 4 ; BND( : , 1 ) = XT ; BND( : , 2 ) = XT
!            END WHERE
!          END IF

!  See if an accurate approximation to the minimum of the quadratic
!  model is to be sought

           IF ( S%slvbqp ) THEN

!  Fix the variables which the Cauchy point predicts are active at the solution

             IF ( S%firstc ) THEN
               S%firstc = .FALSE.
               WHERE( INDEX == 1 .OR. INDEX == 2 )
                 INDEX = 4 ; BND( : , 1 ) = XT ; BND( : , 2 ) = XT
               END WHERE
             ELSE

!  Update the step taken and the set of variables which are considered free

               inform%nvar = 0
               DO i = 1, n
                 P( i ) = P( i ) + DELTAX( i )
                 IF ( P( i ) /= zero .OR. INDEX( i ) == 0 ) THEN
                   inform%nvar = inform%nvar + 1
                   IVAR( inform%nvar ) = i
                 END IF
               END DO
               S%nvar2 = inform%nvar
             END IF
           END IF
  
           IF ( S%mortor ) P = XT - X0

!  If required, print the active set at the generalized Cauchy point

  290      CONTINUE
           IF ( S%printw ) THEN
             WRITE( S%out, "( / )" )
             DO i = 1, n
               IF ( INDEX( i ) == 2 .AND.  XT( i ) >=                          &
                 BU( i ) - ABS( BU( i ) ) * S%epstln ) WRITE( S%out,           &
                 "( ' The variable number ', I3, ' is at its upper bound' )" ) i
               IF ( INDEX( i ) == 1 .AND. XT( i ) <=                           &
                 BL( i ) + ABS( BL( i ) ) * S%epstlp ) WRITE( S%out,           &
                 "( ' The variable number ', I3, ' is at its lower bound' )" ) i
               IF ( INDEX( i ) == 4 ) WRITE( S%out,                            &
                 "( ' The variable number ', I3, ' is temporarily fixed ' )" ) i
             END DO
           END IF

!  -----------------------------------
!  Step 3 of the algorithm (see paper)
!  -----------------------------------

           S%jumpto = 1

!  If an iterative method is to be used, set up convergence tolerances

           S%cgstop = MAX( S%resmin, MIN( control%acccg, S%qgnorm )            &
                           * S%qgnorm * S%qgnorm ) * S%diamin / S%diamax
!          IF ( S%twonrm .AND. .NOT. S%direct ) S%dxsqr = DOT_PRODUCT( P, P )
           IF ( S%twonrm .AND. .NOT. S%direct ) THEN
             S%dxsqr = zero
             DO i = 1, n ; S%dxsqr = S%dxsqr + P( i ) ** 2 ; END DO
           END IF
           IF ( S%printw .AND. S%twonrm .AND. .NOT. S%direct )                 &
             WRITE( S%out,                                                     &
               "( /, ' Two-norm of step to Cauchy point = ', ES12.4 )" )       &
                 SQRT( S%dxsqr )
           S%step = inform%radius

!  If an incomplete factorization preconditioner is to be used, decide
!  on the semi-bandwidth, nsemib, of the preconditioner. For the expanding
!  band method, the allowable bandwidth increases as the solution is approached

           IF ( S%iprcnd ) THEN
             inform%nsemib = n / 5
             IF ( inform%pjgnrm <= point1 ) inform%nsemib = n / 2
             IF ( inform%pjgnrm <= tenm2 ) inform%nsemib = n
           ELSE
             IF ( S%use_band ) THEN
               inform%nsemib = control%semibandwidth
             ELSE
               inform%nsemib = n
             END IF
           END IF

!  If Munksgaards preconditioner is to be used, set the stability factor
!  required by MA61 to be more stringent as the solution is approached

           IF ( S%munks ) THEN
             inform%ciccg = point1
             IF ( inform%pjgnrm <= point1 ) inform%ciccg = tenm2
             IF ( inform%pjgnrm <= tenm2 ) inform%ciccg = zero
           ELSE
             inform%ciccg = zero
           END IF

!  Set a limit on the number of CG iterations that are to be allowed

           inform%itcgmx = n
           IF ( S%iprcnd .OR. S%use_band )                                     &
             inform%itcgmx = MAX( 5, n / ( inform%nsemib + 1 ) )
           IF ( S%seprec .OR. S%gmpspr ) inform%itcgmx = 5
           S%nobnds = S%mortor .AND. S%twonrm

!  Calculate an approximate minimizer of the model within the specified bounds

  300      CONTINUE
           IF ( S%jumpto == 4 ) GO TO 320

!  The product of the Hessian with the vector P is required

           IF ( S%jumpto /= 2 ) THEN
             S%nbprod = S%nbprod + 1
             nvar1 = 1
             CALL CPU_TIME( S%t )

!  Set the required components of Q to zero

             IF ( S%jumpto == 1 ) THEN
               Q = zero
             ELSE
               Q( IVAR( : S%nvar2 ) ) = zero
             END IF

!  Compute the matrix-vector product with the dense vector P

             S%densep = .TRUE.
             CALL HSPRD_hessian_times_vector(                                  &
                 n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                    &
                 inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                    &
                 S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR, IELING,   &
                 IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                  &
                 P , Q , GVALS( : , 2 )  , GVALS( : , 3 ),                     &
                 GRJAC, GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl,   &
                 GXEQX_used , INTREP, S%densep,                                &
                 IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,       &
                 ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,             &
                 NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t
!            IF ( S%jumpto == 1 ) THEN
!              WRITE(6,"( 'XT-X0 ', 5ES12.4 )" ) XT - X0
!              WRITE(6,"( 'P   ', 5ES12.4 )" ) P
!              WRITE(6,"( 'Q   ', 5ES12.4 )" ) Q
!            END IF

!  If required, print a list of the nonzeros of P

             IF ( S%printd .AND. .NOT. S%alllin )                              &
               WRITE( S%out, 2560 ) S%nbprod, ISWKSP( : S%ntotel )
             IF ( S%jumpto == 1 ) THEN

!  If required, print the step taken

               IF ( S%out > 0 .AND. S%print_level >= 20 )                      &
                 WRITE( S%out, 2530 ) P( : n )

!  Compute the value of the model at the generalized Cauchy point and then
!  reset P to zero

               S%fnew = S%fmodel
!              S%fmodel = inform%aug
!!DIR$ IVDEP  
!              DO j = 1, S%nvar2
!                i = IVAR( j )
!                S%fmodel = S%fmodel + ( FUVALS( S%lggfx + i ) +               &
!                                      half * Q( i ) ) * P( i )
!                P( i ) = zero
!              END DO
               S%fmodel = S%f0
!DIR$ IVDEP  
               DO j = 1, S%nvar2
                 i = IVAR( j )
                 S%fmodel = S%fmodel + ( GX0( i ) + half * Q( i ) ) * P( i )
                 P( i ) = zero
               END DO

!  If required, compare the recurred and calculated model values

               IF ( S%printw ) WRITE( S%out,                                   &
                 "( ' *** Calculated quadratic at CP ', ES22.14, /,            &
              &     ' *** Recurred   quadratic at CP ', ES22.14 )" )           &
                 S%fmodel * S%findmx, S%fnew * S%findmx
             END IF
           ELSE

!  Evaluate the 'preconditioned' gradient. If the user has supplied a
!  preconditioner, return to the calling program

             IF ( S%myprec ) THEN
               inform%status = - 9 ; GO TO 800
             ELSE
               IF ( S%iprcnd .OR. S%munks .OR. S%icfs .OR. S%seprec .OR.       &
                    S%gmpspr .OR. S%use_band ) THEN

!  If required, use a preconditioner

                 CALL CPU_TIME( S%t )
                 CALL PRECN_use_preconditioner(                                &
                     S%ifactr, S%munks, S%use_band, S%seprec, S%icfs,          &
                     n, ng, nel, S%ntotel, S%nnza, S%maxsel,                   &
                     S%nadd, S%nvargp, S%nfreef, S%nfixed,                     &
                     control%io_buffer, S%refact, S%nvar2,                     &
                     IVAR, ISTADH, ICNA, ISTADA, INTVAR, IELVAR, S%nvrels,     &
                     IELING, ISTADG, ISTAEV, IFREE,  A, FUVALS,                &
                     S%lnguvl, FUVALS, S%lnhuvl, GVALS( : , 2 ),               &
                     GVALS( : , 3 ), DGRAD , Q, GSCALE_used, ESCALE,           &
                     GXEQX_used , INTREP, RANGE , S%icfact,                    &
                     inform%ciccg, inform%nsemib, inform%ratio, S%print_level, &
                     S%error, S%out, S%infor, alloc_status, bad_alloc,         &
                     ITYPEE, DIAG, OFFDIA, IW, IKEEP, IW1, IVUSE,              &
                     H_col_ptr, L_col_ptr, W, W1, RHS, RHS2, P2,               &
                     G, ISTAGV, ISVGRP,                                        &
                     lirnh, ljcnh, lh, lirnh_min, ljcnh_min, lh_min,           &
                     LINK_col, POS_in_H, llink, lpos,                          &
                     IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                   &
                     matrix, SILS_data, control%SILS_cntl,                     &
                     inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,  &
                     S%PRECN, SCU_matrix, SCU_data, inform%SCU_info,           &
                     S%ASMBL, S%skipg, KNDOFG )
                 CALL CPU_TIME( tim )
                 S%tls = S%tls + tim - S%t
                 S%ifactr = 0

!  Check for error returns

                 IF ( S%infor == 10 ) THEN
                   inform%status = 4 ; GO TO 820 ; END IF
                 IF ( S%infor == 11 ) THEN
                   inform%status = 5 ; GO TO 820 ; END IF
                 IF ( S%infor == 12 ) GO TO 990
               ELSE

!  If required, use a diagonal preconditioner

                 IF ( S%dprcnd ) THEN
                   Q( IVAR( : S%nvar2 ) ) = DGRAD( : S%nvar2 ) /               &
                      FUVALS( S%ldx + IVAR( : S%nvar2 ) )
                 ELSE

!  No preconditioner is required

                   Q( IVAR( : S%nvar2 ) ) = DGRAD( : S%nvar2 ) *               &
                      VSCALE( IVAR( : S%nvar2 ) )
                 END IF
               END IF
             END IF
           END IF
  320      CONTINUE

!  The minimization will take place over all variables which are not on the
!  trust-region boundary with negative gradients pushing over the boundary

           IF ( S%direct ) THEN

!  - - - - - - - - - - - - direct method - - - - - - - - - - - - - - - -

!  Minimize the quadratic using a direct method. The method used is a
!  multifrontal symmetric indefinite factorization scheme. Evaluate the
!  gradient of the quadratic at XT

             inform%nvar = 0
             S%gmodel = zero
             DO i = 1, n
               IF ( INDEX( i ) == 0 ) THEN
                 inform%nvar = inform%nvar + 1
                 IVAR( inform%nvar ) = i
!                gi = FUVALS( S%lggfx + i ) + Q( i )
                 gi = GX0( i ) + Q( i )
                 DGRAD( inform%nvar ) = gi
                 S%gmodel = MAX( S%gmodel, ABS( gi ) )
               ELSE
                 gi = zero
               END IF
               P( i ) = zero ; QGRAD( i ) = gi
             END DO
             S%nvar2 = inform%nvar

!  Check if the gradient of the model at the generalized Cauchy point
!  is already small enough. Compute the ( scaled ) step moved from the
!  previous to the current iterate

             S%step =                                                          &
               LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE, .TRUE. )

!  If the step taken is small relative to the trust-region radius,
!  ensure that an accurate approximation to the minimizer of the
!  model is found

             IF ( S%step <= stptol * inform%radius ) THEN
               IF ( MAX( S%resmin, S%step * S%cgstop /                         &
                 ( inform%radius * stptol ) ) >= S%gmodel ) GO TO 400
             ELSE
               IF ( S%gmodel * S%gmodel < S%cgstop ) GO TO 400
             END IF

!  Factorize the matrix and obtain the solution to the linear system, a
!  direction of negative curvature or a descent direction for the quadratic
!  model

             CALL CPU_TIME( S%t )
             IF ( S%mortor ) THEN
               CALL FRNTL_get_search_direction(                                &
                   n, ng, nel, S%ntotel, S%nnza, S%maxsel,                     &
                   S%nvargp, control%io_buffer, INTVAR, IELVAR, S%nvrels,      &
                   INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  , ISTADA,     &
                   FUVALS, S%lnguvl, FUVALS, S%lnhuvl, ISTADH, GXEQX_used,     &
                   GVALS( : , 2 ), GVALS( : , 3 ), IVAR, S%nvar2,              &
                   QGRAD , P , XT, BND_radius, S%fmodel, GSCALE_used,          &
                   ESCALE, X0    , S%twonrm, S%nobnds, S%dxsqr , S%rad,        &
                   S%cgstop, S%number, S%next  , S%modchl, RANGE ,             &
                   inform%nsemib, inform%ratio, S%print_level, S%error,        &
                   S%out, S%infor, alloc_status, bad_alloc,                    &
                   ITYPEE, DIAG, OFFDIA,                                       &
                   IVUSE,                                                      &
                   RHS, RHS2, P2, ISTAGV, ISVGRP,                              &
                   lirnh, ljcnh, lh, LINK_col, POS_in_H, llink, lpos,          &
                   IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                     &
                   matrix, SILS_data, control%SILS_cntl,                       &
                   inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,    &
                   SCU_matrix, SCU_data, inform%SCU_info, S%ASMBL,             &
                   S%skipg, KNDOFG )
             ELSE
               CALL FRNTL_get_search_direction(                                &
                   n, ng, nel, S%ntotel, S%nnza, S%maxsel,                     &
                   S%nvargp, control%io_buffer, INTVAR, IELVAR, S%nvrels,      &
                   INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  , ISTADA,     &
                   FUVALS, S%lnguvl, FUVALS, S%lnhuvl, ISTADH, GXEQX_used,     &
                   GVALS( : , 2 ), GVALS( : , 3 ), IVAR, S%nvar2,              &
                   QGRAD , P , XT, BND   , S%fmodel, GSCALE_used,              &
                   ESCALE, X0    , S%twonrm, S%nobnds, S%dxsqr , S%rad,        &
                   S%cgstop, S%number, S%next  , S%modchl, RANGE ,             &
                   inform%nsemib, inform%ratio, S%print_level, S%error,        &
                   S%out, S%infor, alloc_status, bad_alloc,                    &
                   ITYPEE, DIAG, OFFDIA,                                       &
                   IVUSE,                                                      &
                   RHS, RHS2, P2, ISTAGV, ISVGRP,                              &
                   lirnh, ljcnh, lh, LINK_col, POS_in_H, llink, lpos,          &
                   IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                     &
                   matrix, SILS_data, control%SILS_cntl,                       &
                   inform%SILS_infoa, inform%SILS_infof, inform%SILS_infos,    &
                   SCU_matrix, SCU_data, inform%SCU_info, S%ASMBL,             &
                   S%skipg, KNDOFG )
             END IF
             CALL CPU_TIME( tim )
             S%tls = S%tls + tim - S%t
             inform%nvar = S%nvar2

!  Check for error returns

             IF ( S%infor == 10 ) THEN ; inform%status = 4 ; GO TO 820 ; END IF
             IF ( S%infor == 11 ) THEN ; inform%status = 5 ; GO TO 820 ; END IF
             IF ( S%infor == 12 ) GO TO 990
             IF ( S%infor >= 6 ) THEN
               inform%status = S%infor ; GO TO 820 ; END IF

!  Save details of the system solved

             S%fill = MAX( S%fill, inform%ratio )
             S%ISYS( S%infor ) = S%ISYS( S%infor ) + 1
             S%lisend = S%LSENDS( S%infor )

!  Compute the ( scaled ) step from the previous to the current iterate
!  in the appropriate norm

             S%step =                                                          &
               LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE, .TRUE. )

!  For debugging, compute the directional derivative and curvature
!  along the direction P

             IF ( S%printm ) THEN
               IF ( .NOT. S%alllin ) ISWKSP( : S%ntotel ) = 0
               nvar1 = 0
               DO i = 1, S%nvar2
                 IF ( IVAR( i ) > 0 ) THEN
                   nvar1 = nvar1 + 1
                   IVAR( nvar1 ) = IVAR( i )
                 END IF
               END DO
               S%nvar2 = nvar1 ; inform%nvar = S%nvar2
               nvar1 = 1 ; S%nbprod = 1

!  Evaluate the product of the Hessian with the dense vector P

               CALL CPU_TIME( S%t )
               Q = zero
               S%densep = .TRUE.
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels,                            &
                   S%nvargp, inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,        &
                   S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR, IELING, &
                   IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                &
                   P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                     &
                   GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),          &
                   S%lnhuvl, GXEQX_used , INTREP, S%densep,                    &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
               CALL CPU_TIME( tim )
               S%tmv = S%tmv + tim - S%t

!  Compute the curvature

!              S%curv =                                                        &
!                DOT_PRODUCT( Q( IVAR( : S%nvar2 ) ), P( IVAR( : S%nvar2 ) ) )
               S%curv = zero
               DO i = 1, S%nvar2
                 S%curv = S%curv + Q( IVAR( i ) ) * P( IVAR( i ) )
               END DO

!  Compare the calculated and recurred curvature

               WRITE( S%out, "( ' curv  = ', ES12.4 )" ) S%curv
               WRITE( S%out, "( ' FRNTL - infor = ', I1 )" ) S%infor
               IF ( S%infor == 1 .OR. S%infor == 3 .OR. S%infor == 5 ) THEN
                 DO j = 1, S%nvar2
                   i = IVAR( j )
                   WRITE( S%out, "( ' P, H * P( ', I6, ' ), RHS( ', I6,        &
                  &  ' ) = ', 3ES15.7 )" ) i, i, P( i ), Q( i ), QGRAD( i )
                 END DO
               END IF
             END IF
           ELSE

!  - - - - - - - - - - - - iterative method - - - - - - - - - - - - - -

!   Minimize the quadratic using an iterative method. The method used
!   is a safeguarded preconditioned conjugate gradient scheme

             CALL CPU_TIME( S%t )
             IF ( S%mortor ) THEN
               inform%itcgmx = COUNT( INDEX == 0 )
               IF ( S%mortor_its > 0 )                                         &
                 inform%itcgmx = MAX( 10, inform%itcgmx / 2 )
!              IF ( S%mortor_its > - 1 ) inform%itcgmx = 10
               CALL CG_solve(                                                  &
                   n, X0, XT, GX0,  BND_radius, S%nbnd,                        &
                   INDEX, S%cgstop, S%fmodel, VSCALE, DGRAD, inform%status,    &
                   P, Q, IVAR, inform%nvar, S%nvar2, S%twonrm, S%rad, S%nobnds,&
                   S%gmodel, S%dxsqr, S%out, S%jumpto, S%print_level,          &
                   S%findmx, inform%itercg, inform%itcgmx,                     &
                   inform%ifixed, W_ws, S%CG )
             ELSE
               inform%itcgmx = 3 * COUNT( INDEX == 0 )
               CALL CG_solve(                                                  &
                   n, X0, XT, GX0, BND, n,                                     &
                   INDEX, S%cgstop, S%fmodel, VSCALE, DGRAD, inform%status,    &
                   P, Q, IVAR, inform%nvar, S%nvar2, S%twonrm, S%rad, S%nobnds,&
                   S%gmodel, S%dxsqr, S%out, S%jumpto, S%print_level,          &
                   S%findmx, inform%itercg, inform%itcgmx,                     &
                   inform%ifixed, W_ws, S%CG )
             END IF
             CALL CPU_TIME( tim )
             S%tls = S%tls + tim - S%t
             IF ( S%jumpto == 0 .OR. S%jumpto == 4 .OR. S%jumpto == 5 )        &
               S%step =                                                        &
                 LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE, .TRUE. )

!  The norm of the gradient of the quadratic model is smaller than
!  cgstop. Perform additional tests to see if the current iterate
!  is acceptable

             S%nvar2 = inform%nvar
             IF ( S%jumpto == 4 ) THEN

!  If the (scaled) step taken is small relative to the trust-region radius,
!  ensure that an accurate approximation to the minimizer of the model is found

               IF ( S%step <= stptol * inform%radius .AND.                     &
                   .NOT. control%quadratic_problem .AND. .NOT. S%slvbqp ) THEN
                 IF ( MAX( S%resmin, S%step * S%cgstop /                       &
                   ( inform%radius * stptol ) ) >=  S%gmodel ) THEN
                   IF ( S%printw ) WRITE( S%out,                               &
                     "( ' Norm of trial step ', ES12.4 )" ) S%step
                   S%jumpto = 0
                 ELSE
                   gi = S%step * S%cgstop / ( inform%radius * stptol )
                   IF ( S%printw ) WRITE( S%out,                               &
                     "( /, ' C.G. tolerance of ', ES12.4, ' has not been',     &
                  &        ' achieved. ', /, ' Actual step length = ', ES12.4, &
                  &        ' Radius = ', ES12.4 )" ) gi, S%step, inform%radius
                   S%jumpto = 4
                 END IF
               ELSE
                 S%jumpto = 0
               END IF
             END IF

!  A bound has been encountered in CG. If the bound is a trust-region bound,
!  stop the minimization

             IF ( S%jumpto == 5 ) THEN
               S%ifactr = 2
               S%nadd = 1
               IF ( S%twonrm ) THEN
                 S%jumpto = 2
               ELSE
                 IF ( S%slvbqp ) THEN
                   S%jumpto = 2
                 ELSE
                   S%jumpto = 0
                   IF ( S%mortor ) THEN
 
!  The bound encountered is an upper bound

!                    IF ( inform%ifixed > 0 ) THEN
!                      IF ( BU( inform%ifixed ) <                              &
!                        BND_radius( inform%ifixed, 2 ) ) S%jumpto = 2
!                    ELSE

!  The bound encountered is a lower bound

!                      IF ( BL( - inform%ifixed ) >                            &
!                        BND_radius( - inform%ifixed, 1 ) ) S%jumpto = 2
!                    END IF
                   ELSE 
                     IF ( S%strctr ) THEN
                       S%rad = BND_radius( i, 1 )
                     ELSE
                       S%rad = inform%radius
                     END IF

!  The bound encountered is an upper bound

                     IF ( inform%ifixed > 0 ) THEN
                       IF ( S%calcdi ) THEN
                         IF ( BU( inform%ifixed ) < X( inform%ifixed ) +       &
                           S%rad / SQRT( FUVALS( S%ldx + inform%ifixed ) ) )   &
                             S%jumpto = 2
                       ELSE
                         IF ( BU( inform%ifixed ) < X( inform%ifixed ) +       &
                           S%rad * VSCALE( inform%ifixed ) ) S%jumpto = 2
                       END IF
                     ELSE

!  The bound encountered is a lower bound

                       IF ( S%calcdi ) THEN
                         IF ( BL( - inform%ifixed ) > X( - inform%ifixed ) -   &
                           S%rad / SQRT( FUVALS( S%ldx - inform%ifixed ) ) )   &
                             S%jumpto = 2
                       ELSE
                         IF ( BL( - inform%ifixed ) > X( - inform%ifixed ) -   &
                           S%rad * VSCALE( - inform%ifixed ) ) S%jumpto = 2
                       END IF
                     END IF
                   END IF
                 END IF
               END IF
               IF ( S%printw .AND. S%jumpto == 2 ) WRITE( S%out,               &
                 "( /, ' Restarting the conjugate gradient iteration ' )" )
             END IF

!  If the bound encountered was a problem bound, continue minimizing the model

             IF ( S%jumpto > 0 ) GO TO 300
             S%cgend = S%CGENDS( inform%status - 9 )
           END IF

!  If required, compute the value of the model from first principles

           IF ( S%printd ) THEN
             S%nbprod = S%nbprod + 1
             inform%nvar = n ; nvar1 = 1 ; S%nvar2 = inform%nvar

!  Compute the step taken, P

             DO i = 1, n
               IVAR( i ) = i
               P( i ) = XT( i ) - X( i )
             END DO

!  Evaluate the product of the Hessian with the dense vector P

             CALL CPU_TIME( S%t )
             Q = zero
             S%densep = .TRUE.
             CALL HSPRD_hessian_times_vector(                                  &
                 n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                    &
                 inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                    &
                 S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR, IELING,   &
                 IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                  &
                 P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                       &
                 GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ), S%lnhuvl,  &
                 GXEQX_used , INTREP, S%densep,                                &
                 IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,       &
                 ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,             &
                 NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t

!  If required, print the step taken

             IF ( S%out > 0 .AND. S%print_level >= 20 ) WRITE( S%out, 2530 ) P

!  Compute the model value, fnew, and reset P to zero

             S%fnew = inform%aug
!DIR$ IVDEP  
             DO j = 1, S%nvar2
               i = IVAR( j )
               S%fnew =                                                        &
                 S%fnew + ( FUVALS( S%lggfx + i ) + half * Q( i ) ) * P( i )
               P( i ) = zero
             END DO
             WRITE( S%out, "( ' *** Calculated quadratic at end CG ', ES22.14, &
            &   /, ' *** Recurred   quadratic at end CG ', ES22.14 )" )        &
               S%fnew * S%findmx, S%fmodel * S%findmx
           END IF

!  ------------------------------------------------------
!  Step 3.25 of the algorithm - More'-Toraldo projections
!  ------------------------------------------------------

           IF ( S%mortor ) THEN
             j = 0
             DO i = 1, n
               IF ( XT( i ) < BL( i ) .OR. XT( i ) > BU( i ) ) THEN
!                WRITE(6,"(3ES12.4)" ) BL(i), XT( i ), BU( i )
                 IF ( S%printt ) WRITE( S%out,                                 &
                   "( /, '    Problem bound would be violated so .... ' )" )
                 j = 1
                 EXIT 
               END IF
             END DO
             
             IF ( j == 1 ) THEN

!  Compute P, the step taken to the Cauchy point

               inform%nvar = n
               S%nvar2 = inform%nvar
             
               DO i = 1, n
                 IVAR( i ) = i
                 P( i ) = XCP( i ) - X( i )
               END DO

!  Evaluate the product of the Hessian with the dense vector P

               CALL CPU_TIME( S%t )
               Q = zero
               S%densep = .TRUE.
               CALL HSPRD_hessian_times_vector(                                &
                   n , ng, nel, S%ntotel, S%nvrels,                            &
                   S%nvargp, inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,        &
                   S%nbprod, S%alllin, IVAR, ISTAEV, ISTADH, INTVAR, IELING,   &
                   IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                &
                   P , Q , GVALS( : , 2 ), GVALS( : , 3 ),                     &
                   GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),          &
                   S%lnhuvl, GXEQX_used , INTREP, S%densep,                    &
                   IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,     &
                   ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,           &
                   NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
               CALL CPU_TIME( tim )
               S%tmv = S%tmv + tim - S%t

!  Recover the Cauchy point and its function and gradient values

               X0( : n ) = XCP( : n )
               S%f0 = S%fcp
               GX0 = FUVALS( S%lggfx + 1 : S%lggfx + n ) + Q
               
!              WRITE(6,"( 'P   ', 5ES12.4 )" ) P
!              WRITE(6,"( 'Q   ', 5ES12.4 )" ) Q
!              WRITE(6,"( ' gx0 ', 5ES12.4 )" ) GX0

!  Recover the set of free variables at the Cauchy point

!              inform%nvar = S%nfreec
!              IVAR( : S%nfreec ) = IFREEC( : S%nfreec )

!  Set the Cauchy direction

               DGRAD = XT - XCP
               S%stepmx = MIN( S%stepmx, one )
               P = zero
               
               IF ( S%twonrm ) S%rad = SQRT( DOT_PRODUCT( DGRAD, DGRAD ) )

!  If possible, use the existing preconditioner

!              IF ( S%refact ) THEN
!                S%ifactr = 1
!              ELSE

!  Ensure that a new Schur complement is calculated. Restore the complete
!  list of variables that were free when the factorization was calculated

!                S%ifactr = 2 ; S%nadd = 1 ; S%nfixed = 0
!                IFREE( : S%nfreef ) = ABS( IFREE( : S%nfreef ) )
!              END IF
               S%jumpto = 1
               S%mortor_its = S%mortor_its + 1
               GO TO 240
             END IF
           END IF

!  -------------------------------------
!  Step 3.5 of the algorithm (SEE PAPER)
!  -------------------------------------

!  An accurate approximation to the minimum of the quadratic model is to be
!  sought

           IF ( S%slvbqp .AND. S%ibqpst <= 2 ) THEN

!  Compute the gradient value

             inform%nvar = n
             S%nvar2 = inform%nvar

!  Compute the step taken

             DO i = 1, n
               IVAR( i ) = i
               DELTAX( i ) = XT( i ) - X( i )
             END DO

!  Evaluate the product of the Hessian with the dense step vector

             CALL CPU_TIME( S%t )
             Q = zero
             S%densep = .TRUE.
             CALL HSPRD_hessian_times_vector(                                  &
                 n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                    &
                 inform%nvar  , nvar1 , S%nvar2 , S%nnonnz,                    &
                 S%nbprod, S%alllin, IVAR  , ISTAEV, ISTADH, INTVAR, IELING,   &
                 IELVAR, ISWKSP( : S%ntotel ), INNONZ( : n ),                  &
                 DELTAX, Q, GVALS( : , 2 ), GVALS( : , 3 ),                    &
                 GRJAC , GSCALE_used, ESCALE, FUVALS( : S%lnhuvl ),            &
                 S%lnhuvl, GXEQX_used , INTREP, S%densep,                      &
                 IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR, ITYPEE, ISYMMH,       &
                 ISTAJC, IUSED, LIST_elements, LINK_elem_uses_var,             &
                 NZ_comp_w, W_ws, W_el, W_in, H_in, RANGE, S%skipg, KNDOFG )
             CALL CPU_TIME( tim )
             S%tmv = S%tmv + tim - S%t

!  Compute the model gradient at XT

             GX0 = FUVALS( S%lggfx + 1 : S%lggfx + n ) + Q
             dltnrm = MAX( zero, MAXVAL( ABS( Q ) ) )

!  Save the values of the nonzero components of the gradient

             k = 0
             DO j = 1, S%nfreef
               i = IFREE( j )
               IF ( i > 0 ) THEN
                 k = k + 1
                 GX0( i ) = DGRAD( k )
               END IF
             END DO

!  Find the projected gradient of the model and its norm

             CALL LANCELOT_projected_gradient(                                 &
                 n, XT, GX0, VSCALE, BND( : , 1 ), BND( : , 2 ), DGRAD,  &
                 IVAR, inform%nvar, S%gmodel )

!  Check for convergence of the inner iteration

             IF ( S%printt )                                                   &
               WRITE( S%out, "( /, '    ** Model gradient is ', ES12.4,        &
              &  ' Required accuracy is ', ES12.4 )" ) S%gmodel, SQRT( S%cgstop)
             IF ( S%gmodel * S%gmodel > S%cgstop .AND. dltnrm > epsmch ) THEN

!  The approximation to the minimizer of the quadratic model is not yet
!  good enough. Perform another iteration

!  Store the function value at the starting point for the Cauchy search

               S%f0 = S%fmodel

!  Set the staring point for the Cauchy step

               X0 = XT

!  Set the Cauchy direction

               DGRAD = - GX0 * ( VSCALE / S%vscmax ) ** 2
               P = zero

!  If possible, use the existing preconditioner

               IF ( S%refact ) THEN
                 S%ifactr = 1
               ELSE

!  Ensure that a new Schur complement is calculated. Restore the complete
!  list of variables that were free when the factorization was calculated

                  S%ifactr = 2 ; S%nadd = 1 ; S%nfixed = 0
                  IFREE( : S%nfreef ) = ABS( IFREE( : S%nfreef ) )
                END IF
                S%jumpto = 1
                GO TO 240
              END IF
           END IF

!  -----------------------------------
!  Step 4 of the algorithm (see paper)
!  -----------------------------------

!  Test for acceptance of new point and trust-region management

  400      CONTINUE
!          WRITE(6,"( 4ES12.4 )") ( BND( i, 1 ), X( i ),                       &
!            XT( i ) - X( i ), BND( i, 2 ), i = 1, n )

!  Determine which nonlinear elements and non-trivial groups need to
!  be re-evaluated by considering which of the variables have changed

           CALL OTHERS_which_variables_changed(                                &
               S%unsucc, n, ng, nel, inform%ncalcf, inform%ncalcg, ISTAEV,     &
               ISTADG, IELING, ICALCF, ICALCG, X , XT, ISTAJC, IGCOLJ,         &
               LIST_elements, LINK_elem_uses_var )

!  If required, print a list of the nonlinear elements and groups
!  which have changed

           IF ( S%printw .AND. .NOT. S%alllin ) THEN
             WRITE( S%out,                                                     &
               "( /, ' Functions for the following elements need to be',       &
            &      ' re-evaluated ', /, ( 12I6 ) )" ) ICALCF( : inform%ncalcf )
             WRITE( S%out,                                                     &
               "( /, ' Functions for the following groups need to be',         &
            &      ' re-evaluated ', /, ( 12I6 ) )" ) ICALCG( : inform%ncalcg )
           END IF

!  If the step taken is ridiculously small, exit

           IF ( S%step <= S%stpmin ) THEN
             inform%status = 3
             GO TO 600
           END IF

!  Return to the calling program to obtain the function value at the new point

           inform%status = - 3
           IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF

!  Compute the group argument values FT

  430      CONTINUE
           DO ig = 1, ng

!  Include the contribution from the linear element

!            ftt = SUM( A( ISTADA( ig ) : ISTADA( ig + 1 ) - 1 ) *             &
!              XT( ICNA( ISTADA( ig ) : ISTADA( ig + 1 ) - 1 ) ) ) - B( ig )
             ftt = - B( ig )
             DO i = ISTADA( ig ), ISTADA( ig + 1 ) - 1
               ftt = ftt + A( i ) * XT( ICNA( i ) )
             END DO
  
!  Inclu  de the contributions from the nonlinear elements
  
!            ftt = ftt + SUM( ESCALE( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) *  &
!              FUVALS( IELING( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) ) )
             DO i = ISTADG( ig ), ISTADG( ig + 1 ) - 1
               ftt = ftt + ESCALE( i ) * FUVALS( IELING( i ) )
             END DO
             FT( ig ) = ftt
           END DO

!  Compute the group function values

           IF ( S%altriv ) THEN
!            S%fnew = DOT_PRODUCT( GSCALE_used, FT )
             S%fnew = zero
             IF ( S%skipg ) THEN
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 0 )                                       &
                   S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
                END DO
             ELSE
               DO ig = 1, ng 
                 S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
               END DO
             END IF
           ELSE

!  If necessary, return to the calling program to obtain the group
!  function and derivative values at the initial point

             inform%status = - 4
             IF ( external_gr ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
           END IF
  470      CONTINUE
           IF ( .NOT. S%altriv ) THEN
             S%fnew = zero
             IF ( S%p_type == 2 ) THEN
               IF ( S%skipg ) THEN
                 DO ig = 1, ng
                   IF ( KNDOFG( ig ) > 0 )                                     &
                     S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                 END DO
               ELSE
                 DO ig = 1, ng
                   S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                 END DO
               END IF
             ELSE
               IF ( S%skipg ) THEN
                 DO ig = 1, ng
                   IF ( KNDOFG( ig ) > 0 ) THEN
                     IF ( GXEQX_used( ig ) ) THEN
                        S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
                     ELSE
                        S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                     END IF
                   END IF
                 END DO
               ELSE
                 DO ig = 1, ng
                   IF ( GXEQX_used( ig ) ) THEN
                      S%fnew = S%fnew + GSCALE_used( ig ) * FT( ig )
                   ELSE
                      S%fnew = S%fnew + GSCALE_used( ig ) * GVALS( ig, 1 )
                   END IF
                 END DO
               END IF
             END IF
           END IF

!  Compute the actual and predicted reductions in the function value.
!  Ensure that rounding errors do not dominate

           S%ared =                                                            &
             ( inform%aug - S%fnew ) + MAX( one, ABS( inform%aug ) ) * S%teneps
           S%prered =                                                          &
            ( inform%aug - S%fmodel ) + MAX( one, ABS( inform%aug ) ) * S%teneps
!          write(6,"(A,3ES12.4)") ' orig, new_f, new_m ', inform%aug, S%fnew, S%fmodel
           IF ( ABS( S%ared ) < S%teneps .AND. ABS( inform%aug ) > S%teneps )  &
             S%ared = S%prered
           IF ( control%quadratic_problem ) THEN
             S%rho = one
           ELSE
             S%rho = S%ared / S%prered
           END IF
           IF ( S%out > 0 .AND. S%print_level >= 100 ) WRITE( S%out,           &
             "( /, ' Old f = ', ES20.12, ' New   f = ', ES20.12, /,            &
          &        ' Old f = ', ES20.12, ' Model f = ', ES20.12 )" )           &
             inform%aug, S%fnew, inform%aug, S%fmodel
           IF ( S%printm ) WRITE( S%out,                                       &
             "( /, ' Actual change    = ', ES20.12, /,                         &
          &        ' Predicted change = ', ES20.12, /                          &
          &        ' Ratio ( rho )    = ', ES20.12 )" ) S%ared, S%prered, S%rho
             

!  Adjust rho in the non-monotone case

           IF ( S%nmhist > 0 ) THEN
             ar_h = ( S%f_r - S%fnew ) + MAX( one, ABS( S%f_r ) ) * S%teneps
             pr_h = S%sigma_r + S%prered
             IF ( ABS( ar_h ) < S%teneps .AND. ABS( S%f_r ) > S%teneps )       &
               ar_h = pr_h
             S%rho = MAX( S%rho, ar_h / pr_h )
           END IF

!  Compute the actual and predicted reductions in each of the
!  group values in the structured trust-region case

!          IF ( S%strctr ) THEN
!            CALL STRUTR_changes( DIMEN , D_model, D_function, XT - X,         &
!                IELING, ISTADG, IELVAR, ISTAEV, INTVAR, ISTADH, ISTADA, ICNA, &
!                A, ESCALE, GSCALE_used, FT, GXEQX_used, INTREP,               &
!                FUVALS, S%lnhuvl, GV_old, GVALS( : , 1 ),                     &
!                GVALS( : , 2 ), GVALS( : , 3 ), GRJAC , S%nvargp, RANGE )
!          END IF

!  -----------------------------------
!  Step 5 of the algorithm (see paper)
!  -----------------------------------

           S%oldrad = inform%radius

!  - - - - step management when the iteration has proved unsuccessful -

           IF ( S%rho < control%eta_successful .OR. S%prered <= zero ) THEN

!  unsuccessful step. Calculate the radius which would just include the newly
!  found point, XT

             S%unsucc = .TRUE.
             IF ( S%rho >= zero .AND. S%prered > zero ) THEN
               S%radmin = S%step
             ELSE

!  Very unsuccessful step. Obtain an estimate of the radius required to obtain
!  a successful step along the step taken, radmin, if such a step were taken
!  at the next iteration

!              slope =                                                         &
!                DOT_PRODUCT( FUVALS( S%lggfx + 1: S%lggfx + n ), XT - X )
               slope = zero
               DO i = 1, n
                 slope = slope + FUVALS( S%lggfx + i ) * ( XT( i ) - X( i ) )
               END DO
               S%curv = S%fmodel - inform%aug - slope
               S%radmin = S%step * ( control%eta_very_successful - one ) *     &
                 slope / ( S%fnew - inform%aug - slope -                       &
                 control%eta_very_successful * S%curv )
             END IF

!  Update the trust-region radius/radii

             IF ( S%strctr ) THEN

!  Structured trust-region case:

!              CALL STRUTR_radius_update(                                      &
!                  DIMEN, D_model, D_function, S%ared, S%prered, control,      &
!                  RADII )
              
               CALL STRUTR_radius_update(                                      &
                   n, ng, nel, XT - X, IELING, ISTADG, IELVAR, ISTAEV, INTVAR, &
                   ISTADH, ISTADA, ICNA, A, ESCALE, GSCALE_used, FT,           &
                   GXEQX_used, ITYPEE, INTREP, FUVALS, S%lnhuvl, GV_old,       &
                   GVALS( : , 1 ),  GVALS( : , 2 ), GVALS( : , 3 ), GRJAC,     &
                   S%nvargp, S%ared, S%prered, RADII, S%maximum_radius,        &
                   control%eta_successful, control%eta_very_successful,        &
                   control%eta_extremely_successful, control%gamma_decrease,   &
                   control%gamma_increase, control%mu_meaningful_model,        &
                   control%mu_meaningful_group, ISTAGV, ISVGRP, IVALJR,        &
                   ISYMMH, W_el, W_in, H_in, W_ws, RANGE )
               inform%radius = MAXVAL( RADII )
             ELSE

!  Unstructured trust-region case:

!  Compute an upper bound on the new trust-region radius. Radmin, the actual
!  radius will be the current radius multiplied by the largest power of 
!  gamma_decrease for which the product is smaller than radmin.

               S%radmin = MIN( S%step, MAX(                                    &
                 inform%radius * control%gamma_smallest, S%radmin ) )

!  If the trust-region radius has shrunk too much, exit. this may indicate a
!  derivative bug or that the user is asking for too much accuracy in the
!  final gradient

               IF ( S%radmin < S%radtol ) THEN
                 IF ( S%printe ) WRITE( S%out, 2540 )
                 inform%status = 2 ; GO TO 600
               END IF

!  Continue reducing the radius by the factor gamma_decrease until it is 
!  smaller than radmin

               DO
                 inform%radius = control%gamma_decrease * inform%radius
                 IF ( inform%radius < S%radmin ) EXIT
               END DO
             
             END IF

!  Compute the distance of the generalized Cauchy point from the
!  initial point

             IF ( S%calcdi ) THEN
               IF ( n > 0 ) QGRAD( : n ) = one /                   &
                 SQRT( FUVALS( S%ldx + 1 : S%ldx + n ) )
               S%step =                                                        &
                 LANCELOT_norm_diff( n, XT, X, S%twonrm, QGRAD, .TRUE. )
             ELSE
               S%step =                                                        &
                 LANCELOT_norm_diff( n, XT, X, S%twonrm, VSCALE,.TRUE. )
             END IF

!  If the generalized Cauchy point lies within the new trust region,
!  it may be reused

!            S%reusec = S%step < inform%radius

!  Start a further iteration using the newly reduced trust region

             IF ( S%direct ) S%next = .FALSE.
             GO TO 120

!  - - - - - - - - - - - successful step - - - - - - - - - - - - - - - -

           ELSE
             S%unsucc = .FALSE.

!  In the non-monotone case, update the sum of predicted models

             IF ( S%nmhist > 0 ) THEN
               S%sigma_c = S%sigma_c + ( inform%aug - S%fmodel )
               S%sigma_r = S%sigma_r + ( inform%aug - S%fmodel )

!  If appropriate, update the best value found

               IF ( S%fnew < S%f_min ) THEN
                 S%f_min = S%fnew ; S%f_c = S%f_min
                 S%sigma_c = zero ; S%l_suc = 0
               ELSE
                 S%l_suc = S%l_suc + 1

!  Check to see if there is a new candidate for the next reference value

                 IF ( S%fnew > S%f_c ) THEN
                   S%f_c = S%fnew ; S%sigma_c = zero ; END IF

!  Check to see if the reference value needs to be reset

                 IF ( S%l_suc == S%nmhist ) THEN
                   S%f_r = S%f_c ; S%sigma_r = S%sigma_c ; END IF
               END IF
             END IF

!  Update the trust-region radius/radii

             IF ( S%strctr ) THEN

!  Structured trust-region case:

!              CALL STRUTR_radius_update( DIMEN, D_model, D_function, S%ared,  &
!                                         S%prered, control, RADII )
               CALL STRUTR_radius_update(                                      &
                   n, ng, nel, XT - X, IELING, ISTADG, IELVAR, ISTAEV, INTVAR, &
                   ISTADH, ISTADA, ICNA, A, ESCALE, GSCALE_used, FT,           &
                   GXEQX_used, ITYPEE, INTREP, FUVALS, S%lnhuvl, GV_old,       &
                   GVALS( : , 1 ), GVALS( : , 2 ), GVALS( : , 3 ), GRJAC,      &
                   S%nvargp, S%ared, S%prered, RADII, S%maximum_radius,        &
                   control%eta_successful, control%eta_very_successful,        &
                   control%eta_extremely_successful, control%gamma_decrease,   &
                   control%gamma_increase, control%mu_meaningful_model,        &
                   control%mu_meaningful_group, ISTAGV, ISVGRP, IVALJR,        &
                   ISYMMH, W_el, W_in, H_in, W_ws, RANGE )
               inform%radius = MAXVAL( RADII )
             ELSE

!  Unstructured trust-region case:
!  Increase the trust-region radius. Note that we require the step taken to be
!  at least a certain multiple of the distance to the trust-region boundary

               IF ( S%rho >= control%eta_very_successful )                     &
                 inform%radius = MIN( MAX( inform%radius,                      &
                   control%gamma_increase * S%step ), S%maximum_radius )
             END IF 

!  - - derivative evaluations when the iteration has proved successful -

!  Evaluate the gradient and approximate Hessian. Firstly, save the
!  old element gradients if approximate Hessians are to be used

             CALL CPU_TIME( S%t )
             IF ( .NOT. S%second .AND. .NOT. S%alllin ) THEN
               IF ( use_elders ) THEN
                 QGRAD( : S%ntotin ) = FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin )
               ELSE
                 QGRAD( : S%ntotin ) = FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin )
               END IF

!  If they are used, update the second derivative approximations.
!  Form the differences in the iterates, P

               P = XT - X
             END IF
             CALL CPU_TIME( tim )
             S%tup = S%tup + tim - S%t

!  Accept the computed point and function value

             inform%aug = S%fnew ; X = XT

!  Return to the calling program to obtain the derivative
!  values at the new point

             IF ( S%fdgrad ) S%igetfd = 0
             IF ( .NOT. ( S%altriv .AND. S%alllin ) ) THEN
               inform%ngeval = inform%ngeval + 1
               IF ( S%altriv ) THEN
                 inform%status = - 6
                 IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
               ELSE
                 inform%status = - 5
                 IF ( external_el .AND. external_gr ) THEN
                   GO TO 800 ; ELSE ; GO TO 700 ; END IF
               END IF
             END IF
           END IF

  540      CONTINUE

!  If a structured trust-region is being used, store the current values
!  of the group functions

           IF ( .NOT. S%unsucc ) THEN
             IF ( S%strctr ) THEN
               DO ig = 1, ng
                 IF ( GXEQX_used( ig ) ) THEN
                   GV_old( ig ) = FT( ig )
                 ELSE
                   GV_old( ig ) = GVALS( ig, 1 )
                 END IF
               END DO
             END IF
           END IF

!  If finite-difference gradients are used, compute their values

           IF ( S%fdgrad .AND. .NOT. S%alllin ) THEN

!  Store the values of the nonlinear elements for future use

             IF ( S%igetfd == 0 ) THEN
               FUVALS_temp( : nel ) = FUVALS( : nel )
               S%centrl =                                                      &
                 S%first_derivatives == 2 .OR. inform%pjgnrm < epsmch ** 0.25
             END IF

!  Obtain a further set of differences

             IF ( use_elders ) THEN
               CALL OTHERS_fdgrad_flexible(                                    &
                                   n, nel, lfuval, S%ntotel, S%nvrels,         &
                                   S%nsets, IELVAR, ISTAEV, IELING,            &
                                   ICALCF, inform%ncalcf, INTVAR,              &
                                   S%ntype, X, XT, FUVALS, S%centrl, S%igetfd, &
                                   S%OTHERS, ISVSET, ISET, INVSET, ISSWTR,     &
                                   ISSITR, ITYPER, LIST_elements,              &
                                   LINK_elem_uses_var, WTRANS, ITRANS,         &
                                   ELDERS( 1, : ) )
             ELSE
               CALL OTHERS_fdgrad( n, nel, lfuval, S%ntotel, S%nvrels,         &
                                   S%nsets, IELVAR, ISTAEV, IELING,            &
                                   ICALCF, inform%ncalcf, INTVAR,              &
                                   S%ntype, X, XT, FUVALS, S%centrl, S%igetfd, &
                                   S%OTHERS, ISVSET, ISET, INVSET, ISSWTR,     &
                                   ISSITR, ITYPER, LIST_elements,              &
                                   LINK_elem_uses_var, WTRANS, ITRANS )
             END IF
             IF ( S%igetfd > 0 ) THEN
               inform%status = - 7
               IF ( external_el ) THEN ; GO TO 800 ; ELSE ; GO TO 700 ; END IF
             END IF

!  Restore the values of the nonlinear elements at X

             S%igetfd = S%nsets + 1
             FUVALS( : nel ) = FUVALS_temp( : nel )
           END IF

!  Compute the gradient value

           CALL CPU_TIME( S%t )
           CALL LANCELOT_form_gradients(                                       &
               n, ng, nel, S%ntotel, S%nvrels, S%nnza,                         &
               S%nvargp, .FALSE., ICNA, ISTADA, IELING, ISTADG, ISTAEV,        &
               IELVAR, INTVAR, A, GVALS( : , 2 ), FUVALS( : S%lnguvl ),        &
               S%lnguvl, FUVALS( S%lggfx  + 1 : S%lggfx + n ),                 &
               GSCALE_used, ESCALE, GRJAC, GXEQX_used, INTREP,                 &
               ISVGRP, ISTAGV, ITYPEE, ISTAJC, W_ws, W_el, RANGE, KNDOFG )

!  If they are used, update the second derivative approximations

           IF ( .NOT. S%second .AND. .NOT.S%alllin ) THEN
             IF ( use_elders ) THEN

!  Form the differences in the gradients, QGRAD

               QGRAD( : S%ntotin ) =                                           &
                 FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin ) - QGRAD( : S%ntotin )
               IF ( S%firsup ) THEN

!  If a secant method is to be used, scale the initial second derivative
!  matrix for each element so as to satisfy the weak secant condition

                 CALL OTHERS_scaleh_flexible(                                  &
                     .FALSE., n, nel, lfuval, S%nvrels,                        &
                     S%ntotin, inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR,  &
                     IELVAR, ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el,   &
                     H_in, ELDERS( 2, : ), RANGE )
                 S%firsup = .FALSE.
               END IF

!  Update the second derivative approximations using one of four
!  possible secant updating formulae, BFGS, DFP, PSB and SR1.

               CALL OTHERS_secant_flexible(                                    &
                   n, nel, lfuval, S%nvrels, S%ntotin, IELVAR, ISTAEV, INTVAR, &
                   ITYPEE, INTREP, ISTADH, FUVALS, ICALCF, inform%ncalcf, P,   &
                   QGRAD, inform%iskip, S%print_level, S%out, W_el, W_in,      &
                   H_in, ELDERS( 2, : ), RANGE )
             ELSE

!  Form the differences in the gradients, QGRAD

               QGRAD( : S%ntotin ) =                                           &
                 FUVALS( S%lgxi + 1 : S%lgxi + S%ntotin ) - QGRAD( : S%ntotin )
               IF ( S%firsup ) THEN

!  If a secant method is to be used, scale the initial second derivative
!  matrix for each element so as to satisfy the weak secant condition

                 CALL OTHERS_scaleh( .FALSE., n, nel, lfuval, S%nvrels,        &
                     S%ntotin, inform%ncalcf, ISTAEV, ISTADH, ICALCF, INTVAR,  &
                     IELVAR, ITYPEE, INTREP, FUVALS, P, QGRAD, ISYMMD, W_el,   &
                     H_in, RANGE )
                 S%firsup = .FALSE.
               END IF

!  Update the second derivative approximations using one of four
!  possible secant updating formulae, BFGS, DFP, PSB and SR1.

               CALL OTHERS_secant(                                             &
                   n, nel, lfuval, S%nvrels, S%ntotin, IELVAR, ISTAEV, INTVAR, &
                   ITYPEE, INTREP, ISTADH, FUVALS, ICALCF, inform%ncalcf, P,   &
                   QGRAD, S%second_derivatives, inform%iskip,                  &
                   S%print_level, S%out, W_el, W_in, H_in, RANGE )
             END IF
           END IF
           CALL CPU_TIME( tim )
           S%tup = S%tup + tim - S%t

!  Compute the projected gradient and its norm

           CALL LANCELOT_projected_gradient(                                   &
               n, X, FUVALS( S%lggfx + 1 : S%lggfx + n ), VSCALE,              &
               BL, BU, DGRAD, IVAR, inform%nvar, inform%pjgnrm )
           S%nfree = inform%nvar

!  If required, use the users preconditioner

           IF ( S%prcond .AND. inform%nvar > 0 .AND. S%myprec ) THEN
             inform%status = - 10 ; GO TO 800
           END IF
  570      CONTINUE

!  Find the norm of the 'preconditioned' projected gradient. Also,
!  if required, find the diagonal elements of the assembled Hessian

           CALL LANCELOT_norm_proj_grad(                                       &
               n , ng, nel, S%ntotel, S%nvrels, S%nvargp,                      &
               inform%nvar, S%smallh, inform%pjgnrm, S%calcdi, S%dprcnd,       &
               S%myprec, IVAR(:inform%nvar ), ISTADH, ISTAEV, IELVAR, INTVAR,  &
               IELING, DGRAD( : inform%nvar ), Q, GVALS( : , 2 ),              &
               GVALS( : , 3 ), FUVALS( S%ldx + 1 : S%ldx + n ),                &
               GSCALE_used, ESCALE, GRJAC, FUVALS( : S%lnhuvl ), S%lnhuvl,     &
               S%qgnorm, GXEQX_used, INTREP, ISYMMD, ISYMMH, ISTAGV, ISLGRP,   &
               ISVGRP, IVALJR, ITYPEE, W_el, W_in, H_in, RANGE, KNDOFG )

           IF ( S%direct ) S%next = S%step < tenten * epsmch .AND. S%infor == 2
           GO TO 120

!  If the user's computed group function values are inadequate, reduce
!  the trust-region radius

  590      CONTINUE
           S%unsucc = .TRUE.
           S%oldrad = inform%radius 
           inform%radius = control%gamma_decrease * inform%radius
           IF ( S%strctr ) RADII = control%gamma_decrease * RADII

!  If the trust-region radius has shrunk too much, exit. this may indicate a
!  derivative bug or that the user is asking for too much accuracy in the
!  final gradient

           IF ( inform%radius < S%radtol ) THEN
             IF ( S%printe ) WRITE( S%out, 2540 )
             inform%status = 2 ; GO TO 600
           END IF
           GO TO 120

! ---------------------
!
!   End the main loop
!
! ---------------------

  600    CONTINUE

!  Print details of the solution

         IF ( S%printi ) THEN
           inform%iter = MIN0( control%maxit, inform%iter )
           IF ( inform%iter == 0 ) THEN
             WRITE( S%out, 2570 ) inform%iter, inform%aug * S%findmx,          &
               inform%ngeval, inform%pjgnrm, inform%itercg, inform%iskip
           ELSE
             WRITE( S%out,2550 ) inform%iter, inform%aug * S%findmx,           &
               inform%ngeval, inform%pjgnrm, inform%itercg, S%oldrad,          &
               inform%iskip
           END IF
           k = COUNT( X <= BL * ( one + SIGN( S%epstlp, BL ) ) .OR.            &
                      X >= BU * ( one - SIGN( S%epstln, BU ) ) )
           WRITE( S%out, "( /, ' There are ', I6, ' variables and ', I6,       &
          &  ' active bounds')" ) n, k
           IF ( S%printm ) THEN
             WRITE( S%out, 2500 ) X
             WRITE( S%out, 2510 )                                              &
               FUVALS( S%lggfx + 1 : S%lggfx + n ) * S%findmx
           END IF
           WRITE( S%out, "( /, ' Times for Cauchy, systems, products and',     &
          &   ' updates', 0P, 4F8.2 )" ) S%tca, S%tls, S%tmv, S%tup
           IF ( S%xactcp ) THEN
             WRITE( S%out, "( /, ' Exact Cauchy step computed ' )" )
           ELSE
             WRITE( S%out, "( /, ' Approximate Cauchy step computed ' )" )
           END IF
           IF ( S%slvbqp )                                                     &
             WRITE( S%out, "( ' Accuarate solution of BQP computed ' )" )
           IF ( S%mortor ) WRITE( S%out,                                       &
             "( ' More''-Toraldo projected search technique used ' )" )
           IF ( control%linear_solver ==  1 ) WRITE( S%out,                    &
             "( ' Conjugate gradients without preconditioner used ' )" )
           IF ( control%linear_solver ==  2 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with diagonal preconditioner used ' )" )
           IF ( control%linear_solver ==  3 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with user-supplied',                     &
          &     ' preconditioner used ' )" )
           IF ( control%linear_solver ==  4 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with band inverse',                      &
          &     ' preconditioner used ' )" )
           IF ( control%linear_solver ==  5 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with Munksgaards',                       &
          &     ' preconditioner used ' )" )
           IF ( control%linear_solver ==  6 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with Schnabel-Eskow ',                   &
          &     ' modified Cholesky preconditioner used ' )" )
           IF ( control%linear_solver ==  7 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with GMPS modified Cholesky',            &
          &     ' preconditioner used ' )" )
           IF ( control%linear_solver ==  8 ) WRITE( S%out,                    &
             "( ' Bandsolver preconditioned C.G.',                             &
          &     ' ( semi-bandwidth = ', I6, ' ) used ' )" ) inform%nsemib
           IF ( control%linear_solver ==  9 ) WRITE( S%out,                    &
             "( ' Conjugate gradients with Lin and More`s',                    &
          &     ' preconditioner used ( memory = ', I6, ' )' )" ) S%icfact
           IF ( control%linear_solver == 11 ) WRITE( S%out,                    &
             "( ' Exact matrix factorization used ' )" )
           IF ( control%linear_solver == 12 ) WRITE( S%out,                    &
             "( ' Modified matrix factorization used ' )" )
           IF ( S%twonrm ) THEN
             WRITE( S%out, "( ' Two-norm trust region used ' )" )
           ELSE
             IF ( S%strctr ) THEN
               WRITE( S%out, "( ' Structured infinty-norm trust region used ')")
             ELSE
               WRITE( S%out, "( ' Infinity-norm trust region used ' )" )
             END IF
           END IF
           IF ( S%nmhist > 0 ) WRITE( S%out,                                   &
             "( ' Non-monotone descent strategy ( history =', I3,              &
          &     ' ) used ' )" ) S%nmhist
           IF ( S%first_derivatives >= 1 )  WRITE( S%out,                      &
             "( ' Finite-difference approximations to',                        &
          &     ' nonlinear-element gradients used' )" )
           IF ( S%second_derivatives <= 0 ) WRITE( S%out,                      &
             "( ' Exact second derivatives used ' )" )
           IF ( S%second_derivatives == 1 ) WRITE( S%out,                      &
             "( ' B.F.G.S. approximation to second derivatives used ' )" )
           IF ( S%second_derivatives == 2 ) WRITE( S%out,                      &
             "( ' D.F.P. approximation to second derivatives used ' )" )
           IF ( S%second_derivatives == 3 ) WRITE( S%out,                      &
             "( ' P.S.B. approximation to second derivatives used ' )" )
           IF ( S%second_derivatives >= 4 ) WRITE( S%out,                      &
             "( ' S.R.1 Approximation to second derivatives used ' )" )
           IF ( S%direct ) THEN
             IF ( S%modchl ) THEN
               WRITE( S%out, "( ' No. pos. def. systems = ', I4,               &
              &  ' No. indef. systems = ', I4, /, ' Ratio ( fill-in ) = ',     &
              &  ES11.2 )" ) S%ISYS( 1 ), S%ISYS( 5 ), S%fill
             ELSE
               WRITE( S%out, "( ' Positive definite   = ', I6,                 &
              &  ' indefinite', 12X, '= ', I6, /, ' Singular consistent = ',   &
              &  I6, ' singular inconsistent = ', I6,                          &
              &  /, ' Ratio ( fill-in ) = ', ES11.2 )" ) S%ISYS( : 4 ), S%fill
             END IF
           END IF
         END IF
         GO TO 820

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!     F U N C T I O N   A N D   D E R I V A T I V E   E V A L U A T I O N S
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!  ======================
!
!   Internal evaluations
!
!  ======================

  700    CONTINUE

!  Check to see if we are still "alive"

         IF ( control%alive_unit > 0 ) THEN
           INQUIRE( FILE = control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             S%inform_status = inform%status
             inform%status = 14
             RETURN
           ELSE
             S%inform_status = inform%status
           END IF
         ELSE
           S%inform_status = inform%status
         END IF

!        WRITE( 6, "( ' internal evaluation ' )" )

!  Further problem information is required

         IF ( inform%status == - 1 .OR. inform%status == - 3 .OR.              &
              inform%status == - 7 ) THEN
           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating element functions ' )" )

!  Evaluate the element function values

           i = 0
           IF ( use_elders ) THEN
             CALL ELFUN_flexible(                                              &
                          FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE,           &
                          ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1, nel, &
                          1, ELDERS, i )
           ELSE
             CALL ELFUN ( FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE,           &
                          ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel, &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1, 1, i )
           END IF
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 1 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
         END IF

         IF ( ( inform%status == - 1 .OR. inform%status == - 6 .OR.            &
              ( inform%status == - 5 .AND. .NOT. external_el ) )               &
              .AND. S%getders ) THEN
           ifflag = 2
           IF ( S%second ) ifflag = 3

!  Evaluate the element function derivatives

           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating derivatives of element functions ' )" )
           i = 0
           IF ( use_elders ) THEN
             CALL ELFUN_flexible(                                              &
                          FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE, ISTAEV,   &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel,         &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1,      &
                          nel, ifflag, ELDERS, i )
           ELSE
             CALL ELFUN ( FUVALS, XT, EPVALU, inform%ncalcf, ITYPEE, ISTAEV,   &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, nel,         &
                          nel + 1, ISTAEV( nel + 1 ) - 1, nel + 1, nel + 1,    &
                          nel + 1, nel, lfuval, n, ISTEPA( nel + 1 ) - 1,      &
                          ifflag, i )
           END IF
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 1 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
         END IF

         IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN

!  Evaluate the group function values

           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating group functions ' )" )
           IF ( S%out > 0 .AND. S%print_level >= 100 ) WRITE( S%out,           &
             "( /, ' Group values ', /, ( 6ES12.4 ) )" ) FT( 1 : ng )
           IF ( S%p_type == 2 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 1 ) = FT( i )
             END DO
           END IF
           i = 0
           CALL GROUP ( GVALS , ng, FT, GPVALU, inform%ncalcg, ITYPEG,         &
                        ISTGPA, ICALCG, ng, ng + 1, ng, ng,                    &
                        ISTGPA( ng + 1 ) - 1, .FALSE., i )
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 2 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
           IF ( S%p_type == 2 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               Y( i ) = GVALS( i, 1 )
               GVALS( i, 1 ) = GVALS( i, 1 ) ** 2
             END DO
           END IF
           IF ( S%p_type == 2 ) THEN
!            IF ( S%out > 0  ) WRITE( S%out,                                   &
             IF ( S%out > 0 .AND. S%print_level >= 100 ) WRITE( S%out,         &
               "( /, ' Group function values ', /, ( 6ES12.4 ) )" )            &
                  GVALS( ICALCG(  1 : inform%ncalcg ) , 1 )
           END IF
         END IF

         IF ( inform%status == - 2 .OR.                                        &
              ( inform%status == - 5 .AND. .NOT. external_gr ) ) THEN

!  Evaluate the group function derivatives

           IF ( S%printd ) WRITE( S%out,                                       &
             "( /, ' Evaluating derivatives of group functions ' )" )
           IF ( S%p_type == 2 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 2 ) = one
               GVALS( i, 3 ) = zero
             END DO
           END IF
           i = 0
           CALL GROUP ( GVALS , ng, FT, GPVALU, inform%ncalcg, ITYPEG,         &
                        ISTGPA, ICALCG, ng, ng + 1, ng, ng,                    &
                        ISTGPA( ng + 1 ) - 1, .TRUE., i )
           IF ( i /= 0 ) THEN 
             IF ( inform%status == - 2 ) THEN
               inform%status = 13 ; RETURN
             ELSE
               inform%status = - 11 ; GO TO 590
             END IF
           END IF
           IF ( S%p_type == 2 ) THEN
             DO j = 1,  inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 3 ) = two * Y( i ) * GVALS( i, 3 ) +                  &
                               two * GVALS( i, 2 ) ** 2
               GVALS( i, 2 ) = two * Y( i ) * GVALS( i, 2 )
             END DO
           END IF
         END IF

!  Rejoin the iteration

         IF ( inform%status == - 5 .AND. ( external_el .OR. external_gr ) ) THEN
           GO TO 800
         ELSE IF ( S%p_type == 3 ) THEN
           GO TO 810
         ELSE  
           GO TO 20
         END IF

!  =======================================
!
!   Evaluations via reverse communication
!
!  =======================================

 800     CONTINUE

!  Check to see if we are still "alive"

         IF ( control%alive_unit > 0 ) THEN
           INQUIRE( FILE = control%alive_file, EXIST = alive )
           IF ( .NOT. alive ) THEN
             S%inform_status = - inform%status
             inform%status = 14
             RETURN
           ELSE
             S%inform_status = inform%status
           END IF
         ELSE
           S%inform_status = inform%status
         END IF

!  First, make sure that the data is correct for feasibility problems

         IF ( S%p_type == 2 ) THEN
           IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 1 ) = FT( i )
             END DO
           END IF
           IF ( inform%status == - 2 .OR.                                      &
              ( inform%status == - 5 .AND. external_gr ) ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 2 ) = one
               GVALS( i, 3 ) = zero
             END DO
           END IF
         END IF

!  Return to the user to obtain problem dependent information

         RETURN
!        IF ( inform%status <= - 1 .AND. inform%status >= - 13 ) RETURN

!  =============================
!
!   R E - E N T R Y   P O I N T
!
!  =============================

 810     CONTINUE

         IF ( inform%status == - 11 ) THEN
           IF ( S%inform_status == - 1 .OR. S%inform_status == - 2 ) THEN
             inform%status = 13 ; RETURN
           ELSE
             GO TO 590
           END IF
         END IF

!  For constrained problems:

         IF ( S%p_type == 3 ) THEN

!  Calculate problem related information

           IF ( inform%status == - 1 ) THEN

!  If there are slack variables, initialize  them to minimize the
!  infeasibility of their associated constraints

             IF ( S%itzero ) THEN
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) >= 3 ) THEN

!  Calculate the constraint value for the inequality constraints. It is
!  assumed that the slack variable occurs last in the list of variables in the
!  linear element

!  Include the contribution from the linear element

!                  ctt = SUM( A( ISTADA( ig ) : ISTADA( ig + 1 ) - 2 ) *       &
!                             X( ICNA( ISTADA( ig ) :                          &
!                                      ISTADA( ig + 1 ) - 2 ) ) ) - B( ig )
                   ctt = - B( ig )
                   DO i = ISTADA( ig ), ISTADA( ig + 1 ) - 2
                     ctt = ctt + A( i ) * X( ICNA( i ) )
                   END DO

!  Include the contributions from the nonlinear elements

!                  ctt = ctt + SUM( ESCALE( ISTADG( ig ) :                     &
!                                           ISTADG( ig + 1 ) - 1 ) *           &
!                                   FUVALS( IELING( ISTADG( ig ) :             &
!                                           ISTADG( ig + 1 ) - 1 ) ) )
                   DO i = ISTADG( ig ), ISTADG( ig + 1 ) - 1
                     ctt = ctt + ESCALE( i ) * FUVALS( IELING( i ) )
                   END DO

!  The slack variable corresponds to a less-than-or-equal-to constraint. Set
!  its value as close as possible to the constraint value

                   j = ISTADA( ig + 1 ) - 1
                   ic = ICNA( j )
                   IF ( KNDOFG( ig ) == 3 ) THEN
                     X( ic ) = MIN( MAX( BL( ic ), - ctt ), BU( ic ) )
  
!  The slack variable corresponds to a greater-than-or-equal-to constraint.
!  Set its value as close as possible to the constraint value
  
                   ELSE
                     X( ic ) = MIN( MAX( BL( ic ), ctt ), BU( ic ) )
                   END IF

!  Compute a suitable scale factor for the slack variable

                   IF ( X( ic ) > one ) THEN
                     VSCALE( ic ) = ten ** ANINT( LOG10( X( ic ) ) )
                   ELSE
                     VSCALE( ic ) = one
                   END IF
                 END IF
               END DO
               S%itzero = .FALSE.
             END IF
           END IF
           IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN

!  Record the unscaled constraint values in C

             WHERE ( GXEQX )
               C = FT
             ELSEWHERE
               C = GVALS( : , 1 )
             END WHERE

!  Print the constraint values on the first iteration

             IF ( S%m > 0 ) THEN
               IF ( inform%iter == 0 ) THEN
                 IF ( S%printi ) THEN
                   inform%obj = zero ; j = 1
                   IF ( S%printm ) WRITE( S%out, 2120 )
                   DO i = 1, ng
                     IF ( KNDOFG( i ) <= 1 ) THEN
                       IF ( i - 1 >= j .AND. S%printm )                        &
                         WRITE( S%out, 2090 ) ( GNAMES( ig ), ig,              &
                           C( ig ) * GSCALE( ig ), ig = j, i - 1 )
                       j = i + 1
                       IF ( KNDOFG( i ) == 1 )                                 &
                         inform%obj = inform%obj + C( i ) * GSCALE( i )
                     END IF
                   END DO
                   IF ( ng >= j .AND. S%printm )                               &
                     WRITE( S%out, 2090 )  ( GNAMES( ig ), ig,                 &
                       C( ig ) * GSCALE( ig ), ig = j, ng )

!  Print the objective function value on the first iteration

                   IF ( S%nobjgr > 0 ) THEN
                     WRITE( S%out, 2010 ) inform%obj * S%findmx
                   ELSE
                     WRITE( S%out, 2020 )
                   END IF
                 END IF

!  Calculate the constraint norm

                 inform%cnorm = zero
                 DO ig = 1, ng
                   IF ( KNDOFG( ig ) > 1 ) inform%cnorm =                     &
                     MAX( inform%cnorm, ABS( GSCALE( ig ) * C( ig ) ) )
                 END DO
                 IF ( S%out > 0 .AND. S%print_level == 1 ) THEN
                   IF ( inform%iter == 0 ) THEN
                     WRITE( S%out,                                             &
                  &  "( ' Constraint norm           ', ES22.14 )" ) inform%cnorm
                   ELSE
                     WRITE( S%out, 2180 ) inform%mu, S%omegak, inform%cnorm,   &
                       S%etak
                   END IF
                 END IF
               END IF

!  Calculate the terms involving the constraints for the augmented Lagrangian
!  function

               hmuinv = half / inform%mu
               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) THEN
                   yiui = GSCALE( ig ) * C( ig ) + inform%mu * Y( ig )
                   GVALS( ig, 1 ) = ( hmuinv * yiui ) * yiui
                 END IF
               END DO
             END IF
           END IF
           IF ( inform%status == - 2 .OR. inform%status == - 5 ) THEN
             IF ( S%m > 0 ) THEN

!  Calculate the derivatives of the terms involving the constraints for the
!  augmented Lagrangian function

               DO ig = 1, ng
                 IF ( KNDOFG( ig ) > 1 ) THEN
                   scaleg = GSCALE( ig )
                   IF ( GXEQX( ig ) ) THEN
                     GVALS( ig, 3 ) = scaleg * ( scaleg / inform%mu )
                     GVALS( ig, 2 ) = scaleg * ( FT( ig ) *                    &
                                    ( scaleg / inform%mu ) + Y( ig ) )
                   ELSE
                     hdash = scaleg * ( Y( ig ) + C( ig ) *                    &
                                      ( scaleg / inform%mu ) )
                     GVALS( ig, 3 ) = hdash * GVALS( ig, 3 ) +                 &
                                ( scaleg * GVALS( ig, 2 ) ) ** 2 / inform%mu
                     GVALS( ig, 2 ) = hdash * GVALS( ig, 2 )
                   END IF
                 END IF
               END DO
             END IF
           END IF

!  For feasibility problems:

         ELSE IF ( S%p_type == 2 ) THEN
           IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               Y( i ) = GVALS( i, 1 )
               GVALS( i, 1 ) = GVALS( i, 1 ) ** 2
             END DO
           END IF
           IF ( inform%status == - 2 .OR.                                      &
              ( inform%status == - 5 .AND. external_gr ) ) THEN
             DO j = 1, inform%ncalcg
               i = ICALCG( j )
               GVALS( i, 3 ) = two * Y( i ) * GVALS( i, 3 ) +                  &
                               two * GVALS( i, 2 ) ** 2
               GVALS( i, 2 ) = two * Y( i ) * GVALS( i, 2 )
             END DO
           END IF
         END IF
         GO TO 20

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    I N N E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 820   CONTINUE

!  Compute the residuals for feasibility problems

       IF ( S%p_type == 2 ) THEN
         WHERE ( GXEQX )
           C = FT
         ELSEWHERE
           C = GVALS( : , 1 )
         END WHERE
       END IF

!  Test whether the maximum allowed number of iterations has been reached

       IF ( inform%status > 3 ) RETURN

!  Print the values of the constraint functions

       IF ( S%p_type == 3 ) THEN
         IF ( S%out > 0 ) THEN
           inform%obj = zero ; j = 1
           IF ( S%printm ) WRITE( S%out, 2120 )
           DO i = 1, ng
             IF ( KNDOFG( i ) == 1 ) THEN
               IF ( i - 1 >= j .AND. S%printm )                                &
                 WRITE( S%out, 2090 ) ( GNAMES( ig ), ig,                      &
                        C( ig ) * GSCALE( ig ), ig = j, i - 1 )
               j = i + 1
               IF ( GXEQX( i ) ) THEN
                 inform%obj = inform%obj + FT( i ) * GSCALE( i )
               ELSE
                 inform%obj = inform%obj + GVALS( i, 1 ) * GSCALE( i )
               END IF
             END IF
           END DO
           IF ( ng >= j .AND. S%printm )                                       &
             WRITE( S%out, 2090 ) ( GNAMES( ig ), ig,                          &
               C( ig ) * GSCALE( ig ), ig = j, ng )

!  Print the objective function value

           IF ( S%printi ) THEN
             IF ( S%nobjgr > 0 ) THEN
               WRITE( S%out, 2010 ) inform%obj * S%findmx
             ELSE
               WRITE( S%out, 2020 )
             END IF
           END IF
         END IF
       END IF

!  Calculate the constraint norm

       IF ( S%p_type == 3 ) THEN
         S%ocnorm = inform%cnorm ; inform%cnorm = zero
         DO ig = 1, ng
           IF ( KNDOFG( ig ) > 1 ) inform%cnorm =                              &
             MAX( inform%cnorm, ABS( C( ig ) * GSCALE( ig ) ) )
         END DO
         IF ( inform%status == 1 ) GO TO 900
         IF ( S%printi )                                                       &
           WRITE( S%out, 2060 ) inform%mu, inform%pjgnrm, S%omegak,            &
                                inform%cnorm, S%etak
       ELSE
         GO TO 900
       END IF

!  Test for convergence of the outer iteration

       IF ( ( S%omegak <= control%stopg .OR. inform%pjgnrm <= control%stopg )  &
              .AND. inform%cnorm <= control%stopc ) GO TO 900

!  Test to see if the merit function has become too small

       IF ( inform%aug < control%min_aug ) THEN
         inform%status = 18
         GO TO 900
       END IF

!  Compute the ratio of successive norms of constraint violations. If this
!  ratio is not substantially decreased over NCRIT iterations, exit with the
!  warning that no feasible point can be found

       IF ( inform%cnorm > point99 * S%ocnorm ) THEN
         S%icrit = S%icrit + 1
         IF ( S%icrit >= S%ncrit ) THEN
           inform%status = 8
           IF ( S%printi ) WRITE( S%out, 2160 ) S%ncrit
           GO TO 900
         END IF
       ELSE
         S%icrit = 0
       END IF

!  Record that an approximate minimizer of the augmented Lagrangian function
!  has been found

       inform%newsol = .TRUE.
       IF ( S%printm ) WRITE( S%out, 2070 ) ( X( i ), i = 1, n )

!  Another iteration will be performed

       inform%status = - 1

!  Check to see if the constraint has been sufficiently reduced

       IF ( inform%cnorm < S%etak .AND. inform%mu <= control%mu_tol ) THEN
         IF ( S%ocnorm > tenm10 .AND. S%printm ) WRITE( S%out, 2080 )          &
           inform%cnorm / S%ocnorm, S%alphak ** S%betae

!  The constraint norm has been reduced sufficiently. Update the Lagrange
!  multiplier estimates, Y

         IF ( S%m > 0 ) THEN
           WHERE ( KNDOFG( : ng ) > 1 )                                        &
             Y( : ng ) =  Y( : ng ) + C( : ng ) *                              &
                ( GSCALE( : ng ) / inform%mu )
         END IF
         IF ( S%printi ) WRITE( S%out, 2040 )
         IF ( S%printm ) THEN
           j = 1
           DO i = 1, ng
             IF ( KNDOFG( i ) == 1 ) THEN
               IF ( i - 1 >= j ) WRITE( S%out, 2030 )                          &
                    ( Y( ig ), ig = j, i - 1 )
               j = i + 1
             END IF
           END DO
           IF ( ng >= j ) WRITE( S%out, 2030 ) Y( j : ng )
         END IF

!  Decrease the convergence tolerances

         S%alphak = MIN( inform%mu, S%gamma1 )
         S%etak   = MAX( S%eta_min, S%etak * S%alphak ** S%betae )
         S%omegak = MAX( S%omega_min, S%omegak * S%alphak ** S%betao )

!  Prepare for the next outer iteration

         IF ( S%printi ) WRITE( S%out, 2000 ) inform%mu, S%omegak, S%etak

!  Move variables which are close to their bounds onto the bound

         lgfx = ISTADH( nel + 1 ) - 1
         S%reeval = .FALSE.
         DO i = 1, n
           XT( i ) = X( i )
           IF ( X( i ) /= BL( i ) .AND.                                        &
                X( i ) - BL( i ) <= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BL( i )
           END IF
           IF ( X( i ) /= BU( i ) .AND.                                        &
                X( i ) - BU( i ) >= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BU( i )
           END IF
         END DO
       ELSE
         
!  Reduce the penalty parameter and reset the convergence tolerances
         
         inform%mu = S%tau * inform%mu
         S%alphak = MIN( inform%mu, S%gamma1 )
         S%etak   = MAX( S%eta_min, S%eta0 * S%alphak ** S%alphae )
         S%omegak = MAX( S%omega_min, S%omega0 * S%alphak ** S%alphao )

!  Prepare for the next outer iteration

         IF ( S%printi ) WRITE( S%out, 2150 )
         IF ( S%printi ) WRITE( S%out, 2000 ) inform%mu, S%omegak, S%etak

!  Move variables which are close to their bounds onto the bound

         S%reeval = .FALSE.
         lgfx = ISTADH( nel + 1 ) - 1
         DO i = 1, n
           XT( i ) = X( i )
           IF ( X( i ) /= BL( i ) .AND.                                        &
                X( i ) - BL( i ) <= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BL( i )
           END IF
           IF ( X( i ) /= BU( i ) .AND.                                        &
                X( i ) - BU( i ) >= theta * FUVALS( lgfx + i ) ) THEN
             S%reeval = .TRUE.
             XT( i ) = BU( i )
           END IF
         END DO

!  If finite-difference gradients are used, use central differences
!  whenever the penalty parameter is small

         IF ( S%first_derivatives >= 1 .AND. inform%mu < epsmch ** 0.25 )      &
           S%first_derivatives = 2
       END IF

!  See if we need to re-evaluate the problem functions

       IF ( S%reeval ) THEN
         IF ( S%printi ) WRITE( S%out, 2190 )
         inform%ngeval = inform%ngeval + 1
         IF ( S%first_derivatives >= 1 ) S%igetfd = 0
         CALL OTHERS_which_variables_changed(                                  &
                     S%unsucc, n, ng, nel, inform%ncalcf, inform%ncalcg,       &
                     ISTAEV, ISTADG, IELING, ICALCF, ICALCG, X , XT,           &
                     ISTAJC, IGCOLJ, LIST_elements, LINK_elem_uses_var )
         X = XT
         GO TO 800
       END IF
       GO TO 10

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!               E N D    O F    O U T E R    I T E R A T I O N 
!
!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

 900 CONTINUE

!  Compute the final, unweighted, multiplier estimates

     ipdgen = 0 ; epslam = control%stopc
     IF ( ( inform%status == 0 .OR. inform%status == 18 ) .AND. S%m > 0 ) THEN
       IF ( S%p_type == 3 ) THEN
         DO ig = 1, ng
           IF ( KNDOFG( ig ) >  1 ) THEN
             scaleg = GSCALE( ig )
             Y( ig ) = scaleg * ( Y( ig ) + C( ig ) * ( scaleg / inform%mu ) )
             IF ( ABS( Y( ig ) ) <= epslam ) ipdgen = ipdgen + 1
           END IF
         END DO
       ELSE
         DO ig = 1, ng
           Y( ig ) = GSCALE( ig ) * C( ig ) 
           IF ( ABS( Y( ig ) ) <= epslam ) ipdgen = ipdgen + 1
         END DO
       END IF
     END IF

     IF ( S%printi ) THEN
       lgfx = ISTADH( nel + 1 ) - 1
       ifixd = COUNT( X >= BU - S%epstol * MAX( one, ABS( BU ) ) .OR.          &
                      X <= BL + S%epstol * MAX( one, ABS( BL ) ) .OR.          &
                      BU - BL <= two * epsmch )      
       IF ( control%maxit >= 0 ) iddgen = COUNT(                               &
               ( X >= BU - S%epstol * MAX( one, ABS( BU ) ) .OR.               &
                 X <= BL + S%epstol * MAX( one, ABS( BL ) ) ) .AND.            &
                 ABS( FUVALS( lgfx + 1 : lgfx + n ) ) <= S%epsgrd )
       IF ( S%printi ) THEN
         WRITE( S%out, 2100 )
!        IF ( S%printt ) THEN ; l = n ; ELSE ; l = 2 ; END IF
         IF ( S%full_solution ) THEN ; l = n ; ELSE ; l = 2 ; END IF
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1
             ic = MIN( l, n )
           ELSE
             IF ( ic < n - l ) WRITE( S%out, 2220 )
             ir = MAX( ic + 1, n - ic + 1 )
             ic = n
           END IF
           DO i = ir, ic
             istate = 1
             IF ( X( i ) >= BU( i ) - S%epstol *                               &
                            MAX( one, ABS( BU( i ) ) ) ) istate = 3
             IF ( X( i ) <= BL( i ) + S%epstol *                               &
                            MAX( one, ABS( BL( i ) ) ) ) istate = 2
             IF ( BU( i ) - BL( i ) <= two * epsmch ) istate = 4
             IF ( control%maxit >= 0 ) THEN
               IF ( ( istate == 2 .OR. istate == 3 ) .AND.                     &
                    ABS( FUVALS( lgfx + i ) ) <= S%epsgrd ) istate = 5
             END IF
             IF ( control%maxit >= 0 ) THEN
               WRITE( S%out, 2110 ) VNAMES( i ), i, S%STATE( istate ),         &
                 X( i ), BL( i ), BU( i ), FUVALS( lgfx + i )
             ELSE
               WRITE( S%out, 2210 ) VNAMES( i ), i, S%STATE( istate ),         &
                 X( i ), BL( i ), BU( i )
             END IF
           END DO
         END DO
       END IF
     END IF

!  Compute the objective function value

     IF ( S%p_type == 3 ) THEN
       inform%obj = zero
       DO ig = 1, ng
         IF ( KNDOFG( ig ) == 1 ) THEN
           IF ( GXEQX( ig ) ) THEN
             inform%obj = inform%obj + FT( ig ) * GSCALE( ig )
           ELSE
             inform%obj = inform%obj + GVALS( ig, 1 ) * GSCALE( ig )
           END IF
         END IF
       END DO
     ELSE
       inform%obj = inform%aug
     END IF
     IF ( S%p_type >= 2 ) THEN
       IF ( S%printi )THEN
         WRITE( S%out, 2130 )
         start_p = .FALSE.
!        IF ( S%printt )THEN ; l = ng ; ELSE ; l = 2 ; END IF
         IF ( S%full_solution ) THEN ; l = ng ; ELSE ; l = 2 ; END IF
         DO j = 1, 2
           IF ( j == 1 ) THEN
             ir = 1
             ic = MIN( l, ng )
           ELSE
             IF ( ic < ng - l .AND. start_p ) WRITE( S%out, 2230 )
             ir = MAX( ic + 1, ng - ic + 1 )
             ic = ng
           END IF
           DO ig = ir, ic
             IF ( KNDOFG( ig ) > 1 ) THEN
               WRITE( S%out, 2140 )                                            &
                 GNAMES( ig ), ig, C( ig ), GSCALE( ig ), Y( ig )
               start_p = .TRUE.
             END IF
           END DO
         END DO
         IF ( S%nobjgr > 0 ) THEN
           WRITE( S%out, 2010 ) inform%obj * S%findmx
         ELSE
           WRITE( S%out, 2020 )
         END IF
         WRITE( S%out, 2170 ) n, S%m, ipdgen, ifixd, iddgen
       END IF
     END IF

     RETURN

!  Unsuccessful returns

 990 CONTINUE
     inform%status = 12
     inform%alloc_status = alloc_status
     inform%bad_alloc = bad_alloc

!  Non-executable statements

 2000  FORMAT( /, ' Penalty parameter ', ES12.4,                               &
                  ' Required projected gradient norm = ', ES12.4, /,           &
                  '                   ', 12X,                                  &
                  ' Required constraint         norm = ', ES12.4 )             
 2010  FORMAT( /, ' Objective function value  ', ES22.14 )                     
 2020  FORMAT( /, ' There is no objective function ' )                         
 2030  FORMAT( /, ' Multiplier values ', /, ( 5ES12.4 ) )                       
 2040  FORMAT( /, ' ******** Updating multiplier estimates ********** ' )      
 2060  FORMAT( /, ' Penalty parameter       = ', ES12.4, /,                    &
                  ' Projected gradient norm = ', ES12.4,                       &
                  ' Required gradient   norm = ', ES12.4, /,                   &
                  ' Constraint         norm = ', ES12.4,                       &
                  ' Required constraint norm = ', ES12.4 )                     
 2070  FORMAT( /, ' Solution   values ', /, ( 5ES12.4 ) )                      
 2080  FORMAT( /, ' ||c|| / ||c( old )|| = ', ES12.4,                          &
                  ' vs ALPHA ** betae = ', ES12.4 )                            
 2090  FORMAT( ( 4X, A10, I7, 6X, ES22.14 ) )                                  
 2100  FORMAT( /, ' Variable name Number Status     Value',                    &
                  '   Lower bound Upper bound  |  Dual value ', /,             &
                  ' ------------- ------ ------     -----',                    &
                  '   ----------- -----------  |  ----------' )                
 2110  FORMAT( 2X, A10, I7, 4X, A5, 3ES12.4, '  |', ES12.4 )                   
 2120  FORMAT( /, ' Constraint name Number        Value ' )                    
 2130  FORMAT( /, ' Constraint name Number    Value    Scale factor ',         &
                  '| Lagrange multiplier', /,                                  &
                  ' --------------- ------    -----    ----- ------ ',         &
                  '| -------------------' )                                    
 2140  FORMAT( 4X, A10, I7, 2X, 2ES12.4, '  |   ', ES12.4 )                    
 2150  FORMAT( /, ' ***********    Reducing mu    *************** ' )          
 2160  FORMAT( /, ' Constraint violations have not decreased',                 &
                  ' substantially over ', I4, ' major iterations. ', /,        &
                  ' Problem possibly infeasible, terminating run. ' )          
 2170  FORMAT( /, ' There are ', I7, ' variables in total. ', /,               &
                  ' There are ', I7, ' equality constraints. ', /,             &
                  ' Of these  ', I7, ' are primal degenerate. ', /,            &
                  ' There are ', I7, ' variables on their bounds. ', /,        &
                  ' Of these  ', I7, ' are dual degenerate. ' )                
 2180  FORMAT( /, ' Penalty parameter       = ', ES12.4, /,                    &
                  '                           ', 12X,                          &
                  ' Required gradient norm   = ', ES12.4, /,                   &
                  ' Constraint norm         = ', ES12.4,                       &
                  ' Required constraint norm = ', ES12.4 )                     
 2190  FORMAT( /, ' Using the shifted starting point. ' )                      
 2210  FORMAT( 2X, A10, I7, 4X, A5, 3ES12.4, '  |      - ' )                   
 2220  FORMAT( '  .               .    .....  ..........  ..........',         &
               '  ..........  |  ..........' )
 2230  FORMAT( '    .               .   ........... ...........',              &
               '  |    ........... ' )                                         
 2500  FORMAT( /, ' X = ', / (  6ES12.4 ) )
 2510  FORMAT( /, ' G = ', / (  6ES12.4 ) )
 2530  FORMAT( /, ' Change in X = ', / ( 6ES12.4 ) )
 2540  FORMAT( /, ' LANCELOT_solve : trust-region radius too small ' )
 2550  FORMAT( /, ' Iteration number      ', I10,                              &
                  '  Merit function value    = ',  ES19.11, /,                 &
                  ' No. derivative evals  ', I10,                              &
                  '  Projected gradient norm = ',  ES19.11, /,                 &
                  ' C.G. iterations       ', I10,                              &
                  '  Trust-region radius     = ',  ES19.11, /,                 &
                  ' Number of updates skipped  ', I5 )
 2560  FORMAT( /, ' The matrix-vector product used elements', ' marked ',      &
               I5, ' in the following list ', /, ( 20I4 ) )   
 2570  FORMAT( /, ' Iteration number      ', I10,                              &
                  '  Merit function value    = ',  ES19.11, /,                 &
                  ' No. derivative evals  ', I10,                              &
                  '  Projected gradient norm = ',  ES19.11, /,                 &
                  ' C.G. iterations       ', I10, /,                           &
                  ' Number of updates skipped  ', I5 )
!               , '  Correct active set after', I5, ' iteration( s ) ' )

!  End of subroutine LANCELOT_solve_main

     END SUBROUTINE LANCELOT_solve_main

!-*-*-*-*  L A N C E L O T -B- LANCELOT_terminate  S U B R O U T I N E -*-*-*-*

     SUBROUTINE LANCELOT_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( LANCELOT_control_type ), INTENT( IN ) :: control
     TYPE ( LANCELOT_inform_type ), INTENT( INOUT ) :: inform
 
!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: alloc_status
     LOGICAL:: alive

     inform%status = 0
     IF ( ASSOCIATED( data%GXEQX_AUG ) ) THEN
       DEALLOCATE( data%GXEQX_AUG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GXEQX_AUG'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ASSOCIATED( data%GROUP_SCALING ) ) THEN
       DEALLOCATE( data%GROUP_SCALING, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GROUP_SCALING'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ITRANS ) ) THEN
       DEALLOCATE( data%ITRANS, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ITRANS'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%LINK_elem_uses_var ) ) THEN
       DEALLOCATE( data%LINK_elem_uses_var, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%LINK_elem_uses_var'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%WTRANS ) ) THEN
       DEALLOCATE( data%WTRANS, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%WTRANS'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISYMMD ) ) THEN
       DEALLOCATE( data%ISYMMD, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISYMMD'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISWKSP ) ) THEN
       DEALLOCATE( data%ISWKSP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISWKSP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISTAJC ) ) THEN
       DEALLOCATE( data%ISTAJC, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISTAJC'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISTAGV ) ) THEN
       DEALLOCATE( data%ISTAGV, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISTAGV'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISVGRP ) ) THEN
       DEALLOCATE( data%ISVGRP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISVGRP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISLGRP ) ) THEN
       DEALLOCATE( data%ISLGRP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISLGRP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IGCOLJ ) ) THEN
       DEALLOCATE( data%IGCOLJ, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IGCOLJ'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IVALJR ) ) THEN
       DEALLOCATE( data%IVALJR, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IVALJR'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IUSED  ) ) THEN
       DEALLOCATE( data%IUSED , STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IUSED '
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ITYPER ) ) THEN
       DEALLOCATE( data%ITYPER, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ITYPER'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISSWTR ) ) THEN
       DEALLOCATE( data%ISSWTR, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISSWTR'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISSITR ) ) THEN
       DEALLOCATE( data%ISSITR, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISSITR'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISET   ) ) THEN
       DEALLOCATE( data%ISET  , STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISET  '
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISVSET ) ) THEN
       DEALLOCATE( data%ISVSET, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISVSET'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%INVSET ) ) THEN
       DEALLOCATE( data%INVSET, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%INVSET'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%LIST_elements ) ) THEN
       DEALLOCATE( data%LIST_elements, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%LIST_elements'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%ISYMMH ) ) THEN
       DEALLOCATE( data%ISYMMH, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%ISYMMH'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IW_asmbl ) ) THEN
       DEALLOCATE( data%IW_asmbl, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IW_asmbl'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%NZ_comp_w ) ) THEN
       DEALLOCATE( data%NZ_comp_w, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%NZ_comp_w'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W_ws ) ) THEN
       DEALLOCATE( data%W_ws, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W_ws'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W_el ) ) THEN
       DEALLOCATE( data%W_el, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W_el'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W_in ) ) THEN
       DEALLOCATE( data%W_in, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W_in'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%H_el ) ) THEN
       DEALLOCATE( data%H_el, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%H_el'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%H_in ) ) THEN
       DEALLOCATE( data%H_in, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%H_in'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%DIAG ) ) THEN
       DEALLOCATE( data%DIAG, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%DIAG'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%OFFDIA ) ) THEN
       DEALLOCATE( data%OFFDIA, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%OFFDIA'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IW ) ) THEN
       DEALLOCATE( data%IW, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IW'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IKEEP ) ) THEN
       DEALLOCATE( data%IKEEP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IKEEP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IW1 ) ) THEN
       DEALLOCATE( data%IW1, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IW1'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%IVUSE ) ) THEN
       DEALLOCATE( data%IVUSE, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%IVUSE'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%H_col_ptr ) ) THEN
       DEALLOCATE( data%H_col_ptr, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%H_col_ptr'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%L_col_ptr ) ) THEN
       DEALLOCATE( data%L_col_ptr, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%L_col_ptr'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W ) ) THEN
       DEALLOCATE( data%W, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%W1 ) ) THEN
       DEALLOCATE( data%W1, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%W1'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%RHS ) ) THEN
       DEALLOCATE( data%RHS, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%RHS'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%RHS2 ) ) THEN
       DEALLOCATE( data%RHS2, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%RHS2'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%XCP ) ) THEN
       DEALLOCATE( data%XCP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%XCP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

      IF ( ALLOCATED( data%X0 ) ) THEN
        DEALLOCATE( data%X0, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%X0'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%BND ) ) THEN
        DEALLOCATE( data%BND, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%BND'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%BND_radius ) ) THEN
        DEALLOCATE( data%BND_radius, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%BND_radius'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%BREAKP ) ) THEN
        DEALLOCATE( data%BREAKP, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%BREAKP'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%DELTAX ) ) THEN
        DEALLOCATE( data%DELTAX, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%DELTAX'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%GRAD ) ) THEN
        DEALLOCATE( data%GRAD, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%GRAD'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%GRJAC ) ) THEN
        DEALLOCATE( data%GRJAC, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%GRJAC'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%INDEX ) ) THEN
        DEALLOCATE( data%INDEX, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%INDEX'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%QGRAD ) ) THEN
        DEALLOCATE( data%QGRAD, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%QGRAD'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%IFREE ) ) THEN
        DEALLOCATE( data%IFREE, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%IFREE'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%RADII ) ) THEN
        DEALLOCATE( data%RADII, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%RADII'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%GX0 ) ) THEN
        DEALLOCATE( data%GX0, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%GX0'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%IFREEC ) ) THEN
        DEALLOCATE( data%IFREEC, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%IFREEC'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%SCU_matrix%BD_row ) ) THEN
        DEALLOCATE( data%SCU_matrix%BD_row, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%SCU_matrix%BD_row'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%SCU_matrix%BD_val ) ) THEN
        DEALLOCATE( data%SCU_matrix%BD_val, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%SCU_matrix%BD_val'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%SCU_matrix%BD_col_start ) ) THEN
        DEALLOCATE( data%SCU_matrix%BD_col_start, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%SCU_matrix%BD_col_start'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%FUVALS_temp ) ) THEN
        DEALLOCATE( data%FUVALS_temp, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%FUVALS_temp'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%INNONZ ) ) THEN
        DEALLOCATE( data%INNONZ, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%INNONZ'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
      IF ( ALLOCATED( data%GV_old ) ) THEN
        DEALLOCATE( data%GV_old, STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN
          inform%status = 12
          inform%alloc_status = alloc_status
          inform%bad_alloc = 'data%GV_old'
          WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
        END IF
      END IF
 
     IF ( ALLOCATED( data%P ) ) THEN
       DEALLOCATE( data%P, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%P'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%P2 ) ) THEN
       DEALLOCATE( data%P2, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%P2'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%G ) ) THEN
       DEALLOCATE( data%G, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%G'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%BREAKP ) ) THEN
       DEALLOCATE( data%BREAKP, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%BREAKP'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%GRAD ) ) THEN
       DEALLOCATE( data%GRAD, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%GRAD'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%LINK_col ) ) THEN
       DEALLOCATE( data%LINK_col, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%LINK_col'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%POS_in_H ) ) THEN
       DEALLOCATE( data%POS_in_H, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%POS_in_H'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%matrix%row ) ) THEN
       DEALLOCATE( data%matrix%row, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%matrix%row'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%matrix%col ) ) THEN
       DEALLOCATE( data%matrix%col, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%matrix%col'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     IF ( ALLOCATED( data%matrix%val ) ) THEN
       DEALLOCATE( data%matrix%val, STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN
         inform%status = 12
         inform%alloc_status = alloc_status
         inform%bad_alloc = 'data%matrix%val'
         WRITE( data%S%error, 2990 ) alloc_status, inform%bad_alloc
       END IF
     END IF

     CALL SCU_terminate( data%SCU_data, inform%status, inform%SCU_info )
     CALL SILS_finalize( data%SILS_data, control%SILS_cntl, inform%status )

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

!  Non-executable statement

 2990  FORMAT( ' ** Message from -LANCELOT_terminate-', /,                     &
               ' Deallocation error (status = ', I6, ') for ', A24 )

!  End of subroutine LANCELOT_terminate

     END SUBROUTINE LANCELOT_terminate

!-*-*-  L A N C E L O T  -B-  LANCELOT_form_gradients  S U B R O U T I N E -*-*

     SUBROUTINE LANCELOT_form_gradients(                                       &
                       n , ng, nel   , ntotel, nvrels, nnza  , nvargp,         &
                       firstg, ICNA  , ISTADA, IELING, ISTADG, ISTAEV,         &
                       IELVAR, INTVAR, A     , GVALS2, GUVALS, lguval,         &
                       GRAD  , GSCALE, ESCALE, GRJAC , GXEQX , INTREP,         &
                       ISVGRP, ISTAGV, ITYPEE, ISTAJC, GRAD_el, W_el ,         &
                       RANGE , KNDOFG )

!  Calculate the the gradient, GRAD, of the objective function and the
!  Jacobian matrix of gradients, GRJAC, of each group

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN    ) :: n , ng, nel   , ntotel, nnza, nvargp
     INTEGER, INTENT( IN    ) :: nvrels, lguval
     LOGICAL, INTENT( IN    ) :: firstg
     INTEGER, INTENT( IN    ), DIMENSION( ng  + 1 ) :: ISTADA, ISTADG
     INTEGER, INTENT( IN    ), DIMENSION( nel + 1 ) :: ISTAEV, INTVAR
     INTEGER, INTENT( IN    ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN    ), DIMENSION( nnza    ) :: ICNA
     INTEGER, INTENT( IN    ), DIMENSION( ntotel  ) :: IELING
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( nnza ) :: A
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GVALS2
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( lguval ) :: GUVALS
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GRAD
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( nvargp ) :: GRJAC
     LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: ISTAJC
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: GRAD_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig, ii, k, ig1, j, jj, l , ll
     INTEGER :: nin   , nvarel, nelow , nelup, istrgv, iendgv
     REAL ( KIND = wp ) :: gi, scalee
     LOGICAL :: nontrv

!  Initialize the gradient as zero

     GRAD = zero

!  Consider the IG-th group

     DO ig = 1, ng
       IF ( PRESENT( KNDOFG ) ) THEN
         IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
       ig1 = ig + 1
       istrgv = ISTAGV( ig ) ; iendgv = ISTAGV( ig1 ) - 1
       nelow = ISTADG( ig ) ; nelup = ISTADG( ig1 ) - 1
       nontrv = .NOT. GXEQX( ig )

!  Compute the first derivative of the group

       IF ( nontrv ) THEN
         gi = GSCALE( ig ) * GVALS2( ig )
       ELSE
         gi = GSCALE( ig )
       END IF

!  This is the first gradient evaluation or the group has nonlinear elements

       IF ( firstg .OR. nelow <= nelup ) THEN
         GRAD_el( ISVGRP( istrgv : iendgv ) ) = zero

!  Loop over the group's nonlinear elements

         DO ii = nelow, nelup
           iel = IELING( ii )
           k = INTVAR( iel ) ; l = ISTAEV( iel )
           nvarel = ISTAEV( iel + 1 ) - l
           scalee = ESCALE( ii )
           IF ( INTREP( iel ) ) THEN

!  The IEL-th element has an internal representation

             nin = INTVAR( iel + 1 ) - k
             CALL RANGE ( iel, .TRUE., GUVALS( k : k + nin - 1 ),              &
                          W_el( : nvarel ), nvarel, nin, ITYPEE( iel ),        &
                          nin, nvarel )
!DIR$ IVDEP
             DO i = 1, nvarel
               j = IELVAR( l )
               GRAD_el( j ) = GRAD_el( j ) + scalee * W_el( i )
               l = l + 1
             END DO
           ELSE

!  The IEL-th element has no internal representation

!DIR$ IVDEP
             DO i = 1, nvarel
               j = IELVAR( l )
               GRAD_el( j ) = GRAD_el( j ) + scalee * GUVALS( k )
               k = k + 1
               l = l + 1
             END DO
           END IF
         END DO

!  Include the contribution from the linear element

!DIR$ IVDEP
         DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
           GRAD_el( ICNA( k ) ) = GRAD_el( ICNA( k ) ) + A( k )
         END DO

!  Find the gradient of the group

         IF ( nontrv ) THEN

!  The group is non-trivial

!DIR$ IVDEP
           DO i = istrgv, iendgv
             ll = ISVGRP( i )
             GRAD( ll ) = GRAD( ll ) + gi * GRAD_el( ll )

!  As the group is non-trivial, also store the nonzero entries of the
!  gradient of the function in GRJAC

             jj = ISTAJC( ll )
             GRJAC( jj ) = GRAD_el( ll )

!  Increment the address for the next nonzero in the column of
!  the Jacobian for variable LL

             ISTAJC( ll ) = jj + 1
           END DO
         ELSE

!  The group is trivial

!DIR$ IVDEP
           DO i = istrgv, iendgv
             ll = ISVGRP( i )
             GRAD( ll ) = GRAD( ll ) + gi * GRAD_el( ll )
           END DO
         END IF

!  This is not the first gradient evaluation and there is only a linear element

       ELSE

!  Add the gradient of the linear element to the overall gradient

!DIR$ IVDEP
         DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
           GRAD( ICNA( k ) ) = GRAD( ICNA( k ) ) + gi * A( k )
         END DO

!  The group is non-trivial; increment the starting addresses for
!  the groups used by each variable in the (unchanged) linear
!  element to avoid resetting the nonzeros in the Jacobian

         IF ( nontrv ) THEN
!DIR$ IVDEP
           DO i = istrgv, iendgv
             ISTAJC( ISVGRP( i ) ) = ISTAJC( ISVGRP( i ) ) + 1
           END DO
         END IF
       END IF
     END DO

!  Reset the starting addresses for the lists of groups using each variable to
!  their values on entry

     DO i = n, 2, - 1
       ISTAJC( i ) = ISTAJC( i - 1 )
     END DO
     ISTAJC( 1 ) = 1

     RETURN

!  End of subroutine LANCELOT_form_gradients

     END SUBROUTINE LANCELOT_form_gradients

!-*-  L A N C E L O T  -B-  LANCELOT_projected_gradient  S U B R O U T I N E -*-

     SUBROUTINE LANCELOT_projected_gradient( n , X , G   , XSCALE, BL, BU,     &
                                             GRAD  , ivar, nvar  , pjgnrm )

!  Compute the projection of the gradient into the feasible box and its norm

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) ::  n
     INTEGER, INTENT( OUT ) ::  nvar
     REAL ( KIND = wp ), INTENT( OUT ) :: pjgnrm
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: ivar
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: X, G, XSCALE
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: BL, BU
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: GRAD

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: gi, epsmch

     epsmch = EPSILON( one )

     nvar = 0
     pjgnrm = zero
     DO i = 1, n
       gi = G( i ) * XSCALE( i )
       IF ( gi == zero ) CYCLE

!  Compute the projection of the gradient within the box

       IF ( gi < zero ) THEN
         gi = - MIN( ABS( BU( i ) - X( i ) ), - gi )
       ELSE
         gi = MIN( ABS( BL( i ) - X( i ) ), gi )
       END IF

!  Record the nonzero components of the Cauchy direction in GRAD

       IF ( ABS( gi ) > epsmch ) THEN
         nvar = nvar + 1
         pjgnrm = MAX( pjgnrm, ABS( gi ) )
         ivar( nvar ) = i
         GRAD( nvar ) = gi
       END IF
     END DO

     RETURN

!  End of LANCELOT_projected_gradient

     END SUBROUTINE LANCELOT_projected_gradient

!-*-*-*- L A N C E L O T -B- LANCELOT_norm_proj_grad S U B R O U T I N E -*-*-*-

     SUBROUTINE LANCELOT_norm_proj_grad(                                       &
                        n , ng, nel   , ntotel, nvrels,                        &
                        nvargp, nvar  , smallh, pjgnrm, calcdi, dprcnd,        &
                        myprec, IVAR  , ISTADH, ISTAEV, IELVAR, INTVAR,        &
                        IELING, DGRAD , Q     , GVALS2, GVALS3, DIAG  ,        &
                        GSCALE, ESCALE, GRJAC , HUVALS, lnhuvl, qgnorm,        &
                        GXEQX , INTREP, ISYMMD, ISYMMH, ISTAGV, ISLGRP,        &
                        ISVGRP, IVALJR, ITYPEE, W_el  , W_in  , H_in  ,        &
                        RANGE , KNDOFG )

!  Find the norm of the projected gradient, scaled if desired.
!  If required, also find the diagonal elements of the Hessian matrix

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n , ng, nel   , ntotel, nvar
     INTEGER, INTENT( IN ) :: nvrels, nvargp, lnhuvl
     REAL ( KIND = wp ), INTENT( IN    ) :: smallh, pjgnrm
     REAL ( KIND = wp ), INTENT( OUT   ) :: qgnorm
     LOGICAL, INTENT( IN ) :: calcdi, dprcnd, myprec
     INTEGER, INTENT( IN ), DIMENSION( nvar    ) :: IVAR
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH, ISTAEV, INTVAR
     INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( ntotel  ) :: IELING
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Q
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvar ) :: DGRAD
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS2, GVALS3, GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvargp ) :: GRJAC
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lnhuvl ) :: HUVALS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: DIAG
     LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISYMMD
     INTEGER, INTENT( IN ), DIMENSION( : , : ) :: ISYMMH
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISLGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IVALJR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_in

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig , j , irow  , ijhess, k , kk, ll
     INTEGER :: iell  , nin    , nvarel, jcol  , ielhst
     REAL ( KIND = wp ) :: gdash, g2dash, temp
     
     IF ( myprec ) THEN
!      qgnorm = SQRT( DOT_PRODUCT( DGRAD( : nvar ), Q( IVAR( : nvar ) ) ) )
       qgnorm = zero
       DO i = 1, nvar
         qgnorm = qgnorm + DGRAD( i ) * Q( IVAR( i ) )
       END DO
       qgnorm = SQRT( qgnorm )
     ELSE
       IF ( calcdi ) THEN

!  Obtain the diagonal elements of the Hessian.
!  Initialize the diagonals as zero

         DIAG = zero ; W_el = zero

!  Obtain the contributions from the second derivatives of the elements

         DO iell = 1, ntotel
           iel = IELING( iell )
           ig = ISLGRP( iell )
           IF ( PRESENT( KNDOFG ) ) THEN
             IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
           nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
           ll = ISTAEV( iel )
           IF ( GXEQX( ig ) ) THEN
             gdash = ESCALE( iell ) * GSCALE( ig )
           ELSE
             gdash = ESCALE( iell ) * GSCALE( ig ) * GVALS2( ig )
           END IF
           IF ( INTREP( iel ) ) THEN
             nin = INTVAR( iel + 1 ) - INTVAR( iel )
             DO kk = 1, nvarel

!  The IEL-th element Hessian has an internal representation.
!  Set W_el as the KK-th column of the identity matrix

               W_el( kk ) = one

!  Gather W_el into its internal variables, W_in

               CALL RANGE ( iel, .FALSE., W_el, W_in, nvarel, nin,             &
                            ITYPEE( iel ), nvarel, nin )
               W_el( kk ) = zero

!  Multiply the internal variables by the element Hessian.
!  Consider the first column of the Hessian

               ielhst = ISTADH( iel )
               H_in( : nin ) = W_in( 1 ) * HUVALS( ISYMMH( 1, : nin ) + ielhst )

!  Now consider the remaining columns of the Hessian

               DO jcol = 2, nin
                 H_in( : nin ) = H_in( : nin ) + W_in( jcol ) *                &
                   HUVALS( ISYMMH( jcol, : nin ) + ielhst )
               END DO

!  Add the KK-th diagonal of the IEL-th element Hessian

               j = IELVAR( ll )
               ll = ll + 1
!              DIAG( j ) =                                                     &
!                DIAG( j ) + gdash * DOT_PRODUCT( W_in( : nin ), H_in( : nin ) )
               temp = zero
               DO i = 1, nin
                  temp = temp + W_in( i ) * H_in( i )
               END DO
               DIAG( j ) = DIAG( j ) + gdash * temp
             END DO
           ELSE

!  The IEL-th element Hessian has no internal representation

             ielhst = ISTADH( iel )
!DIR$ IVDEP
             DO irow = 1, nvarel
               ijhess = ISYMMD( irow ) + ielhst
               j = IELVAR( ll )
               ll = ll + 1
               DIAG( j ) = DIAG( j ) + gdash * HUVALS( ijhess )
             END DO
           END IF
         END DO

!  If the group is non-trivial, add on rank-one first order terms

         DO ig = 1, ng
           IF ( PRESENT( KNDOFG ) ) THEN
             IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
           IF ( .NOT. GXEQX( ig ) ) THEN
             g2dash = GSCALE( ig ) * GVALS3( ig )
!DIR$ IVDEP
             DO k = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
               DIAG( ISVGRP( k ) ) = DIAG( ISVGRP( k ) ) + g2dash *            &
                 GRJAC( IVALJR( k ) ) ** 2
             END DO
           END IF
         END DO

!  Take the absolute values of all the diagonal entries, ensuring that all
!  entries are larger than the tolerance SMALLH

         DIAG( : n ) = MAX( smallh, ABS( DIAG( : n ) ) )
       END IF

!  Use the diagonals to calculate a scaled norm of the gradient

       IF ( dprcnd ) THEN
!        qgnorm = SQRT( DOT_PRODUCT( DGRAD( : nvar ),                          &
!          ( DGRAD( : nvar ) / DIAG( IVAR( : nvar ) ) ) ) )
         qgnorm = zero
         DO i = 1, nvar
           qgnorm = qgnorm + ( DGRAD( i ) ** 2 ) /  DIAG( IVAR( i ) )
         END DO
         qgnorm = SQRT( qgnorm )
       ELSE
         qgnorm = pjgnrm
       END IF
     END IF

     RETURN

!  End of subroutine LANCELOT_norm_proj_grad

     END SUBROUTINE LANCELOT_norm_proj_grad

!-*-*-*-*  L A N C E L O T  -B-   LANCELOT_norm_diff   F U N C T I O N -*-*-*-*

     FUNCTION LANCELOT_norm_diff( n, X, Y, twonrm, RSCALE, scaled )
     REAL ( KIND = wp ) :: LANCELOT_norm_diff

!  Compute the scaled (or unscaled) two (or infinity) norm distance
!  between the vectors X and Y

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n
     LOGICAL, INTENT( IN ) :: twonrm, scaled
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, Y
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: RSCALE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i

     IF ( scaled ) THEN

!  Compute the scaled two-norm distance between X and Y

       IF ( twonrm ) THEN
!        LANCELOT_norm_diff = SQRT( SUM( ( ( X - Y ) / RSCALE ) ** 2 ) )
         LANCELOT_norm_diff = zero
         DO i = 1, n
           LANCELOT_norm_diff =                                                &
             LANCELOT_norm_diff + ( ( X( i ) - Y( i ) ) / RSCALE( i ) ) ** 2 
         END DO
         LANCELOT_norm_diff = SQRT( LANCELOT_norm_diff )

!  Compute the scaled infinity-norm distance between X and Y

       ELSE
         LANCELOT_norm_diff = MAXVAL( ABS( ( X - Y ) / RSCALE ) )
       END IF
     ELSE

!  Compute the two-norm distance between X and Y

       IF ( twonrm ) THEN
!        LANCELOT_norm_diff = SQRT( SUM( ( X - Y ) ** 2 ) )
         LANCELOT_norm_diff = zero
         DO i = 1, n
            LANCELOT_norm_diff = LANCELOT_norm_diff + ( X( i ) - Y( i ) ) ** 2
         END DO
         LANCELOT_norm_diff = SQRT( LANCELOT_norm_diff )

!  Compute the infinity-norm distance between X and Y

       ELSE
         LANCELOT_norm_diff = MAXVAL( ABS( X - Y ) )
       END IF
     END IF
     RETURN

!  End of function LANCELOT_norm_diff

     END FUNCTION LANCELOT_norm_diff

!  End of module LANCELOT

   END MODULE LANCELOT_double























