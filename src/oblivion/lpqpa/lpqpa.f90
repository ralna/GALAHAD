! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L P Q P A   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started August 12th 2002
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LPQPA_double

!      --------------------------------------------------
!     |                                                  |
!     | Solve the l_p quadratic program                  |
!     |                                                  |
!     |    minimize     1/2 x(T) H x + g(T) x + f        |
!     |     + rho || max( 0, c_l - A x, A x - c_u ) ||_p |
!     |    subject to     x_l <=  x  <= x_u              |
!     |                                                  |
!     | using a working-set approach                     |
!     |                                                  |
!      --------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_SYMBOLS
      USE GALAHAD_QPT_double
      USE GALAHAD_QPA_double
      USE GALAHAD_LSQP_double
      USE GALAHAD_LPQP_double
      USE GALAHAD_SPECFILE_double 
  
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LPQPA_initialize, LPQPA_read_specfile, LPQPA_solve,            &
                LPQPA_terminate, QPT_problem_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: LPQPA_control_type
        INTEGER :: out, error, print_level
        TYPE ( QPA_control_type ) :: QPA_control
        TYPE ( LPQP_control_type ) :: LPQP_control
      END TYPE

      TYPE, PUBLIC :: LPQPA_inform_type
        INTEGER :: status
        TYPE ( QPA_inform_type ) :: QPA_inform
        TYPE ( LPQP_inform_type ) :: LPQP_inform
      END TYPE

      TYPE, PUBLIC :: LPQPA_data_type
        TYPE ( QPA_data_type ) :: QPA_data
        TYPE ( LPQP_data_type ) :: LPQP_data
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L P Q P A _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*

      SUBROUTINE LPQPA_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LPQPA. This routine should be called before
!  LPQPA_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. Components are -
!
!    LPQP_control control parameter for LPQP (see LPQP for details)
!    QPA_control control parameter for QPA (see QPA for details)
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LPQPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPQPA_control_type ), INTENT( OUT ) :: control        
      TYPE ( LPQPA_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set control parameters

      control%out = 6
      control%error = 6
      control%print_level = 0

!  Set control parameters for LPQP

!     CALL LPQP_initialize( data%LPQP_data, control%LPQP_control )
      CALL LPQP_initialize( control%LPQP_control )

!  Set control parameters for QPA

      CALL QPA_initialize( data%QPA_data, control%QPA_control,                 &
                           inform%QPA_inform )

      RETURN  

!  End of LPQPA_initialize

      END SUBROUTINE LPQPA_initialize

!-*-*-*-   L P Q P A _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE LPQPA_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LPQPA_initialize could (roughly) 
!  have been set as:

!  BEGIN LPQPA SPECIFICATIONS (DEFAULT)
!    error-printout-device                             6
!    printout-device                                   6
!    print-level                                       0
!    maximum-number-of-iterations                      1000
!    start-print                                       -1
!    stop-print                                        -1
!    factorization-used                                0
!    maximum-column-nonzeros-in-schur-complement       35
!    maximum-dimension-of-schur-complement             75
!    initial-integer-workspace                         1000
!    initial-real-workspace                            1000
!    maximum-refinements                               1
!    maximum-infeasible-iterations-before-rho-increase 100
!    maximum-number-of-cg-iterations                   -1
!    preconditioner-used                               0
!    semi-bandwidth-for-band-preconditioner            5
!    full-max-fill-ratio                               10
!    deletion-strategy                                 0
!    restore-problem-on-output                         2
!    residual-monitor-interval                         1
!    cold-start-strategy                               3
!    infinity-value                                    1.0D+19
!    feasibility-tolerance                             1.0D-12
!    minimum-objective-before-unbounded                -1.0D+32
!    increase-rho-g-factor                             2.0
!    increase-rho-b-factor                             2.0
!    infeasible-g-required-improvement-factor          0.75
!    infeasible-b-required-improvement-factor          0.75
!    pivot-tolerance-used                              1.0D-8
!    pivot-tolerance-used-for-dependencies             0.5
!    zero-pivot-tolerance                              1.0D-12
!    inner-iteration-relative-accuracy-required        0.0
!    inner-iteration-absolute-accuracy-required        1.0E-8
!    treat-zero-bounds-as-general                      F
!    solve-qp                                          F
!    solve-within-bounds                               F
!    temporarily-perturb-constraint-bounds             T
!    array-syntax-worse-than-do-loop                   F
!  END LPQPA SPECIFICATIONS

!  Dummy arguments

      TYPE ( LPQPA_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 38
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'LPQPA'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level' 
      spec(  4 )%keyword = 'maximum-number-of-iterations'
      spec(  5 )%keyword = 'start-print'
      spec(  6 )%keyword = 'stop-print'
      spec(  7 )%keyword = 'factorization-used'
      spec(  8 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec(  9 )%keyword = 'maximum-dimension-of-schur-complement'
      spec( 10 )%keyword = 'initial-integer-workspace'
      spec( 11 )%keyword = 'initial-real-workspace'
      spec( 12 )%keyword = 'maximum-refinements'
      spec( 13 )%keyword = 'maximum-infeasible-iterations-before-rho-increase'
      spec( 14 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 15 )%keyword = 'preconditioner-used'
      spec( 16 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 17 )%keyword = 'full-max-fill-ratio'
      spec( 18 )%keyword = 'deletion-strategy'
      spec( 19 )%keyword = 'restore-problem-on-output'
      spec( 20 )%keyword = 'residual-monitor-interval'
      spec( 21 )%keyword = 'cold-start-strategy'

!  Real key-words

      spec( 22 )%keyword = 'infinity-value'
      spec( 23 )%keyword = 'feasibility-tolerance'
      spec( 24 )%keyword = 'minimum-objective-before-unbounded'
      spec( 25 )%keyword = 'increase-rho-g-factor'
      spec( 26 )%keyword = 'increase-rho-b-factor'
      spec( 27 )%keyword = 'infeasible-g-required-improvement-factor'
      spec( 28 )%keyword = 'infeasible-b-required-improvement-factor'
      spec( 29 )%keyword = 'pivot-tolerance-used'
      spec( 30 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 31 )%keyword = 'zero-pivot-tolerance'
      spec( 32 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 33 )%keyword = 'inner-iteration-absolute-accuracy-required'

!  Logical key-words

      spec( 34 )%keyword = 'treat-zero-bounds-as-general'
      spec( 35 )%keyword = 'solve-qp'
      spec( 36 )%keyword = 'solve-within-bounds'
      spec( 37 )%keyword = 'temporarily-perturb-constraint-bounds'
      spec( 38 )%keyword = 'array-syntax-worse-than-do-loop'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec,                 &
                            control%QPA_control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec,                     &
                            control%QPA_control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_integer( spec( 1 ), control%error, control%error )
      CALL SPECFILE_assign_integer( spec( 1 ), control%QPA_control%error,      &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 1 ), control%LPQP_control%error,     &
                                    control%LPQP_control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%out, control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%QPA_control%out,        &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%LPQP_control%out,       &
                                    control%LPQP_control%error )
      CALL SPECFILE_assign_integer( spec( 3 ), control%print_level,            &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 3 ),                                 &
                                    control%QPA_control%print_level,           &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 3 ),                                 &
                                    control%LPQP_control%print_level,          &
                                    control%LPQP_control%error )
      CALL SPECFILE_assign_integer( spec( 4 ), control%QPA_control%maxit,      &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 5 ),                                 &
                                    control%QPA_control%start_print,           &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 6 ), control%QPA_control%stop_print, &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 7 ), control%QPA_control%factor,     &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 8 ), control%QPA_control%max_col,    &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 9 ), control%QPA_control%max_sc,     &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 10 ), control%QPA_control%indmin,    &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 11 ), control%QPA_control%valmin,    &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 12 ), control%QPA_control%itref_max, &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 13 ),                                &
                                    control%QPA_control%infeas_check_interval, &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 14 ), control%QPA_control%cg_maxit,  &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 15 ), control%QPA_control%precon,    &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 16 ), control%QPA_control%nsemib,    &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 17 ),                                &
                                    control%QPA_control%full_max_fill,         &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 18 ),                                &
                                    control%QPA_control%deletion_strategy,     &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 19 ),                                &
                                    control%QPA_control%restore_problem,       &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 20 ),                                &
                                    control%QPA_control%monitor_residuals,     &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_integer( spec( 21 ), control%QPA_control%cold_start,&
                                    control%QPA_control%error )

!  Set real values


      CALL SPECFILE_assign_real( spec( 22 ), control%QPA_control%infinity,     &
                                 control%QPA_control%error )
      CALL SPECFILE_assign_real( spec( 22 ), control%LPQP_control%infinity,    &
                                 control%LPQP_control%error )
      CALL SPECFILE_assign_real( spec( 23 ), control%QPA_control%feas_tol,     &
                                 control%QPA_control%error )     
      CALL SPECFILE_assign_real( spec( 24 ), control%QPA_control%obj_unbounded,&
                                 control%QPA_control%error )
      CALL SPECFILE_assign_real( spec( 25 ),                                   &
                                 control%QPA_control%increase_rho_g_factor,    &
                                 control%QPA_control%error )     
      CALL SPECFILE_assign_real( spec( 26 ),                                   &
                                 control%QPA_control%increase_rho_g_factor,    &
                                 control%QPA_control%error )     
      CALL SPECFILE_assign_real( spec( 27 ),                                   &
                              control%QPA_control%infeas_g_improved_by_factor, &
                                 control%QPA_control%error )     
      CALL SPECFILE_assign_real( spec( 28 ),                                   &
                              control%QPA_control%infeas_b_improved_by_factor, &
                                 control%QPA_control%error )     


      CALL SPECFILE_assign_real( spec( 29 ), control%QPA_control%pivot_tol,    &
                                 control%QPA_control%error )
      CALL SPECFILE_assign_real( spec( 30 ),                                   &
                               control%QPA_control%pivot_tol_for_dependencies, &
                                 control%QPA_control%error )
      CALL SPECFILE_assign_real( spec( 31 ), control%QPA_control%zero_pivot,   &
                                 control%QPA_control%error )

      CALL SPECFILE_assign_real( spec( 32 ),                                   &
                                 control%QPA_control%inner_stop_relative,      &
                                 control%QPA_control%error )
      CALL SPECFILE_assign_real( spec( 33 ),                                   &
                                 control%QPA_control%inner_stop_absolute,      &
                                 control%QPA_control%error )

!  Set logical values

      CALL SPECFILE_assign_logical( spec( 34 ),                                &
                             control%QPA_control%treat_zero_bounds_as_general, &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_logical( spec( 35 ), control%QPA_control%solve_qp,  &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_logical( spec( 36 ),                                &
                                    control%QPA_control%solve_within_bounds,   &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_logical( spec( 37 ), control%QPA_control%randomize, &
                                    control%QPA_control%error )
      CALL SPECFILE_assign_logical( spec( 38 ),                                &
                          control%QPA_control%array_syntax_worse_than_do_loop, &
                                    control%QPA_control%error )

!  Read the specfiles for QPA and LPQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL QPA_read_specfile( control%QPA_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPA' )
        CALL LPQP_read_specfile( control%LPQP_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-LPQP' )
      ELSE
        CALL QPA_read_specfile( control%QPA_control, device )
        CALL LPQP_read_specfile( control%LPQP_control, device )
      END IF

      RETURN

      END SUBROUTINE LPQPA_read_specfile

!-*-*-*-*-*-*-*-   L P Q P A _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LPQPA_solve( prob, rho, one_norm, C_stat, B_stat,             &
                              data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Given the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric matrix, 
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, solve the related l_p QP problem
!
!      minimize     1/2 x(T) H x + g(T) x + f       
!                     + rho || max( 0, c_l - A x, A x - c_u ) ||_p
!
!     subject to     x_l <=  x  <= x_u
!
!  using an active-set method.
!  The subroutine is particularly appropriate when A and H are sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold 
!   information about the problem on input, and its solution on output.
!   See QPA for details.
!
!  rho is a REAL variable that holds the required value of the penalty 
!   parameter for the l_p qp.
!
!  one-norm is a LOGICAL variable that is true if the l_1 norm is to be
!   used and false if the l_infinity norm is to be used.
!
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to QPA_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; C_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   C_stat( i ) < 0, the i-th constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to QPA_solve to indicate which of the simple bound constraints 
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; B_stat
!   need not be set if control%cold_start is nonzero. On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are 
!   B_stat( i ) < 0, the i-th bound constraint is in the working set, 
!                    on its lower bound, 
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  control is a structure of type LPQPA_control_type that contains
!   control parameters. The components are
!
!     QPA_control, a structure of type QPA_control_type. See QPA for details
!     LPQP_control, a structure of type LPQP_control_type. See LPQP for details
!
!  inform is a structure of type LPQPA_inform_type that provides 
!    information on exit from LPQBP_formulate. The components are
!
!     QPA_inform, a structure of type QPA_inform_type. See QPA for details
!     LPQP_inform, a structure of type LPQP_inform_type. See LPQP for details
!
!  data is a structure of type LPQPA_data_type which holds private internal data
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      REAL ( KIND = wp ), INTENT( IN ) :: rho
      LOGICAL, INTENT( IN ) :: one_norm
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat, C_stat
      TYPE ( LPQPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPQPA_control_type ), INTENT( INOUT ) :: control
      TYPE ( LPQPA_inform_type ), INTENT( OUT ) :: inform

      inform%status = 0

!  Convert the QP problem into an l_p QP

      CALL LPQP_formulate( prob, rho, one_norm, data%LPQP_data,                &
                           control%LPQP_control, inform%LPQP_inform,           &
                           B_stat = B_stat, C_stat = C_stat,                   &
                           cold = control%QPA_control%cold_start )

      IF ( inform%LPQP_inform%status /= 0 ) THEN
        inform%status = - 1
        IF ( control%error > 0 .AND. control%print_level >= 0 )                &
          WRITE( control%error, "( ' On exit from LPQP_formulate, status = ',  &
         &  I6 )" ) inform%LPQP_inform%status
        RETURN
      END IF

!  Solve the problem

      prob%rho_g = ten ** 4
      prob%rho_b = ten ** 6
      control%QPA_control%solve_qp = .TRUE.

      CALL QPA_solve( prob, C_stat, B_stat, data%QPA_data,                     &
                      control%QPA_control, inform%QPA_inform )

      IF ( inform%QPA_inform%status /=   0 .AND.                               &
           inform%QPA_inform%status /= - 8 ) THEN
        inform%status = - 2
        IF ( control%error > 0 .AND. control%print_level >= 0 )                &
          WRITE( control%error, "( ' On exit from QPA_solve status = ',        &
         &  I6 )" ) inform%QPA_inform%status
        RETURN
      END IF

!  Restore the problem

      CALL LPQP_restore( prob, data%LPQP_data )

!  End of LPQPA_solve

      END SUBROUTINE LPQPA_solve

!-*-*-*-*-*-*-   L P Q P A _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LPQPA_terminate( data, control, inform )

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
!   data    see Subroutine LPQPA_initialize
!   control see Subroutine LPQPA_initialize
!   inform  see Subroutine LPQPA_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LPQPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPQPA_control_type ), INTENT( IN ) :: control        
      TYPE ( LPQPA_inform_type ), INTENT( INOUT ) :: inform

!  Deallocate components for QPA

      CALL QPA_terminate( data%QPA_data, control%QPA_control,                  &
                          inform%QPA_inform )

!  Deallocate components for LPQP

      CALL LPQP_terminate( data%LPQP_data, control%LPQP_control,               &
                           inform%LPQP_inform )

      RETURN

!  End of subroutine LPQPA_terminate

      END SUBROUTINE LPQPA_terminate

!  End of module LPQPA

   END MODULE GALAHAD_LPQPA_double
