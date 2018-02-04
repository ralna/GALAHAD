! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L P Q P B   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started August 5th 2002
!   originally released GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_LPQPB_double

!      --------------------------------------------------
!     |                                                  |
!     | Solve the l_p quadratic program                  |
!     |                                                  |
!     |    minimize     1/2 x(T) H x + g(T) x + f        |
!     |     + rho || max( 0, c_l - A x, A x - c_u ) ||_p |
!     |    subject to     x_l <=  x  <= x_u              |
!     |                                                  |
!     | using an interior-point trust-region approach    |
!     |                                                  |
!      --------------------------------------------------

      USE GALAHAD_SYMBOLS
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_QPT_double
      USE GALAHAD_QPB_double
      USE GALAHAD_LSQP_double
      USE GALAHAD_LPQP_double
      USE GALAHAD_SPECFILE_double 
  
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LPQPB_initialize, LPQPB_read_specfile, LPQPB_solve,            &
                LPQPB_terminate, QPT_problem_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: LPQPB_control_type
        INTEGER :: out, error, print_level
        LOGICAL :: reformulate
        TYPE ( QPB_control_type ) :: QPB_control
        TYPE ( LPQP_control_type ) :: LPQP_control
      END TYPE

      TYPE, PUBLIC :: LPQPB_inform_type
        INTEGER :: status
        TYPE ( QPB_inform_type ) :: QPB_inform
        TYPE ( LPQP_inform_type ) :: LPQP_inform
      END TYPE

      TYPE, PUBLIC :: LPQPB_data_type
        TYPE ( QPB_data_type ) :: QPB_data
        TYPE ( LPQP_data_type ) :: LPQP_data
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L P Q P B _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*

      SUBROUTINE LPQPB_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LPQPB. This routine should be called before
!  LPQPB_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. Components are -
!
!    LPQP_control control parameter for LPQP (see LPQP for details)
!    QPB_control control parameter for QPB (see QPB for details)
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LPQPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPQPB_control_type ), INTENT( OUT ) :: control        
      TYPE ( LPQPB_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set control parameters

      control%out = 6
      control%error = 6
      control%print_level = 0

!  Set control parameters for LPQP

!     CALL LPQP_initialize( data%LPQP_data, control%LPQP_control )
      CALL LPQP_initialize( control%LPQP_control )

!  Set control parameters for QPB

      CALL QPB_initialize( data%QPB_data, control%QPB_control,                 &
                           inform%QPB_inform )


!  reformulate should be set true if the problem needs to be reformulated 
!  as an l_p QP, and false if it is already the right format

      control%reformulate = .TRUE.

      RETURN  

!  End of LPQPB_initialize

      END SUBROUTINE LPQPB_initialize

!-*-*-*-   L P Q P B _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE LPQPB_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LPQPB_initialize could (roughly) 
!  have been set as:

!  BEGIN LPQPB SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   maximum-number-of-iterations                    1000
!   start-print                                     -1
!   stop-print                                      -1
!   factorization-used                              0
!   maximum-column-nonzeros-in-schur-complement     35
!   initial-integer-workspace                       1000
!   initial-real-workspace                          1000
!   maximum-refinements                             1
!   maximum-poor-iterations-before-infeasible       200
!   maximum-number-of-cg-iterations                 200
!   preconditioner-used                             0
!   semi-bandwidth-for-band-preconditioner          5
!   restore-problem-on-output                       0
!   infinity-value                                  1.0D+19
!   primal-accuracy-required                        1.0D-5
!   dual-accuracy-required                          1.0D-5
!   complementary-slackness-accuracy-required       1.0D-5
!   mininum-initial-primal-feasibility              1.0
!   mininum-initial-dual-feasibility                1.0
!   initial-barrier-parameter                       -1.0
!   poor-iteration-tolerance                        0.98
!   minimum-objective-before-unbounded              -1.0D+32
!   pivot-tolerance-used                            1.0D-12
!   pivot-tolerance-used-for-dependencies           0.5
!   zero-pivot-tolerance                            1.0D-12
!   initial-trust-region-radius                     -1.0
!   inner-iteration-fraction-optimality-required    0.1
!   inner-iteration-relative-accuracy-required      0.01
!   inner-iteration-absolute-accuracy-required      1.0E-8
!   reformulate-as-lp-qp                            T
!   remove-linear-dependencies                      T
!   treat-zero-bounds-as-general                    F
!   start-at-analytic-center                        T
!   primal-barrier-used                             F
!   move-final-solution-onto-bound                  F
!   array-syntax-worse-than-do-loop                 F
!   remove-original-problem-data                    F
!  END LPQPB SPECIFICATIONS

!  Dummy arguments

      TYPE ( LPQPB_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 39
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'LPQPB'
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
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 12 )%keyword = 'maximum-poor-iterations-before-infeasible'
      spec( 13 )%keyword = 'maximum-number-of-cg-iterations'
      spec( 14 )%keyword = 'preconditioner-used'
      spec( 15 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 16 )%keyword = 'restore-problem-on-output'

!  Real key-words

      spec( 17 )%keyword = 'infinity-value'
      spec( 18 )%keyword = 'primal-accuracy-required'
      spec( 19 )%keyword = 'dual-accuracy-required'
      spec( 20 )%keyword = 'complementary-slackness-accuracy-required'
      spec( 21 )%keyword = 'mininum-initial-primal-feasibility'
      spec( 22 )%keyword = 'mininum-initial-dual-feasibility'
      spec( 23 )%keyword = 'initial-barrier-parameter'
      spec( 24 )%keyword = 'poor-iteration-tolerance'
      spec( 25 )%keyword = 'minimum-objective-before-unbounded'
      spec( 26 )%keyword = 'pivot-tolerance-used'
      spec( 27 )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( 28 )%keyword = 'zero-pivot-tolerance'
      spec( 29 )%keyword = 'initial-trust-region-radius'
      spec( 30 )%keyword = 'inner-iteration-fraction-optimality-required'
      spec( 31 )%keyword = 'inner-iteration-relative-accuracy-required'
      spec( 32 )%keyword = 'inner-iteration-absolute-accuracy-required'

!  Logical key-words

      spec( 33 )%keyword = 'remove-linear-dependencies'
      spec( 34 )%keyword = 'treat-zero-bounds-as-general'
      spec( 35 )%keyword = 'start-at-analytic-center'
      spec( 36 )%keyword = 'primal-barrier-used'
      spec( 37 )%keyword = 'move-final-solution-onto-bound'
      spec( 38 )%keyword = 'array-syntax-worse-than-do-loop'
      spec( 39 )%keyword = 'reformulate-as-lp-qp'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec,                 &
                            control%QPB_control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec,                     &
                            control%QPB_control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_integer( spec( 1 ), control%error, control%error )
      CALL SPECFILE_assign_integer( spec( 1 ), control%QPB_control%error,      &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 1 ), control%LPQP_control%error,     &
                                    control%LPQP_control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%out, control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%QPB_control%out,        &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 2 ), control%LPQP_control%out,       &
                                    control%LPQP_control%error )
      CALL SPECFILE_assign_integer( spec( 3 ), control%print_level,            &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( 3 ), control%QPB_control%print_level,&
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 3 ),                                 &
                                    control%LPQP_control%print_level,          &
                                    control%LPQP_control%error )
      CALL SPECFILE_assign_integer( spec( 4 ), control%QPB_control%maxit,      &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 5 ),                                 &
                                    control%QPB_control%start_print,           &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 6 ), control%QPB_control%stop_print, &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 7 ), control%QPB_control%factor,     &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 8 ), control%QPB_control%max_col,    &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 9 ), control%QPB_control%indmin,     &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 10 ), control%QPB_control%valmin,    &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 11 ), control%QPB_control%itref_max, &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 12 ), control%QPB_control%infeas_max,&
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 13 ), control%QPB_control%cg_maxit,  &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 14 ), control%QPB_control%precon,    &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 15 ), control%QPB_control%nsemib,    &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_integer( spec( 16 ),                                &
                                    control%QPB_control%restore_problem,       &
                                    control%QPB_control%error )

!  Set real values

      CALL SPECFILE_assign_real( spec( 17 ), control%QPB_control%infinity,     &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 17 ), control%LPQP_control%infinity,    &
                                 control%LPQP_control%error )
      CALL SPECFILE_assign_real( spec( 18 ), control%QPB_control%stop_p,       &
                                 control%QPB_control%error )     
      CALL SPECFILE_assign_real( spec( 19 ), control%QPB_control%stop_d,       &
                                 control%QPB_control%error )     
      CALL SPECFILE_assign_real( spec( 20 ), control%QPB_control%stop_c,       &
                                 control%QPB_control%error )     
      CALL SPECFILE_assign_real( spec( 21 ), control%QPB_control%prfeas,       &
                                 control%QPB_control%error )     
      CALL SPECFILE_assign_real( spec( 22 ), control%QPB_control%dufeas,       &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 23 ), control%QPB_control%muzero,       &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 24 ), control%QPB_control%reduce_infeas,&
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 25 ), control%QPB_control%obj_unbounded,&
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 26 ), control%QPB_control%pivot_tol,    &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 27 ),                                   &
                               control%QPB_control%pivot_tol_for_dependencies, &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 28 ), control%QPB_control%zero_pivot,   &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 29 ),                                   &
                                 control%QPB_control%initial_radius,           &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 30 ),                                   &
                                 control%QPB_control%inner_fraction_opt,       &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 31 ),                                   &
                                 control%QPB_control%inner_stop_relative,      &
                                 control%QPB_control%error )
      CALL SPECFILE_assign_real( spec( 32 ),                                   &
                                 control%QPB_control%inner_stop_absolute,      &
                                 control%QPB_control%error )

!  Set logical values

      CALL SPECFILE_assign_logical( spec( 33 ),                                &
                                    control%QPB_control%remove_dependencies,   &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_logical( spec( 34 ),                                &
                              control%QPB_control%treat_zero_bounds_as_general,&
                                    control%QPB_control%error )
      CALL SPECFILE_assign_logical( spec( 35 ), control%QPB_control%center,    &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_logical( spec( 36 ), control%QPB_control%primal,    &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_logical( spec( 38 ),                                &
                          control%QPB_control%array_syntax_worse_than_do_loop, &
                                    control%QPB_control%error )
      CALL SPECFILE_assign_logical( spec( 39 ), control%reformulate,           &
                                    control%error )

!  Read the specfiles for QPB and LPQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL QPB_read_specfile( control%QPB_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPB' )
        CALL LPQP_read_specfile( control%LPQP_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-LPQP' )
      ELSE
        CALL QPB_read_specfile( control%QPB_control, device )
        CALL LPQP_read_specfile( control%LPQP_control, device )
      END IF

      RETURN

      END SUBROUTINE LPQPB_read_specfile

!-*-*-*-*-*-*-*-   L P Q P B _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE LPQPB_solve( prob, rho, one_norm, data, control, inform )

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
!  using a primal-dual method.
!  The subroutine is particularly appropriate when A and H are sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold 
!   information about the problem on input, and its solution on output.
!   See QPB for details.
!
!  rho is a REAL variable that holds the required value of the penalty 
!   parameter for the l_p qp.
!
!  one-norm is a LOGICAL variable that is true if the l_1 norm is to be
!   used and false if the l_infinity norm is to be used.
!
!  control is a structure of type LPQPB_control_type that contains
!   control parameters. The components are
!
!     QPB_control, a structure of type QPB_control_type. See QPB for details
!     LPQP_control, a structure of type LPQP_control_type. See LPQP for details
!
!  inform is a structure of type LPQPB_inform_type that provides 
!    information on exit from LPQBP_formulate. The components are
!
!     QPB_inform, a structure of type QPB_inform_type. See QPB for details
!     LPQP_inform, a structure of type LPQP_inform_type. See LPQP for details
!
!  data is a structure of type LPQPB_data_type which holds private internal data
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      REAL ( KIND = wp ), INTENT( IN ) :: rho
      LOGICAL, INTENT( IN ) :: one_norm
      TYPE ( LPQPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPQPB_control_type ), INTENT( INOUT ) :: control
      TYPE ( LPQPB_inform_type ), INTENT( OUT ) :: inform

      inform%status = 0

!  Convert the QP problem into an l_p QP

!      IF ( control%reformulate ) THEN
        CALL LPQP_formulate( prob, rho, one_norm, data%LPQP_data,              &
                             control%LPQP_control, inform%LPQP_inform )

        IF ( inform%LPQP_inform%status /= 0 ) THEN
          inform%status = - 1
          IF ( control%error > 0 .AND. control%print_level >= 0 )              &
            WRITE( control%error, "( ' On exit from LPQP_formulate, status = ',&
           &  I6 )" ) inform%LPQP_inform%status
          RETURN
          END IF
!        END IF

!  Solve the problem

      CALL QPB_solve( prob, data%QPB_data, control%QPB_control,                &
                      inform%QPB_inform )
      inform%status = inform%QPB_inform%status
      IF ( inform%QPB_inform%status /= GALAHAD_ok .AND.                        &
           inform%QPB_inform%status /= GALAHAD_error_max_iterations .AND.      &
           inform%QPB_inform%status /= GALAHAD_error_ill_conditioned .AND.     &
           inform%QPB_inform%status /= GALAHAD_error_tiny_step ) THEN
        IF ( control%error > 0 .AND. control%print_level >= 0 )                &
          WRITE( control%error, "( ' On exit from QPB_solve status = ',        &
         &  I6 )" ) inform%QPB_inform%status
      END IF

!  Restore the problem

!     IF ( control%reformulate )                                               &
      CALL LPQP_restore( prob, data%LPQP_data )

!  End of LPQPB_solve

      END SUBROUTINE LPQPB_solve

!-*-*-*-*-*-*-   L P Q P B _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LPQPB_terminate( data, control, inform )

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
!   data    see Subroutine LPQPB_initialize
!   control see Subroutine LPQPB_initialize
!   inform  see Subroutine LPQPB_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LPQPB_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPQPB_control_type ), INTENT( IN ) :: control        
      TYPE ( LPQPB_inform_type ), INTENT( INOUT ) :: inform

!  Deallocate components for QPB

      CALL QPB_terminate( data%QPB_data, control%QPB_control,                  &
                          inform%QPB_inform )

!  Deallocate components for LPQP

      CALL LPQP_terminate( data%LPQP_data, control%LPQP_control,               &
                           inform%LPQP_inform )

      RETURN

!  End of subroutine LPQPB_terminate

      END SUBROUTINE LPQPB_terminate

!  End of module LPQPB

   END MODULE GALAHAD_LPQPB_double
