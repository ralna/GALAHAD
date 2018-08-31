! THIS VERSION: GALAHAD 3.1 - 31/08/2018 AT 07:40 GMT.

!-*--*-*-  G A L A H A D _ L A N C E L O T _ S I M P L E    M O D U L E  -*-*-*-

   MODULE LANCELOT_simple_double
      PRIVATE
      PUBLIC :: LANCELOT_simple
   CONTAINS
!===============================================================================
!
       SUBROUTINE LANCELOT_simple( n, X, MY_FUN, fx, exit_code,                &
                                   MY_GRAD, MY_HESS,                           &
                                   BL, BU, VNAMES, CNAMES, neq, nin, CX, Y,    &
                                   iters, maxit, gradtol, feastol, print_level )
!
!-------------------------------------------------------------------------------
!                                                                              !
!      PURPOSE:                                                                !
!      ========                                                                !
!                                                                              !
!      A simple and somewhat NAIVE interface to LANCELOT B for solving the     !
!      nonlinear optimization problem                                          !
!                                                                              !
!          minimize f( x )                                                     !
!              x                                                               !
!                                                                              !
!      possibly subject to constraints of the one or more of the forms         !
!                                                                              !
!          bl <=  x <= bu,                                                     !
!                                                                              !
!          c_e( x )  = 0,                                                      !
!                                                                              !
!          c_i( x ) <= 0,                                                      !
!                                                                              !
!      where f: R^n --> R, c_e: R^n --> R^neq and c_i: R^n --> R^nin are       !
!      twice-continuously differentiable functions.                            !
!                                                                              !
!                                                                              !
!      WHY NAIVE?                                                              !
!      ==========                                                              !
!                                                                              !
!      At variance with more elaborate interfaces for LANCELOT, the            !
!      present one completely IGNORES UNDERLYING PARTIAL SEPARABILITY OR       !
!      SPARSITY STRUCTURE, RESTRICTS THE POSSIBLE FORMS UNDER WHICH THE        !
!      PROBLEM MAY BE PRESENTED TO THE SOLVER, and drastically LIMITS THE      !
!      RANGE OF AVAILABLE ALGORITHMIC OPTIONS. If simpler to use than its      !
!      more elaborate counterparts, it therefore provides a possibly           !
!      substantially inferior numerical performance, especially for            !
!      difficult/large problems, where structure exploitation and/or careful   !
!      selection of algorithmic variants matter. Thus, be warned that          !
!      --------------------------------------------------------------------    !
!      !  THE BEST PERFORMANCE OBTAINABLE WITH LANCELOT B IS PROBABLY NOT !    !
!      !                   WITH THE PRESENT INTERFACE.                    !    !
!      --------------------------------------------------------------------    !
!                                                                              !
!                                                                              !
!      HOW TO USE IT?                                                          !
!      ==============                                                          !
!                                                                              !
!         USE LANCELOT_simple_double                                           !
!                                                                              !
!      and then                                                                !
!                                                                              !
!         CALL LANCELOT_simple( n, X, MY_FUN, fx, exit_code  [,MY_GRAD]      & !
!                             [,MY_HESS] [,BL] [,BU] [,VNAMES] [,CNAMES]     & !
!                             [,neq] [,nin] [,CX] [,Y] [,iters] [,maxit]     & !
!                             [,gradtol] [,feastol] [,print_level]           ) !
!                                                                              !
!                                                                              !
!      1) -----------   Unconstrained problems   ----------------------------- !
!                                                                              !
!                                                                              !
!      The user should provide, at the very minimum, suitable values for the   !
!      following input arguments:                                              !
!                                                                              !
!      n ( integer) : the number of variables,                                 !
!                                                                              !
!      X ( double precision vector of size n ): the starting point for the     !
!                    minimization                                              !
!                                                                              !
!      and a subroutine for computing the objective function value for any X,  !
!      whose interface has the default form                                    !
!                                                                              !
!       MY_FUN( X, fx )                                                        !
!                                                                              !
!      where X(1:n) contains the values of the variables on input, and where   !
!      fx is a double precision scalar returning the value f(X). The actual    !
!      subroutine corresponding to the variable MY_FUN should be declared as   !
!      external.                                                               !
!                                                                              !
!      If the gradient of f can be computed, then the (optional) input         !
!      argument MY_GRAD must be specified and given the name of the            !
!      user-supplied routine computing the gradient, whose interface must be   !
!      of the form                                                             !
!                                                                              !
!       GRADPROB( X, G )                                                       !
!                                                                              !
!      where G is a double precision vector of size n in which the subroutine  !
!      returns the value of the gradient of f at X. The calling sequence to    !
!      LANCELOT_simple must thus contain (in this case), MY_GRAD = GRADPROB,   !
!      the LANCELOT_simple interface bloc and a declaration of GRADPROB as     !
!      external.                                                               !
!                                                                              !
!      If, additionally, the second-derivative matrix of f at X can be         !
!      computed, the (optional) input argument MY_HESS must be                 !
!      specified and given the name of the user-supplied routine computing     !
!      the Hessian, whose interface must be of the form                        !
!                                                                              !
!       HESSPROB( X, H )                                                       !
!                                                                              !
!      where H is a double precision vector of size n*(n+1)/2 in which the     !
!      subroutine returns the entries of the upper triangular part of the      !
!      Hessian of f at X, stored by columns. The calling sequence to           !
!      LANCELOT_simple must thus contain (in this case), MY_HESS = HESSPROB,   !
!      the LANCELOT_simple interface bloc and a declaration of HESSPROB as     !
!      external.                                                               !
!                                                                              !
!      The names of the problem variables may be specified in the (optional)   !
!      input argument                                                          !
!                                                                              !
!      VNAMES ( a size n vector of character fields of length 10 ),            !
!                                                                              !
!      by inserting VNAMES = mynames in the calling sequence (and as for all   !
!      keyword specified arguments, inserting the interface bloc).             !
!                                                                              !
!      In all cases, the best value of X found by LANCELOT B is returned to    !
!      the user in the vector X and the associated objective function value in !
!      the double precision output argument fx.                                !
!                                                                              !
!      The (optional) integer output argument iters reports the number of      !
!      iterations performed by LANCELOT before exiting. Finally, the integer   !
!      output argument exit_code  contains the exit status of the LANCELOT     !
!      run, the value 0 indicating a successful run. Other values indicate     !
!      errors in the input or unsuccessful runs, and are detailed in the       !
!      specsheet of LANCELOT B (with the exception of the value 19, which      !
!      reports a negative value for one or both input arguments NIN and NEQ).  !
!                                                                              !
!      Example                                                                 !
!                                                                              !
!      Let us consider the optimization problem                                !
!                                                                              !
!         minimize   f( x1, x2 ) = 100 * ( x2 - x1**2 )**2 + ( 1 - x1 )**2,    !
!          x1, x2                                                              !
!                                                                              !
!      which is the ever-famous Rosenbrock "banana" problem.                   !
!      The most basic way to solve the problem (but NOT the most efficient)    !
!      is, assuming the starting point X = (/ -1.2d0, 1.0d0 /) known, to       !
!      perform the call                                                        !
!                                                                              !
!       CALL LANCELOT_simple( 2, X, FUN, fx, exit_code )                       !
!                                                                              !
!      where the user-provided subroutine FUN is given by                      !
!                                                                              !
!       SUBROUTINE FUN( X, F )                                                 !
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                              !
!       REAL( KIND = wp ), INTENT( IN )   :: X( : )                            !
!       REAL( KIND = wp ), INTENT( OUT )  :: F                                 !
!       F = 100.0_wp*(X(2)-X(1)**2)**2 +(1.0_wp-X(1))**2                       !
!       RETURN                                                                 !
!       END SUBROUTINE FUN                                                     !
!                                                                              !
!      After compiling and linking (with the GALAHAD modules and library),     !
!      the solution is returned in 60 iterations (with exit_code = 0).         !
!                                                                              !
!      If we now wish to use first and second derivatives of the objective     !
!      function, one should use the call                                       !
!                                                                              !
!       CALL LANCELOT_simple( 2, X, FUN, fx, exit_code,                      & !
!                             MY_GRAD = GRAD, MY_HESS = HESS )                 !
!                                                                              !
!      provide the additional routines                                         !
!                                                                              !
!       SUBROUTINE GRAD( X, G )                                                !
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                              !
!       REAL( KIND = wp ), INTENT( IN )  :: X( : )                             !
!       REAL( KIND = wp ), INTENT( OUT ) :: G( : )                             !
!       G( 1 ) = -400.0_wp*(X(2)-X(1)**2)*X(1)-2.0_wp*(1.0_wp-X(1))            !
!       G( 2 ) =  200.0_wp*(X(2)-X(1)**2)                                      !
!       RETURN                                                                 !
!       END SUBROUTINE GRAD                                                    !
!                                                                              !
!      and                                                                     !
!                                                                              !
!       SUBROUTINE HESS( X, H )                                                !
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                              !
!       REAL( KIND = wp ), INTENT( IN )  :: X( : )                             !
!       REAL( KIND = wp ), INTENT( OUT ) :: H( : )                             !
!       H( 1 ) = -400.0_wp*(X(2)-3.0_wp*X(1)**2)+2.0_wp                        !
!       H( 2 ) = -400.0_wp*X(1)                                                !
!       H( 3 ) =  200.0_wp                                                     !
!       RETURN                                                                 !
!       END SUBROUTINE HESS                                                    !
!                                                                              !
!      and declare them external in the calling program.                       !
!                                                                              !
!      Convergence is then obtained in 23 iterations. Note that using exact    !
!      first-derivatives only is also possible: MY_HESS should then be absent  !
!      from the calling sequence and providing the subroutine HESS unnecessary.!
!                                                                              !
!                                                                              !
!      2) -----------   Bound constrained problems   ------------------------- !
!                                                                              !
!                                                                              !
!      Bound on the problem variables may be imposed by specifying one or      !
!      both of                                                                 !
!                                                                              !
!      BL (double precision vector of size n): the lower bounds on X,          !
!                                                                              !
!      BU (double precision vector of size n): the upper bounds on X.          !
!                                                                              !
!      Note that infinite bounds (represented by a number larger than 10**20   !
!      in absolute value) are acceptable, as well as equal lower and upper     !
!      bounds, which amounts to fixing the corresponding variables. Except for !
!      the specification of BL and/or BU, the interface is identical to that   !
!      for unconstrained problems (1).                                         !
!                                                                              !
!      Example                                                                 !
!                                                                              !
!      If one now wishes to impose zero upper bounds on the variables of our   !
!      unconstrained problem and give names to the variables and the           !
!      objective function, one could use the following call                    !
!                                                                              !
!       VNAMES(1) = 'x1'                                                       !
!       VNAMES(2) = 'x2'                                                       !
!       CALL LANCELOT_simple( 2, X, FUN, fx, exit_code,                      & !
!                        MY_GRAD = GRAD, MY_HESS = HESS,                     & !
!                        BU = (/ 0.0d0, 0.0d0/), VNAMES = VNAMES         )     !
!                                                                              !
!      in which case convergence is obtained in 6 iterations.                  !
!                                                                              !
!                                                                              !
!      3) -----------   Equality constrained problems   ---------------------- !
!                                                                              !
!                                                                              !
!      If, additionally, general equality constraints are also present in the  !
!      problem, this must be declared by specifying the following (optional)   !
!      input arguments                                                         !
!                                                                              !
!      neq (integer): the number of equality constraints.                      !
!                                                                              !
!      In this case, the equality constraints are numbered from 1 to neq       !
!      and the value of the i-th equality constraint must be computed by a     !
!      user-supplied routine of the form                                       !
!                                                                              !
!       FUN( X, fx, i )                     ( i = 1,...,neq )                  !
!                                                                              !
!      where the fx now returns the value of the i-th equality constraint      !
!      evaluated at X if i is specified. (This extension of the unconstrained  !
!      case is best implemented by adding an optional argument i to the        !
!      unconstrained version of FUN.) If derivatives are available, then       !
!      the GRAD and HESS subroutines must be adapted as well                   !
!                                                                              !
!       GRAD( X, G, i )   HESS( X, H, i )   ( i = 1,...,neq )                  !
!                                                                              !
!      for computing the gradient and Hessian of the i-th constraint at X.     !
!      The constraints may be given an individual name (a string of length 10) !
!      which is specified in                                                   !
!                                                                              !
!      CNAMES ( a size neq vector of character fields of length 10 ).          !
!                                                                              !
!      Note that, if the gradient of the objective function is available, so   !
!      must be the gradients of the equality constraints. The same level of    !
!      derivative availability is assumed for all problem functions (objective !
!      and constraints). The final values of the constraints and the values of !
!      their associated Lagrange multipliers is optionally returned to the     !
!      user in the (optional) double precision output arguments CX and Y,      !
!      respectively (both being of size neq).                                  !
!                                                                              !
!                                                                              !
!      4) -----------   Inequality constrained problems   -------------------- !
!                                                                              !
!                                                                              !
!      If inequality constraints are present in the problem, their inclusion   !
!      is similar to that of equality constraints.  One then needs to specify  !
!      the (optional) input argument                                           !
!                                                                              !
!      nin (integer): the number of inequality constraints.                    !
!                                                                              !
!      The inequality constraints are then numbered from neq+1 to neq+nin      !
!      and their values or that of their derivatives is again computing by     !
!      calling, for i = 1,..., nin,                                            !
!                                                                              !
!       FUN( X, fx, i )  GRAD( X, G, i )  HESS( X, H, i )                      !
!                                                                              !
!      The inequality constraints are internally converted in equality ones    !
!      by the addition of a slack variables, whose names are set to 'Slack_i', !
!      where the character i in this string takes the integers values 1  to    !
!      nin.) As for in the equality constrained case, the components (1:nin)   !
!      of the vector CNAMES may now contain the names of the nin inequality    !
!      constraints. The values of the inequality constraints at the final X    !
!      are finally returned (as for equalities) in the optional double         !
!      precision output argument CX of size nin. The values of the Lagrange    !
!      multipliers are returned in the optional double precision output        !
!      argument Y of size nin.                                                 !
!                                                                              !
!                                                                              !
!      5) -------   Problems with equality and inequality constraints -------- !
!                                                                              !
!                                                                              !
!      If they are both equalities and inequalities, neq and nin must be       !
!      specified and the values and derivatives of the constraints are         !
!      computed by                                                             !
!                                                                              !
!       FUN( X, fx, i )  GRAD( X, G, i )  HESS( X, H, i )  (i=1,...,neq)       !
!                                                                              !
!      for the equality constraints, and                                       !
!                                                                              !
!       FUN( X, fx, i )  GRAD( X, G, i )  HESS( X, H, i ) (i=neq+1,neq+nin)    !
!                                                                              !
!      for the inequality constraints. As above, the components (1:neq)        !
!      of the vector CNAMES may now contain the names of the neq equality      !
!      constraints and the following (neq+1:neq+nin) ones the names of the     !
!      nin inequality constraints. Again, the same level of derivative         !
!      availability is assumed for all problem functions (objective and        !
!      constraints). Finally, the optional arguments CX and/or Y, if used,     !
!      are then of size neq+nin.                                               !
!                                                                              !
!      Example                                                                 !
!                                                                              !
!      If we now wish the add to the unconstrained version the new constraints !
!                                                                              !
!                          0  <= x1                                            !
!                                                                              !
!          x1 + 3 * x2   - 3   = 0                                             !
!                                                                              !
!          x1**2 + x2**2 - 4  <= 0,                                            !
!                                                                              !
!      we may transform our call to                                            !
!                                                                              !
!       CALL LANCELOT_simple( 2, X, FUN, fx, exit_code,
!                            MY_GRAD = GRAD, MY_HESS = HESS,                 & !
!                            BL = (/0.0D0, -1.0D20/),                        & !
!                            NEQ = 1, NIN = 1, CX =  cx, Y = y )             & !
!                                                                              !
!      (assuming we need cx and y), and modify the FUN, GRAD and HESS          !
!      functions as follows                                                    !
!                                                                              !
!       SUBROUTINE FUN ( X, F, i )                                             !
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                              !
!       REAL( KIND = wp ), INTENT( IN )   :: X( : )                            !
!       REAL( KIND = wp ), INTENT( OUT )  :: F                                 !
!       INTEGER, INTENT( IN ), OPTIONAL   :: i                                 !
!       IF ( .NOT. PRESENT( i ) ) THEN ! objective function                    !
!          F = 100.0_wp*(X(2)-X(1)**2)**2 +(1.0_wp-X(1))**2                    !
!       ELSE                                                                   !
!          SELECT CASE ( i )                                                   !
!          CASE ( 1 )   ! the equality constraint                              !
!              F = X(1)+3.0_wp*X(2)-3.0_wp                                     !
!          CASE ( 2 )   ! the inequality constraint                            !
!              F = X(1)**2+X(2)**2-4.0_wp                                      !
!          END SELECT                                                          !
!       END IF                                                                 !
!       RETURN                                                                 !
!       END SUBROUTINE FUN                                                     !
!                                                                              !
!       SUBROUTINE GRAD( X, G, i )                                             !
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                              !
!       REAL( KIND = wp ), INTENT( IN )  :: X( : )                             !
!       REAL( KIND = wp ), INTENT( OUT ) :: G( : )                             !
!       INTEGER, INTENT( IN ), OPTIONAL  :: i                                  !
!       IF ( .NOT. PRESENT( i ) ) THEN !objective function's grad              !
!           G( 1 ) = -400.0_wp*(X(2)-X(1)**2)*X(1)-2.0_wp*(1.0_wp-X(1))        !
!           G( 2 ) =  200.0_wp*(X(2)-X(1)**2)                                  !
!       ELSE                                                                   !
!          SELECT CASE ( i )                                                   !
!          CASE ( 1 )    !  equality constraint's gradient components          !
!              G( 1 ) =  1.0_wp                                                !
!              G( 2 ) =  3.0_wp                                                !
!          CASE ( 2 )    !  inequality constraint's gradient components        !
!              G( 1 ) =  2.0_wp*X(1)                                           !
!              G( 2 ) =  2.0_wp*X(2)                                           !
!          END SELECT                                                          !
!       END IF                                                                 !
!       RETURN                                                                 !
!       END SUBROUTINE GRAD                                                    !
!                                                                              !
!       SUBROUTINE HESS( X, H, i )                                             !
!       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )                              !
!       REAL( KIND = wp ), INTENT( IN )  :: X( : )                             !
!       REAL( KIND = wp ), INTENT( OUT ) :: H( : )                             !
!       INTEGER, INTENT( IN ), OPTIONAL  :: i                                  !
!       IF ( .NOT. PRESENT( i ) ) THEN ! objective's Hessian                   !
!          H( 1 ) = -400.0_wp*(X(2)-3.0_wp*X(1)**2)+2.0_wp                     !
!          H( 2 ) = -400.0_wp*X(1)                                             !
!          H( 3 ) =  200.0_wp                                                  !
!       ELSE                                                                   !
!          SELECT CASE ( i )                                                   !
!          CASE ( 1 ) ! equality constraint's Hessian                          !
!              H( 1 ) = 0.0_wp                                                 !
!              H( 2 ) = 0.0_wp                                                 !
!              H( 3 ) = 0.0_wp                                                 !
!          CASE ( 2 ) ! inequality constraint's Hessian                        !
!              H( 1 ) = 2.0_wp                                                 !
!              H( 2 ) = 0.0_wp                                                 !
!              H( 3 ) = 2.0_wp                                                 !
!          END SELECT                                                          !
!       END IF                                                                 !
!       RETURN                                                                 !
!       END SUBROUTINE HESS                                                    !
!                                                                              !
!      Convergence is then obtained in 8 iterations.   Note that, in our       !
!      example, the objective function or its derivatives is/are computed if   !
!      the index i is omitted (see above). The full call could be              !
!                                                                              !
!       CALL LANCELOT_simple( 2, X, FUN, fx, exit_code,                      & !
!                             FUN, MY_GRAD = GRAD, MY_HESS = HESS,           & !
!                             BL  =  BL, BU = BU, VNAMES = VNAMES,           & !
!                             CNAMES = CNAMES,  NEQ = 1, NIN = 1,            & !
!                             CX = cx, Y = y, ITERS = iters , MAXIT = 100,   & !
!                             GRADTOL = 0.00001d0, FEASTOL = 0.00001d0,      & !
!                             PRINT_LEVEL = 1 )                                !
!                                                                              !
!      Of course, the above examples can easily be modified to represent new   !
!      minimization problems :-).                                              !
!                                                                              !
!      AVAILABLE ALGORITHMIC OPTIONS                                           !
!      =============================                                           !
!                                                                              !
!      Beyond the choice of derivative level for the problem functions, the    !
!      following arguments allow a (very limited) control of the algorithmic   !
!      choices used in LANCELOT.                                               !
!                                                                              !
!      maxit ( integer ): maximum number of iterations (default: 1000)         !
!                                                                              !
!      gradtol ( double precision ): the threshold on the infinity norm of     !
!              the gradient (or of the lagrangian's gradient) for declaring    !
!              convergence  (default: 1.0d-5),                                 !
!                                                                              !
!      feastol ( double precision ): the threshold on the infinity norm of     !
!              the constraint violation for declaring convergence (for         !
!              constrained problems) (default: 1.0d-5)                         !
!                                                                              !
!      print_level ( integer ): a positive number proportional to the amount   !
!              of output by the package: 0 corresponds to the silent mode, 1   !
!              to a single line of information per iteration (default), while  !
!              higher values progressively produce more output.                !
!                                                                              !
!                                                                              !
!      OTHER SOURCES                                                           !
!      =============                                                           !
!                                                                              !
!      The user is encouraged to consult the specsheet of the (non-naive)      !
!      interface to LANCELOT within the GALAHAD software library for a better  !
!      view of all possibilities offered by an intelligent use of the package. !
!      The library is described in the paper                                   !
!                                                                              !
!       N. I. M. Gould, D. Orban, Ph. L. Toint,                                !
!       GALAHAD, a library of thread-sage Fortran 90 packages for large-scale  !
!       nonlinear optimization,                                                !
!       Transactions of the AMS on Mathematical Software, vol 29(4),           !
!       pp. 353-372, 2003                                                      !
!                                                                              !
!      The book                                                                !
!                                                                              !
!       A. R. Conn, N. I. M. Gould, Ph. L. Toint,                              !
!       LANCELOT, A Fortan Package for Large-Scale Nonlinear Optimization      !
!       (Release A),                                                           !
!       Springer Verlag, Heidelberg, 1992                                      !
!                                                                              !
!      is also a good source of additional information.                        !
!                                                                              !
!                                                                              !
!      Main author: Ph. Toint, November 2007.                                  !
!     Copyright reserved, Gould/Orban/Toint, for GALAHAD productions           !
!                                                                              !
!-------------------------------------------------------------------------------

       USE LANCELOT_double                       ! double precision version

       IMPLICIT NONE
       INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
       INTEGER,                INTENT( IN )    :: n
       INTEGER,                INTENT( OUT )   :: exit_code
       REAL ( KIND = wp ),     INTENT( INOUT ) :: X( : )
       REAL ( KIND = wp ),     INTENT( OUT )   :: fx
       CHARACTER ( LEN = 10 ), OPTIONAL :: VNAMES( : ), CNAMES( : )
       INTEGER,                OPTIONAL :: maxit, print_level
       INTEGER,                OPTIONAL :: nin, neq, iters
       REAL ( KIND = wp ),     OPTIONAL :: gradtol, feastol
       REAL ( KIND = wp ),     OPTIONAL :: BL( : ), BU( : ), CX( : ), Y( : )
                               OPTIONAL :: MY_GRAD, MY_HESS

       REAL ( KIND = wp ), PARAMETER  :: infinity = 10.0_wp ** 20
       REAL ( KIND = wp ), PARAMETER  :: one = 1.0_wp, zero = 0.0_wp
       TYPE ( LANCELOT_control_type ) :: control
       TYPE ( LANCELOT_inform_type )  :: info
       TYPE ( LANCELOT_data_type )    :: data
       TYPE ( LANCELOT_problem_type ) :: prob
       REAL ( KIND = wp )             :: fiel
       INTEGER :: i, lfuval, iv, ig, iel, ng, nel, lelvar
       INTEGER :: ngpvlu, nepvlu, ninr, neqr, nh, igstrt, ihstrt
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVAR, ICALCF, ICALCG
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )    :: Q, XT,  DGRAD, S
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : )    :: FVALUE, FUVALS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( :, : ) :: GVALUE

       INTERFACE

          SUBROUTINE MY_FUN( X, F, i )
          INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
          REAL( KIND = wp ), INTENT( IN )   :: X( : )
          REAL( KIND = wp ), INTENT( OUT )  :: F
          INTEGER, INTENT( IN ), OPTIONAL   :: i
          END SUBROUTINE MY_FUN

          SUBROUTINE MY_GRAD( X, G, i )
          INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
          REAL( KIND = wp ), INTENT( IN )   :: X( : )
          REAL( KIND = wp ), INTENT( OUT )  :: G( : )
          INTEGER, INTENT( IN ), OPTIONAL   :: i
          END SUBROUTINE MY_GRAD

          SUBROUTINE MY_HESS( X, H, i )
          INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
          REAL( KIND = wp ), INTENT( IN )   :: X( : )
          REAL( KIND = wp ), INTENT( OUT )  :: H( : )
          INTEGER, INTENT( IN ), OPTIONAL   :: i
          END SUBROUTINE MY_HESS

       END INTERFACE

! print banner
!
       IF ( PRESENT( print_level ) ) THEN
          IF( print_level > 0 ) THEN
          WRITE(6,'('' '' )' )
          WRITE(6,'(15x,''**********************************************'' )')
          WRITE(6,'(15x,''*                                            *'' )')
          WRITE(6,'(15x,''*               LANCELOT_simple              *'' )')
          WRITE(6,'(15x,''*                                            *'' )')
          WRITE(6,'(15x,''*  a simple interface to GALAHAD/LANCELOT B  *'' )')
          WRITE(6,'(15x,''*                                            *'' )')
          WRITE(6,'(15x,''**********************************************'' )')
          WRITE(6,'('' '' )' )
          END IF
       END IF
!
! various sizes
!
       IF ( n < 1 ) THEN        ! there must be at least one variable
          exit_code = 15
          RETURN
       END IF
       IF ( PRESENT( nin ) ) THEN
          IF ( nin < 0 ) THEN   ! the number of inequality constraints...
             exit_code = 19     ! ... must be nonnegative
             RETURN
          END IF
          ninr = nin
       ELSE
          ninr = 0              ! no inequality constraints if nin not present
       END IF
       IF ( PRESENT( neq ) ) THEN
          IF ( neq < 0 ) THEN   ! the number of equality constraints...
             exit_code = 19     ! ... must be nonnegative
             RETURN
          END IF
          neqr = neq
       ELSE
          neqr = 0              ! no equality constraints if neq not present
       END IF
       ng       = neqr+ninr+1   ! the number of groups
       nel      = ng            ! the number of nonlinear elements
       lelvar   = ng * n        ! the total number of elemental variables
       ngpvlu   = 0             ! the number of group parameters
       nepvlu   = 0             ! the number of element parameters
       prob%n   = n + ninr ;    ! new variables <= original variables + slacks
       prob%ng  = ng
       prob%nel = nel
       nh       = ( ( n + 1 ) * n ) / 2  ! Hessians' size
!
! make space for problem data
!
       ALLOCATE( prob%ISTADG( ng  + 1 ), prob%ISTGPA( ng  + 1 ) )
       ALLOCATE( prob%ISTADA( ng  + 1 ), prob%ISTAEV( nel + 1 ) )
       ALLOCATE( prob%ISTEPA( nel + 1 ), prob%ITYPEG( ng ) )
       ALLOCATE( prob%KNDOFG( ng      ), prob%ITYPEE( nel ) )
       ALLOCATE( prob%INTVAR( nel + 1 ), prob%INTREP( nel ), prob%GXEQX( ng ) )
       ALLOCATE( prob%ISTADH( nel + 1 ), prob%A( ninr ) )
       ALLOCATE( prob%GPVALU( ngpvlu  ), prob%EPVALU( nepvlu ) )
       ALLOCATE( prob%ESCALE( nel     ), prob%GSCALE( ng ) )
       ALLOCATE( prob%GNAMES( ng      ), prob%VNAMES( prob%n ) )
       ALLOCATE( prob%VSCALE( prob%n  ), prob%BL( prob%n ), prob%BU( prob%n ) )
       ALLOCATE( prob%IELING( nel ), prob%IELVAR( lelvar ), prob%ICNA( ninr ) )
       ALLOCATE( prob%X( prob%n ), Q( prob%n ), XT( prob%n ), DGRAD( prob%n ) )
       ALLOCATE( IVAR( prob%n ), ICALCF( ng ), ICALCG( ng ), S( nh ) )
       ALLOCATE( prob%Y( ng )  , prob%C( ng ), prob%B( ng ) )
       ALLOCATE( FVALUE( ng )  , GVALUE( ng, 3 ) )
!
! set problem data for the simple "1-element" representation
!
       prob%X(1:n) = X(1:n)                ! starting point
       IF ( PRESENT( BL ) ) THEN           ! lower bounds
          prob%BL(1:n) = BL(1:n)
       ELSE
          prob%BL = -infinity
       END IF
       IF ( PRESENT( BU ) ) THEN           ! upper bounds
          prob%BU(1:n) = BU(1:n)
       ELSE
          prob%BU =  infinity
       END IF
       IF ( PRESENT( VNAMES ) ) THEN       ! variable names
          prob%VNAMES(1:n) = VNAMES(1:n)
       ELSE                                ! give default names
          DO iv = 1, prob%n
             IF ( iv < 10 ) THEN
                WRITE( prob%VNAMES( iv ), '( A2,I1 )' ) 'X_', iv
             ELSEIF ( i < 100 ) THEN
                WRITE( prob%VNAMES( iv ), '( A2,I2 )' ) 'X_', iv
             ELSEIF ( i < 1000 ) THEN
                WRITE( prob%VNAMES( iv ), '( A2,I3 )' ) 'X_', iv
             ELSEIF ( i < 10000 ) THEN
                WRITE( prob%VNAMES( iv ), '( A2,I4 )' ) 'X_', iv
             ELSE
                WRITE( prob%VNAMES( iv ), ' (A2,I8 )' ) 'X_', iv
             END IF
          ENDDO
       END IF
       IF ( PRESENT( CNAMES ) ) THEN       ! function names
          IF ( ng > 1 ) THEN
             prob%GNAMES( 1:ng-1 ) = CNAMES( 1:ng-1 )
          END If
       ELSE                                ! give default names
          DO ig = 1, ninr                  ! ... inequalities
             IF ( ig < 10 ) THEN
                WRITE( prob%GNAMES( ig ), '( A6,I1 )' ) 'Inequ_', ig
             ELSEIF ( i < 100 ) THEN
                WRITE( prob%GNAMES( ig ), '( A6,I2 )' ) 'Inequ_', ig
             ELSEIF ( ig < 1000 ) THEN
                WRITE( prob%GNAMES( ig ), '( A6,I3 )' ) 'Inequ_', iv
             ELSE
                WRITE( prob%GNAMES( ig ), '( A6,I4 )' ) 'Inequ_', ig
             END IF
          ENDDO
          DO i = 1, neqr                   ! ... equalities
             ig = ninr + i
             IF ( i < 10 ) THEN
                WRITE( prob%GNAMES( ig ), '( A6,I1 )' ) 'Equal_', i
             ELSEIF ( i < 100 ) THEN
                WRITE( prob%GNAMES( ig ), '( A6,I2 )' ) 'Equal_', i
             ELSEIF ( i < 1000 ) THEN
                WRITE( prob%GNAMES( ig ), '( A6,I3 )' ) 'Equal_', i
             ELSE
                WRITE( prob%GNAMES( ig ), '( A6,I4 )' ) 'Equal_', i
             END IF
          ENDDO
       END IF
       prob%GNAMES( ng ) = 'Objective'     ! the objective function
       DO iv = 1, prob%n
          prob%VSCALE( iv ) = one          ! all variables are scaled by 1.0
       ENDDO
       DO ig = 1, ng
          IF ( ig < ng )THEN
              prob%KNDOFG( ig ) = 2        ! first groups <= constraints
              prob%Y(ig)        = zero     ! initial Lagrange multipliers
          ELSE
              prob%KNDOFG( ig ) = 1        ! last group <= objective function
          END IF
          prob%IELING( ig )  = ig          ! each group contains one element
          prob%ISTADG( ig )  = ig          ! start of the list of elements
                                           ! in group ig
          prob%ITYPEG( ig )  = 0           ! all groups are trivial
          prob%GXEQX ( ig )  = .TRUE.      ! all groups are of the TRIVIAL type
          prob%GSCALE( ig )  = one         ! all groups are scaled by 1.0
          prob%ISTGPA( ig )  = 1           ! there are no group parameters
          prob%B( ig )       = zero        ! there are no constant in groups
       ENDDO
       prob%ISTADG( ng+1 )   = ng+1        ! start of the last phantom group
       prob%ISTGPA( ng+1 )   = 1           ! there are no group parameters
       i = 1
       DO iel = 1, nel
          prob%INTREP( iel ) = .FALSE.     ! no element has internal variables
          prob%ITYPEE( iel ) = iel         ! each element defines its own type
          prob%ESCALE( iel ) = one         ! all elements are scaled by 1.0
          prob%ISTEPA( iel ) = 1           ! there are no element parameters
          prob%INTVAR( iel ) = n           ! each element involves all variables
          prob%ISTAEV( iel ) = i           ! start of the list of variables
                                           ! for element iel
          DO iv = 1, n
            prob%IELVAR( i ) = iv          ! each element involves each one
            i = i + 1                      ! of the variables
          ENDDO
       ENDDO
       prob%ISTAEV( nel+1 )  = i           ! start of the last phantom element
       prob%ISTEPA( nel+1 )  = 1           ! there are no element parameters
       DO i = 1, neqr
          prob%ISTADA( i )   = 1           ! no linear var. in eq. constraints
       ENDDO
       DO i = 1, ninr
          ig                 = neqr + i    ! index of the i-th inequality
          iv                 = n + i       ! index of the corresponding slack
          prob%ISTADA( ig )  = i           ! 1 linear slack per inequality...
          prob%A( i )        = one         ! ... with coefficient 1.0 ...
          prob%ICNA( i )     = iv          ! ... and index iv = n+i
          prob%BL( iv )      = zero        ! slacks are bounded below by 0.0 ...
          prob%BU( iv )      = infinity    ! ... are unbounded above
          prob%X( iv )       = one         ! ... are initialized to 1.0...
          IF ( i < 10 ) THEN               ! ... and are given a name
             WRITE( prob%VNAMES( iv ), '( A6,I1 )' ) 'Slack_', i
          ELSEIF ( i < 100 ) THEN
             WRITE( prob%VNAMES( iv ), '( A6,I2 )' ) 'Slack_', i
          ELSEIF ( i < 1000 ) THEN
             WRITE( prob%VNAMES( iv ), '( A6,I3 )' ) 'Slack_', i
          ELSE
             WRITE( prob%VNAMES( iv ), '( A6,I4 )' ) 'Slack_', i
          END IF
       ENDDO
       prob%ISTADA( ng )     = ninr + 1    ! no linear variable in the objective
       prob%ISTADA( ng+1 )   = ninr + 1    ! start of last phantom linear term
!
! allocate space for FUVALS (see LANCELOT specsheet)
!
       lfuval = nel + 2 * prob%n + ( nel * ( n * ( n + 3 ) ) / 2 )
       ALLOCATE( FUVALS( lfuval ) )
!
! set default algorithmic parameters
!
       CALL LANCELOT_initialize( data, control )
!
! possibly overwrite default settings
!
       control%initial_radius = 1.0_wp     ! initial trust-region radius
!
       IF ( PRESENT( maxit ) )  THEN       ! maximum number of iterations
           control%maxit  = MAX( 0, maxit )
       END IF
!
       IF ( PRESENT( print_level ) ) THEN  ! printout amount
          control%print_level  = MAX( 0, print_level )
       ELSE
          control%print_level  = 1         ! (one line per minor iteration)
       END IF
!
       IF ( PRESENT( gradtol ) ) THEN
          control%stopg = ABS( gradtol )   ! gradient tolerance
       END IF
!
       IF ( PRESENT( feastol ) ) THEN
          control%stopc = ABS( feastol )   ! constraint violation tolerance
       END IF
!
       control%linear_solver = 8           ! linear solver = preconditioned CG
!
       IF ( PRESENT ( MY_GRAD ) ) THEN     ! derivative availability
          control%first_derivatives = 0    ! use exact gradients
          IF ( PRESENT ( MY_HESS ) ) THEN
             control%second_derivatives = 0  ! use exact Hessians
          ELSE
             control%second_derivatives = 4  ! use SR1 update for Hessians
          END IF
       ELSE
          control%first_derivatives  = 1   ! use forward differences for grads
          control%second_derivatives = 4   ! use SR1 update for Hessians
       END IF
!
! solve the problem by using LANCELOT_solve in reverse communication mode
!
       info%status = 0
       DO                                  ! loop until LANCELOT exits

          CALL LANCELOT_solve( prob, RANGE, GVALUE, FVALUE, XT, FUVALS, lfuval,&
                               ICALCF, ICALCG, IVAR, Q, DGRAD , control, info, &
                               data )

          IF ( info%status >= 0 ) EXIT     ! LANCELOT has finished
!
!         Compute function values
!
          IF ( info%status == -1 .OR. &
               info%status == -3 .OR. &
               info%status == -7      )  THEN

             DO i = 1, info%ncalcf
                iel  = ICALCF( i )
!
!                  the objective function
!
                IF ( iel == prob%ng ) THEN
                   CALL MY_FUN( XT(1:n), fiel )      ! user defined
!
!                  the constraint value
!
                ELSE
                   CALL MY_FUN( XT(1:n), fiel, iel ) ! user defined
                END IF
                FUVALS( iel ) = fiel
             END DO

          END IF

          IF ( control%first_derivatives > 0 ) CYCLE
!
!         Compute (first and, possibly, second) derivatives values
!
          IF ( info%status == -1 .OR. &
               info%status == -5 .OR. &
               info%status == -6      ) THEN

             DO i = 1, info%ncalcf
               iel    = ICALCF( i )
               igstrt = prob%INTVAR( iel ) - 1
               ihstrt = prob%ISTADH( iel ) - 1
               IF ( iel == prob%ng ) THEN
!
!                 the gradient of the objective function
!
                  CALL MY_GRAD( XT(1:n), S(1:n) )
                  FUVALS( igstrt+1:igstrt+n ) = S(1:n)

                  IF ( control%second_derivatives > 0 ) CYCLE
!
!                 the entries of the upper triangle of the objective
!                 function's Hessian matrix, stored by columns (user defined)
!
                  CALL MY_HESS( XT(1:n), S(1:nh) )
                  FUVALS( ihstrt+1:ihstrt+nh ) = S(1:nh)

               ELSE
!
!                 the constraint's gradient components (user defined)
!
                  CALL MY_GRAD( XT(1:n), S(1:n), iel )
                  FUVALS( igstrt+1:igstrt+n ) = S(1:n)

                  IF ( control%second_derivatives > 0 ) CYCLE
!
!                 the entries of the upper triangle of constraint's
!                 Hessian matrix, stored by columns (user defined)
!
                  CALL MY_HESS( XT(1:n), S(1:nh), iel )
                  FUVALS( ihstrt+1:ihstrt+nh ) = S(1:nh)

               END IF
             END DO

          END IF

       END DO                              ! loop to reenter LANCELOT
!
! print message on error return status
!
       IF ( control%print_level > 0 .AND. info%status /= 0 ) THEN
          WRITE( 6, "( ' LANCELOT_solve exit status = ', I6 ) " ) info%status
       END IF
!
! define output arguments
!
       fx        = info%obj                             ! objective's value
       X         = prob%X(1:n)                          ! variables' values
       exit_code = info%status                          ! success/failure indic.
       IF ( PRESENT( CX ) .AND. neqr+ninr > 0 ) THEN    ! constraints' values
          CX(1:neqr)           = prob%C(1:neqr)
          CX(neqr+1:neqr+ninr) = prob%C(neqr+1:neqr+ninr) - prob%X(n+1:n+nin)
       END IF
       IF ( PRESENT( Y ) .AND. neqr+ninr > 0 ) THEN     ! Lagrange multipliers
          Y(1:neqr+ninr) = prob%Y(1:neqr+ninr)
       END IF
       IF ( PRESENT( iters ) ) THEN                     ! number of iterations
          iters = info%iter
       END IF
!
! clean up
!
       CALL LANCELOT_terminate( data, control, info )
       DEALLOCATE( prob%GNAMES, prob%VNAMES )         !  delete problem space
       DEALLOCATE( prob%VSCALE, prob%ESCALE, prob%GSCALE, prob%INTREP )
       DEALLOCATE( prob%GPVALU, prob%EPVALU, prob%ITYPEG, prob%GXEQX )
       DEALLOCATE( prob%X, prob%Y, prob%C, prob%A, prob%B, prob%BL, prob%BU )
       DEALLOCATE( prob%ISTADH, prob%IELING, prob%IELVAR, prob%ICNA )
       DEALLOCATE( prob%INTVAR, prob%KNDOFG, prob%ITYPEE, prob%ISTEPA )
       DEALLOCATE( prob%ISTADA, prob%ISTAEV, prob%ISTADG, prob%ISTGPA )
       DEALLOCATE( IVAR, ICALCF, ICALCG )
       DEALLOCATE( S, Q, XT, DGRAD, FVALUE, GVALUE, FUVALS )
!
       RETURN
!
       END SUBROUTINE LANCELOT_simple
!
!===============================================================================
!
       SUBROUTINE RANGE( iel, transp, W1, W2, nelv, ninv, ielt, lw1, lw2 )
       IMPLICIT NONE
       INTEGER, PARAMETER    :: wp = KIND( 1.0D+0 )
       INTEGER, INTENT( IN ) :: iel, nelv, ninv, ielt, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = wp ), INTENT(  IN ), DIMENSION ( lw1 ) :: W1
       REAL ( KIND = wp ), DIMENSION ( lw2 ) :: W2
!      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       RETURN
       END SUBROUTINE RANGE
!
!===============================================================================
   END MODULE LANCELOT_simple_double
