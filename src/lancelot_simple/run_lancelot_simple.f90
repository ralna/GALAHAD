PROGRAM RUN_LANCELOT_simple

!-----------------------------------------------------------------------------
!                                                                            !
!    This programs provides a simple (naive) way to run LANCELOT B on an     !
!    optimization problem without interaction with CUTEr and using a simple  !
!    representation of the problem where the objective function and          !
!    constraints comprise each a single group with a single nonlinear        !
!    element.                                                                !
!                                                                            !
!    The use of this program and the accompanying script lanb_simple         !
!    requires that the GALAHAD package has prealably be installed.           !
!                                                                            !
!    Ph. L. Toint, November 2007.                                            !
!    Copyright reserved, Gould/Orban/Toint, for GALAHAD productions          !
!                                                                            !
!-----------------------------------------------------------------------------
!
    USE LANCELOT_simple_double
!
    IMPLICIT NONE
    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
    REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
    INTEGER :: n, neq, nin, iters, maxit, print_level, exit_code
    REAL ( KIND = wp ) :: gradtol, feastol, fx
    CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION(:) :: VNAMES, CNAMES
    REAL ( KIND = wp ),     ALLOCATABLE, DIMENSION(:) :: BL, BU, X, CX, Y
    EXTERNAL :: FUN, GRAD, HESS
!
! THE TEST PROBLEM DIMENSIONS (user defined)
!
    n   = 2  ! number of variables
    neq = 1  ! number of equality constraints, excluding fixed variables
    nin = 1  ! number of inequality (<= 0) constraints, excluding bounds
!
! allocate space for problem defining vectors
!
    ALLOCATE( X( n  ), BL( n ), BU( n ), CX( neq+nin ), Y( neq+nin ) )
    ALLOCATE( VNAMES( n  ),CNAMES( neq+nin ) )
!
! starting point
!
    X(1) = -1.2_wp                 ! starting point (componentwise)
    X(2) =  1.0_wp
!
! bounds on the variables
!
    BL(1) =  0.0_wp                ! lower bounds (componentwise)
    BL(2) = -infinity
    BU(1) =  infinity              ! upper bounds (componentwise)
    BU(2) =  3.0_wp
!
! names
!
    VNAMES(1) = 'x1'               ! variables
    VNAMES(2) = 'x2'
    CNAMES(1) = 'Equality'         ! equality constraints
    CNAMES(2) = 'Inequality'       ! inequality constraints
!
! algorithmic parameters
!
    maxit       = 100
    gradtol     = 0.00001_wp       ! identical to default
    feastol     = 0.00001_wp       ! identical to default
    print_level = 1                ! identical to default
!
! solve by calling LANCELOT B
!
    CALL LANCELOT_simple( n,  X, fx, exit_code,                          &
                          MY_FUN = FUN, MY_GRAD = GRAD , MY_HESS = HESS, &
                          BL = BL, BU = BU, VNAMES   =  VNAMES,          &
                          CNAMES =  CNAMES, neq = neq, nin = nin,        &
                          CX = CX, Y = Y, ITERS  = iters, MAXIT = maxit, &
                          GRADTOL = gradtol, FEASTOL = feastol,          &
                          PRINT_LEVEL = print_level )
!
! clean up
!
    DEALLOCATE( X, BL, BU, CX, Y )
    DEALLOCATE( VNAMES, CNAMES )
    STOP
!
END PROGRAM RUN_LANCELOT_simple
!
!.............................................................................
    SUBROUTINE FUN ( X, F, i )
!.............................................................................
    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
    REAL( KIND = wp ), INTENT( IN )   :: X( : )
    REAL( KIND = wp ), INTENT( OUT )  :: F
    INTEGER, INTENT( IN ), OPTIONAL   :: i
    IF ( .NOT. PRESENT( i ) ) THEN
!       the objective function value (user defined)
!==============================================================================
        F = 100.0_wp*(X(2)-X(1)**2)**2 +(1.0_wp-X(1))**2                      !
!==============================================================================
     ELSE
        SELECT CASE ( i )
            CASE ( 1 )
!               the equality constraint value (user defined)
!==============================================================================
                F = X(1)+3.0_wp*X(2)-3.0_wp                                   !
!==============================================================================
            CASE ( 2 )
!               the inequality constraint value (user defined)
!==============================================================================
                F = X(1)**2+X(2)**2-4.0_wp                                    !
!==============================================================================
        END SELECT
     END IF
     RETURN
    END SUBROUTINE FUN
!
!.............................................................................
    SUBROUTINE GRAD( X, G, i )
!.............................................................................
    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
    REAL( KIND = wp ), INTENT( IN )  :: X( : )
    REAL( KIND = wp ), INTENT( OUT ) :: G( : )
    INTEGER, INTENT( IN ), OPTIONAL  :: i
    IF ( .NOT. PRESENT( i ) ) THEN
!       the objective functions's gradient components (user defined)
!==============================================================================
        G( 1 ) = -400.0_wp*(X(2)-X(1)**2)*X(1)-2.0_wp*(1.0_wp-X(1))           !
        G( 2 ) =  200.0_wp*(X(2)-X(1)**2)                                     !
!==============================================================================
    ELSE
        SELECT CASE ( i )
            CASE ( 1 )
!               the equality constraint's gradient components (user defined)
!==============================================================================
                G( 1 ) =  1.0_wp                                              !
                G( 2 ) =  3.0_wp                                              !
!==============================================================================
            CASE ( 2 )
!               the inequality constraint's gradient components (user defined)
!==============================================================================
                G( 1 ) =  2.0_wp*X(1)                                         !
                G( 2 ) =  2.0_wp*X(2)                                         !
!==============================================================================
        END SELECT
    END IF
    RETURN
    END SUBROUTINE GRAD
!
!.............................................................................
    SUBROUTINE HESS( X, H, i )
!.............................................................................
    INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
    REAL( KIND = wp ), INTENT( IN )  :: X( : )
    REAL( KIND = wp ), INTENT( OUT ) :: H( : )
    INTEGER, INTENT( IN ), OPTIONAL  :: i
    IF ( .NOT. PRESENT( i ) ) THEN
!       the entries of the upper triangle of the objective function's
!       Hessian  matrix,  stored by columns  (user defined)
!==============================================================================
        H( 1 ) = -400.0_wp*(X(2)-3.0_wp*X(1)**2)+2.0_wp                       !
        H( 2 ) = -400.0_wp*X(1)                                               !
        H( 3 ) =  200.0_wp                                                    !
!==============================================================================
    ELSE
        SELECT CASE ( i )
            CASE ( 1 )
!               the entries of the upper triangle of the equality
!               constraint's Hessian matrix, stored by columns (user defined)
!==============================================================================
                H( 1 ) = 0.0_wp                                               !
                H( 2 ) = 0.0_wp                                               !
                H( 3 ) = 0.0_wp                                               !
!==============================================================================
            CASE ( 2 )
!               the entries of the upper triangle of the inequality
!               constraint's Hessian matrix, stored by columns (user defined)
!==============================================================================
                H( 1 ) = 2.0_wp                                               !
                H( 2 ) = 0.0_wp                                               !
                H( 3 ) = 2.0_wp                                               !
!==============================================================================
        END SELECT
    END IF
    RETURN
    END SUBROUTINE HESS
!
!.............................................................................
