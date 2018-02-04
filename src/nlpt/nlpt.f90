! THIS VERSION: GALAHAD 2.1 - 13/02/2008 AT 09:20 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*    GALAHAD NLPT  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Philippe Toint

!  History -
!   originally released with GALAHAD Version 1.2. November 17th 2002
!   update released with GALAHAD Version 2.0. February 16th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_NLPT_double

!           +--------------------------------------------+
!           |                                            |
!           |      Provide a derived data type for       |
!           |                                            |
!           |       unconstrained and constrained        |
!           |                                            |
!           |       nonlinear programming problems       |
!           |                                            |
!           +--------------------------------------------+

!-------------------------
!  M o d u l e s   u s e d
!-------------------------

      USE GALAHAD_NORMS_double
      USE GALAHAD_TOOLS_double
      USE GALAHAD_SMT_double
      USE GALAHAD_SYMBOLS,                                                     &
          SILENT              => GALAHAD_SILENT,                               &
          TRACE               => GALAHAD_TRACE,                                &
          ACTION              => GALAHAD_ACTION,                               &
          DETAILS             => GALAHAD_DETAILS,                              &
          COORDINATE          => GALAHAD_COORDINATE,                           &
          SPARSE              => GALAHAD_SPARSE_BY_ROWS,                       &
          DENSE               => GALAHAD_DENSE

      IMPLICIT NONE

      PRIVATE

!     Ensure the private nature of the imported symbols.

      PRIVATE :: SILENT, TRACE, ACTION, SPARSE, DENSE, COORDINATE

!     Make the tools public.

      PUBLIC :: NLPT_write_variables, NLPT_write_stats,                        &
                NLPT_write_constraints, NLPT_write_problem,                    &
                NLPT_J_perm_from_C_to_Srow, NLPT_J_perm_from_C_to_Scol,        &
                NLPT_cleanup, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0E+0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PRIVATE, PARAMETER :: wp = dp
      INTEGER, PARAMETER :: wcp = KIND( ( 1.0D+0, 1.0D+0 ) )

!===============================================================================
!              D e r i v e d   t y p e   d e f i n i t i o n s
!===============================================================================

!-------------------------------------------------------------------------------
!                        The problem structure
!-------------------------------------------------------------------------------

      TYPE, PUBLIC :: NLPT_problem_type

        ! the problem name

        CHARACTER( LEN = 10 ) :: pname = '          '

        ! number of variables

        INTEGER :: n

        ! names of the variables

        CHARACTER( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: vnames

        ! current values for the problem's variables

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x

        ! lower bounds on the problem's variables

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x_l

        ! upper bounds on the problem's variables

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x_u

        ! dual variables associated with the bound constraints

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: z

        ! variables' scaling factors

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x_scale

        ! variables' status

        INTEGER, ALLOCATABLE, DIMENSION( : ) :: x_status

        ! the objective function value

        REAL ( KIND = wp ) :: f

        ! gradient of the objective function

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: g

        ! the value of the Lagrangian

        REAL ( KIND = wp ) :: L

        ! gradient of the Lagrangian

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: gL

        ! the type of storage for the lower triangle of the Hessian of the
        ! objective function (unconstrained) or of the Lagrangian (constrained)

        INTEGER :: H_type

        ! the number of nonzeroes in the lower triangular part of the Hessian
        ! of the objective function (unconstrained) or of the Lagrangian
        ! (constrained)

        INTEGER :: H_ne

        ! Hessian of the objective funtion (unconstrained) or of the
        ! Lagrangian (constrained)

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_val

        ! pointers for sparse storage (sparse by rows or coordinate) of
        ! the (lower triangular part of the) Hessian of the objective function
        ! (unconstrained) or of the Lagrangian (constrained)

        INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_ptr

        ! number of nonlinear constraints

        INTEGER :: m

        ! number of linear constraints (if the user wishes to  separate them
        ! from the nonlinear ones)

        INTEGER :: m_a

        ! names of the linear constraints

        CHARACTER( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: anames

        ! current values of the linear constraints

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ax

        ! lower bounds on the linear constraint values (if desired)

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: a_l

        ! upper bounds on the linear constraints values (if desired)

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: a_u

        ! current values of the constraints

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: c

        ! lower bounds on the constraint values

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: c_l

        ! upper bounds on the constraints values

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: c_u

        ! names of the constraints

        CHARACTER( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: cnames

        ! equation( i ) = .TRUE. if c( i ) is an equality constraint
        ! ( i.e. c_l( i ) = c_u( i ) = 0 ).

        LOGICAL, ALLOCATABLE, DIMENSION( : ) :: equation

        ! linear( i ) = .TRUE. iff c( i ) is linear.

        LOGICAL, ALLOCATABLE, DIMENSION( : ) :: linear

        ! Lagrange multipliers for the linear constraint values (if desired)

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: y_a

        ! Lagrange multipliers associated with the constraints

        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: y

        ! constraints' scaling factors

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: c_scale

        ! constraints' status

        INTEGER, ALLOCATABLE, DIMENSION( : ) :: c_status

        ! the type of storage for J

        INTEGER :: J_type

        ! the number of nonzeroes in the Jacobian

        INTEGER :: J_ne

        ! Jacobian of the constraints

        REAL ( KIND = wp ), ALLOCATABLE,  DIMENSION( : ) :: J_val

        ! pointers for sparse Jacobian storage (sparse by rows or coordinate)

        INTEGER, ALLOCATABLE, DIMENSION( : ) :: J_row
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: J_col
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: J_ptr

        ! Four scalar variables of derived type SMT_type.  Used to hold the
        ! Jacobian of the (linear and nonlinear) constraints, the Hessian of
        ! the Lagrangian, and the matrix of products of each constraint Hessian
        ! with a vector.  These will eventually replace all of the above.

        TYPE ( SMT_type ) :: A, J, H, P

        ! the convential value of infinity (that is the value beyond which
        ! bounds are assumed to be infinite)

        REAL ( KIND = wp ) :: infinity

      END TYPE

!  ===================================
!  The NLPT_userdata_type derived type
!  ===================================

      TYPE, PUBLIC :: NLPT_userdata_type
         INTEGER, ALLOCATABLE, DIMENSION( : ) :: integer
         REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: real
         COMPLEX ( KIND = wcp ), ALLOCATABLE, DIMENSION( : ) :: complex
         CHARACTER, ALLOCATABLE, DIMENSION( : ) :: character
         LOGICAL, ALLOCATABLE, DIMENSION( : ) :: logical
         INTEGER, POINTER, DIMENSION( : ) :: integer_pointer => null( )
         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: real_pointer => null( )
         COMPLEX ( KIND = wcp ), POINTER,                                      &
           DIMENSION( : ) :: complex_pointer => null( )
         CHARACTER, POINTER, DIMENSION( : ) :: character_pointer => null( )
         LOGICAL, POINTER, DIMENSION( : ) :: logical_pointer => null( )
      END TYPE NLPT_userdata_type

!  ----------------
!  Other parameters
!  ----------------

      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ZERO = 0.0_wp

!  ----------------------------------------------------------------------------
!  Interface blocks for the single and double precision BLAS routines
!  giving the two-norm.
!  ----------------------------------------------------------------------------

!     INTERFACE TWO_NORM
!
!        FUNCTION SNRM2( n, x, incx )
!          REAL  :: SNRM2
!          INTEGER, INTENT( IN ) :: n, incx
!          REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
!        END FUNCTION SNRM2
!
!        FUNCTION DNRM2( n, x, incx )
!          DOUBLE PRECISION  :: DNRM2
!          INTEGER, INTENT( IN ) :: n, incx
!          DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
!        END FUNCTION DNRM2
!
!     END INTERFACE

   CONTAINS

!===============================================================================
!===============================================================================

      SUBROUTINE NLPT_J_perm_from_C_to_Srow( problem, perm, col, ptr )

!     Build the permutation that transforms the Jacobian from coordinate
!     storage to sparse by row, as well as the associated ptr and col vectors.

!     Arguments:

      TYPE ( NLPT_problem_type ), INTENT( IN ) :: problem

!            the problem whose Jacobian matrix is considered.

      INTEGER, DIMENSION( : ), INTENT( OUT ) :: perm
      INTEGER, DIMENSION( : ), INTENT( OUT ) :: col
      INTEGER, DIMENSION( : ), INTENT( OUT ) :: ptr

!     Programming: Ph. Toint, November 2002

!==============================================================================

!     Local variables

      INTEGER  :: k, i, ii

      IF ( problem%m <= 0 ) RETURN

!     Count the number of nonzero in each row in ptr.

      ptr( 1:problem%m + 1 ) = 0
      DO k = 1, problem%J_ne
         i = problem%J_row( k )
         ptr( i ) = ptr( i ) + 1
      END DO

!     Assign the pointers to the beginning of each row.

      ii = 1
      DO i = 1, problem%m + 1
         k = ptr( i )
         ptr( i ) = ii
         ii = ii + k
      END DO

!     Build the permutation of the elements of A and
!     its col numbers.

      DO k = 1, problem%J_ne
         i  = problem%J_row( k )
         ii = ptr( i )
         perm( k ) = ii
         col( k )  = problem%J_col( k )
         ptr( i )  = ii + 1
      END DO

!     Rebuild the pointers to the beginning of each row.

      DO i = problem%m, 2, -1
         ptr( i ) = ptr( i - 1 )
      END DO
      ptr( 1 ) = 1

      RETURN

      END SUBROUTINE NLPT_J_perm_from_C_to_Srow

!===============================================================================
!===============================================================================

      SUBROUTINE NLPT_J_perm_from_C_to_Scol( problem, perm, row, ptr )

!     Build the permutation that transforms the Jacobian from coordinate
!     storage to sparse by column, as well as the associated ptr and row
!     vectors.

!     Arguments:

      TYPE ( NLPT_problem_type ), INTENT( IN ) :: problem

!            the problem whose Jacobian matrix is considered.

      INTEGER, DIMENSION( : ), INTENT( OUT ) :: perm
      INTEGER, DIMENSION( : ), INTENT( OUT ) :: row
      INTEGER, DIMENSION( : ), INTENT( OUT ) :: ptr

!     Programming: Ph. Toint, November 2002

!==============================================================================

!     Local variables

      INTEGER  :: k, i, ii

!     Count the number of nonzero in each column in ptr.

      ptr( 1:problem%n + 1 ) = 0
      DO k = 1, problem%J_ne
         i = problem%J_col( k )
         ptr( i ) = ptr( i ) + 1
      END DO

!     Assign the pointers to the beginning of each column.

      ii = 1
      DO i = 1, problem%n + 1
         k = ptr( i )
         ptr( i ) = ii
         ii = ii + k
      END DO

!     Build the permutation of the elements of A and
!     its row numbers.

      DO k = 1, problem%J_ne
         i  = problem%J_col( k )
         ii = ptr( i )
         perm( ii ) = k
         row( ii ) = problem%J_row( k )
         ptr( i )  = ii + 1
      END DO

!     Rebuild the pointers to the beginning of each row.

      DO i = problem%n, 2, -1
         ptr( i ) = ptr( i - 1 )
      END DO
      ptr( 1 ) = 1

      RETURN

      END SUBROUTINE NLPT_J_perm_from_C_to_Scol

!===============================================================================
!===============================================================================

   SUBROUTINE NLPT_write_stats( problem, out )

!  Writes the problem statistics (number and type of variables and
!  constraints). The routine assumes that problem%pname is associated
!  and that both problem%c_l and problem%c_u are associated when problem%m > 0.

!  Arguments

   TYPE ( NLPT_problem_type ), INTENT( IN ) :: problem

!         the setup problem

   INTEGER, INTENT( IN ) :: out

!         the printout device number.

!  Programming: Ph. Toint, November 2002.
!
!===============================================================================

!  Local variables

   INTEGER n_free, n_lower, n_upper, n_range, n_fixed, m_lower, m_upper,       &
           m_range, m_equal, m_linear, i, j

   WRITE( out, 1000 ) problem%pname

!-------------------------------------------------------------------------------
! Analyze the problem variables.
!-------------------------------------------------------------------------------

   n_free  = 0
   n_lower = 0
   n_upper = 0
   n_range = 0
   n_fixed = 0

   IF ( ALLOCATED( problem%x_l ) .AND. ALLOCATED( problem%x_u ) ) THEN
      DO j = 1, problem%n
         IF ( problem%x_l( j ) > - problem%infinity ) THEN
            IF ( problem%x_u( j ) <= problem%x_l( j ) ) THEN
               n_fixed = n_fixed + 1
            ELSE
               IF ( problem%x_u( j ) < problem%infinity ) THEN
                  n_range = n_range + 1
               ELSE
                  n_lower = n_lower + 1
               END IF
            END IF
         ELSE
            IF ( problem%x_u( j ) < problem%infinity ) THEN
               n_upper = n_upper + 1
            ELSE
               n_free  = n_free + 1
            END IF
         END IF
      END DO
   ELSE IF ( ALLOCATED( problem%x_l ) ) THEN
      DO j = 1, problem%n
         IF ( problem%x_l( j ) > - problem%infinity ) THEN
            n_lower = n_lower + 1
         ELSE
            n_free  = n_free + 1
         END IF
      END DO
   ELSE IF ( ALLOCATED( problem%x_u ) ) THEN
      DO j = 1, problem%n
         IF ( problem%x_l( j ) <  problem%infinity ) THEN
            n_upper = n_upper + 1
         ELSE
            n_free  = n_free + 1
         END IF
      END DO
   ELSE
      n_free = problem%n
   END IF

!-------------------------------------------------------------------------------
! Analyze the problem constraints, if any.
!-------------------------------------------------------------------------------

   m_lower  = 0
   m_upper  = 0
   m_range  = 0
   m_equal  = 0
   m_linear = 0

   DO i = 1, problem%m
      IF ( problem%equation( i ) ) THEN
         m_equal = m_equal + 1
      ELSE IF ( problem%c_l( i ) > - problem%infinity ) THEN
         IF ( problem%c_u( i ) < problem%infinity ) THEN
            m_range = m_range + 1
          ELSE
            m_lower = m_lower + 1
         END IF
      ELSE IF ( problem%c_u( i ) < problem%infinity ) THEN
         m_upper = m_upper + 1
      END IF
      IF ( problem%linear( i ) ) m_linear = m_linear + 1
   END DO

!-------------------------------------------------------------------------------
! Write the global stats.
!-------------------------------------------------------------------------------

   IF ( problem%n < 100000000 .AND. problem%m < 100000000 ) THEN
      WRITE( out, 1001 ) n_free, n_lower, n_upper, n_range,n_fixed, problem%n, &
                         m_lower, m_upper, m_range, m_equal, m_linear, problem%m
   ELSE
      WRITE( out, 1002 ) n_free, n_lower, m_lower, n_upper, m_upper, n_range,  &
                       m_range, n_fixed, m_equal, m_linear, problem%n, problem%m
   END IF

   RETURN

!  Formats

1000 FORMAT(/,11x,'+--------------------------------------------------------+',&
            /,11x,'|',18x,'Problem : ',a10,18x,'|',                            &
            /,11x,'+--------------------------------------------------------+',&
            / )
1001 FORMAT(17x,'Free    Lower    Upper    Range     Fixed/   Linear  Total',/,&
            17x,'      bounded  bounded  bounded  equalities',/,               &
            ' Variables  ',5(1x,i8),10x,i8,/,' Constraints',9x,6(1x,i8),/)
1002 FORMAT('                    Variables          Constraints   '/,          &
            ' Free            ',1x,i15,/,' Lower bounded   ',1x,i15,5x,i15,/,  &
            ' Upper bounded   ',1x,i15,5x,i15,/,                               &
            ' Range bounded   ',1x,i15,5x,i15,/                                &
            ' Fixed/equalities',1x,i15,5x,i15,/                                &
            ' Linear          ',21x,i15,' Total           ',1x,i15,5x,i15)


   END SUBROUTINE NLPT_write_stats

!===============================================================================
!===============================================================================

   SUBROUTINE NLPT_write_variables( problem, out )

!  Writes the variables and associated multipliers.  This routine assumes that
!  problem%pname and problem%x are associated. The bounds are printed
!  whenever problem%x_l and problem%x_u are associated. Moreover, it is also
!  assumed in this case that problem%g is associated when problem%m = 0, and
!  that problem%z is associated when problem%m > 0. The variables
!  names are used whenever problem%vnames is associated, but this is not
!  mandatory.

!  Arguments

   TYPE ( NLPT_problem_type ), INTENT( IN ) :: problem

!         the setup problem

   INTEGER, INTENT( IN ) :: out

!         the printout device number.

!  Programming: Ph. Toint, November 2002.
!
!===============================================================================

!  Local variables

   INTEGER :: j, k, jj
   LOGICAL :: has_bounds
   REAL( KIND = wp ) :: xl, xu

   WRITE( out, 5000 ) problem%pname

   has_bounds = ALLOCATED( problem%x_l ) .AND. ALLOCATED( problem%x_u )

   IF ( problem%n < 100000 ) THEN
      IF ( ALLOCATED( problem%vnames ) ) THEN
         IF ( has_bounds ) THEN
            WRITE( out, 5001 )
            DO j = 1, problem%n
               xl = problem%x_l( j )
               xu = problem%x_u( j )
               IF ( problem%m > 0) THEN
                  IF ( xl > - problem%infinity .AND. xu < problem%infinity) THEN
                     WRITE( out, 1000 ) j, problem%vnames( j ),xl,problem%x(j),&
                                        xu, problem%z( j )
                  ELSE IF ( xl > - problem%infinity ) THEN
                     WRITE( out, 1001 ) j, problem%vnames( j ),xl,problem%x(j),&
                                        problem%z( j )
                  ELSE IF ( xu < problem%infinity ) THEN
                     WRITE( out, 1002 ) j, problem%vnames( j ),problem%x(j),xu,&
                                        problem%z( j )
                  ELSE
                     WRITE( out, 1004 ) j, problem%vnames( j ), problem%x( j )
                  END IF
               ELSE
                  IF ( xl > - problem%infinity .AND. xu < problem%infinity )THEN
                     WRITE( out, 1000 ) j, problem%vnames( j ),xl,problem%x(j),&
                                        xu, problem%g( j )
                  ELSE IF ( xl > - problem%infinity ) THEN
                     WRITE( out, 1001 ) j, problem%vnames( j ),xl,problem%x(j),&
                                        problem%g( j )
                  ELSE IF ( xu < problem%infinity ) THEN
                     WRITE( out, 1002 ) j, problem%vnames( j ),problem%x(j),xu,&
                                        problem%g( j )
                  ELSE
                     WRITE( out, 1003 ) j, problem%vnames( j ),problem%x( j ), &
                                        problem%g( j )
                  END IF
               END IF
            END DO
         ELSE
            WRITE( out, 5002 )
            DO j = 1, problem%n
               WRITE( out, 3003 ) j, problem%vnames( j ), problem%x( j ),      &
                                  problem%g( j )
            END DO
         END IF
      ELSE
         IF ( has_bounds ) THEN
            WRITE( out, 5003 )
            DO j = 1, problem%n
               xl = problem%x_l( j )
               xu = problem%x_u( j )
               IF ( problem%m > 0) THEN
                  IF ( xl > - problem%infinity .AND. xu < problem%infinity )THEN
                     WRITE( out, 2000 ) j, xl, problem%x( j ),xu, problem%z( j )
                  ELSE IF ( xl > - problem%infinity ) THEN
                     WRITE( out, 2001 ) j, xl, problem%x( j ), problem%z( j )
                  ELSE IF ( xu < problem%infinity ) THEN
                     WRITE( out, 2002 ) j, problem%x( j ), xu, problem%z( j )
                  ELSE
                     WRITE( out, 2004 ) j, problem%x( j )
                  END IF
               ELSE
                  IF ( xl > - problem%infinity .AND. xu < problem%infinity )THEN
                     WRITE( out, 2000 ) j, xl, problem%x( j ),xu,problem%g( j )
                  ELSE IF ( xl > - problem%infinity ) THEN
                     WRITE( out, 2001 ) j, xl, problem%x( j ), problem%g( j )
                  ELSE IF ( xu < problem%infinity ) THEN
                     WRITE( out, 2002 ) j, problem%x( j ), xu, problem%g( j )
                  ELSE
                     WRITE( out, 2003 ) j, problem%x( j ), problem%g( j )
                  END IF
               END IF
            END DO
         ELSE
            WRITE( out, 5004 )
            DO j = 1, problem%n
               WRITE( out, 4003 ) j, problem%x( j ), problem%g( j )
            END DO
         END IF
      END IF
   ELSE
      k = 1
      DO j = 1, problem%n / 4
         WRITE( out, 5005 )
         IF ( ALLOCATED( problem%vnames ) ) THEN
            WRITE( out, 6005 ) ( jj, jj = j, j+3 )
            WRITE( out, 6004 ) problem%vnames( j:j+3 )
         ELSE
            WRITE( out, 6006 ) ( jj, jj = j, j+3 )
         END IF
         IF ( has_bounds ) WRITE( out, 6000 ) problem%x_l( j:j+3 )
         WRITE( out, 6001 ) problem%x( j:j+3 )
         IF ( has_bounds ) THEN
            WRITE( out, 6002 ) problem%x_u( j:j+3 )
            IF ( problem%m > 0 ) THEN
               WRITE( out, 6003 ) problem%z( j:j+3 )
            ELSE
               WRITE( out, 6003 ) problem%g( j:j+3 )
            END IF
         END IF
         k = k + 4
      END DO
      IF ( k <= problem%n ) THEN
         WRITE( out, 5005 )
         IF ( ALLOCATED( problem%vnames ) ) THEN
            WRITE( out, 6005 ) ( jj, jj = k, problem%n )
            WRITE( out, 6004 ) problem%vnames( k:problem%n )
         ELSE
            WRITE( out, 6006 ) ( jj, jj = k, problem%n )
         END IF
         IF ( has_bounds ) WRITE( out, 6000 ) problem%x_u( k:problem%n)
         WRITE( out, 6001 ) problem%x( k:problem%n )
         IF ( has_bounds ) THEN
            WRITE( out, 6002 ) problem%x_l( k:problem%n )
            IF ( problem%m > 0 ) THEN
               WRITE( out, 6003 ) problem%z( k:problem%n )
            ELSE
               WRITE( out, 6003 ) problem%g( k:problem%n )
            END IF
         END IF
      END IF
   END IF
   WRITE( out, 5005 )

   RETURN

1000 FORMAT( 1x,i5,1x,a10,4(1x,1pE12.4 ) )
1001 FORMAT( 1x,i5,1x,a10,2(1x,1pE12.4 ), 14x, 1pE12.4 )
1002 FORMAT( 1x,i5,1x,a10,1x,1pE12.4, 13x, 2(1x,1pE12.4) )
1003 FORMAT( 1x,i5,1x,a10,14x,1pE12.4, 14x, 1pE12.4 )
1004 FORMAT( 1x,i5,1x,a10,14x,1pE12.4)
2000 FORMAT( 1x,' X(', i5, ') ', 4(1x,1pE12.4 ) )
2001 FORMAT( 1x,' X(', i5, ') ', 2(1x,1pE12.4 ), 14x, 1pE12.4 )
2002 FORMAT( 1x,' X(', i5, ') ', 1x,1pE12.4, 13x, 2(1x,1pE12.4) )
2003 FORMAT( 1x,' X(', i5, ') ', 14x,1pE12.4,14x,1pE12.4 )
2004 FORMAT( 1x,' X(', i5, ') ', 14x,1pE12.4 )
3003 FORMAT( 1x,i5,1x,a10,1x, 1pE12.4,1x,1pE12.4 )
4003 FORMAT( 1x,' X(', i5, ') ', 1pE12.4,1x,1pE12.4 )
5000 FORMAT(/,11x,'+--------------------------------------------------------+',&
            /,11x,'|',18x,'Problem : ',a10,18x,'|',                            &
            /,11x,'+--------------------------------------------------------+',&
            / )
5001 FORMAT(5x,                                                                &
           'j Name           Lower         Value        Upper     Dual value',/)
5002 FORMAT(5x,'j Name           Value      Dual value',/)
5003 FORMAT(' Variable       Lower        Value        Upper     Dual value',/)
5004 FORMAT(' Variable       Value      Dual value',/)
5005 FORMAT(/)
6000 FORMAT(1x,'LOWER BOUND',3x,1pe13.6,3(4x,1pe13.6))
6001 FORMAT(1x,'VALUE      ',3x,1pe13.6,3(4x,1pe13.6))
6002 FORMAT(1x,'UPPER BOUND',3x,1pe13.6,3(4x,1pe13.6))
6003 FORMAT(1x,'DUAL VALUE ',3x,1pe13.6,3(4x,1pe13.6))
6004 FORMAT(1x,'NAME       ',3x,a13,3(4x,a13))
6005 FORMAT(1x,'INDEX      ',3x,i13,3(4x,i13))
6006 FORMAT(1x,'           ',4(1x,'X(',i13,')'))

   END SUBROUTINE NLPT_write_variables

!===============================================================================
!===============================================================================

   SUBROUTINE NLPT_write_constraints( problem, out )

!  Writes the constraints and associated multipliers. This routine assumes that
!  problem%c, problem%c_l, problem%c_u, problem%y, problem%equation and
!  problem%linear are associated and contain relevant values.  The
!  constraints' names are used whenever problem%cnames is associated, but this
!  is not mandatory.

!  Arguments

   TYPE ( NLPT_problem_type ), INTENT( IN ) :: problem

!         the setup problem

   INTEGER, INTENT( IN ) :: out

!         the printout device number.

!  Programming: Ph. Toint, November 2002.
!
!===============================================================================

!  Local variables

   INTEGER               :: i, k, ii
   LOGICAL               :: has_types
   REAL( KIND = wp )     :: cl, cu
   CHARACTER( LEN = 10 ) :: type
   CHARACTER( LEN = 80 ) :: types

!  Return if there is no constraint.

   IF ( problem%m <= 0 ) RETURN

!  Print banner.

   WRITE( out, 5000 ) problem%pname

!  Print constraints' values, bounds and multipliers.

   IF ( problem%m < 1000000 ) THEN
      IF ( ALLOCATED( problem%cnames ) ) THEN
         WRITE( out, 5001 )
         DO i = 1, problem%m
            cl = problem%c_l( i )
            cu = problem%c_u( i )
            IF ( ALLOCATED( problem%equation ) ) THEN
               IF( problem%equation( i ) ) THEN
                 type( 1:4 ) = '[=] '
               ELSE
                 type( 1:4 ) = '    '
               END IF
            ELSE
               type( 1:4 ) = '    '
            END IF
            IF ( ALLOCATED( problem%linear ) ) THEN
               IF ( problem%linear( i ) ) THEN
                  type( 5:10 ) = 'linear'
               ELSE
                  type( 5:10 ) = '      '
               END IF
            ELSE
               type( 5:10 ) = '      '
            END IF
            IF ( cl > - problem%infinity .AND. cu < problem%infinity ) THEN
               WRITE( out, 1000 ) i, problem%cnames( i ), cl, problem%c( i ),  &
                                  cu, problem%y( i ), type
            ELSE IF ( cl > - problem%infinity ) THEN
               WRITE( out, 1001 ) i, problem%cnames( i ), cl, problem%c( i ),  &
                                  problem%y( i ), type
            ELSE IF ( cu < problem%infinity ) THEN
               WRITE( out, 1002 ) i, problem%cnames( i ), problem%c( i ), cu,  &
                                  problem%y( i ), type
            ELSE
               WRITE( out, 1003 ) i, problem%cnames( i ), problem%c( i ), type
            END IF
         END DO
      ELSE
         WRITE( out, 5002 )
         DO i = 1, problem%m
            cl = problem%c_l( i )
            cu = problem%c_u( i )
            IF ( ALLOCATED( problem%equation ) ) THEN
               IF( problem%equation( i ) ) THEN
                 type( 1:4 ) = '[=] '
               ELSE
                 type( 1:4 ) = '    '
               END IF
            ELSE
               type( 1:4 ) = '    '
            END IF
            IF ( ALLOCATED( problem%linear ) ) THEN
               IF ( problem%linear( i ) ) THEN
                  type( 5:10 ) = 'linear'
               ELSE
                  type( 5:10 ) = '      '
               END IF
            ELSE
               type( 5:10 ) = '      '
            END IF
            IF ( cl > - problem%infinity .AND. cu < problem%infinity ) THEN
               WRITE( out, 3000 ) i, cl, problem%c( i ), cu, problem%y( i ),type
            ELSE IF ( cl > - problem%infinity ) THEN
               WRITE( out, 3001 ) i, cl, problem%c( i ), problem%y( i ), type
            ELSE IF ( cu < problem%infinity ) THEN
               WRITE( out, 3002 ) i, problem%c( i ), cu, problem%y( i ), type
            ELSE
               WRITE( out, 3003 ) i, problem%c( i ), type
            END IF
         END DO
      END IF
   ELSE
      k = 1
      has_types = ALLOCATED( problem%equation ) .OR.                          &
                  ALLOCATED( problem%linear )
      DO i = 1, problem%m / 4
         WRITE( out, 5004 )
         IF ( ALLOCATED( problem%vnames ) ) THEN
            WRITE( out, 6005 ) ( ii, ii = i, i+3 )
            WRITE( out, 6004 ) problem%vnames( i:i+3 )
         ELSE
            WRITE( out, 6006 ) ( ii, ii = i, i+3 )
         END IF
         WRITE( out, 6000 ) problem%c_u( i:i+3 )
         WRITE( out, 6001 ) problem%c( i:i+3 )
         WRITE( out, 6002 ) problem%c_l( i:i+3 )
         WRITE( out, 6003 ) problem%y( i:i+3 )
         IF ( has_types ) THEN
            WRITE( types, 6007 )
            IF ( ALLOCATED( problem%equation ) ) THEN
               IF ( problem%equation( i   ) ) types( 15:19 ) = '[=] '
               IF ( problem%equation( i+1 ) ) types( 33:36 ) = '[=] '
               IF ( problem%equation( i+2 ) ) types( 50:53 ) = '[=] '
               IF ( problem%equation( i+3 ) ) types( 67:70 ) = '[=] '
            END IF
            IF ( ALLOCATED( problem%linear ) ) THEN
               IF ( problem%linear( i   ) ) types( 20:26 ) = ' linear'
               IF ( problem%linear( i+1 ) ) types( 37:43 ) = ' linear'
               IF ( problem%linear( i+2 ) ) types( 54:60 ) = ' linear'
               IF ( problem%linear( i+3 ) ) types( 71:77 ) = ' linear'
            END IF
            WRITE( out, 6008 ) types
         END IF
         k = k + 4
      END DO
      IF ( k <= problem%n ) THEN
         WRITE( out, 5004 )
         IF ( ALLOCATED( problem%vnames ) ) THEN
            WRITE( out, 6005 ) ( ii, ii = k, problem%m )
            WRITE( out, 6004 ) problem%vnames( k:problem%m )
         ELSE
            WRITE( out, 6006 ) ( ii, ii = k, problem%m )
         END IF
         WRITE( out, 6000 ) problem%c_u( k:problem%m )
         WRITE( out, 6001 ) problem%c( k:problem%m )
         WRITE( out, 6002 ) problem%c_l( k:problem%m )
         WRITE( out, 6003 ) problem%y( k:problem%m )
         IF ( has_types ) THEN
            WRITE( types, 6007 )
            IF ( ALLOCATED( problem%equation ) ) THEN
               IF ( problem%equation( k   ) ) types( 15:19 ) = '[=] '
               IF ( k + 1 <= problem%m ) THEN
                  IF ( problem%equation( k+1 ) ) types( 33:36 ) = '[=] '
                  IF ( k + 2 <= problem%m ) THEN
                     IF ( problem%equation( k+2 ) ) types( 50:53 ) = '[=] '
                  END IF
               END IF
            END IF
            IF ( ALLOCATED( problem%linear ) ) THEN
               IF ( problem%linear( k   ) ) types( 20:26 ) = ' linear'
               IF ( k + 1 <= problem%m ) THEN
                  IF ( problem%linear( k+1 ) ) types( 37:43 ) = ' linear'
                  IF ( k + 2 <= problem%m ) THEN
                     IF ( problem%linear( k+2 ) ) types( 54:60 ) = ' linear'
                  END IF
               END IF
            END IF
            WRITE( out, 6008 ) types
         END IF
      END IF
   END IF
   WRITE( out, 5004 )

   RETURN

1000 FORMAT( 1x,i5,1x,a10,4(1x,1pE12.4 ),1x,a10 )
1001 FORMAT( 1x,i5,1x,a10,2(1x,1pE12.4 ),14x, 1pE12.4,1x,a10 )
1002 FORMAT( 1x,i5,1x,a10,13x,3(1x,1pE12.4),1x,a10 )
1003 FORMAT( 1x,i5,1x,a10,14x,1pE12.4,1x,a10 )
3000 FORMAT( 1x,' C(',i5,') ',4(1x,1pE12.4 ),1x,a10 )
3001 FORMAT( 1x,' C(',i5,') ',2(1x,1pE12.4 ),14x,1pE12.4,1x,a10 )
3002 FORMAT( 1x,' C(',i5,') ',13x,3(1x,1pE12.4),1x,a10 )
3003 FORMAT( 1x,' C(',i5,') ',14x,1pE12.4,1x,a10 )
5000 FORMAT(/,11x,'+--------------------------------------------------------+',&
            /,11x,'|',18x,'Problem : ',a10,18x,'|',                            &
            /,11x,'+--------------------------------------------------------+',&
            / )
5001 FORMAT(5x,                                                                &
           'i Name           Lower         Value        Upper     Dual value',/)
5002 FORMAT(' Constraint     Lower        Value        Upper     Dual value',/)
5004 FORMAT(/)
6000 FORMAT(1x,'LOWER BOUND',3x,1pe13.6,3(4x,1pe13.6))
6001 FORMAT(1x,'VALUE      ',3x,1pe13.6,3(4x,1pe13.6))
6002 FORMAT(1x,'UPPER BOUND',3x,1pe13.6,3(4x,1pe13.6))
6003 FORMAT(1x,'DUAL VALUE ',3x,1pe13.6,3(4x,1pe13.6))
6004 FORMAT(1x,'NAME       ',3x,a13,3(4x,a13))
6005 FORMAT(1x,'INDEX      ',3x,i13,3(4x,i13))
6006 FORMAT(1x,'           ',4(1x,'X(',i13,')'))
6007 FORMAT('TYPE',76x)
6008 FORMAT(a80)

   END SUBROUTINE NLPT_write_constraints

!===============================================================================
!===============================================================================

   SUBROUTINE NLPT_write_problem( problem, out, print_level )

!  Write the problem's current values. This routine assumes that
!  problem%x, is associated and contain relevant. The bounds are printed
!  whenever problem%x_l and problem%x_u are associated. Moreover, it is also
!  assumed in this case that problem%g is associated when problem%m = 0, and
!  that problem%z, problem%c, problem%c_l, problem%c_u, problem%y,
!  problem%equation and problem%linear are associated when problem%m > 0. The
!  variables/constraints' names are used whenever
!  problem%vnames/problem%cnames is associated, but this is not mandatory.

!  Arguments:

   TYPE ( NLPT_problem_type ), INTENT( IN  ) :: problem

!             The problem.

   INTEGER, INTENT( IN ) :: out

!             The device number for output.

   INTEGER, INTENT( IN ) :: print_level

!             The level of requested printout.

!  Programming: Ph. Toint, November 2002.

!===============================================================================

!  Local variables

   INTEGER            :: i, j
   REAL ( KIND = wp ) :: max_violation

   IF ( print_level == SILENT .OR. problem%n <= 0 ) RETURN

   CALL NLPT_write_stats( problem, out )

   IF ( print_level >= ACTION ) THEN
      CALL NLPT_write_variables( problem, out )
      WRITE( out, 105 ) problem%f
      WRITE( out, 106 )
      CALL TOOLS_output_vector( problem%n, problem%g, out )
      IF ( print_level >= DETAILS .AND. ALLOCATED( problem%H_val ) ) THEN
         WRITE( out, 107 )
         SELECT CASE ( problem%H_type )
         CASE ( COORDINATE )
            CALL TOOLS_output_matrix_C( problem%H_ne, problem%H_val,           &
                                        problem%H_row, problem%H_col, out )
         CASE ( SPARSE )
            CALL TOOLS_output_matrix_S( problem%H_ne, problem%H_val,           &
                                        problem%H_ptr, problem%H_col, out )
         CASE ( DENSE )
            CALL TOOLS_output_matrix_D( problem%n, problem%m,                  &
                                        problem%H_val, .TRUE., out )
         END SELECT
      END IF
      IF ( problem%m > 0 ) THEN
         CALL NLPT_write_constraints( problem, out )
         IF ( print_level >= DETAILS .AND. ALLOCATED( problem%J_val ) )THEN
            WRITE( out, 108 )
            SELECT CASE ( problem%J_type )
            CASE ( COORDINATE )
               CALL TOOLS_output_matrix_C( problem%J_ne,problem%J_val,         &
                                           problem%J_row, problem%J_col, out )
            CASE ( SPARSE )
               CALL TOOLS_output_matrix_S( problem%J_ne,problem%J_val,         &
                                           problem%J_ptr, problem%J_col, out )
            CASE ( DENSE )
               CALL TOOLS_output_matrix_D( problem%m, problem%m,               &
                                           problem%J_val, .FALSE., out )
            END SELECT
         END IF
      END IF
   ELSE
      WRITE( out, 102 ) NRM2( problem%n, problem%x, 1 )
      WRITE( out, 101 ) problem%f, NRM2( problem%n, problem%g, 1 )
      max_violation = ZERO
      IF ( ALLOCATED( problem%x_l ) .AND. ALLOCATED( problem%x_u ) ) THEN
         DO j = 1, problem%n
            IF ( problem%x( j ) > problem%x_u( j ) ) THEN
               max_violation = MAX( max_violation, problem%x(j)-problem%x_u(j) )
            ELSE IF ( problem%x( j ) < problem%x_l( j ) ) THEN
               max_violation = MAX( max_violation, problem%x_l(j)-problem%x(j) )
            END IF
         END DO
         WRITE( out, 103 ) max_violation
      END IF
      IF ( problem%m > 0 ) THEN
         max_violation = ZERO
         DO i = 1, problem%m
            IF ( problem%c( i ) > problem%c_u( i ) ) THEN
               max_violation = MAX( max_violation,                             &
                                             problem%c( i ) - problem%c_u( i ) )
            ELSE IF ( problem%c( i ) < problem%c_l( i ) ) THEN
               max_violation = MAX( max_violation,                             &
                                             problem%c_l( i ) - problem%c( i ) )
            END IF
         END DO
         WRITE( out, 104 ) max_violation
      END IF
   END IF

!  Indicate end of problem.

   WRITE( out, 109 )

   RETURN

!  Formats

101 FORMAT( ' Objective function value     = ',1PE14.7,/,                      &
            ' Norm of its gradient         = ',1PE14.7 )
102 FORMAT( ' Norm of the current X        = ',1PE14.7 )
103 FORMAT( ' Maximal bound violation      = ',1PE14.7 )
104 FORMAT( ' Maximal constraint violation = ',1PE14.7 )
105 FORMAT( ' OBJECTIVE FUNCTION value     = ',1PE14.7 )
106 FORMAT( /, ' GRADIENT of the objective function:' )
107 FORMAT( /, ' Lower triangle of the HESSIAN of the Lagrangian:')
108 FORMAT( /, ' JACOBIAN matrix:' )
109 FORMAT( /,11x,'-------------------- END OF PROBLEM ----------------------',&
            /)

   END SUBROUTINE NLPT_write_problem

!===============================================================================
!===============================================================================

   SUBROUTINE NLPT_cleanup( problem, print_level, out )

!  Deallocates the pointers used by problem.

      TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: problem

!            the setup problem;

      INTEGER, INTENT( IN ), OPTIONAL :: print_level

!            the level of print out requested;

      INTEGER, INTENT( IN ), OPTIONAL :: out

!            the device number for output;

!     Programming: Ph. Toint, November 2002.
!
!===============================================================================
!===============================================================================

!  Local variables

   INTEGER :: iout, plevel

   IF ( PRESENT( out ) ) THEN
       iout = out
   ELSE
       iout = 6
   END IF

   IF ( PRESENT( print_level ) )  THEN
      plevel = print_level
   ELSE
      plevel = SILENT
   END IF

   IF ( plevel >= TRACE ) WRITE( iout, 1000 )

!  Clean up the various workspace arrays.

   IF ( plevel >= DETAILS ) WRITE( iout, 1001 ) problem%pname

   IF ( ALLOCATED( problem%vnames   ) ) DEALLOCATE( problem%vnames   )
   IF ( ALLOCATED( problem%x        ) ) DEALLOCATE( problem%x        )
   IF ( ALLOCATED( problem%x_scale  ) ) DEALLOCATE( problem%x_scale  )
   IF ( ALLOCATED( problem%x_status ) ) DEALLOCATE( problem%x_status )
   IF ( ALLOCATED( problem%x_l      ) ) DEALLOCATE( problem%x_l      )
   IF ( ALLOCATED( problem%x_u      ) ) DEALLOCATE( problem%x_u      )
   IF ( ALLOCATED( problem%z        ) ) DEALLOCATE( problem%z        )
   IF ( ALLOCATED( problem%g        ) ) DEALLOCATE( problem%g        )
   IF ( ALLOCATED( problem%H_val    ) ) DEALLOCATE( problem%H_val    )
   IF ( ALLOCATED( problem%H_row    ) ) DEALLOCATE( problem%H_row    )
   IF ( ALLOCATED( problem%H_col    ) ) DEALLOCATE( problem%H_col    )
   IF ( ALLOCATED( problem%H_ptr    ) ) DEALLOCATE( problem%H_ptr    )
   IF ( ALLOCATED( problem%H%id     ) ) DEALLOCATE( problem%H%id     )
   IF ( ALLOCATED( problem%H%type   ) ) DEALLOCATE( problem%H%type   )
   IF ( ALLOCATED( problem%H%val    ) ) DEALLOCATE( problem%H%val    )
   IF ( ALLOCATED( problem%H%row    ) ) DEALLOCATE( problem%H%row    )
   IF ( ALLOCATED( problem%H%col    ) ) DEALLOCATE( problem%H%col    )
   IF ( ALLOCATED( problem%H%ptr    ) ) DEALLOCATE( problem%H%ptr    )
   IF ( ALLOCATED( problem%c_scale  ) ) DEALLOCATE( problem%c_scale  )
   IF ( ALLOCATED( problem%c_status ) ) DEALLOCATE( problem%c_status )
   IF ( ALLOCATED( problem%cnames   ) ) DEALLOCATE( problem%cnames   )
   IF ( ALLOCATED( problem%c        ) ) DEALLOCATE( problem%c        )
   IF ( ALLOCATED( problem%c_l      ) ) DEALLOCATE( problem%c_l      )
   IF ( ALLOCATED( problem%c_u      ) ) DEALLOCATE( problem%c_u      )
   IF ( ALLOCATED( problem%equation ) ) DEALLOCATE( problem%equation )
   IF ( ALLOCATED( problem%linear   ) ) DEALLOCATE( problem%linear   )
   IF ( ALLOCATED( problem%J_val    ) ) DEALLOCATE( problem%J_val    )
   IF ( ALLOCATED( problem%J_row    ) ) DEALLOCATE( problem%J_row    )
   IF ( ALLOCATED( problem%J_col    ) ) DEALLOCATE( problem%J_col    )
   IF ( ALLOCATED( problem%J_ptr    ) ) DEALLOCATE( problem%J_ptr    )
   IF ( ALLOCATED( problem%J%id     ) ) DEALLOCATE( problem%J%id     )
   IF ( ALLOCATED( problem%J%type   ) ) DEALLOCATE( problem%J%type   )
   IF ( ALLOCATED( problem%J%val    ) ) DEALLOCATE( problem%J%val    )
   IF ( ALLOCATED( problem%J%row    ) ) DEALLOCATE( problem%J%row    )
   IF ( ALLOCATED( problem%J%col    ) ) DEALLOCATE( problem%J%col    )
   IF ( ALLOCATED( problem%J%ptr    ) ) DEALLOCATE( problem%J%ptr    )
   IF ( ALLOCATED( problem%y        ) ) DEALLOCATE( problem%y        )
   IF ( ALLOCATED( problem%gL       ) ) DEALLOCATE( problem%gL       )

   IF ( plevel >= TRACE ) THEN
      IF ( plevel >= DETAILS ) WRITE( iout, 1002 )
      WRITE( iout, 1003 )
   END IF

   RETURN

!  Formats

1000 FORMAT( /,' **************************************************',/,        &
               ' *                                                *',/,        &
               ' *               NLPT problem cleanup             *',/,        &
               ' *                                                *',/,        &
               ' **************************************************',/ )
1001 FORMAT( ' cleaning up problem', a )
1002 FORMAT( '   problem cleanup successful' )
1003 FORMAT( /,' *********************** Bye **********************',/ )

   END SUBROUTINE NLPT_cleanup

!===============================================================================
!===============================================================================

   END MODULE GALAHAD_NLPT_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*  END GALAHAD NLPT  M O D U L E  *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*                                 *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
