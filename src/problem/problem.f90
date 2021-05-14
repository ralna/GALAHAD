   SUBROUTINE FUN( X, f, C, data )
   USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: X
   REAL ( KIND = wp ), INTENT( OUT ) :: f
   REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ), DIMENSION( : ) :: C
   TYPE ( NLPT_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE FUN


   SUBROUTINE GRAD( X, G, Y, J, data )
   USE GALAHAD_SMT_double
   USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: X
   REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: G
   TYPE( SMT_type ), OPTIONAL, INTENT( OUT ) :: J
   TYPE ( NLPT_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE GRAD


   SUBROUTINE HESS( X, H, Y, i, data )
   USE GALAHAD_SMT_double
   USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   INTEGER, OPTIONAL, INTENT( IN ) :: i
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: X
   REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Y
   TYPE( SMT_type ), INTENT( OUT ) :: H
   TYPE ( NLPT_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE HESS


   SUBROUTINE HPROD( P, R, X, Y, data )
   USE GALAHAD_SMT_double
   USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: P
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: R
   REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: X, Y
   TYPE ( NLPT_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
!  P product with
!  R result of product added to R (i.e, R <- R + prod)
!  transpose if transpose wanted

   END SUBROUTINE HPROD


   SUBROUTINE JPROD( P, R, transpose, X, data )
   USE GALAHAD_SMT_double
   USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: P
   REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: R
   REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: X
   TYPE ( NLPT_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE JPROD


!  SUBROUTINE PREC( U, V[, X, Y, data ] )
!  USE GALAHAD_SMT_double
!  USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type
!  INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!  TYPE ( NLPT_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
!  END SUBROUTINE PREC

!arguments:

!  X primal variables
!  Y dual variables
!  [] optional
!  F objective value
!  C constraint values
!  G objective/Lagrangian gradient (obj if Y absent)
!  J constraint Jacobian
!  H Hessian (obj if Y is absent, constraint i if present)
!  U vector to be preconditioned
!  V result of preconditioning
!  data - derived type containing (rank-1?) allocatable/pointer
!      arrays of type integer, real, logical (etc) for user-provided data 

