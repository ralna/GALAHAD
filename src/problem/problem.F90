! THIS VERSION: GALAHAD 4.1 - 2022-12-20 AT 13:00 GMT.

#include "galahad_modules.h"

   SUBROUTINE FUN( X, f, C, data )
   USE GALAHAD_USERDATA_precision
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: X
   REAL ( KIND = rp_ ), INTENT( OUT ) :: f
   REAL ( KIND = rp_ ), OPTIONAL, INTENT( OUT ), DIMENSION( : ) :: C
   TYPE ( GALAHAD_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE FUN

   SUBROUTINE GRAD( X, G, Y, J, data )
   USE GALAHAD_SMT_precision
   USE GALAHAD_USERDATA_precision
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: X
   REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( : ) :: G
   TYPE( SMT_type ), OPTIONAL, INTENT( OUT ) :: J
   TYPE ( GALAHAD_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE GRAD

   SUBROUTINE HESS( X, H, Y, i, data )
   USE GALAHAD_SMT_precision
   USE GALAHAD_USERDATA_precision
   INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: i
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: X
   REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: Y
   TYPE( SMT_type ), INTENT( OUT ) :: H
   TYPE ( GALAHAD_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE HESS

   SUBROUTINE HPROD( P, R, X, Y, data )
   USE GALAHAD_SMT_precision
   USE GALAHAD_USERDATA_precision
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: P
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: R
   REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: X, Y
   TYPE ( GALAHAD_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
!  P product with
!  R result of product added to R (i.e, R <- R + prod)
!  transpose if transpose wanted
   END SUBROUTINE HPROD

   SUBROUTINE JPROD( P, R, transpose, X, data )
   USE GALAHAD_SMT_precision
   USE GALAHAD_USERDATA_precision
   LOGICAL, INTENT( IN ) :: transpose
   REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: P
   REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: R
   REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( : ) :: X
   TYPE ( GALAHAD_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
   END SUBROUTINE JPROD

!  SUBROUTINE PREC( U, V[, X, Y, data ] )
!  USE GALAHAD_SMT_precision
!  USE GALAHAD_USERDATA_precision
!  TYPE ( GALAHAD_userdata_type ), OPTIONAL, INTENT( INOUT ) :: data
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

