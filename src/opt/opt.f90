! THIS VERSION: GALAHAD 2.6 - 10/12/2014 AT 08:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ O P T   M O D U L E  *-*-*-*-*-*-*-*

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. March 1st 2012

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_OPT_double

!     ------------------------------------------------
!    |                                                |
!    | Compute criticality measures for the nonlinear |
!    | programming problem                            |
!    |                                                |
!    |  minimize          f (x)                       |
!    |  subject to c^l <= c (x) <= c^u                |
!    |  and        x^l <=    x  <= x^u                |
!    |                                                |
!     ------------------------------------------------

     USE GALAHAD_NORMS_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: OPT_primal_infeasibility, OPT_dual_infeasibility,               &
               OPT_complementary_slackness, OPT_multiplier_norm

     INTERFACE OPT_primal_infeasibility
       MODULE PROCEDURE OPT_primal_infeasibility_general,                      &
                        OPT_primal_infeasibility_bounds
     END INTERFACE OPT_primal_infeasibility

     INTERFACE OPT_complementary_slackness
       MODULE PROCEDURE OPT_complementary_slackness_general,                   &
                        OPT_complementary_slackness_bounds
     END INTERFACE OPT_complementary_slackness

     INTERFACE OPT_multiplier_norm
       MODULE PROCEDURE OPT_multiplier_norm_general,                           &
                        OPT_multiplier_norm_bounds
     END INTERFACE OPT_multiplier_norm

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp

   CONTAINS

!-*-*-  O P T _ P R I M A L _ I N F E A S I B I L I T Y   F U C T I O N S  -*-*-

     FUNCTION OPT_primal_infeasibility_bounds( n, X, X_l, X_u, norm, SCALE )

!  Find an appropriate norm of the infeasibility of x in
!    x^l <= x <= x_u

     REAL ( KIND = wp ) :: OPT_primal_infeasibility_bounds

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u, X
     INTEGER, INTENT( IN ), OPTIONAL :: norm
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( n ) :: SCALE

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used

!  trivial case

     IF ( n <= 0 ) THEN
       OPT_primal_infeasibility_bounds = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  use the scaled ||scale*vector|| norm

       IF ( PRESENT( SCALE ) ) THEN

!  find the one-norm of the infeasibility

         IF ( norm_used == 1 ) THEN
           OPT_primal_infeasibility_bounds = ONE_NORM( SCALE( : n ) *          &
             MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )

!  find the two-norm of the infeasibility

         ELSE IF ( norm_used == 2 ) THEN
           OPT_primal_infeasibility_bounds = TWO_NORM( SCALE( : n ) *          &
             MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )

!  find the infinity-norm of the infeasibility

         ELSE
           OPT_primal_infeasibility_bounds = INFINITY_NORM( SCALE( : n ) *     &
             MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )
         END IF

!  use the unscaled ||vector|| norm

       ELSE

!  find the one-norm of the infeasibility

         IF ( norm_used == 1 ) THEN
           OPT_primal_infeasibility_bounds = ONE_NORM(                         &
             MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )

!  find the two-norm of the infeasibility

         ELSE IF ( norm_used == 2 ) THEN
           OPT_primal_infeasibility_bounds = TWO_NORM(                         &
             MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )

!  find the infinity-norm of the infeasibility

         ELSE
           OPT_primal_infeasibility_bounds = INFINITY_NORM(                    &
             MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )
         END IF
       END IF
     END IF

     RETURN

!  End of function OPT_primal_infeasibility_bounds

     END FUNCTION OPT_primal_infeasibility_bounds

     FUNCTION OPT_primal_infeasibility_general( n, X, X_l, X_u,                &
                                                m, C, C_l, C_u, norm, SCALE )

!  Find an appropriate norm of the infeasibility of x in
!    x^l <= x <= x_u and c^l <= c(x) <= c^u

     REAL ( KIND = wp ) :: OPT_primal_infeasibility_general

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n, m
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u, X
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u, C
     INTEGER, INTENT( IN ), OPTIONAL :: norm
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( m ) :: SCALE

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used
     REAL ( KIND = wp ) :: norm_x, norm_c, norm_max

!  trivial cases

     IF ( m <= 0 ) THEN
       OPT_primal_infeasibility_general =                                      &
         OPT_primal_infeasibility_bounds( n, X, X_l, X_u, norm )
     ELSE IF ( n <= 0 ) THEN
       OPT_primal_infeasibility_general = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  find the one-norm of the infeasibility

       IF ( norm_used == 1 ) THEN
         OPT_primal_infeasibility_general =                                    &
           ONE_NORM( MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ),        &
                          zero ) ) +                                           &
           ONE_NORM( MAX( C_l( : m ) - C( : m ), C( : m ) - C_u( : m ),        &
                          zero ) )

!  find the two-norm of the infeasibility

       ELSE IF ( norm_used == 2 ) THEN
         norm_x =                                                              &
           TWO_NORM( MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ), zero ) )
         norm_c =                                                              &
           TWO_NORM( MAX( C_l( : m ) - C( : m ), C( : m ) - C_u( : m ), zero ) )
         norm_max = MAX( norm_x, norm_c )
         IF ( norm_max > zero ) THEN
           OPT_primal_infeasibility_general = norm_max *                       &
             SQRT( ( norm_x / norm_max ) ** 2 + ( norm_c / norm_max ) ** 2 )
         ELSE
           OPT_primal_infeasibility_general = zero
         END IF

!  find the infinity-norm of the infeasibility

       ELSE
         OPT_primal_infeasibility_general = MAX(                               &
           INFINITY_NORM( MAX( X_l( : n ) - X( : n ), X( : n ) - X_u( : n ),   &
                               zero ) ),                                       &
           INFINITY_NORM( MAX( C_l( : m ) - C( : m ), C( : m ) - C_u( : m ),   &
                               zero ) ) )
       END IF
     END IF

     RETURN

!  End of function OPT_primal_infeasibility_general

     END FUNCTION OPT_primal_infeasibility_general

!-*-*-*-  O P T _ D U A L _ I F E A S I B I L I T Y   F U C T I O N  -*-*-*-

     FUNCTION OPT_dual_infeasibility( n, G, norm )

!  Find an appropriate norm of the infeasibility of g(x)

     REAL ( KIND = wp ) :: OPT_dual_infeasibility

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: G
     INTEGER, INTENT( IN ), OPTIONAL :: norm

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used

!  trivial case

     IF ( n <= 0 ) THEN
       OPT_dual_infeasibility = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  find the one-norm of the infeasibility

       IF ( norm_used == 1 ) THEN
         OPT_dual_infeasibility = ONE_NORM( G( : n ) )

!  find the two-norm of the infeasibility

       ELSE IF ( norm_used == 2 ) THEN
         OPT_dual_infeasibility = TWO_NORM( G( : n ) )

!  find the infinity-norm of the infeasibility

       ELSE
         OPT_dual_infeasibility = INFINITY_NORM( G( : n ) )
       END IF
     END IF

     RETURN

!  End of function OPT_dual_infeasibility

     END FUNCTION OPT_dual_infeasibility

!-   O P T _ C O M P L E M E N T A R Y _ S L A C K N E S S   F U C T I O N S  -

     FUNCTION OPT_complementary_slackness_bounds( n, X, X_l, X_u, Z, norm )

!  Find an appropriate norm of the complementary slackness
!    min( x - x^l, x_u - x ) . z

     REAL ( KIND = wp ) :: OPT_complementary_slackness_bounds

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u, X, Z
     INTEGER, INTENT( IN ), OPTIONAL :: norm

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used

!  trivial case

     IF ( n <= 0 ) THEN
       OPT_complementary_slackness_bounds = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  find the one-norm of the complementarity

       IF ( norm_used == 1 ) THEN
         OPT_complementary_slackness_bounds =                                  &
           ONE_NORM( MIN( ABS( X_l( : n ) - X( : n ) ),                        &
                          ABS( X( : n ) - X_u( : n ) ) ) * Z( : n ) )

!  find the two-norm of the complementarity

       ELSE IF ( norm_used == 2 ) THEN
         OPT_complementary_slackness_bounds =                                  &
           TWO_NORM( MIN( ABS( X_l( : n ) - X( : n ) ),                        &
                          ABS( X( : n ) - X_u( : n ) ) ) * Z( : n ) )

!  find the infinity-norm of the complementarity

       ELSE
         OPT_complementary_slackness_bounds =                                  &
           INFINITY_NORM( MIN( ABS( X_l( : n ) - X( : n ) ),                   &
                               ABS( X( : n ) - X_u( : n ) ) ) * Z( : n ) )
       END IF
     END IF

     RETURN

!  End of function OPT_complementary_slackness_bounds

     END FUNCTION OPT_complementary_slackness_bounds

     FUNCTION OPT_complementary_slackness_general( n, X, X_l, X_u, Z,          &
                                                   m, C, C_l, C_u, Y, norm )

!  Find an appropriate norm of the complementary slackness
!   ( min( x - x^l, x_u - x ) . z )
!   ( min( c - c^l, c_u - c ) . y )

     REAL ( KIND = wp ) :: OPT_complementary_slackness_general

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n, m
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X_l, X_u, X, Z
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C_l, C_u, C, Y
     INTEGER, INTENT( IN ), OPTIONAL :: norm

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used
     REAL ( KIND = wp ) :: norm_x, norm_c, norm_max

!  trivial cases

     IF ( m <= 0 ) THEN
       OPT_complementary_slackness_general =                                   &
         OPT_complementary_slackness_bounds( n, X, X_l, X_u, Z, norm )
     ELSE IF ( n <= 0 ) THEN
       OPT_complementary_slackness_general = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  find the one-norm of the complementarity

       IF ( norm_used == 1 ) THEN
         OPT_complementary_slackness_general =                                 &
           ONE_NORM( MIN( ABS( X_l( : n ) - X( : n ) ),                        &
                          ABS( X( : n ) - X_u( : n ) ) ) * Z( : n ) ) +        &
           ONE_NORM( MIN( ABS( C_l( : m ) - C( : m ) ),                        &
                          ABS( C( : m ) - C_u( : m ) ) ) * Y( : m ) )

!  find the two-norm of the complementarity

       ELSE IF ( norm_used == 2 ) THEN
         norm_x =                                                              &
           TWO_NORM( MIN( ABS( X_l( : n ) - X( : n ) ),                        &
                          ABS( X( : n ) - X_u( : n ) ) ) * Z( : n ) )
         norm_c =                                                              &
           TWO_NORM( MIN( ABS( C_l( : m ) - C( : m ) ),                        &
                          ABS( C( : m ) - C_u( : m ) ) ) * Y( : m ) )
         norm_max = MAX( norm_x, norm_c )
         IF ( norm_max > zero ) THEN
           OPT_complementary_slackness_general = norm_max *                    &
             SQRT( ( norm_x / norm_max ) ** 2 + ( norm_c / norm_max ) ** 2 )
         ELSE
           OPT_complementary_slackness_general = zero
         END IF

!  find the infinity-norm of the complementarity

       ELSE
         OPT_complementary_slackness_general = MAX(                            &
           INFINITY_NORM( MIN( ABS( X_l( : n ) - X( : n ) ),                   &
                               ABS( X( : n ) - X_u( : n ) ) ) * Z( : n ) ),    &
           INFINITY_NORM( MIN( ABS( C_l( : m ) - C( : m ) ),                   &
                               ABS( C( : m ) - C_u( : m ) ) ) * Y( : m ) ) )
       END IF
     END IF

     RETURN

!  End of function OPT_complementary_slackness_general

     END FUNCTION OPT_complementary_slackness_general

!-*-*-*-   O P T _ M U L T I P L I E R _ N O R M    F U C T I O N S  -*-*-*-

     FUNCTION OPT_multiplier_norm_bounds( n, Z, norm )

!  Find an appropriate norm of the Lagrange multipliers, z

     REAL ( KIND = wp ) :: OPT_multiplier_norm_bounds

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Z
     INTEGER, INTENT( IN ), OPTIONAL :: norm

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used

!  trivial case

     IF ( n <= 0 ) THEN
       OPT_multiplier_norm_bounds = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  find the one-norm of the complementarity

       IF ( norm_used == 1 ) THEN
         OPT_multiplier_norm_bounds = ONE_NORM( Z( : n ) )

!  find the two-norm of the complementarity

       ELSE IF ( norm_used == 2 ) THEN
         OPT_multiplier_norm_bounds = TWO_NORM( Z( : n ) )

!  find the infinity-norm of the complementarity

       ELSE
         OPT_multiplier_norm_bounds = INFINITY_NORM( Z( : n ) )
       END IF
     END IF

     RETURN

!  End of function OPT_multiplier_norm_bounds

     END FUNCTION OPT_multiplier_norm_bounds

     FUNCTION OPT_multiplier_norm_general( n, Z, m, Y, norm )

!  Find an appropriate norm of the Lagrange multipliers ( z^T, y^T )^T

     REAL ( KIND = wp ) :: OPT_multiplier_norm_general

!--------------------------------
!   D u m m y   A r g u m e n t s
!---------------------------------

     INTEGER, INTENT( IN ) :: n, m
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: Z
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: Y
     INTEGER, INTENT( IN ), OPTIONAL :: norm

!--------------------------------
!   L o c a l   V a r i a b l e s
!--------------------------------

     INTEGER :: norm_used
     REAL ( KIND = wp ) :: norm_z, norm_y, norm_max

!  trivial cases

     IF ( m <= 0 ) THEN
       OPT_multiplier_norm_general = OPT_multiplier_norm_bounds( n, Z, norm )
     ELSE IF ( n <= 0 ) THEN
       OPT_multiplier_norm_general = zero

!  select the appropriate norm

     ELSE
       IF ( PRESENT( norm ) ) THEN
         IF ( norm == 1 .OR. norm == 2 ) THEN
           norm_used = norm
         ELSE
           norm_used = 0
         END IF
       ELSE
         norm_used = 0
       END IF

!  find the one-norm of the complementarity

       IF ( norm_used == 1 ) THEN
         OPT_multiplier_norm_general                                           &
           = ONE_NORM( Z( : n ) ) + ONE_NORM( Y( : m ) )

!  find the two-norm of the complementarity

       ELSE IF ( norm_used == 2 ) THEN
         norm_z = TWO_NORM( Z( : n ) )
         norm_y = TWO_NORM( Y( : m ) )
         norm_max = MAX( norm_z, norm_y )
         IF ( norm_max > zero ) THEN
           OPT_multiplier_norm_general = norm_max *                            &
             SQRT( ( norm_z / norm_max ) ** 2 + ( norm_y / norm_max ) ** 2 )
         ELSE
           OPT_multiplier_norm_general = zero
         END IF

!  find the infinity-norm of the complementarity

       ELSE
         OPT_multiplier_norm_general                                           &
           = MAX( INFINITY_NORM( Z( : n ) ), INFINITY_NORM( Y( : m ) ) )
       END IF
     END IF

     RETURN

!  End of function OPT_multiplier_norm_general

     END FUNCTION OPT_multiplier_norm_general

!  End of module GALAHAD_OPT

   END MODULE GALAHAD_OPT_double
