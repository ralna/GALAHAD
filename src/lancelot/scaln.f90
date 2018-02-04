! THIS VERSION: GALAHAD 2.6 - 12/03/2014 AT 13:30 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  S C A L N   M O D U L E  -*-*-*-*-*-*-*-

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 1st 1995

   MODULE LANCELOT_SCALN_double

     USE LANCELOT_types_double, ONLY: LANCELOT_problem_type, LANCELOT_data_type
     USE LANCELOT_OTHERS_double, ONLY: OTHERS_fdgrad_save_type, OTHERS_fdgrad
     USE LANCELOT_INITW_double
     IMPLICIT NONE
     
     PRIVATE
     PUBLIC :: SCALN_save_type, SCALN_get_scalings
     
!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!  The SCALN_save_type derived type

     TYPE :: SCALN_save_type
       INTEGER :: nsets, igetfd, ntype, ntotel, nvrels, nnza
       LOGICAL :: altriv
       TYPE ( OTHERS_fdgrad_save_type ) :: OTHERS
     END TYPE SCALN_save_type

   CONTAINS

!-*-*-*-  L A N C E L O T  -B-  SCALN_get_scalings  S U B R O U T I N E  -*-*

     SUBROUTINE SCALN_get_scalings(                                            &
                      prob, RANGE , data, ncalc, ICALC, FT, GVALS,             &
                      FUVALS, lfuval, stopga, stopca, scaleg, scalev,          &
                      iout, iprint, buffer, fdgrad, status, S, ELFUN , GROUP )

!  To calculate suitable variable and constraint scalings for a nonlinear
!  programming problem input to LANCELOT

!  status must be set to 0 on entry, and will have the value 0 on
!  successful exit. If status is negative on exit, the user must set
!  appropriate values and re-enter. Possible values of status are:
!   -1  compute the values of the element functions and their derivatives
!       at prob%X, and return the values in FUVALS. All other arguments 
!       must be unaltered on re-entry
!   -2  compute the values of the first derivatives of the the group 
!       functions evaluated at FT, and return the values in GVALS( :, 2 ). 
!       All other arguments must be unaltered on re-entry

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LANCELOT_problem_type ), INTENT( INOUT ) :: prob
     TYPE ( LANCELOT_data_type ), INTENT( INOUT ) :: data
     TYPE ( SCALN_save_type ), INTENT( INOUT ) :: S
     INTEGER, INTENT( IN ) :: lfuval, iprint, iout, buffer
     INTEGER, INTENT( INOUT ) :: status, ncalc
     REAL ( KIND = wp ), INTENT( INOUT ) :: stopca, stopga
     LOGICAL, INTENT( IN ) :: scaleg, scalev, fdgrad
     INTEGER, INTENT( INOUT ), DIMENSION( MAX( prob%ng, prob%nel ) ) :: ICALC
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng ) :: FT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( prob%ng, 3 ) :: GVALS
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS

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
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
       INTEGER, INTENT( IN ) :: INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
       INTEGER, INTENT( IN ) :: ICALCF(LCALCF)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(LXVALU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(LEPVLU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(LFUVAL)
       END SUBROUTINE ELFUN 

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

     OPTIONAL :: ELFUN , GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER, PARAMETER :: lmin = 10000
     LOGICAL :: internal_el, internal_gr

     internal_el = PRESENT( ELFUN )
     internal_gr = PRESENT( GROUP )

! Initial entry: set up data

     IF ( status == 0 ) THEN 

       IF ( prob%n <= 0 .OR. prob%ng <= 0 ) RETURN

!  If the element functions are to be evaluated internally, check that
!  the user has supplied appropriate information

       IF ( internal_el ) THEN
         IF ( ALLOCATED( prob%ISTEPA ) .AND. ALLOCATED( prob%ITYPEE ) .AND.  &
              ALLOCATED( prob%EPVALU ) ) THEN
           IF ( SIZE( prob%ISTEPA ) < prob%nel + 1 .OR.                        &
                SIZE( prob%ITYPEE ) < prob%nel ) THEN
             status = 10 ; RETURN ; END IF
           IF ( SIZE( prob%EPVALU ) < prob%ISTEPA( prob%nel + 1 ) - 1) THEN
             status = 10 ; RETURN ; END IF
         ELSE
           status = 10 ; RETURN
         END IF
       END IF

!  Do the same if the group functions are to be evaluated internally.

       IF ( internal_gr ) THEN
         IF ( ALLOCATED( prob%ISTGPA ) .AND. ALLOCATED( prob%ITYPEG ) .AND.  &
              ALLOCATED( prob%GPVALU ) ) THEN
           IF ( SIZE( prob%ISTGPA ) < prob%ng + 1 .OR.                         &
                SIZE( prob%ITYPEG ) < prob%ng ) THEN
             status = 11 ; RETURN ; END IF
           IF ( SIZE( prob%GPVALU ) < prob%ISTGPA( prob%ng + 1 ) - 1 ) THEN
             status = 11 ; RETURN ; END IF
         ELSE
           status = 11 ; RETURN
         END IF
       END IF

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

     END IF
     
!  ========================================================
!  Call the main subroutine to perform the bulk of the work
!  ========================================================

!  Both internal element and group evaluations will be performed
!  -------------------------------------------------------------

     IF ( internal_el .AND. internal_gr ) THEN
       CALL SCALN_get_scalings_main(                                           &
           prob%n, prob%ng, prob%nel, lfuval,                                  &
           prob%X , prob%Y, ncalc, ICALC, prob%ICNA,                           &
           prob%ISTADA, prob%IELING, prob%INTVAR, prob%ISTADG, prob%ISTAEV,    &
           prob%IELVAR, prob%ISTADH, prob%A, prob%B, FT, GVALS, FUVALS,        &
           stopga, stopca, prob%GSCALE, prob%ESCALE, prob%VSCALE, scaleg,      &
           scalev, prob%KNDOFG, prob%ITYPEE, prob%GXEQX, prob%INTREP, RANGE ,  &
           iout, iprint, buffer, fdgrad, status, S,                            &
!    workspace                                                               
           data%ISYMMD, data%ISYMMH, data%ISWKSP, data%ISTAJC, data%ISTAGV,    &
           data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR, data%IUSED ,    &
           data%ITYPER, data%ISSWTR, data%ISSITR, data%ISET  , data%ISVSET,    &
           data%INVSET, data%LIST_elements, data%FUVALS_temp, data%X0,         &
           data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,                &
           data%W_in, data%H_el, data%H_in, data%WTRANS,                       &
           data%S%EXTEND%lwtran, data%ITRANS, data%S%EXTEND%litran,            &
           data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,                 &
           data%LINK_elem_uses_var,                                            &
           data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,                &
           ELFUN = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU,         &
           GROUP = GROUP , ISTGPA = prob%ISTGPA,                               &
           ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
     
!  Just internal element evaluations will be performed
!  ---------------------------------------------------

     ELSE IF ( internal_el ) THEN
       CALL SCALN_get_scalings_main(                                           &
           prob%n, prob%ng, prob%nel, lfuval,                                  &
           prob%X , prob%Y, ncalc, ICALC, prob%ICNA,                           &
           prob%ISTADA, prob%IELING, prob%INTVAR, prob%ISTADG, prob%ISTAEV,    &
           prob%IELVAR, prob%ISTADH, prob%A, prob%B, FT, GVALS, FUVALS,        &
           stopga, stopca, prob%GSCALE, prob%ESCALE, prob%VSCALE, scaleg,      &
           scalev, prob%KNDOFG, prob%ITYPEE, prob%GXEQX, prob%INTREP, RANGE ,  &
           iout, iprint, buffer, fdgrad, status, S,                            &
!    workspace                                                               
           data%ISYMMD, data%ISYMMH, data%ISWKSP, data%ISTAJC, data%ISTAGV,    &
           data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR, data%IUSED ,    &
           data%ITYPER, data%ISSWTR, data%ISSITR, data%ISET  , data%ISVSET,    &
           data%INVSET, data%LIST_elements, data%FUVALS_temp, data%X0,         &
           data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,                &
           data%W_in, data%H_el, data%H_in, data%WTRANS,                       &
           data%S%EXTEND%lwtran, data%ITRANS, data%S%EXTEND%litran,            &
           data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,                 &
           data%LINK_elem_uses_var,                                            &
           data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,                &
           ELFUN = ELFUN , ISTEPA = prob%ISTEPA, EPVALU = prob%EPVALU )
     
!  Just internal group evaluations will be performed
!  -------------------------------------------------

     ELSE IF ( internal_gr ) THEN
       CALL SCALN_get_scalings_main(                                           &
           prob%n, prob%ng, prob%nel, lfuval,                                  &
           prob%X , prob%Y, ncalc, ICALC, prob%ICNA,                           &
           prob%ISTADA, prob%IELING, prob%INTVAR, prob%ISTADG, prob%ISTAEV,    &
           prob%IELVAR, prob%ISTADH, prob%A, prob%B, FT, GVALS, FUVALS,        &
           stopga, stopca, prob%GSCALE, prob%ESCALE, prob%VSCALE, scaleg,      &
           scalev, prob%KNDOFG, prob%ITYPEE, prob%GXEQX, prob%INTREP, RANGE ,  &
           iout, iprint, buffer, fdgrad, status, S,                            &
!    workspace                                                               
           data%ISYMMD, data%ISYMMH, data%ISWKSP, data%ISTAJC, data%ISTAGV,    &
           data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR, data%IUSED ,    &
           data%ITYPER, data%ISSWTR, data%ISSITR, data%ISET  , data%ISVSET,    &
           data%INVSET, data%LIST_elements, data%FUVALS_temp, data%X0,         &
           data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,                &
           data%W_in, data%H_el, data%H_in, data%WTRANS,                       &
           data%S%EXTEND%lwtran, data%ITRANS, data%S%EXTEND%litran,            &
           data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,                 &
           data%LINK_elem_uses_var,                                            &
           data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min,                &
           GROUP = GROUP, ISTGPA = prob%ISTGPA,                                &
           ITYPEG = prob%ITYPEG, GPVALU = prob%GPVALU )
     
!  Element and group evaluations will be performed via reverse communication
!  -------------------------------------------------------------------------

     ELSE
       CALL SCALN_get_scalings_main(                                           &
           prob%n, prob%ng, prob%nel, lfuval,                                  &
           prob%X , prob%Y, ncalc, ICALC, prob%ICNA,                           &
           prob%ISTADA, prob%IELING, prob%INTVAR, prob%ISTADG, prob%ISTAEV,    &
           prob%IELVAR, prob%ISTADH, prob%A, prob%B, FT, GVALS, FUVALS,        &
           stopga, stopca, prob%GSCALE, prob%ESCALE, prob%VSCALE, scaleg,      &
           scalev, prob%KNDOFG, prob%ITYPEE, prob%GXEQX, prob%INTREP, RANGE ,  &
           iout, iprint, buffer, fdgrad, status, S,                            &
!    workspace                                                               
           data%ISYMMD, data%ISYMMH, data%ISWKSP, data%ISTAJC, data%ISTAGV,    &
           data%ISVGRP, data%ISLGRP, data%IGCOLJ, data%IVALJR, data%IUSED ,    &
           data%ITYPER, data%ISSWTR, data%ISSITR, data%ISET  , data%ISVSET,    &
           data%INVSET, data%LIST_elements, data%FUVALS_temp, data%X0,         &
           data%IW_asmbl, data%NZ_comp_w, data%W_ws, data%W_el,                &
           data%W_in, data%H_el, data%H_in, data%WTRANS,                       &
           data%S%EXTEND%lwtran, data%ITRANS, data%S%EXTEND%litran,            &
           data%S%EXTEND%lwtran_min, data%S%EXTEND%litran_min,                 &
           data%LINK_elem_uses_var,                                            &
           data%S%EXTEND%l_link_e_u_v, data%S%EXTEND%llink_min )
     END IF

     RETURN

!  End of subroutine SCALN_get_scalings

     END SUBROUTINE SCALN_get_scalings

!-*-*-  L A N C E L O T  -B-  SCALN_get_scalings_main  S U B R O U T I N E  -*-*

     SUBROUTINE SCALN_get_scalings_main(                                       &
                      n, ng, nel, lfuval,                                      &
                      X , Y , ncalc , ICALC , ICNA  , ISTADA,                  &
                      IELING, INTVAR, ISTADG, ISTAEV, IELVAR, ISTADH,          &
                      A , B , FT    , GVALS , FUVALS, stopga, stopca,          &
                      GSCALE, ESCALE, VSCALE, scaleg, scalev,                  &
                      KNDOFG, ITYPEE, GXEQX , INTREP, RANGE , iout  , iprint,  &
                      buffer, fdgrad, status, S,                               &
!  workspace
                      ISYMMD, ISYMMH, ISWKSP, ISTAJC, ISTAGV, ISVGRP,          &
                      ISLGRP, IGCOLJ, IVALJR, IUSED , ITYPER, ISSWTR,          &
                      ISSITR, ISET  , ISVSET, INVSET, LIST_elements,           &
                      FUVALS_temp, X_temp,                                     &
                      IW_asmbl, NZ_comp_w, W_ws, W_el, W_in, H_el, H_in,       &
                      WTRANS, lwtran, ITRANS, litran,                          &
                      lwtran_min, litran_min, LINK_elem_uses_var  ,            &
                      l_link_e_u_v, llink_min,                                 &
                      ELFUN , ISTEPA, EPVALU, GROUP , ISTGPA, ITYPEG, GPVALU )

!  To calculate suitable variable and constraint scalings for a nonlinear
!  programming problem input to LANCELOT

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, ng, nel, lfuval
     TYPE ( SCALN_save_type ), INTENT( INOUT ) :: S
     INTEGER, INTENT( IN ) :: iprint, iout, buffer
     INTEGER, INTENT( INOUT ) :: status, ncalc
     REAL ( KIND = wp ), INTENT( INOUT ) :: stopca, stopga
     LOGICAL, INTENT( IN ) :: scaleg, scalev, fdgrad
     INTEGER, INTENT( IN ), DIMENSION( ng      ) :: KNDOFG
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( IN ), DIMENSION( ng  + 1 ) :: ISTADA, ISTADG
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTAEV, INTVAR
     INTEGER, INTENT( IN ), DIMENSION( ISTADG( ng + 1 ) - 1 ) :: IELING
     INTEGER, INTENT( IN ), DIMENSION( ISTAEV( nel + 1 ) - 1 ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( ISTADA( ng  + 1 ) - 1 ) :: ICNA
     INTEGER, INTENT( OUT ), DIMENSION( nel + 1 ) :: ISTADH
     INTEGER, INTENT( INOUT ),                                                 &
              DIMENSION( MAX( ng, nel ) ) :: ICALC
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( IN ),                                         &
            DIMENSION( ISTADA( ng + 1 ) - 1  ) :: A
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: B
     REAL ( KIND = wp ), INTENT( IN  ),                                        &
            DIMENSION( ISTADG( ng + 1 ) - 1  ) :: ESCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: VSCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng ) :: FT, Y, GSCALE
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng, 3 ) :: GVALS
     REAL ( KIND = wp ), INTENT( INOUT ),                                      &
            DIMENSION( lfuval ) :: FUVALS
     LOGICAL, INTENT( IN    ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN    ), DIMENSION( nel ) :: INTREP

!--------------------------------------------------------------
!   D u m m y   A r g u m e n t s  f o r   W o r k s p a c e 
!--------------------------------------------------------------

     INTEGER, INTENT( INOUT ) :: lwtran, litran, lwtran_min, litran_min
     INTEGER, INTENT( INOUT ) :: l_link_e_u_v, llink_min
     INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: ISYMMH
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISYMMD, ISWKSP, ISTAJC 
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTAGV, ISVGRP, ISLGRP, IGCOLJ 
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVALJR, IUSED , ITYPER, ISSWTR
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISSITR, ISET  , ISVSET, INVSET 
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: LIST_elements
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW_asmbl
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_comp_w
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_ws
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_el
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_in
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ITRANS, LINK_elem_uses_var
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FUVALS_temp, X_temp
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WTRANS 

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE

       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE

!  Interface block for ELFUN 

       SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, ncalcf, ITYPEE, ISTAEV,      &
                          IELVAR, INTVAR, ISTADH, ISTEPA, ICALCF, ltypee,      &
                          lstaev, lelvar, lntvar, lstadh, lstepa, lcalcf,      &
                          lfuval, lxvalu, lepvlu, ifflag, ifstat )
       INTEGER, INTENT( IN ) :: ncalcf, ifflag, ltypee, lstaev, lelvar, lntvar
       INTEGER, INTENT( IN ) :: lstadh, lstepa, lcalcf, lfuval, lxvalu, lepvlu
       INTEGER, INTENT( OUT ) :: ifstat
       INTEGER, INTENT( IN ) :: ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
       INTEGER, INTENT( IN ) :: INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
       INTEGER, INTENT( IN ) :: ICALCF(LCALCF)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: XVALUE(LXVALU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ) :: EPVALU(LEPVLU)
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( INOUT ) :: FUVALS(LFUVAL)
       END SUBROUTINE ELFUN 

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

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( nel + 1 ) :: ISTEPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng + 1 ) :: ISTGPA
     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: ITYPEG
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: EPVALU
     REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( : ) :: GPVALU
     OPTIONAL :: ELFUN , GROUP

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i , ig, nnzgrj, lstagv, lendgv, iestev, nvarel, iel, l
     INTEGER :: ig1, j, ii, k , ieintv, nvargp, lfxi  , lgxi
     INTEGER :: lhxi  , lggfx , ldx   , lgrjac, maxsel, nin
     INTEGER :: ntotin, lnguvl, lnhuvl, alloc_status
     REAL ( KIND = wp ) :: gi    , scale , scalee, ftt   , grpmax
     REAL ( KIND = wp ) :: varmin, tol
     LOGICAL :: centrl, reallocate, external_el, external_gr
     CHARACTER ( LEN = 80 ) :: bad_alloc

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IRNGRJ, ICNGRJ
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRPSCA, GRJAC
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WK_n, WK_t

     external_el = .NOT. PRESENT( ELFUN )
     external_gr = .NOT. PRESENT( GROUP )

!  Branch to the interior of the code if a re-entry is being made

     SELECT CASE ( status )
     CASE ( - 1 ) ; GO TO 110
     CASE ( - 2 ) ; GO TO 160
     END SELECT

!  0. Initialize data

!  Ensure that all element functions are evaluated

     DO i = 1, nel ; ICALC( i ) = i ; END DO

!  If finite-difference gradients are required, set up the necessary
!  data structures (this is a bit wasteful as not all the partitions
!  are used)

     S%ntotel = ISTADG( ng  + 1 ) - 1
     S%nvrels = ISTAEV( nel + 1 ) - 1
     S%nnza   = ISTADA( ng  + 1 ) - 1
     CALL INITW_initialize_workspace(                                          &
         n, ng, nel, S%ntotel, S%nvrels, S%nnza  , n,  &
         nvargp, IELING, ISTADG, IELVAR, ISTAEV, INTVAR, ISTADH, ICNA, ISTADA, &
         ITYPEE, GXEQX , INTREP, S%altriv,.FALSE., fdgrad, lfxi, lgxi, lhxi  , &
         lggfx , ldx   , lnguvl, lnhuvl, ntotin, S%ntype ,                     &
         S%nsets , maxsel, RANGE , iprint, iout, buffer,                       &
! workspace
         lwtran, litran, lwtran_min, litran_min,  l_link_e_u_v, llink_min,     &
         ITRANS, LINK_elem_uses_var, WTRANS, ISYMMD, ISWKSP, ISTAJC, ISTAGV,   &
         ISVGRP, ISLGRP, IGCOLJ, IVALJR, IUSED , ITYPER, ISSWTR, ISSITR,       &
         ISET  , ISVSET, INVSET, LIST_elements, ISYMMH, IW_asmbl, NZ_comp_w,   &
         W_ws, W_el, W_in, H_el, H_in, status, alloc_status, bad_alloc,        &
         .FALSE. ) 

     IF ( status > 0 ) THEN ; WRITE( iout, 2100 ) ; GO TO 610 ; END IF
   
     S%igetfd = 0

!  Set up the starting addresses for the element gradients with respect to
!  their internal variables

!  1. Calculate the required element and group derivatives

!  Return to the calling program to obtain the element function and derivative
!  values at the initial point

     ncalc = nel
     status = - 1
     IF ( external_el ) RETURN

!  Evaluate the element function and derivative value
       
 100 CONTINUE
     CALL ELFUN ( FUVALS, X, EPVALU, ncalc, ITYPEE, ISTAEV, IELVAR, INTVAR,    &
                  ISTADH, ISTEPA, ICALC, nel, nel + 1, ISTAEV( nel + 1 ) - 1,  &
                  nel + 1, nel + 1, nel + 1, nel, lfuval, n,                   &
                  ISTEPA( nel + 1 ) - 1, 1, i )
     IF ( .NOT. fdgrad )                                                       &
       CALL ELFUN ( FUVALS, X, EPVALU, ncalc, ITYPEE, ISTAEV, IELVAR, INTVAR,  &
                    ISTADH, ISTEPA, ICALC, nel, nel + 1, ISTAEV( nel + 1 ) - 1,&
                    nel + 1, nel + 1, nel + 1, nel, lfuval, n,                 &
                    ISTEPA( nel + 1 ) - 1, 2, i )

 110 CONTINUE

!  If finite-difference gradients are used, compute their values

     IF ( fdgrad .AND. nel > 0 ) THEN

!  Store the values of the nonlinear elements for future use

       IF ( S%igetfd == 0 ) THEN
     
          reallocate = .TRUE.
          IF ( ALLOCATED( X_temp ) ) THEN
            IF ( SIZE( X_temp ) < n ) THEN ; DEALLOCATE( X_temp )
            ELSE ; reallocate = .FALSE. ; END IF
          END IF
          IF ( reallocate ) THEN 
            ALLOCATE( X_temp( n ), STAT = alloc_status )
            IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'X_temp'; GO TO 620
            END IF
          END IF
       
          reallocate = .TRUE.
          IF ( ALLOCATED( FUVALS_temp ) ) THEN
            IF ( SIZE( FUVALS_temp ) < nel ) THEN
               DEALLOCATE( FUVALS_temp )
            ELSE ; reallocate = .FALSE. ; END IF
          END IF
          IF ( reallocate ) THEN 
            ALLOCATE( FUVALS_temp( nel ), STAT = alloc_status )
            IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'FUVAL_'; GO TO 620
            END IF
          END IF
       
          X_temp( : n ) = X( : n )
          FUVALS_temp( : nel ) = FUVALS( : nel )
          centrl = .TRUE.
       END IF

!  Obtain a further set of differences

       CALL OTHERS_fdgrad( n, nel, lfuval,                                     &
                           S%ntotel, S%nvrels, S%nsets , IELVAR,               &
                           ISTAEV, IELING, ICALC, ncalc, INTVAR,               &
                           S%ntype, X_temp, X, FUVALS, centrl, S%igetfd,       &
                           S%OTHERS, ISVSET, ISET, INVSET, ISSWTR,             &
                           ISSITR, ITYPER, LIST_elements,                      &
                           LINK_elem_uses_var, WTRANS, ITRANS )
       IF ( S%igetfd > 0 ) THEN
         IF ( external_el ) THEN ; RETURN ; ELSE ; GO TO 100 ; END IF
       END IF

!  Restore the values of the nonlinear elements at X

       S%igetfd = S%nsets + 1
       FUVALS( : nel ) = FUVALS_temp( : nel )
     END IF

!  Compute the group argument values FT

     DO ig = 1, ng
       ftt = - B( ig )

!  Include the contribution from the linear element

       DO j = ISTADA( ig ), ISTADA( ig + 1 ) - 1
         IF ( ICNA( j ) <= n ) ftt = ftt + A( j ) * X( ICNA( j ) )
       END DO

!  Include the contributions from the nonlinear elements

!      ftt = ftt + SUM( ESCALE( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 )          &
!            * FUVALS( IELING( ISTADG( ig ) : ISTADG( ig + 1 ) - 1 ) ) )
       DO j = ISTADG( ig ), ISTADG( ig + 1 ) - 1
         ftt = ftt + ESCALE( j ) * FUVALS( IELING( j ) )
       END DO
       FT( ig ) = ftt
       ICALC( ig ) = ig
     END DO

!  Compute the group function values

     IF ( S%altriv ) THEN 
       GVALS( : ng, 2 ) = FT( : ng ) 
     ELSE

!  If necessary, return to the calling program to obtain the group function
!  and derivative values at the initial point

       ncalc = ng
       status = - 2
       IF ( external_gr ) RETURN

!  Evaluate the group function derivatives

       CALL GROUP ( GVALS, ng, FT, GPVALU, ncalc, ITYPEG, ISTGPA, ICALC,       &
                    ng, ng + 1, ng, ng, ISTGPA( ng + 1 ) - 1, .TRUE., i )
     END IF

 160 CONTINUE
     IF ( .NOT. S%altriv ) THEN
       WHERE ( GXEQX( : ng ) ) GVALS( : ng, 2 ) = FT( : ng)
     END IF

!  2. Form the Jacobian matrix of the nonlinear function (group( 1 ), .... ,
!     group(NG))(transpose). 

     lgrjac = ISTAGV( ng + 1 ) - 1

!    Allocate space to hold the Jacobian in co-ordinate form

     reallocate = .TRUE.
     IF ( ALLOCATED( IRNGRJ ) ) THEN
       IF ( SIZE( IRNGRJ ) < lgrjac ) THEN ; DEALLOCATE( IRNGRJ )
       ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( IRNGRJ( lgrjac ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IRNGRJ'; GO TO 620 ; END IF
     END IF
     
     reallocate = .TRUE.
     IF ( ALLOCATED( ICNGRJ ) ) THEN
       IF ( SIZE( ICNGRJ ) < lgrjac ) THEN ; DEALLOCATE( ICNGRJ )
       ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( ICNGRJ( lgrjac ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ICNGRJ'; GO TO 620 ; END IF
     END IF
     
     reallocate = .TRUE.
     IF ( ALLOCATED( GRJAC ) ) THEN
       IF ( SIZE( GRJAC ) < lgrjac ) THEN ; DEALLOCATE( GRJAC )
       ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( GRJAC( lgrjac ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'GRJAC'; GO TO 620 ; END IF
     END IF
     
     reallocate = .TRUE.
     IF ( ALLOCATED( GRPSCA ) ) THEN
       IF ( SIZE( GRPSCA ) < ng ) THEN ; DEALLOCATE( GRPSCA )
       ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( GRPSCA( ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'GRPSCA'; GO TO 620 ; END IF
     END IF
     
     reallocate = .TRUE.
     IF ( ALLOCATED( WK_n ) ) THEN
       IF ( SIZE( WK_n ) < n ) THEN ; DEALLOCATE( WK_n )
       ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( WK_n( n ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'WK_n'; GO TO 620 ; END IF
     END IF
     
     reallocate = .TRUE.
     IF ( ALLOCATED( WK_t ) ) THEN
       IF ( SIZE( WK_t ) < MAX( ng, n ) ) THEN ; DEALLOCATE( WK_t )
       ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( WK_t( MAX( ng, n ) ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'WK_t'; GO TO 620 ; END IF
     END IF
  
!  Consider the IG-th group

     nnzgrj = 0
     DO ig = 1, ng
       ig1 = ig + 1
       lstagv = ISTAGV( ig )
       lendgv = ISTAGV( ig1 ) - 1

!  Initialize the group derivative to zero

       WK_n( ISVGRP( lstagv : lendgv ) ) = zero

!  See if the group has any nonlinear elements

       DO ii = ISTADG( ig ), ISTADG( ig1 ) - 1
         iel = IELING( ii )
         ieintv = INTVAR( iel )
         iestev = ISTAEV( iel )
         nvarel = ISTAEV( iel + 1 ) - iestev
         scalee = ESCALE( ii )
         IF ( INTREP( iel ) ) THEN

!  The IEL-th element has an internal representation

           nin = INTVAR( iel + 1 ) - ieintv
           CALL RANGE ( iel, .TRUE., FUVALS( ieintv : ieintv + nin - 1 ),      &
                        WK_t( : nvarel ), nvarel, nin, ITYPEE( iel ),          &
                        nin, nvarel )
!DIR$ IVDEP
           DO i = 1, nvarel
             j = IELVAR( iestev )
             WK_n( j ) = WK_n( j ) + scalee * WK_t( i )
             iestev = iestev + 1
           END DO
         ELSE

!  The IEL-th element has no internal representation

!DIR$ IVDEP
           DO i = 1, nvarel
             j = IELVAR( iestev )
             WK_n( j ) = WK_n( j ) + scalee * FUVALS( ieintv )
             ieintv = ieintv + 1
             iestev = iestev + 1
           END DO
         END IF
       END DO

!  Include the contribution from the linear element

!DIR$ IVDEP
       DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
         IF ( ICNA( k ) <= n ) THEN
           j = ICNA( k )
           WK_n( j ) = WK_n( j ) + A( k )
         END IF
       END DO

!  Store the values of the nonzero entries of the gradient of the IG-th group
!  in GRJAC, along with their row and column indices in the overall Jacobian

       IF ( .NOT. GXEQX( ig ) ) THEN
         gi = GSCALE( ig ) * GVALS( ig , 2 )
       ELSE
         gi = GSCALE( ig )
       END IF
!DIR$ IVDEP
       DO j = lstagv, lendgv
         i = ISVGRP( j )
         nnzgrj = nnzgrj + 1
         IRNGRJ( nnzgrj ) = ig
         ICNGRJ( nnzgrj ) = i
         GRJAC( nnzgrj ) = gi * WK_n( i )
         IF ( iprint >= 100 ) WRITE( iout, 2070 ) ig, i, GRJAC( nnzgrj )
       END DO
     END DO

!  3. Remove all entries which are smaller than TOL times the
!     largest entries in both their rows and columns

!  Find the largest entries, in absolute value, in each row and column

     tol = SQRT( EPSILON( one ) )
     WK_n = zero; WK_t( : ng ) = zero
     DO k = 1, nnzgrj
       i = IRNGRJ( k )
       j = ICNGRJ( k )
       gi = ABS( GRJAC( k ) )
       WK_t( i ) = MAX( WK_t( i ), gi )
       WK_n( j ) = MAX( WK_n( j ), gi )
     END DO

!  Remove small entries

     l = 0
     DO k = 1, nnzgrj
       i = IRNGRJ( k )
       j = ICNGRJ( k )
       gi = ABS( GRJAC( k ) ) / tol
       IF ( gi > WK_t( i ) .OR. gi > WK_n( j ) ) THEN
         l = l + 1
         IRNGRJ( l ) = i
         ICNGRJ( l ) = j
         GRJAC( l ) = GRJAC( k )
       ELSE
         IF ( iprint >= 100 ) WRITE( iout, 2080 ) i, j, GRJAC( k )
       END IF
     END DO
     nnzgrj = l

!  4. Apply the row and column equilibrating scheme of Curtis and Reid to the
!     Jacobian

     CALL SCALN_curtis_reid( ng, n, nnzgrj, GRJAC, IRNGRJ,                     &
                             ICNGRJ, GRPSCA, VSCALE( : n ) )

!  5. Calculate the scale factors

!  Obtain the smallest variable and group scale factor

     grpmax = zero
     varmin = HUGE( one )
     VSCALE( : n ) = EXP( VSCALE( : n ) )
     varmin = MIN( MINVAL( VSCALE( : n ) ), varmin )
!DIR$ IVDEP
     DO ig = 1, ng
       IF ( KNDOFG( ig ) /= 1 ) THEN
         GRPSCA( ig ) = EXP( GRPSCA( ig ) )
         grpmax = MAX( GRPSCA( ig ),grpmax )
       END IF
     END DO

!  Scale the factors relative to their largest members

     VSCALE( : n ) = ten ** ANINT( LOG10( VSCALE( : n ) / varmin ) )
     WHERE ( KNDOFG( : ng ) /= 1 ) GRPSCA( : ng ) =                            &
             ten ** ANINT( LOG10( GRPSCA( : ng ) / grpmax ) )

!  Use the variable scalings within LANCELOT

     IF ( scalev ) THEN
       scale = zero
!DIR$ IVDEP
       VSCALE( : n ) = VSCALE( : n )
       scale = MAX( scale, MAXVAL( VSCALE( : n ) ) )
       stopga = stopga / scale
     END IF

!  Use the group scalings within LANCELOT. Scale the Lagrange multipliers

     IF ( scaleg ) THEN
       scale = one
!DIR$ IVDEP
       DO ig = 1, ng
         IF ( KNDOFG( ig ) /= 1 ) THEN
           GSCALE( ig ) = GSCALE( ig ) * GRPSCA( ig )
           Y( ig ) = Y( ig ) / GRPSCA( ig )
           scale = MIN( scale, GSCALE( ig ) )
         END IF
       END DO
       stopca = stopca * scale
     END IF

!  Successful conclusion to the calculation

     IF ( iprint > 0 ) THEN
       IF ( scaleg ) THEN
         WRITE( iout, 2040 )
         j = 1
         DO i = 1, ng
           IF ( KNDOFG( i ) == 1 ) THEN
             IF ( i - 1 >= j )                                                 &
               WRITE( iout, 2060 ) ( ig, GRPSCA( ig ), ig = j, i - 1 )
              j = i + 1
           END IF
         END DO
         IF ( ng >= j ) WRITE( iout, 2060 ) ( ig, GRPSCA( ig ), ig = j, ng )
       END IF
       IF ( scalev ) WRITE( iout, 2050 ) ( i, VSCALE( i ),i = 1, n )
       WRITE( iout, 2090 ) stopga, stopca
     END IF
     status = 0

!  De-allocate workspace arrays

     DEALLOCATE( IRNGRJ, ICNGRJ, GRJAC, GRPSCA, WK_n, WK_t )
     IF ( fdgrad .AND. nel > 0 ) DEALLOCATE( X_temp, FUVALS_temp )

 610 CONTINUE
     DEALLOCATE( ISTAGV, ISVGRP )
     
     RETURN

!  Unsuccessful returns

 620 CONTINUE
     status = 1000 + alloc_status
     WRITE( iout, 2600 ) TRIM( bad_alloc ), alloc_status
     RETURN

!  Non-executable statements

 2040  FORMAT( /,' Multiply groups by the factors:' )
 2050  FORMAT( /,' Divide variables by the factors:', / 4( I6, ES12.4 ) )
 2060  FORMAT( 4( I6, ES12.4 ) )
 2070  FORMAT( ' Row ', I6, ' column ', I6, ' value ', ES12.4 )
 2080  FORMAT( ' Row ', I6, ' column ', I6, ' value ', ES12.4,' removed' )
 2090  FORMAT( /,' Scaled projected gradient tolerance = ', ES12.4, /,         &
                 ' Scaled constraint tolerance         = ', ES12.4 )
 2100  FORMAT( ' ** Return from SCALN. Insufficient workspace ' )
 2600  FORMAT( ' ** Message from -SCALN_get_scalings-', /,                     &
               ' Allocation error, for ', A, ', status = ', I0 )

!  End of subroutine SCALN_get_scalings_main

     END SUBROUTINE SCALN_get_scalings_main

!-*-*-*-  L A N C E L O T  -B-  SCALN_curtis_reid  S U B R O U T I N E  -*-*

     SUBROUTINE SCALN_curtis_reid(  m , n , ne, A , IRNA  , ICNA  , R , C )

!   Compute row and column scalings using the algorithm of Curtis and Reid
!   (J.I.M.A. 10 (1972) 118-124) by approximately minimizing the function

!        sum (nonzero A) ( log(|a_ij|) - r_i - c_j)^2

!   The required scalings are then e^-r and e^-c respectively

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m , n , ne
     INTEGER, INTENT( IN  ), DIMENSION( ne ) :: IRNA, ICNA
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ne ) :: A
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m  ) :: R
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n  ) :: C

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

     INTEGER, PARAMETER :: itmax = 100

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k, iter
     REAL ( KIND = wp ) :: e, e_old, eprod, q, q_old
     REAL ( KIND = wp ) :: ss, ss_old, stop_tol, loga, val, absa

!-----------------------------------------------
!   A u t o m a t i c   A r r a y s
!-----------------------------------------------

     REAL ( KIND = wp ), DIMENSION( m ) :: ROW_COUNT, M_INV_SIG
     REAL ( KIND = wp ), DIMENSION( n ) :: COL_COUNT, COL_RHS, SHIFT_C
     
     IF ( m < 1 .OR. n < 1 ) RETURN

!  Set the stopping tolerance

     stop_tol = ne * point1

!  Initialise for accumulation of sums and products

     R = zero ; C = zero
     ROW_COUNT = zero ; COL_COUNT = zero ; COL_RHS = zero

!  Count non-zeros in the rows, and compute r.h.s. vectors; use R to store
!  the row r.h.s. (sigma in Curtis+Reid)

     DO k = 1, ne
       absa = ABS( A( k ) )
       IF ( absa /= zero ) THEN
         i = IRNA( k ) ; j = ICNA( k )
         loga = LOG( absa )
         ROW_COUNT( i ) = ROW_COUNT( i ) + one
         COL_COUNT( j ) = COL_COUNT( j ) + one
         R ( i )      = R ( i ) + loga
         COL_RHS( j ) = COL_RHS( j ) + loga
       END IF
     END DO

!  Account for structural singularity

     WHERE ( ROW_COUNT == zero ) ROW_COUNT = one
     WHERE ( COL_COUNT == zero ) COL_COUNT = one

!  Use Reid's special purpose method for matrices with property "A". Comments
!  refer to equation numbers in Curtis and Reid's paper

!  Form M^-1 sigma and N^-1 tau (in C+R's notation)

     M_INV_SIG = R / ROW_COUNT
     COL_RHS = COL_RHS / COL_COUNT

!  Compute initial residual vector

     R = M_INV_SIG
     DO k = 1, ne
       IF ( A( k ) /= zero ) THEN
         i = IRNA( k ) ; j = ICNA( k )
         R( i ) = R( i ) - COL_RHS( j ) / ROW_COUNT( i )  ! (4.3)
       END IF
     END DO

!  Set initial values

     e = zero ; q = one ; 
!    ss = DOT_PRODUCT( ROW_COUNT, R ** 2 )
     ss = zero ; DO k = 1, m ; ss = ss + ROW_COUNT( k ) * R( k ) ** 2 ; END DO
     IF ( ss > stop_tol ) THEN

!  Iteration loop

       SHIFT_C = zero
       DO iter = 1, itmax

!  Update column residual vector

         DO k = 1, ne
           IF ( A( k ) /= zero ) THEN
             i = IRNA( k ) ; j = ICNA( k )
             C( j ) = C( j ) + R( i )
           END IF
         END DO

!  Rescale column residual

         ss_old = ss ; ss = zero
         DO j = 1, n
           val = - C( j ) / q
           C( j ) = val / COL_COUNT( j )     ! (4.4a)
           ss = ss + val * C( j )            ! (4.5a)
         END DO

!  Test for termination

          IF ( ss <= stop_tol ) THEN
            R = zero ; GO TO 100
          END IF
          e_old = e ; e = q * ss / ss_old   ! (4.6)
          q = one - e                       ! (4.7)
          R = R * e * ROW_COUNT

!  Rescale row residual vector

          eprod = e * e_old

!  Update row residual

          DO k = 1, ne
            IF ( A( k ) /= zero ) THEN
              i = IRNA( k ) ; j = ICNA( k )
              R( i ) = R( i ) + C( j )
            END IF
          END DO

!  Again, rescale row residual

          ss_old = ss ; ss = zero
          DO i = 1, m
            val = - R( i ) / q
            R( i ) = val / ROW_COUNT( i )    ! (4.4b)
            ss = ss + val * R( i )           ! (4.5b)
          END DO

!  Test for termination

          IF ( ss <= stop_tol ) THEN
            COL_RHS = COL_RHS + ( C + eprod * SHIFT_C ) / q ; EXIT
          END IF
          e_old = e ; e = q * ss / ss_old   ! (4.6)
          q_old = q ; q = one - e           ! (4.7)

!  Rescale column residual vector

          SHIFT_C = ( C + eprod * SHIFT_C ) / ( q * q_old )
          COL_RHS = COL_RHS + SHIFT_C

!  Update column scaling factors

          C = C * e * COL_COUNT
        END DO
      END IF

!  Rescale row scaling factors

     R = R * ROW_COUNT

!  Rescale row scaling factors

 100 CONTINUE
     DO k = 1, ne
       IF ( A( k ) /= zero ) THEN
         i = IRNA( k )
         j = ICNA( k )
         R( i ) = R( i ) + COL_RHS( j )
       END IF
     END DO

!  Final rescaling of row and column scaling factors

     R = R / ROW_COUNT - M_INV_SIG ; C = - COL_RHS
     RETURN

!  End of subroutine SCALN_curtis_reid

     END SUBROUTINE SCALN_curtis_reid

!  End of module LANCELOT_SCALN

   END MODULE LANCELOT_SCALN_double




 
