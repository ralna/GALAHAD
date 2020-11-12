#include <fintrf.h>

!  THIS VERSION: GALAHAD 2.4 - 09/03/2010 AT 11:45 GMT.

!-*-*-*  G A L A H A D _ S I L S _ M A T L A B _ T Y P E S   M O D U L E  *-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. March 9th, 2010

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SILS_MATLAB_TYPES

!  provide control and inform values for Matlab interfaces to SILS

      USE GALAHAD_MATLAB
      USE GALAHAD_SILS_double, SILS_control_type => SILS_control

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SILS_matlab_control_set, SILS_matlab_control_get,              &
                SILS_matlab_inform_create, SILS_matlab_inform_get_ainfo,       &
                SILS_matlab_inform_get_finfo, SILS_matlab_inform_get_sinfo

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: slen = 30

!--------------------------
!  Derived type definitions
!--------------------------

      TYPE, PUBLIC :: SILS_pointer_type
        mwPointer :: pointer
        mwPointer :: flag, more, stat, nsteps, nrltot, nirtot, nrlnec
        mwPointer :: nirnec, nrladu, niradu, ncmpa, oor, dup, maxfrt
        mwPointer :: nebdu, nrlbdu, nirbdu, ncmpbr, ncmpbi, ntwo, neig, delay
        mwPointer :: signc, modstep, rank, opsa, opse, opsb, maxchange
      END TYPE 

    CONTAINS

!-*-*  S I L S _ M A T L A B _ C O N T R O L _ S E T  S U B R O U T I N E   *-*-

!     SUBROUTINE SILS_matlab_control_set( ps, SILS_contro1, len )
      SUBROUTINE SILS_matlab_control_set( ps, SILS_contro1 )

!  --------------------------------------------------------------

!  Set matlab control arguments from values provided to SILS

!  Arguments

!  ps - given pointer to the structure
!  SILS_contro1 - SILS control structure
!  len - length of any character component

!  --------------------------------------------------------------

      mwPointer :: ps
!     mwSize :: len
      TYPE ( SILS_control_type ) :: SILS_contro1

!  local variables

      INTEGER :: i, nfields
      mwPointer :: pc
      mwSize :: mxGetNumberOfFields
      CHARACTER ( LEN = slen ) :: name, mxGetFieldNameByNumber

      nfields = mxGetNumberOfFields( ps )
      DO i = 1, nfields
        name = mxGetFieldNameByNumber( ps, i )
        SELECT CASE ( TRIM( name ) )
        CASE( 'lp' )
          CALL MATLAB_get_value( ps, 'lp',                                     &
                                 pc, SILS_contro1%lp )
        CASE( 'wp' )
          CALL MATLAB_get_value( ps, 'wp',                                     &
                                 pc, SILS_contro1%wp )
        CASE( 'mp' )
          CALL MATLAB_get_value( ps, 'mp',                                     &
                                 pc, SILS_contro1%mp )
        CASE( 'sp' )
          CALL MATLAB_get_value( ps, 'sp',                                     &
                                 pc, SILS_contro1%sp )
        CASE( 'ldiag' )
          CALL MATLAB_get_value( ps, 'ldiag',                                  &
                                 pc, SILS_contro1%ldiag )
        CASE( 'factorblocking' )
          CALL MATLAB_get_value( ps, 'factorblocking',                         &
                                 pc, SILS_contro1%factorblocking )
        CASE( 'solveblocking' )
          CALL MATLAB_get_value( ps, 'solveblocking',                          &
                                 pc, SILS_contro1%solveblocking )
        CASE( 'la' )
          CALL MATLAB_get_value( ps, 'la',                                     &
                                 pc, SILS_contro1%la )
        CASE( 'liw' )
          CALL MATLAB_get_value( ps, 'liw',                                    &
                                 pc, SILS_contro1%liw )
        CASE( 'maxla' )
          CALL MATLAB_get_value( ps, 'maxla',                                  &
                                 pc, SILS_contro1%maxla )
        CASE( 'maxliw' )
          CALL MATLAB_get_value( ps, 'maxliw',                                 &
                                 pc, SILS_contro1%maxliw )
        CASE( 'pivoting' )
          CALL MATLAB_get_value( ps, 'pivoting',                               &
                                 pc, SILS_contro1%pivoting )
        CASE( 'thresh' )
          CALL MATLAB_get_value( ps, 'thresh',                                 &
                                 pc, SILS_contro1%thresh )
        CASE( 'nemin' )
          CALL MATLAB_get_value( ps, 'nemin',                                  &
                                 pc, SILS_contro1%nemin )
        CASE( 'ordering' )
          CALL MATLAB_get_value( ps, 'ordering',                               &
                                 pc, SILS_contro1%ordering )
        CASE( 'scaling' )
          CALL MATLAB_get_value( ps, 'scaling',                                &
                                 pc, SILS_contro1%scaling )
        CASE( 'multiplier' )
          CALL MATLAB_get_value( ps, 'multiplier',                             &
                                 pc, SILS_contro1%multiplier )
        CASE( 'reduce' )
          CALL MATLAB_get_value( ps, 'reduce',                                 &
                                 pc, SILS_contro1%reduce )
        CASE( 'u' )
          CALL MATLAB_get_value( ps, 'u',                                      &
                                 pc, SILS_contro1%u )
        CASE( 'static_tolerance' )
          CALL MATLAB_get_value( ps, 'static_tolerance',                       &
                                 pc, SILS_contro1%static_tolerance )
        CASE( 'static_level' )
          CALL MATLAB_get_value( ps, 'static_level',                           &
                                 pc, SILS_contro1%static_level )
        CASE( 'tolerance' )
          CALL MATLAB_get_value( ps, 'tolerance',                              &
                                 pc, SILS_contro1%tolerance )
        CASE( 'convergence' )
          CALL MATLAB_get_value( ps, 'convergence',                            &
                                 pc, SILS_contro1%convergence )
        END SELECT
      END DO

      RETURN

!  End of subroutine SILS_matlab_control_set

      END SUBROUTINE SILS_matlab_control_set

!-*-*  S I L S _ M A T L A B _ C O N T R O L _ G E T  S U B R O U T I N E   *-*-

      SUBROUTINE SILS_matlab_control_get( struct, SILS_contro1, name )

!  --------------------------------------------------------------

!  Get matlab control arguments from values provided to SILS

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  SILS_contro1 - SILS control structure
!  name - name of component of the structure

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SILS_control_type ) :: SILS_contro1
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix
      mwPointer :: pointer

      INTEGER * 4, PARAMETER :: ninform = 23
      CHARACTER ( LEN = 28 ), PARAMETER :: finform( ninform ) = (/             &
           'lp                          ', 'wp                          ',     &
           'mp                          ', 'sp                          ',     &
           'ldiag                       ', 'factorblocking              ',     &
           'solveblocking               ', 'la                          ',     &
           'liw                         ', 'maxla                       ',     &
           'maxliw                      ', 'pivoting                    ',     &
           'nemin                       ', 'thresh                      ',     &
           'ordering                    ', 'scaling                     ',     &
           'multiplier                  ', 'reduce                      ',     &
           'u                           ', 'static_tolerance            ',     &
           'static_level                ', 'tolerance                   ',     &
           'convergence                 '                        /)

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, pointer,                &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        pointer = struct
      END IF

!  create the components and get the values

      CALL MATLAB_fill_component( pointer, 'lp',                               &
                                  SILS_contro1%lp )
      CALL MATLAB_fill_component( pointer, 'wp',                               &
                                  SILS_contro1%wp )
      CALL MATLAB_fill_component( pointer, 'mp',                               &
                                  SILS_contro1%mp )
      CALL MATLAB_fill_component( pointer, 'sp',                               &
                                  SILS_contro1%sp )
      CALL MATLAB_fill_component( pointer, 'ldiag',                            &
                                  SILS_contro1%ldiag )
      CALL MATLAB_fill_component( pointer, 'factorblocking',                   &
                                  SILS_contro1%factorblocking )
      CALL MATLAB_fill_component( pointer, 'solveblocking',                    &
                                  SILS_contro1%solveblocking )
      CALL MATLAB_fill_component( pointer, 'la',                               &
                                  SILS_contro1%la )
      CALL MATLAB_fill_component( pointer, 'liw',                              &
                                  SILS_contro1%liw )
      CALL MATLAB_fill_component( pointer, 'maxla',                            &
                                  SILS_contro1%maxla )
      CALL MATLAB_fill_component( pointer, 'maxliw',                           &
                                  SILS_contro1%maxliw )
      CALL MATLAB_fill_component( pointer, 'pivoting',                         &
                                  SILS_contro1%pivoting )
      CALL MATLAB_fill_component( pointer, 'nemin',                            &
                                  SILS_contro1%nemin )
      CALL MATLAB_fill_component( pointer, 'thresh',                           &
                                  SILS_contro1%thresh )
      CALL MATLAB_fill_component( pointer, 'ordering',                         &
                                  SILS_contro1%ordering )
      CALL MATLAB_fill_component( pointer, 'scaling',                          &
                                  SILS_contro1%scaling )
      CALL MATLAB_fill_component( pointer, 'multiplier',                       &
                                  SILS_contro1%multiplier )
      CALL MATLAB_fill_component( pointer, 'reduce',                           &
                                  SILS_contro1%reduce )
      CALL MATLAB_fill_component( pointer, 'u',                                &
                                  SILS_contro1%u )
      CALL MATLAB_fill_component( pointer, 'static_tolerance',                 &
                                  SILS_contro1%static_tolerance )
      CALL MATLAB_fill_component( pointer, 'static_level',                     &
                                  SILS_contro1%static_level )
      CALL MATLAB_fill_component( pointer, 'tolerance',                        &
                                  SILS_contro1%tolerance )
      CALL MATLAB_fill_component( pointer, 'convergence',                      &
                                  SILS_contro1%convergence )

      RETURN

!  End of subroutine SILS_matlab_control_get

      END SUBROUTINE SILS_matlab_control_get

!-*  S I L S _ M A T L A B _ I N F O R M _ C R E A T E  S U B R O U T I N E   *-

      SUBROUTINE SILS_matlab_inform_create( struct, SILS_pointer, name )

!  --------------------------------------------------------------

!  Create matlab pointers to hold SILS_inform values

!  Arguments

!  struct - pointer to the structure for which this will be a component
!  name - name of component of the structure
!  SILS_pointer - SILS pointer sub-structure
!  name - optional name of component of the structure (root of structure if
!         absent)

!  --------------------------------------------------------------

      mwPointer :: struct
      TYPE ( SILS_pointer_type ) :: SILS_pointer
      CHARACTER ( len = * ), OPTIONAL :: name

!  local variables

      mwPointer :: mxCreateStructMatrix

      INTEGER * 4, PARAMETER :: ninform = 29
      CHARACTER ( LEN = 9 ), PARAMETER :: finform( ninform ) = (/              &
           'flag     ', 'more     ', 'stat     ',                              &
           'nsteps   ', 'nrltot   ', 'nirtot   ', 'nrlnec   ',                 &
           'nirnec   ', 'nrladu   ', 'niradu   ', 'ncmpa    ',                 &
           'oor      ', 'dup      ', 'maxfrt   ',                              &
           'nebdu    ', 'nrlbdu   ', 'nirbdu   ', 'ncmpbr   ',                 &
           'ncmpbi   ', 'ntwo     ', 'neig     ', 'delay    ',                 &
           'signc    ', 'modstep  ', 'rank     ',                              &
           'opsa     ', 'opse     ', 'opsb     ', 'maxchange'   /)  

!  create the structure

      IF ( PRESENT( name ) ) THEN
        CALL MATLAB_create_substructure( struct, name, SILS_pointer%pointer,   &
                                         ninform, finform )
      ELSE
        struct = mxCreateStructMatrix( 1_mws_, 1_mws_, ninform, finform )
        SILS_pointer%pointer = struct
      END IF

!  create the components

      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
         'flag', SILS_pointer%flag )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'more', SILS_pointer%more )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'stat', SILS_pointer%stat )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nsteps', SILS_pointer%nsteps )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nrltot', SILS_pointer%nrltot )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nirtot', SILS_pointer%nirtot )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nrlnec', SILS_pointer%nrlnec )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nirnec', SILS_pointer%nirnec )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nrladu', SILS_pointer%nrladu )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'niradu', SILS_pointer%niradu )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'ncmpa  ', SILS_pointer%ncmpa )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'oor', SILS_pointer%oor )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'dup', SILS_pointer%dup )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'maxfrt', SILS_pointer%maxfrt )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nebdu', SILS_pointer%nebdu )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nrlbdu', SILS_pointer%nrlbdu )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'nirbdu', SILS_pointer%nirbdu )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'ncmpbr', SILS_pointer%ncmpbr )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'ncmpbi', SILS_pointer%ncmpbi )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'ntwo', SILS_pointer%ntwo )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'neig', SILS_pointer%neig )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'delay', SILS_pointer%delay )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'signc', SILS_pointer%signc )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'modstep', SILS_pointer%modstep )
      CALL MATLAB_create_integer_component( SILS_pointer%pointer,              &
        'rank', SILS_pointer%rank )
      CALL MATLAB_create_real_component( SILS_pointer%pointer,                 &
        'opsa', SILS_pointer%opsa )
      CALL MATLAB_create_real_component( SILS_pointer%pointer,                 &
        'opse', SILS_pointer%opse )
      CALL MATLAB_create_real_component( SILS_pointer%pointer,                 &
        'opsb', SILS_pointer%opsb )
      CALL MATLAB_create_real_component( SILS_pointer%pointer,                 &
        'maxchange', SILS_pointer%maxchange )
      RETURN

!  End of subroutine SILS_matlab_inform_create

      END SUBROUTINE SILS_matlab_inform_create

! S I L S _ M A T L A B _ I N F O R M _ G E T _ A I N F O   S U B R O U T I N E 

      SUBROUTINE SILS_matlab_inform_get_ainfo( AINFO, SILS_pointer )

!  --------------------------------------------------------------

!  Set SILS_inform values from matlab pointers

!  Arguments

!  SILS_inform - SILS inform structure
!  SILS_pointer - SILS pointer structure

!  --------------------------------------------------------------

      TYPE ( SILS_ainfo ) :: AINFO
      TYPE ( SILS_pointer_type ) :: SILS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( AINFO%flag,                                     &
                               mxGetPr( SILS_pointer%flag ) )
      CALL MATLAB_copy_to_ptr( AINFO%more,                                     &
                               mxGetPr( SILS_pointer%more ) )
      CALL MATLAB_copy_to_ptr( AINFO%stat,                                     &
                               mxGetPr( SILS_pointer%stat ) )
      CALL MATLAB_copy_to_ptr( AINFO%nsteps,                                   &
                               mxGetPr( SILS_pointer%nsteps ) )
      CALL MATLAB_copy_to_ptr( AINFO%nrltot,                                   &
                               mxGetPr( SILS_pointer%nrltot ) )
      CALL MATLAB_copy_to_ptr( AINFO%nirtot,                                   &
                               mxGetPr( SILS_pointer%nirtot ) )
      CALL MATLAB_copy_to_ptr( AINFO%nrlnec,                                   &
                               mxGetPr( SILS_pointer%nrlnec ) )
      CALL MATLAB_copy_to_ptr( AINFO%nirnec,                                   &
                               mxGetPr( SILS_pointer%nirnec ) )
      CALL MATLAB_copy_to_ptr( AINFO%nrladu,                                   &
                               mxGetPr( SILS_pointer%nrladu ) )
      CALL MATLAB_copy_to_ptr( AINFO%niradu,                                   &
                               mxGetPr( SILS_pointer%niradu ) )
      CALL MATLAB_copy_to_ptr( AINFO%ncmpa,                                    &
                               mxGetPr( SILS_pointer%ncmpa ) )
      CALL MATLAB_copy_to_ptr( AINFO%oor,                                      &
                               mxGetPr( SILS_pointer%oor ) )
      CALL MATLAB_copy_to_ptr( AINFO%dup,                                      &
                               mxGetPr( SILS_pointer%dup ) )
      CALL MATLAB_copy_to_ptr( AINFO%maxfrt,                                   &
                               mxGetPr( SILS_pointer%maxfrt ) )
      CALL MATLAB_copy_to_ptr( AINFO%opsa,                                     &
                               mxGetPr( SILS_pointer%opsa ) )
      CALL MATLAB_copy_to_ptr( AINFO%opse,                                     &
                               mxGetPr( SILS_pointer%opse ) )

      RETURN

!  End of subroutine SILS_matlab_inform_get_ainfo

      END SUBROUTINE SILS_matlab_inform_get_ainfo

! S I L S _ M A T L A B _ I N F O R M _ G E T _ F I N F O   S U B R O U T I N E 

      SUBROUTINE SILS_matlab_inform_get_finfo( FINFO, SILS_pointer )

!  --------------------------------------------------------------

!  Set SILS_inform values from matlab pointers

!  Arguments

!  SILS_inform - SILS inform structure
!  SILS_pointer - SILS pointer structure

!  --------------------------------------------------------------

      TYPE ( SILS_finfo ) :: FINFO
      TYPE ( SILS_pointer_type ) :: SILS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( FINFO%flag,                                     &
                               mxGetPr( SILS_pointer%flag ) )
      CALL MATLAB_copy_to_ptr( FINFO%more,                                     &
                               mxGetPr( SILS_pointer%more ) )
      CALL MATLAB_copy_to_ptr( FINFO%stat,                                     &
                               mxGetPr( SILS_pointer%stat ) )
      CALL MATLAB_copy_to_ptr( FINFO%maxfrt,                                   &
                               mxGetPr( SILS_pointer%maxfrt ) )
      CALL MATLAB_copy_to_ptr( FINFO%nebdu,                                    &
                               mxGetPr( SILS_pointer%nebdu ) )
      CALL MATLAB_copy_to_ptr( FINFO%nrlbdu,                                   &
                               mxGetPr( SILS_pointer%nrlbdu ) )
      CALL MATLAB_copy_to_ptr( FINFO%nirbdu,                                   &
                               mxGetPr( SILS_pointer%nirbdu ) )
      CALL MATLAB_copy_to_ptr( FINFO%nrltot,                                   &
                               mxGetPr( SILS_pointer%nrltot ) )
      CALL MATLAB_copy_to_ptr( FINFO%nirtot,                                   &
                               mxGetPr( SILS_pointer%nirtot ) )
      CALL MATLAB_copy_to_ptr( FINFO%nrlnec,                                   &
                               mxGetPr( SILS_pointer%nrlnec ) )
      CALL MATLAB_copy_to_ptr( FINFO%nirnec,                                   &
                               mxGetPr( SILS_pointer%nirnec ) )
      CALL MATLAB_copy_to_ptr( FINFO%ncmpbr,                                   &
                               mxGetPr( SILS_pointer%ncmpbr ) )
      CALL MATLAB_copy_to_ptr( FINFO%ncmpbi,                                   &
                               mxGetPr( SILS_pointer%ncmpbi ) )
      CALL MATLAB_copy_to_ptr( FINFO%ntwo,                                     &
                               mxGetPr( SILS_pointer%ntwo ) )
      CALL MATLAB_copy_to_ptr( FINFO%neig,                                     &
                               mxGetPr( SILS_pointer%neig ) )
      CALL MATLAB_copy_to_ptr( FINFO%delay,                                    &
                               mxGetPr( SILS_pointer%delay ) )
      CALL MATLAB_copy_to_ptr( FINFO%signc,                                    &
                               mxGetPr( SILS_pointer%signc ) )
      CALL MATLAB_copy_to_ptr( FINFO%modstep,                                  &
                               mxGetPr( SILS_pointer%modstep ) )
      CALL MATLAB_copy_to_ptr( FINFO%rank,                                     &
                               mxGetPr( SILS_pointer%rank ) )
      CALL MATLAB_copy_to_ptr( FINFO%opsa,                                     &
                               mxGetPr( SILS_pointer%opsa ) )
      CALL MATLAB_copy_to_ptr( FINFO%opse,                                     &
                               mxGetPr( SILS_pointer%opse ) )
      CALL MATLAB_copy_to_ptr( FINFO%opsb,                                     &
                               mxGetPr( SILS_pointer%opsb ) )
      CALL MATLAB_copy_to_ptr( FINFO%maxchange,                                &
                               mxGetPr( SILS_pointer%maxchange ) )

      RETURN

!  End of subroutine SILS_matlab_inform_get_finfo

      END SUBROUTINE SILS_matlab_inform_get_finfo

! S I L S _ M A T L A B _ I N F O R M _ G E T _ S I N F O   S U B R O U T I N E 

      SUBROUTINE SILS_matlab_inform_get_sinfo( SINFO, SILS_pointer )

!  --------------------------------------------------------------

!  Set SILS_inform values from matlab pointers

!  Arguments

!  SILS_inform - SILS inform structure
!  SILS_pointer - SILS pointer structure

!  --------------------------------------------------------------

      TYPE ( SILS_sinfo ) :: SINFO
      TYPE ( SILS_pointer_type ) :: SILS_pointer

!  local variables

      mwPointer :: mxGetPr

      CALL MATLAB_copy_to_ptr( SINFO%flag,                                     &
                               mxGetPr( SILS_pointer%flag ) )
      CALL MATLAB_copy_to_ptr( SINFO%stat,                                     &
                               mxGetPr( SILS_pointer%stat ) )

      RETURN

!  End of subroutine SILS_matlab_inform_get_sinfo

      END SUBROUTINE SILS_matlab_inform_get_sinfo

!-*-*-*-  E N D  o f  G A L A H A D _ S I L S _ T Y P E S   M O D U L E  -*-*-*-

    END MODULE GALAHAD_SILS_MATLAB_TYPES




