! THIS VERSION: GALAHAD 3.0 - 24/10/2017 AT 14:45 GMT.

!-*-*-*-*-*-*-*-*-  L A N C E L O T  - B - U S E L A N C E L O T  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   February 6th 1995 as runlanb
!   March 14th 2003 as uselanb
!   update released with GALAHAD Version 2.0. May 11th 2006

    MODULE GALAHAD_USELANCELOT_double

!  SIF interface to LANCELOT. It opens and closes all the files, allocate
!  arrays, reads and checks data, and calls the appropriate minimizers

!NOT95USE GALAHAD_CPU_time
     USE GALAHAD_RAND_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_COPYRIGHT
     USE LANCELOT_OTHERS_double
     USE LANCELOT_DRCHE_double
     USE LANCELOT_DRCHG_double
     USE LANCELOT_SCALN_double
     USE LANCELOT_double
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: USE_LANCELOT

   CONTAINS

!-*-*-*-*-*-*-*-*-*-   U S E _ L A N B  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

     SUBROUTINE USE_LANCELOT( input )

!  Dummy argument

     INTEGER, INTENT( IN ) :: input

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 31
     CHARACTER ( LEN = 16 ) :: specname = 'RUNLANCELOT'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNLANCELOT.SPC'

!  Default values for specfile-defined parameters

     INTEGER :: dfiledevice = 26
     INTEGER :: rfiledevice = 47
     INTEGER :: sfiledevice = 62
     INTEGER :: vfiledevice = 63
     INTEGER :: wfiledevice = 59
     LOGICAL :: write_problem_data    = .FALSE.
     LOGICAL :: write_solution        = .FALSE.
     LOGICAL :: write_result_summary  = .FALSE.
     LOGICAL :: write_solution_vector = .FALSE.
     CHARACTER ( LEN = 30 ) :: dfilename = 'LANCELOT.data'
     CHARACTER ( LEN = 30 ) :: rfilename = 'LANCELOTRES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'LANCELOTSOL.d'
     CHARACTER ( LEN = 30 ) :: wfilename = 'LANCELOTSAVE.d'
     CHARACTER ( LEN = 30 ) :: vfilename = 'LANCELOTSOLVEC.d'
     LOGICAL :: testal = .FALSE.
     LOGICAL :: dechk  = .FALSE.
     LOGICAL :: dechke = .FALSE.
     LOGICAL :: dechkg = .FALSE.
     LOGICAL :: not_fatal  = .FALSE.
     LOGICAL :: not_fatale = .FALSE.
     LOGICAL :: not_fatalg = .FALSE.
     LOGICAL :: getsca = .FALSE.
     INTEGER :: print_level_scaling = 0
     LOGICAL :: scale  = .FALSE.
     LOGICAL :: scaleg = .FALSE.
     LOGICAL :: scalev = .FALSE.
     LOGICAL :: get_max = .FALSE.
     LOGICAL :: just_feasible = .FALSE.
     LOGICAL :: warm_start = .FALSE.
     INTEGER :: istore = 0

!  The default values for RUNLANCELOT could have been set as:

! BEGIN RUNLANCELOT SPECIFICATIONS (DEFAULT)
!  write-problem-data                       NO
!  problem-data-file-name                   LANCELOT.data
!  problem-data-file-device                 26
!  write-solution                           YES
!  solution-file-name                       LANCELOTSOL.d
!  solution-file-device                     62
!  write-solution-vector                    NO
!  solution-vector-file-name                LANCELOTSOLVEC.d
!  solution-vector-file-device              63
!  write-result-summary                     YES
!  result-summary-file-name                 LANCELOTRES.d
!  result-summary-file-device               47
!  check-all-derivatives                    NO
!  check-derivatives                        YES
!  check-element-derivatives                YES
!  check-group-derivatives                  YES
!  ignore-derivative-bugs                   NO
!  ignore-element-derivative-bugs           NO
!  ignore-group-derivative-bugs             NO
!  get-scaling-factors                      NO
!  scaling-print-level                      1
!  use-scaling-factors                      NO
!  use-constraint-scaling-factors           NO
!  use-variable-scaling-factors             NO
!  maximizer-sought                         NO
!  just-find-feasible-point                 NO
!  restart-from-previous-point              NO
!  restart-data-file-name                   LANCELOTSAVE.d
!  restart-data-file-device                 59
!  save-data-for-restart-every              0
! END RUNLANCELOT SPECIFICATIONS

!  Output file characteristics

     INTEGER :: errout = 6
     CHARACTER ( LEN = 10 ) :: pname

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: ntotel, nvrels, nnza  , ngpvlu, nepvlu
     INTEGER :: neltyp, ngrtyp, ialgor, nnlneq, nlinin, nnlnin, nobjgr
     INTEGER :: ieltyp, lfuval
     INTEGER :: nin   , ninmax, nelmax, iauto , out
     INTEGER :: i , j , ifflag, numvar, igrtyp, iores
     INTEGER :: norder, nfree , nfixed, alloc_status, alive_request
     INTEGER :: nlower, nupper, nboth , nslack, nlinob, nnlnob, nlineq
     REAL    :: time  , t     , timm  , ttotal
     REAL ( KIND = wp ) :: fobj, epsmch, rand
     REAL ( KIND = wp ), DIMENSION( 2 ) :: OBFBND
     LOGICAL :: alive, dsave, second, filexx, fdgrad, is_specfile
!    LOGICAL :: square

     CHARACTER ( LEN = 3  ) :: minmax
     CHARACTER ( LEN = 5  ) :: optimi
     CHARACTER ( LEN = 80 ) :: bad_alloc
     CHARACTER ( LEN = 10 ) :: pname2, tempna

!----------------------------
!   D e r i v e d   T y p e s
!----------------------------

     TYPE ( LANCELOT_control_type ) :: control
     TYPE ( LANCELOT_inform_type ) :: inform
     TYPE ( LANCELOT_data_type ) :: data
     TYPE ( LANCELOT_problem_type ) :: prob
     TYPE ( SCALN_save_type ) :: SCALN_S
     TYPE ( DRCHE_save_type ) :: DRCHE_S
     TYPE ( DRCHG_save_type ) :: DRCHG_S
     TYPE ( RAND_seed ) :: seed

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVAR
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ICALCF
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ICALCG
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: GVALS
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: XT
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DGRAD
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Q
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FT
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: FUVALS
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: ETYPES
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: GTYPES

!  locally used arrays

     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ISTINV, ITEST, IELVAR_temp
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_temp

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

!  ================================
!   O p e n   d a t a   f i l e s
!  ================================

!  ** Open LANCELOT specification file for unix systems

!    OPEN( iinrun, FILE = 'SPEC.SPC', FORM = 'FORMATTED', STATUS = 'UNKNOWN' )

!  ==========================================================================
!   D a t a    f i l e s   o p e n  ;  p a r t i t i o n   w o r k s p a c e
!  ==========================================================================

!  Input the problem dimensions

     READ( input, 1050 ) prob%n, prob%ng, prob%nel, ntotel,                    &
                         nvrels, nnza, ngpvlu, nepvlu, neltyp, ngrtyp
     READ( input, 1060 ) ialgor, pname, iauto
!    iauto = - 1

!  Allocate space for the integer arrays

     ALLOCATE( prob%ISTADG( prob%ng  + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ISTADG' ; GO TO 800; END IF
     ALLOCATE( prob%ISTGPA( prob%ng  + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ISTGPA' ; GO TO 800; END IF
     ALLOCATE( prob%ISTADA( prob%ng  + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ISTADA' ; GO TO 800; END IF
     ALLOCATE( prob%ISTAEV( prob%nel + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ISTAEV' ; GO TO 800; END IF
     ALLOCATE( prob%ISTEPA( prob%nel + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ISTEPA' ; GO TO 800; END IF
     ALLOCATE( prob%ITYPEG( prob%ng  ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ITYPEG' ; GO TO 800; END IF
     IF ( ialgor > 1 ) THEN
       ALLOCATE( prob%KNDOFG( prob%ng  ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'KNDOFG'; GO TO 800; END IF
     END IF
     ALLOCATE( prob%ITYPEE( prob%nel ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ITYPEE' ; GO TO 800; END IF
     ALLOCATE( prob%INTVAR( prob%nel + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%INTVAR' ; GO TO 800; END IF
     ALLOCATE( prob%IELING( ntotel ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%IELING' ; GO TO 800; END IF
     ALLOCATE( prob%IELVAR( nvrels ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%IELVAR' ; GO TO 800; END IF
     ALLOCATE( prob%ICNA( nnza ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ICNA' ; GO TO 800; END IF
     ALLOCATE( prob%ISTADH( prob%nel + 1 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ISTADH' ; GO TO 800; END IF
     ALLOCATE( IVAR( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IVAR' ; GO TO 800; END IF
     ALLOCATE( ICALCG( prob%ng ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ICALCG' ; GO TO 800; END IF
     ALLOCATE( ICALCF( MAX( prob%ng, prob%nel ) ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ICALCF' ; GO TO 800; END IF

!  Allocate space for the real arrays

     ALLOCATE( prob%A( nnza ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%A' ; GO TO 800 ; END IF
     ALLOCATE( prob%B( prob%ng ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%B' ; GO TO 800 ; END IF
     ALLOCATE( prob%BL( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%BL' ; GO TO 800 ; END IF
     ALLOCATE( prob%BU( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%BU' ; GO TO 800 ; END IF
     ALLOCATE( prob%X( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%X' ; GO TO 800 ; END IF
     IF ( ialgor > 1 ) THEN
       ALLOCATE( prob%Y( prob%ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%Y' ; GO TO 800 ; END IF
       ALLOCATE( prob%C( prob%ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'prob%C' ; GO TO 800 ; END IF
     END IF
     ALLOCATE( prob%GPVALU( ngpvlu ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%GPVALU' ; GO TO 800; END IF
     ALLOCATE( prob%EPVALU( nepvlu ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%EPVALU' ; GO TO 800; END IF
     ALLOCATE( prob%ESCALE( ntotel ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%ESCALE' ; GO TO 800; END IF
     ALLOCATE( prob%GSCALE( prob%ng ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%GSCALE' ; GO TO 800; END IF
     ALLOCATE( prob%VSCALE( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%VSCALE' ; GO TO 800; END IF
     ALLOCATE( XT( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'XT' ; GO TO 800 ; END IF
     ALLOCATE( DGRAD( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'DGRAD' ; GO TO 800 ; END IF
     ALLOCATE( Q( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'Q' ; GO TO 800 ; END IF
     ALLOCATE( FT( prob%ng ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'FT' ; GO TO 800 ; END IF
     ALLOCATE( GVALS( prob%ng , 3 ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'GVALS' ; GO TO 800 ; END IF

!  Allocate space for the logical arrays

     ALLOCATE( prob%INTREP( prob%nel ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%INTREP' ; GO TO 800; END IF
     ALLOCATE( prob%GXEQX( prob%ng ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%GXEQX' ; GO TO 800 ; END IF

!  Allocate space for the character arrays

     ALLOCATE( prob%GNAMES( prob%ng ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%GNAMES' ; GO TO 800; END IF
     ALLOCATE( prob%VNAMES( prob%n ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       bad_alloc = 'prob%VNAMES' ; GO TO 800; END IF
     ALLOCATE( ETYPES( neltyp ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ETYPES' ; GO TO 800; END IF
     ALLOCATE( GTYPES( ngrtyp ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'GTYPES' ; GO TO 800; END IF

!  Set up initial data

     epsmch = EPSILON( one )
     inform%newsol = .FALSE.
     control%quadratic_problem = .FALSE.

!  ------------------ Open the specfile for runlancelot ----------------

     INQUIRE( FILE = runspec, EXIST = is_specfile )
     IF ( is_specfile ) THEN
       OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD')

!   Define the keywords

       spec( 1 )%keyword  = 'write-problem-data'
       spec( 2 )%keyword  = 'problem-data-file-name'
       spec( 3 )%keyword  = 'problem-data-file-device'
       spec( 4 )%keyword  = ''
       spec( 5 )%keyword  = 'write-solution'
       spec( 6 )%keyword  = 'solution-file-name'
       spec( 7 )%keyword  = 'solution-file-device'
       spec( 8 )%keyword  = 'write-result-summary'
       spec( 9 )%keyword  = 'result-summary-file-name'
       spec( 10 )%keyword = 'result-summary-file-device'
       spec( 11 )%keyword = 'check-all-derivatives'
       spec( 12 )%keyword = 'check-derivatives'
       spec( 13 )%keyword = 'check-element-derivatives'
       spec( 14 )%keyword = 'check-group-derivatives'
       spec( 15 )%keyword = 'ignore-derivative-bugs'
       spec( 16 )%keyword = 'ignore-element-derivative-bugs'
       spec( 17 )%keyword = 'ignore-group-derivative-bugs'
       spec( 18 )%keyword = 'get-scaling-factors'
       spec( 19 )%keyword = 'scaling-print-level'
       spec( 20 )%keyword = 'use-scaling-factors'
       spec( 21 )%keyword = 'use-constraint-scaling-factors'
       spec( 22 )%keyword = 'use-variable-scaling-factors'
       spec( 23 )%keyword = 'maximizer-sought'
       spec( 24 )%keyword = 'just-find-feasible-point'
       spec( 25 )%keyword = 'restart-from-previous-point'
       spec( 26 )%keyword = 'restart-data-file-name'
       spec( 27 )%keyword = 'restart-data-file-device'
       spec( 28 )%keyword = 'save-data-for-restart-every'
       spec( 29 )%keyword = 'write-solution-vector'
       spec( 30 )%keyword = 'solution-vector-file-name'
       spec( 31 )%keyword = 'solution-vector-file-device'

!   Read the specfile

       CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

       CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
       CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
       CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 5 ), write_solution, errout )
       CALL SPECFILE_assign_string ( spec( 6 ), sfilename, errout )
       CALL SPECFILE_assign_integer( spec( 7 ), sfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 8 ), write_result_summary, errout )
       CALL SPECFILE_assign_string ( spec( 9 ), rfilename, errout )
       CALL SPECFILE_assign_integer( spec( 10 ), rfiledevice, errout )
       CALL SPECFILE_assign_logical( spec( 11 ), testal, errout )
       CALL SPECFILE_assign_logical( spec( 12 ), dechk, errout )
       CALL SPECFILE_assign_logical( spec( 13 ), dechke, errout )
       CALL SPECFILE_assign_logical( spec( 14 ), dechkg, errout )
       CALL SPECFILE_assign_logical( spec( 15 ), not_fatal, errout )
       CALL SPECFILE_assign_logical( spec( 16 ), not_fatale, errout )
       CALL SPECFILE_assign_logical( spec( 17 ), not_fatalg, errout )
       CALL SPECFILE_assign_logical( spec( 18 ), getsca, errout )
       CALL SPECFILE_assign_integer( spec( 19 ), print_level_scaling, errout )
       CALL SPECFILE_assign_logical( spec( 20 ), scale, errout )
       CALL SPECFILE_assign_logical( spec( 21 ), scaleg, errout )
       CALL SPECFILE_assign_logical( spec( 22 ), scalev, errout )
       CALL SPECFILE_assign_logical( spec( 23 ), get_max, errout )
       CALL SPECFILE_assign_logical( spec( 24 ), just_feasible, errout )
       CALL SPECFILE_assign_logical( spec( 25 ), warm_start, errout )
       CALL SPECFILE_assign_string ( spec( 26 ), wfilename, errout )
       CALL SPECFILE_assign_integer( spec( 27 ), wfiledevice, errout )
       CALL SPECFILE_assign_integer( spec( 28 ), istore, errout )
       CALL SPECFILE_assign_logical( spec( 29 ), write_solution_vector, errout )
       CALL SPECFILE_assign_string ( spec( 30 ), vfilename, errout )
       CALL SPECFILE_assign_integer( spec( 31 ), vfiledevice, errout )
     END IF

     IF ( dechk .OR. testal ) THEN ; dechke = .TRUE. ; dechkg = .TRUE. ; END IF
     IF ( not_fatal ) THEN ; not_fatale = .TRUE. ; not_fatalg = .TRUE. ; END IF
     IF ( scale ) THEN ; scaleg = .TRUE. ; scalev = .TRUE. ; END IF

!  If required, print out the (raw) problem data

     IF ( write_problem_data ) THEN
       INQUIRE( FILE = dfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',             &
                STATUS = 'OLD', IOSTAT = iores )
       ELSE
          OPEN( dfiledevice, FILE = dfilename, FORM = 'FORMATTED',             &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( control%out, 2160 ) iores, dfilename
         STOP
       END IF
     END IF

!  If required, append results to a file

     IF ( write_result_summary ) THEN
       INQUIRE( FILE = rfilename, EXIST = filexx )
       IF ( filexx ) THEN
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
       ELSE
          OPEN( rfiledevice, FILE = rfilename, FORM = 'FORMATTED',             &
                STATUS = 'NEW', IOSTAT = iores )
       END IF
       IF ( iores /= 0 ) THEN
         write( control%out, 2160 ) iores, rfilename
         STOP
       END IF
       WRITE( rfiledevice, "( A10 )" ) pname
     END IF

!  Set up data for next problem

     CALL LANCELOT_initialize( data, control )
     IF ( is_specfile )                                                        &
       CALL LANCELOT_read_specfile( control, input_specfile )

!    CALL SPECI_read_spec_file( iinrun, control%print_level, ipstrt, ipstop,   &
!                               ipgap , getsca, dechke, dechkg,                &
!                               fatale, fatalg, print_level_scaling, scaleg,   &
!                               scalev, testal, warmst, istore,                &
!                               inform%status , control )

     fdgrad = control%first_derivatives >= 1
!    IF ( inform%status == 1 ) GO TO 900
     inform%iter = 0
     second = control%second_derivatives == 0
     out = control%out

!  Print out problem data. Input the number of variables, groups,
!  elements and the identity of the objective function group

     IF ( ialgor == 2 ) THEN
       READ( input, 1000 ) nslack
     ELSE
       nslack = 0
     END IF
     numvar = prob%n - nslack
     IF ( ialgor == 1 ) scaleg = .FALSE.
     IF ( write_problem_data )                                                 &
       WRITE( dfiledevice, 1100 ) pname, prob%n, prob%ng, prob%nel

!  Input the starting addresses of the elements in each group, of the
!  parameters used for each group and of the nonzeros of the linear
!  element in each group

     READ( input, 1010 ) prob%ISTADG
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ISTADG', prob%ISTADG
     READ( input, 1010 ) prob%ISTGPA
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ISTGPA ', prob%ISTGPA
     READ( input, 1010 ) prob%ISTADA
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ISTADA', prob%ISTADA

!  Input the starting addresses of the variables and parameters in each element

     READ( input, 1010 ) prob%ISTAEV
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ISTAEV', prob%ISTAEV
     READ( input, 1010 ) prob%ISTEPA
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ISTEPA ', prob%ISTEPA

!  Input the group type of each group

     READ( input, 1010 ) prob%ITYPEG
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ITYPEG', prob%ITYPEG
     IF ( ialgor > 1 ) THEN
       READ( input, 1010 ) prob%KNDOFG
       IF ( write_problem_data )                                               &
         WRITE( dfiledevice, 1110 ) 'KNDOFG', prob%KNDOFG
       IF ( just_feasible ) THEN
         IF ( control%print_level > 0 ) WRITE( out, 2320 )
         WHERE ( prob%KNDOFG == 1 ) prob%KNDOFG = 0
       END IF
     END IF

!  Input the element type of each element

     READ( input, 1010 ) prob%ITYPEE
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ITYPEE', prob%ITYPEE

!  Input the number of internal variables for each element

     READ( input, 1010 ) prob%INTVAR( : prob%nel )
     IF ( write_problem_data )                                                 &
       WRITE( dfiledevice, 1110 ) 'INTVAR', prob%INTVAR( : prob%nel )

!  Determine the required length of FUVALS.

     lfuval = prob%nel + 2 * prob%n
     DO i = 1, prob%nel
       lfuval = lfuval + ( prob%INTVAR( i ) * ( prob%INTVAR( i ) + 3 ) ) / 2
     END DO

!  Allocate FUVALS.

     ALLOCATE( FUVALS( lfuval ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'FUVALS' ; GO TO 800; END IF

!  Input the identity of each individual element

     READ( input, 1010 ) prob%IELING
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'IELING', prob%IELING

!  Input the variables in each group's elements

     READ( input, 1010 ) prob%IELVAR
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'IELVAR', prob%IELVAR

!  Input the column addresses of the nonzeros in each linear element

     READ( input, 1010 ) prob%ICNA
     IF ( write_problem_data ) WRITE( dfiledevice, 1110 ) 'ICNA  ', prob%ICNA

!  Input the values of the nonzeros in each linear element, the constant term
!  in each group, the lower and upper bounds on the variables and the starting
!  point for the minimization

     READ( input, 1020 ) prob%A
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'A     ', prob%A
     READ( input, 1020 ) prob%B
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'B     ', prob%B
     READ( input, 1020 ) prob%BL
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'BL    ', prob%BL
     READ( input, 1020 ) prob%BU
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'BU    ', prob%BU
     READ( input, 1020 ) prob%X
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'X     ', prob%X
     IF ( ialgor > 1 ) THEN
       READ( input, 1020 ) prob%Y
       IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'Y     ', prob%Y
     END IF

!  Input the parameters in each group

     READ( input, 1020 ) prob%GPVALU
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'GPVALU', prob%GPVALU

!  Input the parameters in each individual element

     READ( input, 1020 ) prob%EPVALU
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'EPVALU', prob%EPVALU

!  Input the scale factors for the nonlinear elements

     READ( input, 1020 ) prob%ESCALE
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'ESCALE', prob%ESCALE

!  Input the scale factors for the groups

     READ( input, 1020 ) prob%GSCALE
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'GSCALE', prob%GSCALE

!  If a maximum is required, change the sign of the weight(s) associated with
!  the objective function group(s)

     IF ( get_max ) THEN
       control%print_max = .TRUE.
       IF ( ialgor == 1 ) THEN
         prob%GSCALE = - prob%GSCALE
       ELSE
         WHERE ( prob%KNDOFG == 1 ) prob%GSCALE = - prob%GSCALE
       END IF
     END IF

!  Input the scale factors for the variables

     READ( input, 1020 ) prob%VSCALE
     IF ( write_problem_data ) WRITE( dfiledevice, 1120 ) 'VSCALE', prob%VSCALE

!  Input the lower and upper bounds on the objective function

     READ( input, 1080 ) OBFBND( 1 ), OBFBND( 2 )
     IF ( write_problem_data )                                                 &
       WRITE( dfiledevice, 1180 ) 'OBFBND', OBFBND( 1 ), OBFBND( 2 )

!  Set the lower bound for the minimization

!    IF ( get_max ) THEN
!      objfbn = OBFBND( 1 )
!    ELSE
!      objfbn = - OBFBND( 2 )
!    END IF

!  Input a logical array which says whether an element has internal variables

     READ( input, 1030 ) prob%INTREP
     IF ( write_problem_data ) WRITE( dfiledevice, 1130 ) 'INTREP', prob%INTREP

!  Input a logical array which says whether a group is trivial

     READ( input, 1030 ) prob%GXEQX
     IF ( write_problem_data )                                                 &
       WRITE( dfiledevice, 1130 ) 'GXEQX ', prob%GXEQX

!  Input the names given to the groups and to the variables

     READ( input, 1040 ) prob%GNAMES
     IF ( write_problem_data ) WRITE( dfiledevice, 1140 ) 'GNAMES', prob%GNAMES
     READ( input, 1040 ) prob%VNAMES
     IF ( write_problem_data ) WRITE( dfiledevice, 1140 ) 'VNAMES', prob%VNAMES

!  Input the names given to the element and group types

     READ( input, 1040 ) ETYPES
     IF ( write_problem_data ) WRITE( dfiledevice, 1140 ) 'ETYPES', ETYPES
     READ( input, 1040 ) GTYPES
     IF ( write_problem_data ) WRITE( dfiledevice, 1140 ) 'GTYPES', GTYPES
     READ( input, 1010 ) ( j, i = 1, prob%n )

     CLOSE( dfiledevice )

!  If the problem has no objective function, implicitly restate it as a
!  least-squares problem.

!    square = .FALSE.
!    IF ( ialgor >= 2 ) THEN
!      DO i = 1,  prob%ng
!        IF ( prob%KNDOFG( i ) == 1 ) GO TO 112
!      END DO
!      square = .TRUE.
!112   CONTINUE
!    END IF

     IF ( control%print_level > 0 ) THEN
       WRITE( out, 2100 )
       CALL COPYRIGHT( out, '2002' )
       WRITE( out, 2010 ) TRIM( pname )
       IF ( iauto == 1 ) WRITE( out, 2610 )
       IF ( iauto == 2 ) WRITE( out, 2620 )
     END IF

!  Read a previous solution file for a re-entry

     IF ( warm_start .AND. wfiledevice > 0 ) THEN

       OPEN( wfiledevice, FILE = wfilename, FORM = 'FORMATTED',                &
             STATUS = 'OLD', IOSTAT = iores )
       IF ( iores /= 0 ) THEN
         write( control%out, 2160 ) iores, wfilename
         STOP
       END IF

       REWIND( wfiledevice )
       READ( wfiledevice, 2510 ) pname2
       IF ( pname2 /= pname ) THEN
         WRITE( out, 2500 )
         WRITE( out, 2550 ) pname2, pname
         GO TO 990
       END IF
       READ( wfiledevice, 2520 ) i
       IF ( i /= prob%n ) THEN
         WRITE( out, 2500 )
         WRITE( out, 2560 ) prob%n, i
         GO TO 990
       END IF
       READ( wfiledevice, 2520 ) i
       IF ( i /= prob%ng ) THEN
         WRITE( out, 2500 )
         WRITE( out, 2570 ) prob%ng, i
         GO TO 990
       END IF
       READ( wfiledevice, 2530 ) control%initial_mu
       READ( wfiledevice, 2590 )
       DO i = 1, prob%n
         READ( wfiledevice, 2540 ) prob%X( i ), tempna
         IF ( tempna /= prob%VNAMES( i ) ) THEN
           WRITE( out, 2500 )
           WRITE( out, 2580 ) tempna
           GO TO 990
         END IF
       END DO
       IF ( ialgor >= 2 ) THEN
         READ( wfiledevice, 2590 )
         DO i = 1,  prob%ng
           IF ( prob%KNDOFG( i ) > 1 ) THEN
             READ( wfiledevice, 2540 ) prob%Y( i ), tempna
             IF ( tempna /= prob%GNAMES( i ) ) THEN
               WRITE( out, 2500 )
               WRITE( out, 2600 ) tempna
               GO TO 990
             END IF
           END IF
         END DO
       END IF
!      CLOSE( wfiledevice )
     END IF

!  If required, test the derivatives of the element functions

     CALL RAND_initialize( seed )

     IF ( prob%nel > 0 .AND. dechke .AND. .NOT. fdgrad ) THEN
       inform%status = 0

!  Allocate temporary space for the tests

       ALLOCATE( ITEST( prob%nel ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ITEST' ; GO TO 800
       END IF
       ALLOCATE( IELVAR_temp( MAX( prob%nel, nvrels ) ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IELVAR_temp' ; GO TO 800
       END IF

!  Check the derivatives of the element functions at the point XT

       DO j = 1, prob%n
         CALL RAND_random_real( seed, .TRUE., rand )
         IF ( one <= prob%BL( j ) ) THEN
           XT( j ) = prob%BL( j ) + point1 * rand *                            &
             MIN( one, prob%BU( j ) - prob%BL( j ) )
         ELSE
           IF ( one >= prob%BU( j ) ) THEN
             XT( j ) = prob%BU( j ) - point1 * rand *                          &
               MIN( one, prob%BU( j ) - prob%BL( j ) )
           ELSE
             XT( j ) = one + point1 * rand * MIN( one, prob%BU( j ) - one )
           END IF
         END IF
       END DO

       IF ( testal ) THEN

!  Test all the nonlinear element functions

         inform%ncalcf = prob%nel
         DO j = 1, inform%ncalcf
           ITEST( j ) = j
         END DO
       ELSE

!  Test one nonlinear element function of each type

         inform%ncalcf = 0
         IELVAR_temp = 0
         IF ( control%print_level > 0 .AND. out > 0 ) WRITE( out, 2060 )
         DO j = 1, prob%nel
           ieltyp = prob%ITYPEE( j )
           IF ( IELVAR_temp( ieltyp ) == 0 ) THEN
             IF ( control%print_level > 0 .AND. out > 0 )                      &
               WRITE( out, 2040 ) j, ETYPES( ieltyp )
             inform%ncalcf = inform%ncalcf + 1
             ITEST( inform%ncalcf ) = j
             IELVAR_temp( ieltyp ) = 1
           END IF
         END DO
       END IF

!  Allocate real workspace

       ninmax = 0 ; nelmax = 0
       DO j = 1, prob%nel
         nin = prob%INTVAR( j )
         ninmax = MAX( ninmax, nin )
         nelmax = MAX( nelmax, prob%ISTAEV( j + 1 ) - prob%ISTAEV( j ) )
       END DO
       ALLOCATE( X_temp( nelmax ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'X_temp' ; GO TO 800
       END IF

!  Check the derivatives of the nonlinear element functions
!  --------------------------------------------------------

 100   CONTINUE

       IF ( iauto == 0 .OR. iauto == - 2 .OR. iauto == 1 .OR. iauto == 2 ) THEN
         CALL DRCHE_check_element_derivatives(                                 &
             prob, ICALCF, inform%ncalcf, XT, FUVALS, lfuval, IELVAR_temp,     &
             X_temp, nelmax, ninmax, epsmch, second, ITEST,                    &
             control%print_level, out,                                         &
             RANGE , inform%status, DRCHE_S, ELFUN  = ELFUN  )
       ELSE
         CALL DRCHE_check_element_derivatives(                                 &
             prob, ICALCF, inform%ncalcf, XT, FUVALS, lfuval, IELVAR_temp,     &
             X_temp, nelmax, ninmax, epsmch, second, ITEST,                    &
             control%print_level, out,                                         &
             RANGE , inform%status, DRCHE_S )
       END IF

!  compute group values and derivatives as required

       IF ( inform%status == - 1 ) THEN
         CALL ELFUN ( FUVALS, XT, prob%EPVALU, inform%ncalcf, prob%ITYPEE,     &
                      prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH,      &
                      prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,             &
                      prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,           &
                      prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,    &
                      prob%ISTEPA( prob%nel + 1 ) - 1, 1, i )
         j = 2
         IF ( second ) j = 3
         CALL ELFUN ( FUVALS, XT, prob%EPVALU, inform%ncalcf, prob%ITYPEE,     &
                      prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH,      &
                      prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,             &
                      prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,           &
                      prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,    &
                      prob%ISTEPA( prob%nel + 1 ) - 1, j, i )
       END IF
       IF ( inform%status == - 2 )                                             &
         CALL ELFUN ( FUVALS, X_temp, prob%EPVALU, inform%ncalcf, prob%ITYPEE, &
                      prob%ISTAEV, IELVAR_temp, prob%INTVAR, prob%ISTADH,      &
                      prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,             &
                      prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,           &
                      prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,    &
                      prob%ISTEPA( prob%nel + 1 ) - 1, 1, i )
       IF ( inform%status == - 3 )                                             &
         CALL ELFUN ( FUVALS, X_temp, prob%EPVALU, inform%ncalcf, prob%ITYPEE, &
                      prob%ISTAEV, IELVAR_temp, prob%INTVAR, prob%ISTADH,      &
                      prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,             &
                      prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,           &
                      prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,    &
                      prob%ISTEPA( prob%nel + 1 ) - 1, 2, i )
       IF ( inform%status < 0 ) GO TO 100

       DEALLOCATE( ITEST, IELVAR_temp, X_temp )

!  Stop if there were any warning messages

       IF ( inform%status > 0 .AND. .NOT. not_fatale ) THEN
         IF ( out > 0 .AND. control%print_level == 0 ) WRITE( out, 2370 )
         GO TO 990
       END IF
     END IF

!  If required, test the derivatives of the group functions

     IF ( dechkg .AND. prob%ng > 0 ) THEN
       inform%status = 0

!  Allocate temporary space for the tests

       ALLOCATE( ITEST( prob%ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ITEST' ; GO TO 800
       END IF
       ALLOCATE( IELVAR_temp( prob%ng ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IELVAR_temp' ; GO TO 800
       END IF

!  Check the derivatives of the group functions at the points FT.
!  Test all the nontrivial group functions

       IF ( testal ) THEN
         inform%ncalcg = 0
         DO j = 1, prob%ng
           IF ( prob%ITYPEG( j ) <= 0 ) CYCLE
           inform%ncalcg = inform%ncalcg + 1
           IELVAR_temp( j ) = 1
           ITEST( inform%ncalcg ) = j
           CALL RAND_random_real( seed, .TRUE., rand )
           FT( j ) = rand + point1
         END DO
       ELSE

!  Test one nontrivial group function of each type

         inform%ncalcg = 0
         IELVAR_temp = 0
         IF ( control%print_level > 0 ) WRITE( out, 2060 )
         DO j = 1, prob%ng
           igrtyp = prob%ITYPEG( j )
           IF ( igrtyp <= 0 ) CYCLE
           IF ( IELVAR_temp( igrtyp ) == 0 ) THEN
             IELVAR_temp( igrtyp ) = 1
             inform%ncalcg = inform%ncalcg + 1
             ITEST( inform%ncalcg ) = j
             IF ( control%print_level > 0 )                                    &
               WRITE( out, 2050 ) j, GTYPES( igrtyp )
             CALL RAND_random_real( seed, .TRUE., rand )
             FT( j ) = rand + point1
           END IF
         END DO
       END IF
       DEALLOCATE( IELVAR_temp )

!  Check the derivatives of the group functions
!  --------------------------------------------

  200  CONTINUE

       IF ( iauto == 0 .OR. iauto == - 3 ) THEN
         CALL DRCHG_check_group_derivatives(                                   &
             prob, FT, GVALS, ITEST, inform%ncalcg, epsmch,                    &
             control%print_level, out, inform%status, DRCHG_S, GROUP = GROUP )
       ELSE IF ( iauto == 1 .OR. iauto == 2 ) THEN
         CALL DRCHG_check_group_derivatives(                                   &
             prob, FT, GVALS, ITEST, inform%ncalcg, epsmch,                    &
             control%print_level, out, inform%status, DRCHG_S, GROUP = GROUP )
       ELSE
         CALL DRCHG_check_group_derivatives(                                   &
             prob, FT, GVALS, ITEST, inform%ncalcg, epsmch,                    &
             control%print_level, out, inform%status, DRCHG_S )
       END IF

!  compute group values and derivatives as required

       IF ( inform%status == - 1 .OR. inform%status == - 2 )                   &
         CALL GROUP ( GVALS , prob%ng, FT    , prob%GPVALU,                    &
                      inform%ncalcg, prob%ITYPEG, prob%ISTGPA, ITEST,          &
                      prob%ng, prob%ng + 1, prob%ng, prob%ng,                  &
                      prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., i )
       IF ( inform%status == - 1 .OR. inform%status == - 3 )                   &
         CALL GROUP ( GVALS , prob%ng, FT    , prob%GPVALU,                    &
                      inform%ncalcg, prob%ITYPEG, prob%ISTGPA, ITEST,          &
                      prob%ng, prob%ng + 1, prob%ng, prob%ng,                  &
                      prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE. , i )
       IF ( inform%status < 0 ) GO TO 200
       DEALLOCATE( ITEST )

!  Stop if there were any warning messages

       IF ( inform%status > 0 .AND. .NOT. not_fatalg ) THEN
         IF ( out > 0 .AND. control%print_level == 0 ) WRITE( out, 2380 )
         GO TO 990
       END IF
     END IF

!  If the problem has no objective function, implicitly restate it as a
!  least-squares problem. Ensure that all groups are non-trivial. Remember to
!  square the scaling factors

!    IF ( square ) THEN
!      control%print_max = .FALSE.
!      control%stopg = MIN( control%stopg, control%stopc )
!      prob%GXEQX( : prob%ng ) = .FALSE.
!    END IF

!  Obtain appropriate variable and group scalings, if required

     IF ( getsca .AND. ialgor > 1 ) THEN

!  Set up real workspace addresses

       norder = prob%n
       prob%n = numvar

!  Allocate integer workspace

       ALLOCATE( ISTINV( prob%nel + 1 ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'ISTINV' ; GO TO 800 ; END IF

!  Check the derivatives of the element functions at the point XT

       XT = MAX( prob%BL, MIN( prob%BU, prob%X ) )
       ISTINV( : prob%nel ) = prob%INTVAR( : prob%nel )
       inform%status = 0

!  Calculate the scalings

  220  CONTINUE
!      print_level_scaling = control%print_level
       IF ( iauto == 0 ) THEN
         CALL SCALN_get_scalings(                                              &
             prob, RANGE , data, inform%ncalcf, ICALCF, FT, GVALS, FUVALS,     &
             lfuval, control%stopg, control%stopc, scaleg, scalev, out,        &
             print_level_scaling, control%io_buffer, fdgrad, inform%status,    &
             SCALN_S, ELFUN = ELFUN, GROUP = GROUP )
       ELSE IF ( iauto == 1 .OR. iauto == 2 ) THEN
         CALL SCALN_get_scalings(                                              &
             prob, RANGE , data, inform%ncalcf, ICALCF, FT, GVALS, FUVALS,     &
             lfuval, control%stopg, control%stopc, scaleg, scalev, out,        &
             print_level_scaling, control%io_buffer, fdgrad, inform%status,    &
             SCALN_S, ELFUN = ELFUN, GROUP = GROUP )
       ELSE IF ( iauto == - 2 ) THEN
         CALL SCALN_get_scalings(                                              &
             prob, RANGE , data, inform%ncalcf, ICALCF, FT, GVALS, FUVALS,     &
             lfuval, control%stopg, control%stopc, scaleg, scalev, out,        &
             print_level_scaling, control%io_buffer, fdgrad, inform%status,    &
             SCALN_S, ELFUN = ELFUN )
       ELSE IF ( iauto == - 3 ) THEN
         CALL SCALN_get_scalings(                                              &
             prob, RANGE , data, inform%ncalcf, ICALCF, FT, GVALS,  FUVALS,    &
             lfuval, control%stopg, control%stopc, scaleg, scalev, out,        &
             print_level_scaling, control%io_buffer, fdgrad, inform%status,    &
             SCALN_S, GROUP = GROUP )
       ELSE
         CALL SCALN_get_scalings(                                              &
             prob, RANGE , data, inform%ncalcf, ICALCF, FT, GVALS, FUVALS,     &
             lfuval, control%stopg, control%stopc, scaleg, scalev, out,        &
             print_level_scaling, control%io_buffer, fdgrad, inform%status,    &
             SCALN_S )
       END IF

       IF ( inform%status < 0 ) THEN

!  Further problem information is required

         IF ( inform%status == - 1 ) THEN

!  Evaluate the element function and derivative value

           CALL ELFUN ( FUVALS, XT, prob%EPVALU, inform%ncalcf, prob%ITYPEE,   &
               prob%ISTAEV, prob%IELVAR, prob%INTVAR,                          &
               prob%ISTADH, prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,       &
               prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,                  &
               prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,           &
               prob%ISTEPA( prob%nel + 1 ) - 1, 1, i )
           IF ( .NOT. fdgrad ) CALL ELFUN ( FUVALS, XT, prob%EPVALU,           &
               inform%ncalcf, prob%ITYPEE, prob%ISTAEV, prob%IELVAR,           &
               prob%INTVAR, prob%ISTADH, prob%ISTEPA, ICALCF, prob%nel,        &
               prob%nel + 1, prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,    &
               prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,           &
               prob%ISTEPA( prob%nel + 1 ) - 1, 2, i )
         END IF
         IF ( inform%status == - 2 ) THEN

!  Evaluate the group function derivatives

           CALL GROUP ( GVALS, prob%ng, FT, prob%GPVALU, inform%ncalcf,        &
               prob%ITYPEG, prob%ISTGPA, ICALCF, prob%ng, prob%ng + 1,         &
               prob%ng, prob%ng, prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., i )
         END IF
         GO TO 220
       ELSE
         prob%n = norder
         prob%INTVAR( : prob%nel ) = ISTINV( : prob%nel )
         DEALLOCATE( ISTINV )
         IF ( inform%status > 0 ) GO TO 990
       END IF
     END IF

!  Prepare for the minimization

     inform%status = 0
     time = 0.0
     CALL CPU_TIME( ttotal )
     IF ( get_max ) THEN
       minmax = 'MAX'
       IF ( control%print_level > 0 ) WRITE( out, 2350 )
     ELSE
       minmax = 'MIN'
       IF ( control%print_level > 0 ) WRITE( out, 2340 )
     END IF

!  Call the minimizer

 300 CONTINUE
     CALL CPU_TIME( timm )

     IF ( iauto == 0 .OR. iauto == 1 .OR. iauto == 2 ) THEN
       CALL LANCELOT_solve(                                                    &
           prob, RANGE , GVALS , FT, XT, FUVALS, lfuval, ICALCF, ICALCG, IVAR, &
           Q, DGRAD , control, inform, data, ELFUN = ELFUN , GROUP = GROUP )
     ELSE IF ( iauto == - 2 ) THEN
       CALL LANCELOT_solve(                                                    &
           prob, RANGE , GVALS , FT, XT, FUVALS, lfuval, ICALCF, ICALCG, IVAR, &
           Q, DGRAD , control, inform, data, ELFUN = ELFUN )
     ELSE IF ( iauto == - 3 ) THEN
       CALL LANCELOT_solve(                                                    &
           prob, RANGE , GVALS , FT, XT, FUVALS, lfuval, ICALCF, ICALCG, IVAR, &
           Q, DGRAD , control, inform, data, GROUP = GROUP )
     ELSE
       CALL LANCELOT_solve(                                                    &
           prob, RANGE , GVALS , FT, XT, FUVALS, lfuval, ICALCF, ICALCG, IVAR, &
           Q, DGRAD , control, inform, data )
     END IF

     CALL CPU_TIME( t )
     timm = t - timm
     time = time + timm

!  The user wishes to terminate the minimization

     IF ( inform%status == 14 ) THEN
       IF ( out > 0 ) THEN
         WRITE( out, 2070 ) TRIM( runspec )
         DO
           READ( 5, 1070 ) alive_request
           SELECT CASE ( alive_request )
           CASE( 0 : 3 )
             EXIT
           CASE DEFAULT
             WRITE( out, 2080 )
           END SELECT
         END DO
       ELSE
         alive_request = 3
       END IF
     END IF

!  Write the solution file for possible re-entry.

     IF ( istore > 0 ) THEN
       dsave = wfiledevice > 0 .AND. MOD( inform%iter, istore ) == 0 .AND.     &
               ( inform%status == - 2 .OR. inform%status == - 4 )
     ELSE IF ( wfiledevice > 0 .AND. inform%status == 14 ) THEN
       dsave = alive_request == 0 .OR. alive_request == 3
     ELSE
       dsave = wfiledevice > 0 .AND. inform%status >= 0 .AND. istore == 0
     END IF
     IF ( dsave ) THEN
       INQUIRE( FILE = wfilename, OPENED = filexx )
       IF ( .NOT. filexx ) THEN
         INQUIRE( FILE = wfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( wfiledevice, FILE = wfilename, FORM = 'FORMATTED',           &
                  STATUS = 'OLD', POSITION = 'APPEND', IOSTAT = iores )
         ELSE
            OPEN( wfiledevice, FILE = wfilename, FORM = 'FORMATTED',           &
                  STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( control%out, 2160 ) iores, wfilename
           STOP
         END IF
       END IF

       REWIND( wfiledevice )
       WRITE( wfiledevice, 2450 ) pname, prob%n, prob%ng, inform%mu
       WRITE( wfiledevice, 2300 )
       DO i = 1, prob%n
         WRITE( wfiledevice, 2460 ) prob%X( i ), prob%VNAMES( i )
       END DO
       IF ( ialgor >= 2 ) THEN
         WRITE( wfiledevice, 2290 )
         DO i = 1, prob%ng
           IF ( prob%KNDOFG( i ) > 1 )                                         &
             WRITE( wfiledevice, 2460 ) prob%Y( i ), prob%GNAMES( i )
         END DO
       END IF
!      CLOSE( wfiledevice )
     END IF

!  The user wishes to terminate the minimization

     IF ( inform%status == 14 ) THEN
       SELECT CASE ( alive_request )
       CASE( 0 )
         GO TO 500
       CASE( 1, 3 )
         GO TO 300
       CASE( 2 )
         IF ( is_specfile ) CLOSE( input_specfile )
         INQUIRE( FILE = runspec, EXIST = is_specfile )
         IF ( is_specfile ) THEN
           OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED',           &
                 STATUS = 'OLD' )
           CALL LANCELOT_read_specfile( control, input_specfile )
         END IF
         GO TO 300
       END SELECT
     END IF

!  If there is insufficient workspace, terminate execution

     IF ( inform%status == 4 .OR. inform%status == 5 .OR.                      &
          inform%status == 6 .OR. inform%status == 7 ) GO TO 500

!  If the approximation to the solution has changed appreciably, print out the
!  solution in SIF format

     IF ( inform%newsol .OR. inform%status == 0 .OR. inform%status == 3 ) THEN
       inform%newsol = .FALSE.

!  If required, write the solution to a file

       IF ( write_solution ) THEN
         INQUIRE( FILE = sfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                STATUS = 'OLD', IOSTAT = iores )
         ELSE
            OPEN( sfiledevice, FILE = sfilename, FORM = 'FORMATTED',          &
                STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( out, 2160 ) iores, sfilename
           STOP
         END IF

         REWIND( sfiledevice )
         WRITE( sfiledevice, 2250 ) pname, inform%mu
         WRITE( sfiledevice, 2300 )
         DO i = 1, prob%n
            WRITE( sfiledevice, 2260 ) prob%VNAMES( i ), prob%X( i )
         END DO
         IF ( ialgor >= 2 ) THEN
           WRITE( sfiledevice, 2290 )
           nobjgr = 0
           DO i = 1, prob%ng
             IF ( prob%KNDOFG( i ) > 1 ) THEN
               WRITE( sfiledevice, 2260 ) prob%GNAMES( i ), prob%Y( i )
             ELSE
               nobjgr = nobjgr + 1
             END IF
           END DO
           IF ( nobjgr > 0 ) THEN
             IF ( get_max ) THEN
               WRITE( sfiledevice, 2360 ) - inform%obj
             ELSE
               WRITE( sfiledevice, 2270 ) inform%obj
             END IF
           END IF
         ELSE
           IF ( get_max ) THEN
             WRITE( sfiledevice, 2360 ) - inform%aug
           ELSE
             WRITE( sfiledevice, 2270 ) inform%aug
           END IF
         END IF
         CLOSE( sfiledevice )
       END IF

!  If required, write the solution vector to a file

       IF ( write_solution_vector ) THEN
         INQUIRE( FILE = vfilename, EXIST = filexx )
         IF ( filexx ) THEN
            OPEN( vfiledevice, FILE = vfilename, FORM = 'FORMATTED',          &
                STATUS = 'OLD', IOSTAT = iores )
         ELSE
            OPEN( vfiledevice, FILE = vfilename, FORM = 'FORMATTED',          &
                STATUS = 'NEW', IOSTAT = iores )
         END IF
         IF ( iores /= 0 ) THEN
           write( out, 2160 ) iores, vfilename
           STOP
         END IF

         REWIND( vfiledevice )
         DO i = 1, numvar
           WRITE( vfiledevice, "( ES22.15 )" ) prob%X( i )
         END DO
         CLOSE( vfiledevice )
       END IF
     END IF

!  Further problem information is required

     IF ( inform%status < 0 ) THEN
       IF ( inform%status == - 1 .OR. inform%status == - 3 .OR.                &
            inform%status == - 7 ) THEN
         IF ( control%print_level >= 10 ) WRITE( out, 2200 )

!  Evaluate the element function values

         CALL ELFUN ( FUVALS, XT, prob%EPVALU, inform%ncalcf, prob%ITYPEE,     &
             prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH,               &
             prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,                      &
             prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,                    &
             prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,             &
             prob%ISTEPA( prob%nel + 1 ) - 1, 1, i )
         IF ( i /= 0 ) THEN
           IF ( inform%status == - 1 ) THEN
             inform%status = 13
             IF ( control%print_level > 0 .AND. out > 0 ) WRITE( out, 2110 )
             GO TO 990
           ELSE
             inform%status = - 11 ; GO TO 300
           END IF
         END IF
       END IF

       IF ( ( inform%status == - 1 .OR. inform%status == - 5 .OR.              &
              inform%status == - 6 ) .AND. .NOT. fdgrad ) THEN
         ifflag = 2
         IF ( second ) ifflag = 3

!  Evaluate the element function derivatives

         IF ( control%print_level >= 10 ) WRITE( out, 2210 )
         CALL ELFUN ( FUVALS, XT, prob%EPVALU, inform%ncalcf, prob%ITYPEE,     &
             prob%ISTAEV, prob%IELVAR, prob%INTVAR, prob%ISTADH,               &
             prob%ISTEPA, ICALCF, prob%nel, prob%nel + 1,                      &
             prob%ISTAEV( prob%nel + 1 ) - 1, prob%nel + 1,                    &
             prob%nel + 1, prob%nel + 1, prob%nel, lfuval, prob%n,             &
             prob%ISTEPA( prob%nel + 1 ) - 1, ifflag, i )
         IF ( i /= 0 ) THEN
           IF ( inform%status == - 1 ) THEN
             inform%status = 13
             IF ( control%print_level > 0 .AND. out > 0 ) WRITE( out, 2110 )
             GO TO 990
           ELSE
             inform%status = - 11 ; GO TO 300
           END IF
         END IF
       END IF

       IF ( inform%status == - 2 .OR. inform%status == - 4 ) THEN

!  Evaluate the group function values

         IF ( control%print_level >= 10 ) WRITE( out, 2220 )
         IF ( control%print_level >= 100 )                                     &
           WRITE( out, 2390 ) ( FT( i ), i = 1, prob%ng )
         CALL GROUP ( GVALS, prob%ng, FT, prob%GPVALU, inform%ncalcg,          &
                      prob%ITYPEG, prob%ISTGPA, ICALCG,                        &
                      prob%ng, prob%ng + 1, prob%ng, prob%ng,                  &
                      prob%ISTGPA( prob%ng + 1 ) - 1, .FALSE., i )
         IF ( i /= 0 ) THEN
           IF ( inform%status == - 2 ) THEN
             inform%status = 13
             IF ( control%print_level > 0 .AND. out > 0 ) WRITE( out, 2110 )
             GO TO 990
           ELSE
             inform%status = - 11 ; GO TO 300
           END IF
         END IF
         IF ( control%print_level >= 100 )                                     &
           WRITE( out, 2400 ) ( GVALS( i, 1 ), i = 1, prob%ng )
       END IF
       IF ( inform%status == - 2 .OR. inform%status == - 5 ) THEN

!  Evaluate the group function derivatives

         IF ( control%print_level >= 10 ) WRITE( out, 2230 )
         CALL GROUP ( GVALS, prob%ng, FT, prob%GPVALU, inform%ncalcg,          &
                      prob%ITYPEG, prob%ISTGPA, ICALCG,                        &
                      prob%ng, prob%ng + 1, prob%ng, prob%ng,                  &
                      prob%ISTGPA( prob%ng + 1 ) - 1, .TRUE., i )
         IF ( i /= 0 ) THEN
           IF ( inform%status == - 2 ) THEN
             inform%status = 13
             IF ( control%print_level > 0 .AND. out > 0 ) WRITE( out, 2110 )
             GO TO 990
           ELSE
             inform%status = - 11 ; GO TO 300
           END IF
         END IF
       END IF

       IF ( inform%status <= - 8 ) THEN

!  Evaluate the preconditioned gradient - use a diagonal preconditioner

         IF ( control%print_level >= 10 ) WRITE( out, 2240 )
         Q( IVAR( : inform%nvar ) ) = DGRAD( : inform%nvar )
       END IF
       GO TO 300
     END IF

!  The minimization has been terminated
!  -------------------------------------

!    IF ( square ) THEN
!      IF ( inform%aug > control%stopc ) THEN
!        inform%status = 8
!        IF ( control%print_level > 0 .AND. out > 0 ) WRITE( out, 2150 )
!      END IF
!    END IF

!  Write out any remaining details

 500 CONTINUE

     CALL CPU_TIME( t )
     ttotal = t - ttotal
     IF ( rfiledevice > 0 .OR. ( out > 0 .AND. control%print_level == 0 ) ) THEN
       IF ( ialgor == 1 ) THEN ; fobj = inform%aug
       ELSE ; fobj = inform%obj ; END IF
     END IF
     IF ( out > 0 .AND. control%print_level == 0 ) THEN
       IF ( get_max ) THEN
         WRITE( out, 2180 ) - fobj
       ELSE
         WRITE( out, 2180 ) fobj
       END IF
       DO i = 1, prob%n
         WRITE( out, 2170 ) prob%VNAMES( i ), prob%X( i )
       END DO
       IF ( inform%status /= 0 ) WRITE( out, 2190 ) inform%status
     END IF
     IF ( out > 0 .AND. control%print_level > 0 ) THEN
       WRITE( out, 2000 ) inform%status, inform%iter, time, ttotal - time
       IF ( iauto == 1 ) WRITE( out, 2610 )
       IF ( iauto == 2 ) WRITE( out, 2620 )
       IF ( get_max ) THEN
         WRITE( out, 2350 )
       ELSE
         WRITE( out, 2340 )
       END IF
       WRITE( out, 2010 ) TRIM( pname )
     END IF

!  Write the solution summary file

     IF ( write_result_summary ) THEN
       READ( input, 1090 ) pname, nfree , nfixed, nlower, nupper, nboth ,      &
             nslack, nlinob, nnlnob, nlineq, nnlneq, nlinin, nnlnin
       BACKSPACE( rfiledevice )
       WRITE( rfiledevice, 2090 ) pname, nfree, nfixed, nlower, nupper, nboth, &
             nslack, nlinob, nnlnob, nlineq, nnlneq, nlinin, nnlnin
       IF ( ialgor == 1 ) optimi = 'SBMIN'
       IF ( ialgor == 2 ) optimi = 'AUGLG'
       IF ( get_max ) THEN
         WRITE( rfiledevice, 2310 ) pname, numvar, minmax, optimi,             &
           control%two_norm_tr, control%linear_solver,                         &
           control%first_derivatives,                                          &
           control%second_derivatives, control%exact_gcp,                      &
           control%accurate_bqp, control%structured_tr, control%more_toraldo,  &
           control%non_monotone, inform%iter, inform%ngeval, inform%itercg,    &
           inform%status, pname, ttotal, - fobj
       ELSE
         WRITE( rfiledevice, 2310 ) pname, numvar, minmax, optimi,             &
           control%two_norm_tr, control%linear_solver,                         &
           control%first_derivatives,                                          &
           control%second_derivatives, control%exact_gcp,                      &
           control%accurate_bqp, control%structured_tr, control%more_toraldo,  &
           control%non_monotone, inform%iter, inform%ngeval, inform%itercg,    &
           inform%status, pname, ttotal, fobj
       END IF
     END IF
     GO TO 900

!  Allocation errors

 800 CONTINUE
     WRITE( out, 2990 ) alloc_status, TRIM( bad_alloc )

!  End of execution

 900 CONTINUE

! De-allocate all arrays

     CALL LANCELOT_terminate( data, control, inform )
     DEALLOCATE( prob%ISTADG, prob%ISTGPA, prob%ISTADA, STAT = alloc_status )
     DEALLOCATE( prob%GXEQX , prob%IELING, prob%ISTAEV, STAT = alloc_status )
     DEALLOCATE( prob%ISTEPA, prob%ITYPEG, GTYPES, STAT = alloc_status )
     DEALLOCATE( prob%ITYPEE, prob%IELVAR, prob%ICNA, STAT = alloc_status )
     DEALLOCATE( prob%ISTADH, prob%INTVAR, IVAR, STAT = alloc_status )
     DEALLOCATE( ICALCG, ICALCF, prob%A, prob%B, prob%X, STAT = alloc_status )
     DEALLOCATE( prob%GPVALU, prob%EPVALU, prob%ESCALE, STAT = alloc_status )
     DEALLOCATE( prob%GSCALE, prob%VSCALE, XT    , DGRAD , STAT = alloc_status )
     DEALLOCATE( Q , FT, GVALS , prob%BL, prob%BU, ETYPES, STAT = alloc_status )
     DEALLOCATE( prob%INTREP, prob%GNAMES, prob%VNAMES, STAT = alloc_status )
     IF ( ialgor > 1 ) THEN
       DEALLOCATE( prob%Y, prob%C, STAT = alloc_status )
       DEALLOCATE( prob%KNDOFG, STAT = alloc_status )
     END IF

 990 CONTINUE

!  Close the opened files

     CLOSE( rfiledevice )
     CLOSE( wfiledevice )
     IF ( is_specfile ) CLOSE( input_specfile )

!  Close and delete the remaining files

     INQUIRE( FILE = control%alive_file, EXIST = alive )

     IF ( alive .AND. control%alive_unit > 0 ) THEN
       OPEN( control%alive_unit, FILE = control%alive_file,                    &
             FORM = 'FORMATTED', STATUS = 'UNKNOWN' )
       REWIND( control%alive_unit )
       CLOSE( control%alive_unit, STATUS = 'DELETE' )
     END IF
     RETURN

!  Non-executable statements

 1000  FORMAT( I10 )
 1010  FORMAT( ( 10I8 ) )
 1020  FORMAT( ( 1P, 4D16.8 ) )
 1030  FORMAT( ( 72L1 ) )
 1040  FORMAT( ( 8A10 ) )
 1050  FORMAT( 10I10 )
 1060  FORMAT( I2, A10, I2 )
 1070  FORMAT( I1 )
 1080  FORMAT( 1P, 2D16.8 )
 1090  FORMAT( A10, 12I8 )
 1100  FORMAT( A10, 3I8 )
 1110  FORMAT( 1X, A6, /, ( 1X, 12I6 ) )
 1120  FORMAT( 1X, A6, /, ( 1X, 1P, 4D16.8 ) )
 1130  FORMAT( 1X, A6, /, ( 1X, 72L1 ) )
 1140  FORMAT( 1X, A6, /, ( 1X, 8A10 ) )
 1180  FORMAT( 1X, A6, /, 1P, 2D16.6 )
 2000  FORMAT( /, ' inform             = ', I16,                               &
                  ' Number of iterations = ', I16, /,                          &
                  ' Time( LANCELOT B ) = ', 0P, F16.2,                         &
                  ' Time( other )        = ', 0P, F16.2 )
 2010  FORMAT( /, ' ************* Problem ', A, ' *****************' )
 2040  FORMAT( ' Element number ', I6,' chosen as representative of',          &
               ' element type ', A10 )
 2050  FORMAT( ' Group number ', I6,                                           &
               ' chosen as representative of',' group type ', A10 )
 2060  FORMAT( / )
 2070  FORMAT( /, ' ** You have interupted the execution. Do you wish to:', /, &
                  ' 0. Terminate execution ? ,', /,                            &
                  ' 1. Resume execution without further changes ? , ', /,      &
                  ' 2. Reread the specification file ', A,                     &
                  ' and resume execution ? , or ', /,                          &
                  ' 3. Save the solution estimate and resume execution ? ' )
 2080  FORMAT( ' Please reply 0, 1, 2 or 3 ' )
 2090  FORMAT( A10, 12I8 )
 2100  FORMAT( /, ' *-*-*-*-*-* LANCELOT B -*-*-*-*-*-*' )
 2110  FORMAT( /, ' Error evaluating problem function at the initial point ' )
!2150  FORMAT( /, ' Constraint violations are large. Problem',                 &
!                 ' possibly infeasible. ' )
 2160  FORMAT( ' IOSTAT = ', I6, ' when opening file ', A9, '. Stopping ' )
 2170  FORMAT( 12X, A10, 6X, ES22.14 )
 2180  FORMAT( /, ' objective function value = ', ES22.14, / )
 2190  FORMAT( /, ' ** Warning. Exit from LANCELOT with inform = ', I3 )
 2200  FORMAT( /, ' Evaluating element functions ' )
 2210  FORMAT( /, ' Evaluating derivatives of element functions ' )
 2220  FORMAT( /, ' Evaluating group functions ' )
 2230  FORMAT( /, ' Evaluating derivatives of group functions ' )
 2240  FORMAT( /, ' Evaluating user supplied  preconditioner ' )
 2250  FORMAT( '*   LANCELOT solution for problem name: ', A10, /,             &
               '*   penalty parameter value is ',ES12.4 )
 2260  FORMAT( '    Solution  ', A10, ES12.5 )
 2270  FORMAT( /, ' XL solution  ', 10X, ES12.5 )
 2290  FORMAT( /, '*   Lagrange multipliers ', / )
 2300  FORMAT( /, '*   variables ', / )
 2310  FORMAT( A10, I6, 1X, A3, 1X, A5, '( ', L1, ' ', I2,' ', I1, ' ', I1,    &
               ' ', L1, ' ', L1, ' ', L1, ' ', I4, ' ', I4, ') ',     &
               3I8, I4,                                                        &
               /, A10,' time ',0P, F12.2, ' Final function value ', ES12.4 )
 2320  FORMAT( /, ' Only a feasible point is required ... removing objective ' )
 2340  FORMAT( /, ' *-*-*-*-*-*-* Minimizer sought *-*-*-*-*-*-*-*-*' )
 2350  FORMAT( /, ' *-*-*-*-*-*-* Maximizer sought *-*-*-*-*-*-*-*-*' )
 2360  FORMAT( /, ' XU solution  ', 10X, ES12.5 )
 2370  FORMAT( /, ' Possible error in element derivative. Stopping ' )
 2380  FORMAT( /, ' Possible error in group derivative. Stopping ' )
 2390  FORMAT( /, ' Group values ', /, ( 6ES12.4 ) )
 2400  FORMAT( /, ' Group function values ', /, ( 6ES12.4 ) )
 2450  FORMAT( 16X, A10,' problem name ', /, 16X, I8,                          &
               '   number of variables ', /, 16X, I8,'   number of groups ',   &
               /, ES24.16,'   penalty parameter value ' )
 2460  FORMAT( ES24.16, 2X, A10 )
 2500  FORMAT( /, ' Re-entry requested ' )
 2510  FORMAT( 16X, A10 )
 2520  FORMAT( 16X, I8 )
 2530  FORMAT( ES24.16 )
 2540  FORMAT( ES24.16, 2X, A10 )
 2550  FORMAT( /, ' *** Exit from LANCELOT_main: re-entry requested with',     &
                  ' data for problem ', A10, /,                                &
                  '     while the most recently decoded problem is ', A10 )
 2560  FORMAT( /, ' *** Exit from LANCELOT_main: number of variables changed', &
                  ' from ', I8,' to ', I8,' on re-entry ' )
 2570  FORMAT( /, ' *** Exit from LANCELOT_main: number of groups changed ',   &
                  ' from ', I8, ' to ', I8, ' on re-entry ' )
 2580  FORMAT( /, ' *** Exit from LANCELOT_main: variable named ', A10,        &
                  ' out of order on re-entry ' )
 2590  FORMAT( // )
 2600  FORMAT( /, ' *** Exit from LANCELOT_main: Lagrange multiplier for ',    &
               A10, ' out of order on re-entry ' )
 2610  FORMAT( /, ' =+=+= Automatic derivatives - forward mode  =+=+=' )
 2620  FORMAT( /, ' =+=+= Automatic derivatives - backward mode =+=+=')
 2990  FORMAT( ' ** Message from -USE_LANCELOT-', /,                           &
               ' Allocation error (status = ', I0, ') for ', A )

!  End of subroutine USE_LANCELOT

      END SUBROUTINE USE_LANCELOT

!  End of module USELANCELOT_double

    END MODULE GALAHAD_USELANCELOT_double
