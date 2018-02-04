! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-  L P S Q P  -  R U N L P S Q P  *-*-*-*-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  Started: August 15th 2002

   PROGRAM RUNLPSQP_double

!  This is the driver program for running LPSQP for a variety of computing
!  systems. It opens and closes all the files, allocate arrays, reads and
!  checks data, and calls the appropriate minimizers

!A   USE GALAHAD_LPSQPA_double
!B   USE GALAHAD_LPSQP_double
     USE GALAHAD_SPECFILE_double
     IMPLICIT NONE

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------
!   D e r i v e d   T y p e s
!-------------------------------

     TYPE ( LPSQP_control_type ) :: control
     TYPE ( LPSQP_inform_type ) :: inform
     TYPE ( LPSQP_data_type ) :: data

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

!  Problem input characteristics

     INTEGER, PARAMETER :: io_buffer = 11
     INTEGER, PARAMETER :: input = 55
     CHARACTER ( LEN = 16 ) :: prbdat = 'OUTSDIF.d'

!  Specfile characteristics

     INTEGER, PARAMETER :: input_specfile = 34
     INTEGER, PARAMETER :: lspec = 29
     CHARACTER ( LEN = 16 ) :: specname = 'RUNLPSQP'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec
     CHARACTER ( LEN = 16 ) :: runspec = 'RUNLPSQP.SPC'

!  Default values for specfile-defined parameters

     INTEGER :: dfiledevice = 26
     INTEGER :: rfiledevice = 47
     INTEGER :: sfiledevice = 62
     INTEGER :: wfiledevice = 59
     LOGICAL :: write_problem_data   = .FALSE.
     LOGICAL :: write_solution       = .FALSE.
     LOGICAL :: write_result_summary = .FALSE.
     CHARACTER ( LEN = 30 ) :: dfilename = 'LANB.data'
     CHARACTER ( LEN = 30 ) :: rfilename = 'LANBRES.d'
     CHARACTER ( LEN = 30 ) :: sfilename = 'LANBSOL.d'
     CHARACTER ( LEN = 30 ) :: wfilename = 'LANBSAVE.d'
     LOGICAL :: fulsol = .FALSE.
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
     LOGICAL :: warm_start = .FALSE.
     INTEGER :: istore = 0
     REAL ( KIND = wp ) :: rho = 100000.0
     LOGICAL :: one_norm = .FALSE.

!  Output file characteristics

     INTEGER :: errout = 6

!    INTEGER :: m, n

!  ------------------ Open the specfile for runlpsqp ----------------

     OPEN( input_specfile, FILE = runspec, FORM = 'FORMATTED', STATUS = 'OLD' )

!   Define the keywords

     spec( 1 )%keyword  = 'write-problem-data'
     spec( 2 )%keyword  = 'problem-data-file-name'
     spec( 3 )%keyword  = 'problem-data-file-device'
     spec( 4 )%keyword  = 'print-full-solution'
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
     spec( 24 )%keyword = 'restart-from-previous-point'
     spec( 25 )%keyword = 'restart-data-file-name'
     spec( 26 )%keyword = 'restart-data-file-device'
     spec( 27 )%keyword = 'save-data-for-restart-every'
     spec( 28 )%keyword = 'rho-used'
     spec( 29 )%keyword = 'one-norm-penalty'

!   Read the specfile

     CALL SPECFILE_read( input_specfile, specname, spec, lspec, errout )

!   Interpret the result

     CALL SPECFILE_assign_logical( spec( 1 ), write_problem_data, errout )
     CALL SPECFILE_assign_string ( spec( 2 ), dfilename, errout )
     CALL SPECFILE_assign_integer( spec( 3 ), dfiledevice, errout )
     CALL SPECFILE_assign_logical( spec( 4 ), fulsol, errout )
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
     CALL SPECFILE_assign_logical( spec( 24 ), warm_start, errout )
     CALL SPECFILE_assign_string ( spec( 25 ), wfilename, errout )
     CALL SPECFILE_assign_integer( spec( 26 ), wfiledevice, errout )
     CALL SPECFILE_assign_integer( spec( 27 ), istore, errout )
     CALL SPECFILE_assign_real( spec( 28 ), rho, errout )
     CALL SPECFILE_assign_logical( spec( 29 ), one_norm, errout )

     IF ( dechk .OR. testal ) THEN ; dechke = .TRUE. ; dechkg = .TRUE. ; END IF
     IF ( not_fatal ) THEN ; not_fatale = .TRUE. ; not_fatalg = .TRUE. ; END IF
     IF ( scale ) THEN ; scaleg = .TRUE. ; scalev = .TRUE. ; END IF

!  Open the problem data file

     OPEN( input, FILE = prbdat, FORM = 'FORMATTED', STATUS = 'OLD' )
     REWIND( input )
!    CALL CDIMEN( input, n, m )

!  Set up data for next problem

     CALL LPSQP_initialize( data, control, inform )
     CALL LPSQP_read_specfile( control, input_specfile )

!  Solve the problem

     CALL LPSQP_solve( input, io_buffer, rho, one_norm, control, inform, data )

!  Close any opened files

     CLOSE( input )
     CLOSE( input_specfile )

     STOP

!  End of program RUNLPSQP

   END PROGRAM RUNLPSQP_double
