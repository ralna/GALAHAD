! THIS VERSION: GALAHAD 2.5 - 08/02/2013 AT 16:10 GMT.

Program NLLSRTMA

  ! Version 9 July 2008

  Implicit None
  Integer :: N, M
  Integer, Parameter :: INSPEC = 46, INPUT = 47, IOUT = 6
  INTEGER, PARAMETER :: io_buffer = 11
  INTEGER, PARAMETER :: wp=KIND( 1.0D+0 )  ! set precision
  INTEGER :: IERR, MYMAXIT, iter, iprint, iter_int, iter_int_tot, cutest_status 
  Real( Kind = wp ), Dimension( : ), Allocatable :: X,BL,BU,V,CL,CU,TOL,OUTPUT
  Real, Dimension( 2 ) :: CPU( 2 )
  Real, Dimension( 7 ) :: CALLS( 7 )
  Character( len = 10 ) ::  PNAME
  Character( len = 10 ), Dimension( : ), Allocatable :: VNAMES, GNAMES
  Logical, Dimension( : ), Allocatable :: EQUATN, LINEAR
  !
  !  Open the relevant problem file.
  !
  Open ( INPUT, FILE = 'OUTSDIF.d', FORM = 'FORMATTED', STATUS = 'OLD' )
  Rewind INPUT
  !
  ! Open the results_rt file
  !
  OPEN( UNIT = 100, FILE = 'results_rt', STATUS='old', POSITION='append')
  !
  !  Get problem dimensions 
  !
  Call CUTEST_cdimen( cutest_status, INPUT, N, M )
  IF ( cutest_status /= 0 ) GO TO 910
  !
  !  Set up SIF data from the problem file
  !
  Allocate(TOL(4), OUTPUT(2),  X( N ), BL( N ), BU( N ),V( M+1 ), CL( M+1 ),   &
             CU( M+1 ), EQUATN( M+1 ), LINEAR( M+1 ) )
  Call CUTEST_csetup( cutest_status, INPUT, IOUT, io_buffer, N, M, X, BL, BU,  &
                      V, CL, CU, EQUATN, LINEAR, 1, 0, 0 )
  IF ( cutest_status /= 0 ) GO TO 910
  !
  !  Obtain problem/variables/constraints names.
  !
  Allocate( VNAMES( N ) ,GNAMES( M ))
  Call CUTEST_cnames( cutest_status, N, M, PNAME, VNAMES, GNAMES )
  IF ( cutest_status /= 0 ) GO TO 910
  !
  !  Call the optimizer.
  !
  TOL(1) = 10.0_wp**(-6)   !epsilon_ca  ! input tolerance
  TOL(2) = 10.0_wp**(-12)    !epsilon_cr
  TOL(3) = 10.0_wp**(-6)   !epsilon_ga
  TOL(4) = 10.0_wp**(-12)   !epsilon_gr
  MYMAXIT = 10000
 ! MYMAXIT = 118


  iprint=1  !iprint>0 print one line per iteration

  if (iprint.gt.0) then   
     !
     ! Open the history_rt file
     !
     OPEN( UNIT = 200, FILE = 'history_rt', STATUS='old' , POSITION='append')
     write(200,'(''Problem '',A10, ''N='',I5,'' M='',I5)') PNAME , N,M
  end if
  CALL NLLSRT(N,M, X, MYMAXIT,TOL,OUTPUT, IERR,iter,iter_int,iter_int_tot,iprint)
	
  !
  !  Close the problem file
  !
  Close( INPUT )
  !
  !  Write the standard statistics (of which some may be irrelevant)
  !
  !    CALLS( 1 ): number of calls to the objective function
  !    CALLS( 2 ): number of calls to the objective gradient
  !    CALLS( 3 ): number of calls to the objective Hessian
  !    CALLS( 4 ): number of Hessian times vector products
  !           --constrained problems only--
  !    CALLS( 5 ): number of calls to the constraint functions
  !    CALLS( 6 ): number of calls to the constraint gradients
  !    CALLS( 7 ): number of calls to the constraint Hessians
  !           -----------------------------
  !
  !    CPU( 1 ) : CPU time (in seconds) for USETUP or CSETUP
  !    CPU( 2 ) : CPU time ( in seconds) since the end of USETUP or CSETUP
  !
  !  Note that each constraint function is counted separately.
  !  Evaluating all the constraints thus results in PNC evaluations, where
  !  PNC is the number of constraints in the problem.  Note that PNC does not
  !  include repetitions for constraints having full ranges.
  
  !  (N, is the dimension of the problem, M is the number of constraints)
  !
  Call CUTEST_creport( cutest_status, CALLS, CPU )      
  IF ( cutest_status /= 0 ) GO TO 910
  !
  ! print on results_rt
  !
  write(100,*) 
  write(100,'(''Problem '',A10, ''N='',I5,'' M='',I5)') PNAME , N,M
  Write(100,'(''||C_k||='',d10.5,'' ||C_k||^2='',d10.5,'' ||g_k||='',d10.5 )')&
       sqrt(OUTPUT(1)),OUTPUT(1), OUTPUT(2)
  write(100,'(  ''iter='',I5,''  iter_int='',I8,''  iter_int_tot='',I8 )') &
	 iter, iter_int, iter_int_tot
  write(100,'(''exit='',I2,''    Set up time= '',f8.2, '' solve time= '',f8.2   )') &
	IERR, CPU(1), CPU(2) 
  !
  !
  ! print on history_rt
  !
  if (iprint.gt.0) then
     write(200,'()') 
     write(200,'(''||C_k||='',d10.5,'' ||C_k||^2='',d10.5,'' ||g_k||='', d10.5 )') sqrt(OUTPUT(1)),OUTPUT(1), OUTPUT(2)
     WRITE(200,'( ''iter='',I5, ''  iter_int='',I8,''  iter_int_tot='',I8 )') &
	iter, iter_int, iter_int_tot
     WRITE(200,'( ''exit='',I2,''    Set up time= '',f8.2, '' solve time= '',f8.2   )') & 
	IERR, CPU(1), CPU(2)
     write(200,'( ''************************************************'')')
  end if
  !
  !
  Write ( IOUT, 2000 ) PNAME, N, M,        &
        CALLS( 5 ), CALLS( 6 ), iter, iter_int, iter_int_tot
  Write ( IOUT, 2001 ) IERR, OUTPUT(1), OUTPUT(2),  CPU( 1 ), CPU( 2 )
  !
  !  Free allocated memory
  !
  Deallocate( OUTPUT,TOL,X, BU, BL, VNAMES, EQUATN, LINEAR )
  Deallocate( V, CL, CU, GNAMES )
  CALL CUTEST_cterminate( cutest_status )
  !
  !  Exit
  !
  Stop

910 CONTINUE
    WRITE( out, "( ' CUTEst error, status = ', i0, ', stopping' )" )           &
      cutest_status
    STOP
  !
  !  Non-executable statements.
  !
  !  The following is the complete standard statistics output format
  !
2000 Format( /, 24('*'), ' CUTEst statistics ', 24('*') // &
          ,' Code used                :  NLLSRT',    / &
       !   ,' Variant                  :  name of a variant, if needed',/ &
          ,' Problem                  :  ', A10,  / &
          ,' # variables              =      ', I10 / &
          ,' # constraints            =      ', I10 / &
        !  ,' # linear constraints     =      ', I10 / &
        !  ,' # equality constraints   =      ', I10 / &
        !  ,' # inequality constraints =      ', I10 / &
        !  ,' # bounds                 =      ', I10 / &
          ,' # constraints functions  =        ', F8.2 / &
          ,' # constraints gradients  =        ', F8.2 / &   
          ,' # external iterations    =        ', I10 / &
          ,' # internal iterations    =        ', I10 / &
          ,' # tot internal iterations=        ', I10 )
2001 Format(                                          &
           ' Exit code                =        ', I10 / &
          ,' Final f=||C||^2         = ', E15.7/ &
          ,' Final ||grad||          = ', E15.7 / &
          ,' Set up time              =      ', 0P, F10.2, ' seconds'/ &
          ' Solve time               =      ', 0P, F10.2, ' seconds'// &
          66('*') / )


!IERR   = 0   zero residual solution found
!       = 1   non zero solution found
!       = -3  maxit   
!       = -1  small step


End Program NLLSRTMA

