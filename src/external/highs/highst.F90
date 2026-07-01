! THIS VERSION: GALAHAD 5.6 - 2026-06-30 AT 11:30 GMT.

#include "galahad_modules.h"

  PROGRAM HiGHS_main

!  test program for HiGHS strictly-convex QP package

!  Nick Gould, June 2026

  USE GALAHAD_KINDS_precision, ONLY: ip_, rp_
  USE HIGHS_INTERFACE_precision

  IMPLICIT NONE

!  Parameters

  INTEGER ( KIND = ip_ ), PARAMETER :: input = 55
  INTEGER ( KIND = ip_ ), PARAMETER :: io_buffer = 11
  INTEGER ( KIND = ip_ ), PARAMETER :: out = 6
  INTEGER ( KIND = ip_ ), PARAMETER :: input_specfile = 34
  INTEGER ( KIND = ip_ ), PARAMETER :: spec = 29
  REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
  REAL ( KIND = rp_ ), PARAMETER :: ac_tol = ten ** ( - 6 )
  REAL ( KIND = rp_ ), PARAMETER :: eq_tol = ten ** ( - 10 )
  REAL ( KIND = rp_ ), PARAMETER :: infinity = ten ** 20

!  problem parameters

  INTEGER, PARAMETER :: n = 3, m = 2, h_ne = 4, a_ne = 4

!  Highs parameters

  INTEGER ( KIND = ipc_ ), PARAMETER :: sense = 1
  INTEGER ( KIND = ipc_ ), PARAMETER :: aformat_colwise = 1
  INTEGER ( KIND = ipc_ ), PARAMETER :: qformat_triangular = 1
  INTEGER ( KIND = ipc_ ), PARAMETER :: modelstatus_optimal = 7
  INTEGER ( KIND = ipc_ ), PARAMETER :: runstatus_error = - 1
  INTEGER ( KIND = ipc_ ), PARAMETER :: runstatus_ok = 0
  INTEGER ( KIND = ipc_ ), PARAMETER :: runstatus_warning = - 1
  REAL ( KIND = rpc_ ) :: offset = 0
  LOGICAL, PARAMETER :: no_highs_logging = .TRUE.
  LOGICAL ( KIND = c_bool ), PARAMETER :: logical_false = .false.
  LOGICAL ( KIND = c_bool ), PARAMETER :: logical_true = .true.
  INTEGER ( KIND = ipc_ ) :: iteration_count
  REAL ( KIND = rpc_ ) :: objective_function_value
  TYPE ( c_ptr ) :: highs

!  local variables

  INTEGER ( KIND = ip_ ) :: status, i, ir, ic, iter, j, l, l1, l2, row, col
  INTEGER ( KIND = ip_ ) :: mfixed, mdegen, nfixed, ndegen, mequal, mredun
  REAL ( KIND = rp_ ) :: f, obj, res_p, res_d
  REAL ( KIND = rp_ ) :: gcol, max_d
  LOGICAL :: qp, fulsol = .FALSE.
  CHARACTER ( LEN = 10 ) :: p_name
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_ptr, A_row
  INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_ptr, H_row
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: G, X_0, X_l, X_u
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Z, Y, C_l, C_u, C, G_l
  REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, H_val
  CHARACTER ( LEN =  5 ) :: state

!  local C variables

  INTEGER ( KIND = ip_ ) :: modelstatus, runstatus
  INTEGER ( KIND = ipc_ ) :: num_primal_infeasibilities
  INTEGER ( KIND = ipc_ ) :: num_dual_infeasibilities
  REAL ( KIND = rpc_ ) :: max_primal_infeasibility
  REAL ( KIND = rpc_ ) :: max_dual_infeasibility
  CHARACTER ( KIND = c_char, LEN = 10 ) :: solver_used
  INTEGER ( KIND = ipc_ ) :: numcol, numrow, numnz, hessian_numnz
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: astart
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: aindex
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: qstart
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: qindex
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: integrality
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: colbasisstatus
  INTEGER ( KIND = ipc_ ), ALLOCATABLE, DIMENSION( : ) :: rowbasisstatus
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: colcost
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: collower
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: colupper
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: rowlower
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: rowupper
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: avalue
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: qvalue
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: sol
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: colvalue
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: coldual
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: rowvalue
  REAL ( KIND = rpc_ ), ALLOCATABLE, DIMENSION( : ) :: rowdual
  
!  results summary output if required (set output_summary > 10) 

  INTEGER ( KIND = ip_ ) :: output_summary = 0
! INTEGER ( KIND = ip_ ) :: output_summary = 47
! CHARACTER ( LEN = 10 ) :: summary_filename = 'HIGHS.res'

  ALLOCATE( G( n ), X_l( n ), X_u( n ), STAT = status )
  ALLOCATE( C( m ), C_l( m ), C_u( m ), STAT = status )
  ALLOCATE( X_0( n ), Y( m ), Z( n ), STAT = status )
  ALLOCATE( H_val( h_ne ), H_row( h_ne ), H_ptr( n + 1 ), STAT = status )
  ALLOCATE( A_val( a_ne ), A_row( a_ne ), A_ptr( n + 1 ), STAT = status )

!  input the problem data per GALAHAD's standard QP format

  f = 1.0_rp_                              ! objective constant
  G = [ 0.0_rp_, 2.0_rp_, 0.0_rp_ ]        ! objective gradient
  C_l = [ 1.0_rp_, 2.0_rp_ ]               ! constraint lower bound
  C_u = [ 2.0_rp_, 2.0_rp_ ]               ! constraint upper bound
  X_l = [ - 1.0_rp_, - infinity, - infinity ] ! variable lower bound
  X_u = [ 1.0_rp_, infinity, 2.0_rp_ ]     ! variable upper bound
  X_0 = 0.0_rp_; Y = 0.0_rp_ ; Z = 0.0_rp_ ! start from zero
  H_val = [ 1.0_rp_, 2.0_rp_, 1.0_rp_, 3.0_rp_ ] ! Hessian H, column storage
  H_row = [ 1, 2, 3, 3 ]                         ! NB lower triangle
  H_ptr = [ 1, 2, 4, 5 ] 
  A_val = [ 2.0_rp_, 1.0_rp_, 1.0_rp_, 1.0_rp_ ] ! Jacobian A, column storage
  A_row = [ 1, 1, 2, 2 ]
  A_ptr = [ 1, 2, 4, 5 ]

!  is it an LP or a QP?

  qp = H_ptr( n + 1 ) /= 1

!  transform the data into the structure required by HiGHS, allocating
!  space as required, and deallocating the reciprocal data after it
!  has been transfered

  DEALLOCATE( X_0, Z, STAT = status )
  IF ( status /= 0 ) GO TO 990

!  set dimensions

  numcol = n
  numrow = m
  numnz = A_ptr( n + 1 ) - 1
  hessian_numnz = H_ptr( n + 1 ) - 1

!  transfer the constant and linear term for the objective function

  offset = f
  ALLOCATE( colcost( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  colcost( : numcol ) = G( : numcol )
  DEALLOCATE( G, STAT = status )
  IF ( status /= 0 ) GO TO 990
  
!  transfer the lower and upper variable bounds

  ALLOCATE( collower( numcol ), colupper( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  collower( : numcol ) = X_l( : numcol )
  colupper( : numcol ) = X_u( : numcol )
  DEALLOCATE( X_l, X_u, STAT = status )
  IF ( status /= 0 ) GO TO 990

!  transfer the lower and upper constraint bounds

  ALLOCATE( rowlower( numrow ), rowupper( numrow ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  rowlower( : numrow ) = C_l( : numrow )
  rowupper( : numrow ) = C_u( : numrow )
  DEALLOCATE( C_l, C_u, STAT = status )
  IF ( status /= 0 ) GO TO 990

!  transfer the constraint matrix

!write(6,*) ' A_ptr ', A_ptr
!write(6,*) ' A_row ', A_row
!write(6,*) ' A_val ', A_val

  ALLOCATE( astart( numcol ), aindex( numnz ), avalue( numnz ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  astart( : numcol ) = A_ptr( : numcol ) - 1
  aindex( : numnz ) = A_row( : numnz ) - 1
  avalue( : numnz ) = A_val( : numnz )
!write(6,*) ' astart ', astart
!write(6,*) ' aindex ', aindex
!write(6,*) ' avalue ', avalue
  DEALLOCATE( A_ptr, A_row, A_val, STAT = status )
  IF ( status /= 0 ) GO TO 990

!  transfer the Hessian term for the objective function

!write(6,*) ' qp ', qp
  IF ( qp ) THEN
!DO i = 1, n
!  WRITE( 6, "( I6, ' : ', ( 10I6 ) )" ) i, ( H_row( H_ptr( i ) : H_ptr( i + 1 ) - 1 ) )
!END DO
!write(6,*) ' H_ptr ', H_ptr
!write(6,*) ' H_row ', H_row
!write(6,*) ' H_val ', H_val

    ALLOCATE( qstart( numcol ), qindex( hessian_numnz ),                       &
              qvalue( hessian_numnz ), STAT = status )
    IF ( status /= 0 ) GO TO 990
    qstart( : numcol ) = H_ptr( : numcol ) - 1
    qindex( : hessian_numnz ) = H_row( : hessian_numnz ) - 1
    qvalue( : hessian_numnz ) = H_val( : hessian_numnz )
!write(6,*) ' qstart ', qstart
!write(6,*) ' qindex ', qindex
!write(6,*) ' qvalue ', qvalue
  END IF
  DEALLOCATE( H_ptr, H_row, H_val, STAT = status )
  IF ( status /= 0 ) GO TO 990

!  allocate other arrays needed by HiGHS

  ALLOCATE( integrality( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  integrality = 0

  ALLOCATE( sol( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  sol = 0.0_rpc_
  ALLOCATE( colvalue( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  colvalue = 0.0_rpc_
  ALLOCATE( coldual( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  coldual = 0.0_rpc_
  ALLOCATE( rowvalue( numrow ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  rowvalue = 0.0_rpc_
  ALLOCATE( rowdual( numrow ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  rowdual = 0.0_rpc_
  ALLOCATE( colbasisstatus( numcol ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  colbasisstatus = 0
  ALLOCATE( rowbasisstatus( numrow ), STAT = status )
  IF ( status /= 0 ) GO TO 990
  rowbasisstatus = 0

!  create the HiGHS environment

  CALL Highs_create( highs )

!  make sure that HiGHS is available

  IF ( .NOT. C_ASSOCIATED( highs ) ) THEN
    WRITE( out, "( ' call to HiGHS failed, substitute dummy package called' )" )
    STOP
  END IF

!  record the bound infinity used

  CALL Highs_setDoubleOptionValue( highs, "infinite_bound" // C_NULL_CHAR,     &
                                   infinity, runstatus )

! IF ( no_highs_logging ) THEN
    CALL Highs_setBoolOptionValue( highs, "output_flag" // C_NULL_CHAR,        &
                                   logical_false, runstatus )
! endif

  CALL Highs_passLp( highs, numcol, numrow, numnz, aformat_colwise,            &
                     sense, offset, colcost, collower, colupper,               &
                     rowlower, rowupper, astart, aindex, avalue, runstatus )

  IF ( qp ) THEN
    CALL Highs_passHessian( highs, numcol, hessian_numnz, qformat_triangular,  &
                            qstart, qindex, qvalue, runstatus )

!   CALL Highs_qpCall( numcol, numrow, numnz, hessian_numnz,                   &
!                      aformat_colwise, qformat_triangular, sense,             &
!                      offset, colcost, collower, colupper, rowlower,          &
!                      rowupper, astart, aindex, avalue, qstart, qindex,       &
!                      qvalue, colvalue, coldual, rowvalue, rowdual,           &
!                      colbasisstatus, rowbasisstatus, modelstatus, runstatus )
  ELSE
!   CALL Highs_lpCall( numcol, numrow, numnz, aformat_colwise, sense,          &
!                      offset, colcost, collower, colupper, rowlower,          &
!                      rowupper, astart, aindex, avalue,                       &
!                      colvalue, coldual, rowvalue, rowdual,                   &
!                      colbasisstatus, rowbasisstatus, modelstatus, runstatus )
  END IF

!  solve the problem

  CALL Highs_run( highs, runstatus )
  CALL Highs_getModelStatus( highs, modelstatus )

!  record which HiGHS solver was actually used

  CALL Highs_getStringOptionValue( highs, "solver" // C_NULL_CHAR,             &
                                   solver_used, runstatus )
  DO i = 1, 10
    IF( solver_used( i : i ) == C_NULL_CHAR ) EXIT
  END DO
  IF ( i < 10 ) solver_used( i : 10 ) = ' '

!  recover the objective function value and iteration count

  CALL Highs_getDoubleInfoValue( highs,                                        &
                                 "objective_function_value" // C_NULL_CHAR,    &
                                 objective_function_value , runstatus)
  IF ( qp ) THEN
    IF ( TRIM( solver_used ) == 'hipo' ) THEN
      CALL Highs_getIntInfoValue( highs,                                       &
        "ipm_iteration_count" // C_NULL_CHAR, iteration_count, runstatus )
    ELSE
      CALL Highs_getIntInfoValue( highs,                                       &
        "qp_iteration_count" // C_NULL_CHAR, iteration_count, runstatus )
    END IF
  ELSE
    CALL Highs_getIntInfoValue( highs,                                         &
      "simplex_iteration_count" // C_NULL_CHAR, iteration_count, runstatus )
  END IF

!  compute the number and maximum values of the primal and dual infeasibilities

  CALL Highs_getIntInfoValue( highs,                                           &
                              "num_primal_infeasibilities" // C_NULL_CHAR,     &
                              num_primal_infeasibilities, runstatus )
  CALL Highs_getDoubleInfoValue( highs,                                        &
                                 "max_primal_infeasibility" // C_NULL_CHAR,    &
                                 max_primal_infeasibility, runstatus )
  CALL Highs_getIntInfoValue( highs,                                           &
                              "num_dual_infeasibilities" // C_NULL_CHAR,       &
                               num_dual_infeasibilities, runstatus )
  CALL Highs_getDoubleInfoValue( highs,                                        &
                                 "max_dual_infeasibility" // C_NULL_CHAR,      &
                                 max_dual_infeasibility, runstatus )

!  get the primal and dual solution ...

  CALL Highs_getSolution( highs, colvalue( : numcol ), coldual( : numrow ),    &
                          rowvalue, rowdual, runstatus )

!  ... and the basis

  CALL Highs_getBasis( highs, colbasisstatus, rowbasisstatus, runstatus )

  IF ( runstatus /= runstatus_ok ) THEN
    WRITE( 6, "( ' Highs_lpCall run status is not ', I0, ' but ', I0 )" )      &
      runstatus_ok, runstatus
  ELSE
    WRITE( 6, "( ' Run status = ', I0, ', model status = ', I0 )" )            &
       runstatus, modelstatus
    IF ( modelstatus == modelstatus_optimal ) THEN

! report the column primal and dual values, and basis status
    
!     IF ( qp ) THEN
!       DO col = 1, n
!         WRITE( 6, "( ' Col ', I6, ' = ', ES12.4, ' dual = ', ES12.4 )" )     &
!           col, colvalue( col ), coldual( col )
!       END DO
!     ELSE
!       DO col = 1, n
!         WRITE( 6, "( ' Col ', I6, ' = ', ES12.4, ' dual = ', ES12.4,         &
!        & ' status = ', I6 )" ) col, colvalue( col ), coldual( col ),         &
!           colbasisstatus( col )
!       END DO
!     END IF

! report the row primal and dual values, and basis status

!     IF ( qp ) THEN
!        DO row = 1, m
!         WRITE( 6, "( ' Row ', I6, ' = ', ES12.4, ' dual = ', ES12.4 )" )     &
!           row, rowvalue(row), rowdual( row )
!       END DO
!     ELSE
!       DO row = 1, m
!         WRITE( 6, "( ' Row ', I6, ' = ', ES12.4, ' dual = ', ES12.4,         &
!        & ' status = ',  I6 )" ) row, rowvalue(row), rowdual( row ),          &
!           rowbasisstatus( row )
!       END DO
!     END IF

! compute the objective function value

      obj = f + DOT_PRODUCT( colvalue, colcost )
      IF ( qp ) THEN
        DO col = 1, n
          l1 = qstart( col ) + 1
          IF ( col < n ) THEN
            l2 = qstart( col + 1 )
          ELSE
            l2 = hessian_numnz
          END IF
          obj = obj + 0.5_rp_ * colvalue( col ) * qvalue( l1 ) * colvalue( col )
          DO l = l1 + 1, l2
            row = qindex( l ) + 1
            obj = obj + colvalue( col ) * qvalue( l ) * colvalue( row )
          END DO
        END DO
      END IF
!     WRITE( 6, "( /, ' Optimal objective value =', ES22.14 )" ) obj
    END IF
  END IF

!  compute the primal and dual residuals if necessary

  IF ( output_summary > 10 ) THEN
    ALLOCATE( C( m ), G_l( n ), STAT = status )
    IF ( status /= 0 ) GO TO 990
    G_l( : n ) = colcost( : n )
    IF ( qp ) THEN
      DO col = 1, n
        l1 = qstart( col ) + 1
        IF ( col < n ) THEN
          l2 = qstart( col + 1 )
        ELSE
          l2 = hessian_numnz
        END IF
        G_l( col ) = G_l( col ) + qvalue( l1 ) * colvalue( col )
        DO l = l1 + 1, l2
          row = qindex( l ) + 1
          G_l( col ) = G_l( col ) + qvalue( l ) * colvalue( row )
          G_l( row ) = G_l( row ) + qvalue( l ) * colvalue( col )
        END DO
      END DO
    END IF
    G_l( : n ) = G_l( : n ) - coldual( : n )  ; C( : m ) = 0.0_rp_
    DO col = 1, n
      gcol = G_l( col) + coldual( col )
      l1 = astart( col ) + 1
      IF ( col < n ) THEN
        l2 = astart( col + 1 )
      ELSE
        l2 = numnz
      END IF
      DO l = l1, l2
        row = aindex( l ) + 1
        G_l( col ) = G_l( col ) - avalue( l ) * rowdual( row )
        C( row ) = C( row ) + avalue( l ) * colvalue( col )
      END DO
!     IF ( ABS( G_l( col ) ) > 0.00000001 ) write(6,"(I6, 4ES12.4)") col,      &
!       G_l( col ), gcol, G_l( col ) - gcol - coldual( col ), coldual( col )
    END DO
    C( : m ) = MIN( MAX( rowlower( : m ), C( : m ) ), rowupper( : m ) )        &
                 - C( : m )
    res_p = MAXVAL( ABS( C( : m ) ) ) ; res_d = MAXVAL( ABS( G_l( : n ) ) )
!   WRITE( 6, "( ' Primal and dual infeasibility =', 2ES22.14 )" ) res_p, res_d
    DEALLOCATE( G_l, C, STAT = status )
    IF ( status /= 0 ) GO TO 990

!  WRITE( 6, "( 1X, I0, ' iterations required' )" ) iteration_count
!  WRITE( 6, "( ' number of & maximum primal infeasibilities: ', I0, ES12.4 )")&
!    num_primal_infeasibilities,  max_primal_infeasibility
!  WRITE( 6, "( ' number of & maximum dual infeasibilities: ', I0, ES12.4 )" ) &
!    num_dual_infeasibilities,  max_dual_infeasibility

!  output a summary of the results to a file if required

    BACKSPACE( output_summary )
    iter = iteration_count
    SELECT CASE ( runstatus )
    CASE ( runstatus_ok )
      WRITE( output_summary,                                                   &
        "( A10, 1X, I8, 1X, I8, ES16.8, 2ES9.1, bn, I9, I6 )" )         &
       p_name, n, m, objective_function_value, res_p, res_d, iter,             &
       runstatus
    CASE DEFAULT
      WRITE( output_summary,                                                   &
        "( A10,  1X, I8, 1X, I8, ES16.8, 2ES9.1, bn, I9, I6 )" )        &
        p_name, n, m, objective_function_value, res_p, res_d, -iter,           &
        runstatus
    END SELECT
    CLOSE( output_summary )
  END IF

!  write details

! WRITE( out, "(' Final objective value = ', ES11.3 )" ) obj
! WRITE( out, "(' Optimal X = ', 7F9.2 )" ) X( : n )

  WRITE( out, "( /, 24('*'), ' GALAHAD statistics ', 24('*') //                &
 &              ,' Package used            :  HiGHS (', A, ')',  /             &
 &              ,' Problem                 :  ', A10,    /                     &
 &              ,' # variables             =      ', I10 /                     &
 &              ,' # constraints           =      ', I10 /                     &
 &              ,' Exit code               =      ', I10 /                     &
 &              ,' Final f                 = ', ES15.7 /                       &
 &               67('*') / )" ) TRIM( solver_used ), p_name, n, m, runstatus,  &
     objective_function_value

  l = 4 ; IF ( fulsol ) l = n

!  Print details of the primal and dual variables

  WRITE( 6, "( ' Solution : ', /, '                    ',                      &
 &             '        <------ Bounds ------> ', /                            &
 &             '      #  state    value   ',                                   &
 &             '    Lower       Upper       Dual' )" )
  DO j = 1, 2
    IF ( j == 1 ) THEN
      ir = 1 ; ic = MIN( l, n )
    ELSE
      IF ( ic < n - l ) WRITE( 6, "( '      . .          .....',               &
     & ' ..........  ..........  ..........  .......... ' )" )
      ir = MAX( ic + 1, n - ic + 1 ) ; ic = n
    END IF
    DO i = ir, ic
      state = ' FREE'
      IF ( ABS( colvalue( i ) - collower( i ) ) < ac_tol ) state = 'LOWER'
      IF ( ABS( colvalue( i ) - colupper( i ) ) < ac_tol ) state = 'UPPER'
      IF ( ABS( collower( i ) - colupper( i ) ) < eq_tol ) state = 'FIXED'
      WRITE( 6, "( I7, 1X, A6, 4ES12.4 )" ) i, state,                          &
        colvalue( i ), collower( i ), colupper( i ), coldual( i )
    END DO
  END DO

!  Compute the number of fixed and degenerate variables.

  nfixed = 0 ; ndegen = 0
  DO i = 1, n
    IF ( ABS( colupper( i ) - collower( i ) ) < eq_tol ) THEN
      nfixed = nfixed + 1
      IF ( ABS( coldual( i ) ) < ac_tol ) ndegen = ndegen + 1
    ELSE IF ( MIN( ABS( colvalue( i ) - collower( i ) ),                       &
              ABS( colvalue( i ) - colupper( i ) ) ) <=                        &
              MAX( ac_tol, ABS( coldual( i ) ) ) ) THEN
      nfixed = nfixed + 1
      IF ( ABS( coldual( i ) ) < ac_tol ) ndegen = ndegen + 1
    END IF
  END DO

!  Print details of the constraints.

  IF ( m > 0 ) THEN
    WRITE( out, "( /, ' Constraints : ', /, '        ',                        &
   &       '        <------ Bounds ------> ', /                                &
   &       '      #  state    value   ',                                       &
   &       '    Lower       Upper     Multiplier ' )" )
    l = 2  ; IF ( fulsol ) l = m
    DO j = 1, 2
      IF ( j == 1 ) THEN
        ir = 1 ; ic = MIN( l, m )
      ELSE
        IF ( ic < m - l ) WRITE( out, "( '      . .          .....',           &
       & ' ..........  ..........  ..........  .......... ' )" )
        ir = MAX( ic + 1, m - ic + 1 ) ; ic = m
      END IF
      DO i = ir, ic
        state = ' FREE'
        IF ( ABS( rowvalue( I ) - rowlower( i ) ) < ac_tol ) state = 'LOWER'
        IF ( ABS( rowvalue( I ) - rowupper( i ) ) < ac_tol ) state = 'UPPER'
        IF ( ABS( rowlower( i ) - rowupper( i ) ) < eq_tol ) state = 'EQUAL'
        WRITE( out, "( I7, 1X, A6, 4ES12.4 )" ) i,                             &
          state, rowvalue( i ), rowlower( i ), rowupper( i ), rowdual( i )
      END DO
    END DO

!  Compute the number of equality, fixed inequality and degenerate constraints

    mequal = 0 ; mfixed = 0 ; mdegen = 0 ; mredun = 0
    DO i = 1, m
     IF ( ABS( rowlower( i ) - rowupper( i ) ) < eq_tol ) THEN
        mequal = mequal + 1
        IF ( ABS( rowdual( i ) ) < ac_tol ) mredun = mredun + 1
      ELSE IF ( MIN( ABS( rowvalue( i ) - rowlower( i ) ),                     &
                ABS( rowvalue( i ) - rowupper( i ) ) ) <=                      &
           MAX( ac_tol, ABS( rowdual( i ) ) ) ) THEN
        mfixed = mfixed + 1
        IF ( ABS( rowdual( i ) ) < ac_tol ) mdegen = mdegen + 1
      END IF
    END DO
  END IF
  max_d = MAX( MAXVAL( ABS( rowdual( : m ) ) ),                                &
               MAXVAL( ABS( coldual( : n ) ) ) )
  WRITE( out, "( /, ' Of the ', I0, ' variables, ', I0,                        &
 &  ' are on bounds & ', I0, ' are dual degenerate' )" ) n, nfixed, ndegen
  IF ( m > 0 ) THEN
    WRITE( out, "( ' Of the ', I0, ' constraints, ', I0,                       &
   &  ' are equations, & ', I0, ' are redundant' )" ) m, mequal, mredun
     IF ( m /= mequal ) WRITE( out, "( ' Of the ', I0, ' inequalities, ',      &
   &  I0, ' are on bounds, & ', I0, ' are degenerate' )" ) m - mequal,         &
      mfixed, mdegen
  END IF
  WRITE( out, "( /, ' Final objective function value  ', ES22.14, /,           &
 &                  ' Maximum dual variable           ', ES22.14, /,           &
 &                  ' Maximum constraint violation    ', ES22.14, /,           &
 &                  ' Maximum dual infeasibility      ', ES22.14, /,           &
 &                  ' Number of HiGHS iterations = ', I0 )" )                  &
    objective_function_value, max_d, res_p, res_d, iter

!  destroy the HiGHS environment

  CALL Highs_destroy( highs )

!  deallocate workspace

  DEALLOCATE( colcost, collower, colupper, rowlower, rowupper, astart,         &
              aindex, avalue, sol, colvalue, coldual, rowvalue, rowdual,       &
              colbasisstatus, rowbasisstatus, STAT = status )
  IF ( qp ) DEALLOCATE( qstart, qindex, qvalue, STAT = status )
  STOP

  990 CONTINUE
  WRITE( out, "( ' Allocation error, status = ', i0 )" ) status
  STOP

  END PROGRAM HiGHS_main
