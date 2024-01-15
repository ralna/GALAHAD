   PROGRAM GALAHAD_DGO_EXAMPLE2 !  GALAHAD 4.0 - 2022-03-12 AT 11:10 GMT
   USE GALAHAD_DGO_double                       ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )    ! set precision
   TYPE ( NLPT_problem_type ):: nlp
   TYPE ( DGO_control_type ) :: control
   TYPE ( DGO_inform_type ) :: inform
   TYPE ( DGO_data_type ) :: data
   TYPE ( GALAHAD_userdata_type ) :: userdata
   INTEGER :: s
   INTEGER, PARAMETER :: n = 2, h_ne = 3
   REAL ( KIND = wp ) :: x1, x2
   REAL ( KIND = wp ), PARAMETER :: p = - 2.1_wp
! start problem data
   nlp%pname = 'CAMEL6'                         ! name
   nlp%n = n ; nlp%H%ne = h_ne                  ! dimensions
   ALLOCATE( nlp%X( n ), nlp%G( n ), nlp%X_l( n ), nlp%X_u( n ) )
   nlp%X_l( : n )  = (/ - 3.0_wp, - 2.0_wp /)
   nlp%X_u( : n )  = (/ 3.0_wp, 2.0_wp /)
!  sparse co-ordinate storage format
   CALL SMT_put( nlp%H%type, 'COORDINATE', s )  ! Specify co-ordinate storage
   ALLOCATE( nlp%H%val( h_ne ), nlp%H%row( h_ne ), nlp%H%col( h_ne ) )
   nlp%H%row = (/ 1, 2, 2 /) ! Hessian H
   nlp%H%col = (/ 1, 1, 2 /) ! NB lower triangle
! problem data complete
   CALL DGO_initialize( data, control, inform ) ! Initialize control parameters
! Solve the problem
   inform%status = 1 ! set for initial entry
   DO  ! Solve problem using reverse communication
     CALL DGO_solve( nlp, control, inform, data, userdata )
     IF ( inform%status == 0 ) THEN ! Successful return
       WRITE( 6, "( ' DGO: ', I0, ' evaluations -', /,                         &
      &     ' Best objective value found =', ES12.4, /,                        &
      &     ' Corresponding solution = ', ( 5ES12.4 ) )" )                     &
       inform%iter, inform%obj, nlp%X
       EXIT
     ELSE IF ( inform%status < 0 ) THEN ! Error returns
       WRITE( 6, "( ' DGO_solve exit status = ', I6 ) " ) inform%status
       EXIT
     END IF
     x1 = nlp%X( 1 ) ; x2 = nlp%X( 2 )
     IF ( inform%status == 2 .OR. inform%status == 23 .OR.                     &
          inform%status == 25  .OR. inform%status == 235 ) THEN ! evaluate f
       nlp%f = ( 4.0_wp + p * x1 ** 2 + x1 ** 4 / 3.0_wp ) * x1 ** 2           &
                 + x1 * x2 + ( - 4.0_wp + 4.0_wp * x2 ** 2 ) * x2 ** 2
     END IF
     IF ( inform%status == 3 .OR. inform%status == 23 .OR.                     &
          inform%status == 35 .OR. inform%status == 235 ) THEN ! evaluate g
       nlp%G( 1 ) = ( 8.0_wp + 4.0_wp * p * x1 ** 2 + 2.0_wp * x1 ** 4 ) * x1  &
                    + x2
       nlp%G( 2 ) = x1 + ( - 8.0_wp + 16.0_wp * x2 ** 2 ) * x2
     END IF
     IF ( inform%status == 4 ) THEN ! evaluate H
       nlp%H%val( 1 ) = 8.0_wp + 12.0_wp * p * x1 ** 2 + 10.0_wp * x1 ** 4
       nlp%H%val( 2 ) = 1.0_wp
       nlp%H%val( 3 ) = - 8.0_wp + 48.0_wp * x2 * x2
     END IF
     IF ( inform%status == 5 .OR. inform%status == 25 .OR.                     &
          inform%status == 35 .OR. inform%status == 235 ) THEN ! evaluate u = Hv
       data%U( 1 ) = data%U( 1 ) + ( 8.0_wp + 12.0_wp * p * x1 ** 2            &
                       + 10.0_wp * x1 ** 4 ) * data%V( 1 ) + data%V( 2 )
       data%U( 2 ) = data%U( 2 ) + data%V( 1 )                                 &
                       + ( - 8.0_wp + 48.0_wp * x2 * x2 ) * data%V( 2 )
     END IF
     data%eval_status = 0
   END DO
   CALL DGO_terminate( data, control, inform )  ! delete internal workspace
   DEALLOCATE( nlp%X, nlp%G, nlp%H%val, nlp%H%row, nlp%H%col )
   END PROGRAM GALAHAD_DGO_EXAMPLE2
