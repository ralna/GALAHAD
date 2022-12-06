  MODULE GALAHAD_NLLSRT_double
  CONTAINS
  SUBROUTINE NLLSRT(N,M, X, MAXIT,TOL, OUTPUT,IERR, iter, iter_int, iter_int_tot, iprint)
  ! Version 9 July 2008
    USE GALAHAD_LSRT_DOUBLE                                                

    INTEGER, PARAMETER :: wp=KIND( 1.0D+0 )  ! set precision
       
    ! dummy arguments
    INTEGER, INTENT( IN ) :: N,M , iprint  
    INTEGER, INTENT(OUT) :: IERR, iter, iter_int, iter_int_tot
    INTEGER, INTENT(IN) :: MAXIT       !  maximum number of iterations
    REAL (KIND=wp), INTENT( INOUT ) :: X(N)
    REAL (KIND=wp), INTENT(IN) :: TOL(4)
    REAL (KIND=wp), INTENT(OUT) :: OUTPUT(2)

    !local variables
    REAL (KIND=wp) ::  C(M+1),CJAC(M+1,N+1),CJACT(N+1,M+1), CJACTXPS(N+1,M+1)
    REAL (KIND=wp) ::  GRAD(N),S(N), U(M), V(N), RES(M), XPS(N), CXPS(M+1),AS(N)
    REAL (KIND=wp) ::  sigma,p,  NC, NG, NRES, NCXPS, NS, rho, nrho,drho,NG0,NC0
    REAL (KIND=wp) ::  eps_ca, eps_cr,eps_ga,eps_gr, epsbar, q, e,n2ATb
    INTEGER :: itc ,i ,l, INFO        !iteration counter

    TYPE (LSRT_data_type) :: data           ! used by lsrt_solve
    TYPE (LSRT_control_type) :: control
    TYPE (LSRT_inform_type) :: inform

    !internal parameters
    REAL(KIND=wp),PARAMETER::eta1=0.01_wp, eta2=0.95_wp, gamma1=1.0_wp, &
                             gamma2=2.0_wp, eps_m=10.0_wp**(-15)  
    REAL(KIND=wp),PARAMETER:: one=1.0_wp, zero=0.0_wp, ten=10.0_wp, half=0.5_wp

   !parameter setting
   eps_ca=TOL(1)
   eps_cr=TOL(2)
   eps_ga=TOL(3)
   eps_gr=TOL(4)	
   
       
    !IERR   = 0   zero residual solution found
    !       = 1   non zero solution found
    !       = -3  maxit   
    !       = -1  small step
    
    !initialization
    OUTPUT = 0.0_wp
    sigma = one    !initial regularization parameter
    p = 3.0_wp
    IERR = -3
    iter_int = 0
    iter_int_tot = 0	
    XPS = 0.0_wp
    CXPS = 0.0_wp
    CJACTXPS = 0.0_wp

    ! compute C(x_0) and JAC(x_0)^T  
    CALL CCFG( N, M, X, M+1, C, .TRUE., N+1, M+1,  CJACT, .true. )  
    NC=sqrt(dot_product(C(:M),C(:M)))    ! compute ||C(x_0)||
    NC0=NC  
    GRAD=matmul(CJACT(:N,:M),C(:M))      ! compute grad(x_0)=Jac(x_0)^TC(x_0) 
    NG=SQRT(DOT_PRODUCT(GRAD,GRAD))      ! compute ||grad(x_0)||
    NG0=NG
    CJAC(:M,:N)=transpose(CJACT(:N,:M))  ! compute JAC(x_0) 
       
    !   MAIN LOOP    
         
    DO itc=1,MAXIT
       
          OUTPUT(1) = NC**2
          OUTPUT(2) = NG
          iter = itc - 1 
       
        if (iprint.gt.0) then
	     print '(''It='', I4, '' ||C_k||= '',d10.5,'' ||grad_k||= '',d10.5 ) ', iter,  NC, NG
             write(200, '(''It='', I4, '' ||C_k||= '',d10.5,'' ||grad_k||= '',d10.5 )')iter,  NC, NG
        end if

       IF ((NC.le.max(eps_cr* NC0,eps_ca)).or.(NG.le.max(eps_gr* NG0,eps_ga)))    then 	
	 	
	 if  (NC.le.max(eps_cr* NC0,eps_ca)) THEN
	                IERR=0                  ! zero residual solution found
	             else
	                IERR=1               ! nonzero residual solution found
	             end if
	       RETURN
	     END IF
     
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     !COMPUTE A SOLUTION OF THE REGULARIZED PROBLEM   
     !        
     !             min 0.5 ||Js+C||^2+sigma/p ||s||^p 
     !
     !using the GALAHAD module  GALAHAD_LSRT_DOUBLE   
     
     
     CALL LSRT_initialize(data,control,inform)      ! initialize control parameters
   
	if (NC.le.ten**(-1)) then
             control%fraction_opt = one               
	else
	     control%fraction_opt = 0.9_wp 	  ! only require 90% of the best
	end if

!     control%fraction_opt=0.9_wp               
     control%print_level = 1
     n2ATb=dot_product(matmul(CJACT(:N,:M),-C(:M)),matmul(CJACT(:N,:M),-C(:M)))  ! norma al quadrato grad modello nel punto inz 0
     
    control%stop_relative=min(0.1_wp,sqrt(sqrt(n2ATb)))
      
     U = -C(:M)                                ! the term b in min||Ax-b||
     inform%status = 1
     DO                                      !iteration to find the minimizer
        CALL LSRT_solve(M,N,p,sigma,S,U,V,data,control,inform)
        
        SELECT CASE(inform%status)
        CASE(2)                                ! form  U=U+J*V
           U=U+matmul(CJAC(:M,:N),V)
        CASE(3)                                ! form V=V+J^TU
           V=V+matmul(CJACT(:N,:M),U)  
        CASE(4)                                ! Restart
           U=-C(:M)                            ! reinitialize U to C(x_k) 
        CASE(-2:0)                             ! succesful return
           
       !  WRITE(6,'(1X,I0,'' 1st pass and '',I0, '' 2nd pass iterationsS'')') &
           !       inform%iter, inform%iter_pass2
           
       ! compute the step norm for checking
       !  NS=SQRT(DOT_PRODUCT(S,S))            ! compute ||S||
       !  WRITE(6,'('' ||s|| recurred and calculated = '', 2ES16.8)') &
       !             inform%x_norm, NS
                  
       ! compute the residual for checking   RES=-C(x_k)-JAC(x_k)s
       !      RES=-C(:M)-matmul(CJAC(:M,:N),S)
       !      NRES=SQRT(DOT_PRODUCT(RES,RES))      ! compute ||RES||
       ! WRITE(102,'('' ||Js+C|| recurred and calculated = '', 2ES16.8)') &
       !              inform%r_norm, NRES
      
       ! WRITE(102,'('' objective recurred and calculated = '', 2ES16.8)') &
       !              inform%obj,0.5_wp* NRES+(sigma/p)*NS**p
             CALL LSRT_terminate(data, control, inform) 
             EXIT
          CASE DEFAULT
             !  WRITE(6,'('' LSTR_solve exit status = '', I6)' ) inform%status
             CALL LSRT_terminate(data, control, inform)
             EXIT
          END SELECT
          
       END DO
        NRES = inform%r_norm
        NS = inform%x_norm
        INFO = inform%status
	if ((INFO).lt.0) then 
 		print*,'errore nel sottoprogramma',INFO
	end if
        ! print*,NS,NRES
        iter_int = iter_int + inform%iter
        iter_int_tot = iter_int_tot + inform%iter + inform%iter_pass2
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       ! test to accept the trial step  S
       CALL ABSV(N,S,AS)  ! compute the absolute value 
                          ! of the vector S (componentwise)
                 
       IF (maxval(AS).LT.ten*eps_m) THEN
          if (iprint.gt.0) then
             print*,'     ||s||=', NS
           !  write(200,'(''||s||= '')') NS
             print*,'itc=', itc
           !  write(200,'(''itc= '')') itc
          end if
         
          IERR = -1
          
          return
       END IF

       XPS = X+S           ! compute x_k+s_k       
       ! compute C(x_k+s_k) and JacT(x_k+s_k)
       CALL CCFG( N, M, XPS, M+1, CXPS, .true., N+1, M+1, CJACTXPS, .true. ) 
       NCXPS=SQRT(DOT_PRODUCT(CXPS(:M),CXPS(:M)))       ! ||C(x_k+s_k)||

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       !  compute rho
   
       nrho = half * (NC**2-NCXPS**2)                    ! numerator
       drho = half * NC**2 -(half*NRES**2 + (sigma/p)*NS**p)     ! denominator     
       rho = nrho/drho
        
       ! if rho<0, compute an estimate of the rounding error e
       if (rho.lt.0) then
          q = zero
          e = zero
          do i=1,M
             do l=1,N
                q=q+abs(CJAC(i,l))*abs(S(l))
             end do
             e=e+abs(C(i))+q
          end do
          epsbar=eps_m*e
       
	! set rho=1 if nrho or drho are < eps_m*e  
          if (((abs(nrho).lt.epsbar).and.(abs(drho).lt.epsbar)).or.(NCXPS.eq.NRES)) then
   	 	rho = one 
   	 	print*,'rho===1'
   	 	write(200,'(''rho===1'')')
 		endif
       end if

     
       if (iprint.gt.0) then
       !  print'(3X, ''iter_in='', I3,'' rho='', d15.8, '' ||s||='',d10.5,  '' sigma='',d10.5 )' inform%iter,  rho, NS, sigma
  	  write(200,'(3X,'' pass1='', I4, '' pass2='', I4,'' rho='', d15.8, '' ||s||='',d10.5,  '' sigma='',d10.5 )')  &
		 inform%iter, inform%iter_pass2, rho, NS, sigma
	  end if
    !    read(*,*)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ! sigma update
        
      IF (rho.GE.eta2) THEN

         sigma = max (min (gamma1 * sigma, NG), eps_m)  ! ACO rule
        
         if (iprint.gt.0) then
             write(200,'(''  verys'')')
         end if
 
     ELSE IF (rho.GE.eta1) THEN 
                   
         if (iprint.gt.0) then
            write(200,'(''  succ'')')
         end if

      ELSE

         if (iprint.gt.0) then
            write(200,'(''  uns'')')
         end if

         sigma = gamma2 * sigma

      END IF
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ! iterate update
  
      IF (rho.GE.eta1) THEN
         X = XPS           ! x_k+1=x_k+s_k
         C = CXPS          ! C(x_k+1)
     
         NC = NCXPS        ! ||C(x_k+1)||
              
         CJACT = CJACTXPS                        ! Jac^T(x_k+1)  
         CJAC(:M,:N) = transpose(CJACT(:N,:M))   ! Jac (x_k+1)
                 
         GRAD = matmul(CJACT(:N,:M),C(:M)) ! grad(x_k+1)=Jac(x_k+1)^TC(x_k+1) nx1
     
         NG = SQRT(DOT_PRODUCT(GRAD,GRAD)) ! ||grad(x_k+1)||
	
      END IF
            
   END DO
	
	iter = MAXIT
    
   RETURN
 END SUBROUTINE NLLSRT

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 SUBROUTINE    ABSV(N,V,AV)

   INTEGER, PARAMETER :: wp=KIND( 1.0D+0 )  ! set precision
   !Dummy arguments
   INTEGER, INTENT( IN ) :: N
   REAL (KIND=wp), INTENT( IN ) :: V(N)
   REAL (KIND=wp), INTENT( OUT ) :: AV(N)
   
   !local variables
   INTEGER :: i
   
   DO i=1,N
      AV(i)=ABS(V(i))
   END DO
   
   RETURN
 END SUBROUTINE ABSV
 
  END MODULE GALAHAD_NLLSRT_double
