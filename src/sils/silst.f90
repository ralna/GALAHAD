! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.
      program SILS_test_deck
      use GALAHAD_SILS_double
      implicit none
      type(SMT_type) matrix
      type(SILS_control) cntl
      type(SILS_ainfo) ainfo
      type(SILS_finfo) finfo
      type(SILS_sinfo) sinfo
      type(SILS_factors) factors

      double precision, allocatable :: b(:,:),x(:,:),d(:,:),bb(:),xx(:),pert(:)
      integer, allocatable :: perm(:),pivots(:),ip(:),iw(:)
      integer i,info,jj,j1,j2,kase,n,ne,ne1,nrhs

      EXTERNAL YM01AD

      do kase = 0,5
! Read matrix order and number of entries
        READ(5,*) n,ne
        matrix%n = n
        matrix%ne = ne
! Allocate matrix arrays
        allocate(matrix%val(ne), matrix%row(ne), matrix%col(ne))
        nrhs = 1
        IF (kase .le. 3) THEN
          IF (kase .eq. 1) THEN
            nrhs = 2
            allocate (perm(n))
          ENDIF
          IF (kase.eq.2) THEN
            n = n + 1
            matrix%n = n
          ENDIF

! Allocate arrays of appropriate sizes
          IF (kase .eq. 1)  THEN
            allocate(b(n,nrhs), x(n,nrhs), pivots(n))
          ELSE
            allocate(bb(n), xx(n), d(2,n), pivots(n))
          ENDIF

          IF (kase .eq. 0) allocate(pert(n))

! Read matrix and right-hand side
          READ(5,*) (matrix%row(I),matrix%col(I),matrix%val(I),I=1,ne)
          IF (kase.eq.1) THEN
            READ(5,*) b(:,1)
            b(:,2) = b(:,1)
          ENDIF
          IF (kase.eq.2) THEN
            READ(5,*) bb(1:n-1)
            bb(n) = 1.23456789
            xx = bb
          ENDIF
        ELSE
          allocate(bb(n), xx(n), d(2,n), pivots(n))
          allocate(ip(n+1),iw(2*n))
          ne1 = ne
          CALL YM01AD(n,n,ne1,ne,-1,2,.true.,matrix%row,matrix%val,ip,iw)
          do i=1,n
            xx(i) = 1.0
            j1=ip(i)
            j2=ip(i+1)-1
            do jj=j1,j2
              matrix%col(jj)=i
! Set diagonal to zero
              IF (i==matrix%row(jj)) matrix%val(jj) = 0.0
            enddo
          enddo
          deallocate(ip,iw)
        ENDIF

! Initialize the structures
        IF (kase.lt.5) CALL SILS_initialize(factors,cntl)
        IF (kase.eq.0) cntl%pivoting = 4

! Analyse
        CALL SILS_analyse(matrix,factors,cntl,ainfo)

        IF(ainfo%flag<0) THEN
           WRITE(6,'(A,I2)') &
              'Failure of SILS_analyse with ainfo%flag=', ainfo%flag
           STOP
        END IF

! Recall analyse using input pivot sequence
        IF (kase.eq.1) THEN
          CALL SILS_enquire(factors,perm)
          CALL SILS_analyse(matrix,factors,cntl,ainfo,perm)

          IF(ainfo%flag<0) THEN
             WRITE(6,'(A,I2)') &
              'Failure of SILS_analyse with ainfo%flag=', ainfo%flag
             STOP
          END IF
        ENDIF

        IF (kase.eq.3) matrix%n = matrix%n + 1

! Factorize matrix
        CALL SILS_factorize(matrix,factors,cntl,finfo)
        IF(finfo%flag<0) THEN
           WRITE(6,'(A,I2)') &
              'Failure of SILS_factorize with finfo%flag=', finfo%flag
           go to 50
        END IF

        IF (kase.eq.0) THEN
! Check perturbation enquiry
          CALL SILS_enquire(factors,perturbation=pert)
          WRITE(6,'(A/(5G16.8))') 'Schnabel-Eskow permutation',pert
          go to 50
        ENDIF

        IF (kase.eq.1) THEN
          x = b
! Solve without refinement (2 rhs)
          CALL SILS_solve(matrix,factors,x,cntl,sinfo)
          IF(sinfo%flag.eq.0)WRITE(6,'(/A,/,(3G24.16))')  &
             'Solution without refinement is',x

          x = b
          CALL SILS_part_solve(factors,cntl,'L',x,info)
          WRITE(6,'(/a,/,(3G24.16))')'Solution after L is',x
          CALL SILS_part_solve(factors,cntl,'D',x,info)
          WRITE(6,'(a,/,(3G24.16))')'Solution after D is',x
          CALL SILS_part_solve(factors,cntl,'U',x,info)
          WRITE(6,'(a,/,(3G24.16))')'Solution after U is',x

! Perform one refinement
          CALL SILS_solve(matrix,factors,x,cntl,sinfo,b)
          IF(sinfo%flag.eq.0)WRITE(6,'(/A,/,(3G24.16))') &
              'Solution after one refinement is',x

      ELSE

! One rhs

          CALL SILS_solve(matrix,factors,xx,cntl,sinfo)
          IF(sinfo%flag.eq.0 .and. kase.le.3) WRITE(6,'(/A,/,(3G24.16))')  &
             'Solution without refinement is',xx

          IF (kase.gt.2) GO TO 50

          xx = bb
          CALL SILS_part_solve(factors,cntl,'L',xx,info)
          WRITE(6,'(/a,/,(3G24.16))')'Solution after L is',xx
          CALL SILS_part_solve(factors,cntl,'D',xx,info)
          WRITE(6,'(a,/,(3G24.16))')'Solution after D is',xx
          CALL SILS_part_solve(factors,cntl,'U',xx,info)
          WRITE(6,'(a,/,(3G24.16))')'Solution after U is',xx

! Perform one refinement
          CALL SILS_solve(matrix,factors,xx,cntl,sinfo,bb)
          WRITE(6,'(/a,/,(3G24.16))')'Solution with refinement is',xx

          CALL SILS_enquire(factors,pivots=pivots,d=d)
          WRITE(6,'(/a)')'Pivots and D:'
          i = 1
          do
           IF (i>n) exit
           IF (pivots(i)>0) THEN
              WRITE(6,'(i5,f10.5)') pivots(i), d(1,i)
              i = i + 1
            ELSE
              WRITE(6,'(i5,2f10.5)') pivots(i), d(1:2,i)
              WRITE(6,'(i5,2f10.5)') pivots(i+1), d(2,i), d(1,i+1)
              i = i + 2
            END IF
          end do

          d(1,:) = 1
          CALL SILS_alter_d(factors,d,i)
          IF (i==0) WRITE(6,'(/a)')'Successful call of SILS_alter_d'
          CALL SILS_enquire(factors,pivots=pivots,d=d)
          WRITE(6,'(a)')'Pivots and altered D:'
          i = 1
          do
           IF (i>n) exit
           IF (pivots(i)>0) THEN
              WRITE(6,'(i5,f10.5)') pivots(i), d(1,i)
              i = i + 1
            ELSE
              WRITE(6,'(i5,2f10.5)') pivots(i), d(1:2,i)
              WRITE(6,'(i5,2f10.5)') pivots(i+1), d(2,i), d(1,i+1)
              i = i + 2
            END IF
          end do

        ENDIF

! Deallocate arrays
  50    IF (kase.ne.4) CALL SILS_finalize(factors,cntl,info)
        deallocate(matrix%val, matrix%row, matrix%col)
        IF (kase .eq. 1) THEN
! Deallocate arrays for second pass
          deallocate(pivots,b,x)
        ELSE
! Deallocate arrays for next pass
          deallocate(pivots,bb,xx,d)
        ENDIF

      enddo

      END program SILS_test_deck
