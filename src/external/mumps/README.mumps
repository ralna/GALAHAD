MUMPS 5.5.1 control and info values

----
CNTL:
----

CNTL(1)    Threshold for numerical pivoting
CNTL(2)    Iterative refinement stopping tolerance
CNTL(3)    Null pivot detection threshold
CNTL(4)    Threshold for static pivoting
CNTL(5)    Fixation for null pivots
CNTL(6)    unused
CNTL(7)    Dropping threshold for BLR compression
CNTL(8-15) unused

-----
ICNTL:
-----

ICNTL(1)   Output stream for error messages
ICNTL(2)   Output stream for diagnostic messages
ICNTL(3)   Output stream for global information
ICNTL(4)   Level of printing
ICNTL(5)   Matrix format (0=assembled,1=element)
ICNTL(6)   Maximum transversal
ICNTL(7)   Ordering (1=AMD,2=AMF,3=Scotch,4=Pord,5=Metis,6=AMDD,7=auto)
ICNTL(8)   Scaling strategy (-1=user,0=none,1=diag,4=inf norm,7=equib,77=auto)
ICNTL(9)   Solve A x=b (1) or A^Tx = b (else)
ICNTL(10)  Max steps iterative refinement
ICNTL(11)  Error analysis (1=all,2=some,else=off)
ICNTL(12)  LDLT ordering strat
ICNTL(13)  Parallel root (0=on, 1=off)
ICNTL(14)  Percent of memory increase
ICNTL(15)  Analysis by block
ICNTL(18)  Distributed matrix
ICNTL(19)  Schur option ( 0=off,else=on )
ICNTL(20)  Den.(0)/sparse(1,2,3)/dist.(10,11) RHS
ICNTL(21)  Gathered (0) or distributed(1) solution
ICNTL(22)  Out-of-core option (0=off, >0=on)
ICNTL(23)  Max working memory per processor (0=auto)
ICNTL(24)  Null pivot detection (0=off)
ICNTL(25)  Allow solution of defficient system
ICNTL(31)  Discard factors (0=off, else=on)
ICNTL(32)  Forward elimination during factorization (0=off)
ICNTL(33)  Compute determinant (0=off)
ICNTL(35)  Block Low Rank (BLR, 0=off >0=on)
ICNTL(36)  BLR variant
ICNTL(37)  unused
ICNTL(38)  compression raate for LU
ICNTL(39:57) unused
ICNTL(58)  Symbolic factorization option
ICNTL(59:60) unused

-----
INFOG:
-----

INFOG(1) is 0 if the last call to MUMPS was successful, negative if an
 error occurred or positive if a warning is returned. In particular,
 after successfully saving or restoring an instance (call to MUMPS with
 JOB=7 or JOB=8), INFOG(1) will be 0 even if INFOG(1) was different from
 0 at the moment of saving the MUMPS instance to disk.

 Possible values are

  -1 An error occurred on processor INFO(2).

  -2 NNZ (or NZ) is out of range. INFO(2)=NNZ (or NZ).

  -3 MUMPS was called with an invalid value for JOB. This may happen if
   the analysis (JOB=1) was not performed (or failed) before the
   factorization (JOB=2), or the factorization was not performed (or
   failed) before the solve (JOB=3), or the initialization phase
   (JOB=-1) was performed a second time on an instance not freed
   (JOB=-2).  This error also occurs if JOB does not contain the same
   value on all processes on entry to MUMPS. INFO(2) is then set to the
   local value of JOB.

  -4 Error in user-provided permutation array PERM IN at position
   INFO(2). This error may only occur on the host.

  -5 Problem of real workspace allocation of size INFO(2) during
   analysis. The unit for INFO(2) is the number of real values (single
   precision for SMUMPS/CMUMPS, double precision for DMUMPS/ZMUMPS), in the
   Fortran “ALLOCATE” statement that did not succeed. If INFO(2) is
   negative, then its absolute value should be multiplied by 1 million.

  -6 Matrix is singular in structure. INFO(2) holds the structural rank.

  -7 Problem of integer workspace allocation of size INFO(2) during
   analysis. The unit for INFO(2) is the number of integer values that
   MUMPS tried to allocate in the Fortran ALLOCATE statement that did not
   succeed. If INFO(2) is negative, then its absolute value should be
   multiplied by 1 million.

  -8 Main internal integer workarray IS too small for factorization. This
   may happen, for example, if numerical pivoting leads to significantly
   more fill-in than was predicted by the analysis. The user should
   increase the value of ICNTL(14) before calling the factorization again
   (JOB=2).

  -9 The main internal real/complex workarray S is too small. If INFO(2)
   is positive, then the number of entries that are missing in S at the
   moment when the error is raised is available in INFO(2).  If INFO(2) is
   negative, then its absolute value should be multiplied by 1 million. If
   an error –9 occurs, the user should increase the value of ICNTL(14)
   before calling the factorization (JOB=2) again, except if LWK USER is
   provided LWK USER should be increased.

  -10 Numerically singular matrix. INFO(2) holds the number of eliminated
   pivots.

  -11 Internal real/complex workarray S or LWK USER too small for
   solution. If INFO(2) is positive, then the number of entries that are
   missing in S/LWK USER at the moment when the error is raised is
   available in INFO(2). If the numerical phases are out-of-core and LWK
   USER is provided for the solution phase and is smaller than the value
   provided for the factorization, it should be increased by at least
   INFO(2). In other cases, please contact us.

  -12 Internal real/complex workarray S too small for iterative
   refinement. Please contact us.

  -13 Problem of workspace allocation of size INFO(2) during the
   factorization or solve steps. The size that the package tried to
   allocate with a Fortran ALLOCATE statement is available in INFO(2).  If
   INFO(2) is negative, then the size that the package requested is
   obtained by multiplying the absolute value of INFO(2) by 1 million. In
   general, the unit for INFO(2) is the number of scalar entries of the
   type of the input matrix (real, complex, single or double precision).

  -14 Internal integer workarray IS too small for solution. See error
   INFO(1) = -8.

  -15 Integer workarray IS too small for iterative refinement and/or error
    analysis. See error INFO(1) =-8.

  -16 N is out of range. INFO(2)=N.

  -17 The internal send buffer that was allocated dynamically by MUMPS on
   the processor is too small.  The user should increase the value of
   ICNTL(14) before calling MUMPS again.

  -18 The blocking size for multiple RHS (ICNTL(27)) is too large and may
   lead to an integer overflow.  This error may only occurs for very large
   matrices with large values of ICNTL(27) (e.g., several
   thousands). INFO(2) provides an estimate of the maximum value of
   ICNTL(27) that should be used.

  -19 The maximum allowed size of working memory ICNTL(23) is too small to
   run the factorization phase and should be increased. If INFO(2) is
   positive, then the number of entries that are missing at the moment when
   the error is raised is available in INFO(2). If INFO(2) is negative,
   then its absolute value should be multiplied by 1 million.

  -20 The internal reception buffer that was allocated dynamically by
   MUMPS is too small. Normally, this error is raised on the sender side
   when detecting that the message to be sent is too large for the
   reception buffer on the receiver. INFO(2) holds the minimum size of the
   reception buffer required (in bytes). The user should increase the value
   of ICNTL(14) before calling MUMPS again.

  -21 Value of PAR=0 is not allowed because only one processor is
   available; Running MUMPS in hostnode mode (the host is not a slave
   processor itself) requires at least two processors. The user should
   either set PAR to 1 or increase the number of processors.

  -22 A pointer array is provided by the user that is either
    • not associated, or
    • has insufficient size, or
    • is associated and should not be associated (for example, RHS on
      non-host processors)
    INFO(2) points to the incorrect pointer array as follows:
     INFO(2) array
       1     IRN or ELTPTR
       2     JCN or ELTVAR
       3     PERM IN
       4     A or A ELT
       5     ROWSCA
       6     COLSCA
       7     RHS
       8     LISTVAR SCHUR
       9     SCHUR
       10    RHS SPARSE
       11    IRHS SPARSE
       12    IRHS PTR
       13    ISOL loc
       14    SOL loc
       15    REDRHS
       16    IRN loc, JCN loc or A loc
       17    IRHS loc
       18    RHS loc

  -23 MPI was not initialized by the user prior to a call to MUMPS with
   JOB = –1.

  -24 NELT is out of range. INFO(2)=NELT.

  -25 A problem has occurred in the initialization of the BLACS. This may
   be because you are using a vendor’s BLACS. Try using a BLACS version
   from netlib instead.

  -26 LRHS is out of range. INFO(2)=LRHS.

  -27 NZ RHS and IRHS PTR(NRHS+1) do not match. INFO(2) = IRHS PTR(NRHS+1).

  -28 IRHS PTR(1) is not equal to 1. INFO(2) = IRHS PTR(1).

  -29 LSOL loc is smaller than INFO(23). INFO(2)=LSOL loc.

  -30 SCHUR LLD is out of range. INFO(2) = SCHUR LLD.

  -31 A 2D block cyclic symmetric (SYM=1 or 2) Schur complement is
   required with the option ICNTL(19)=3, but the user has provided a
   process grid that does not satisfy the constraint
   MBLOCK=NBLOCK. INFO(2)=MBLOCK-NBLOCK.

  -32 Incompatible values of NRHS and ICNTL(25). Either ICNTL(25) was set
   to -1 and NRHS is different from INFOG(28); or ICNTL(25) was set to i, 1
   ≤ i ≤ INFOG(28) and NRHS is different from 1. Value of NRHS is stored in
   INFO(2).

  -33 ICNTL(26) was asked for during solve phase (or during the
   factorization – see ICNTL(32)) but the Schur complement was not asked
   for at the analysis phase (ICNTL(19)).  INFO(2)=ICNTL(26).

  -34 LREDRHS is out of range. INFO(2)=LREDRHS.

  -35 This error is raised when the expansion phase is called (ICNTL(26)=2)
   but reduction phase (ICNTL(26)=1) was not called before. This error
   also occurs in case the reduction phase (ICNTL(26) = 1) is asked for at
   the solution phase (JOB=3) but the forward elimination was already
   performed during the factorization phase (JOB=2 and
   ICNTL(32)=1). INFO(2) contains the value of ICNTL(26).

  -36 Incompatible values of ICNTL(25) and INFOG(28). The value of
   ICNTL(25) is stored in INFO(2).

  -37 Value of ICNTL(25) incompatible with some other parameter. If
   ICNTL(25) is incompatible with ICNTL(xx), the index xx is stored in
   INFO(2).

  -38 Parallel analysis was set (i.e., ICNTL(28)=2) but PT-SCOTCH or
   ParMetis were not provided.

  -39 Incompatible values for ICNTL(28) and ICNTL(5) and/or ICNTL(19)
   and/or ICNTL(6).  Parallel analysis is not possible in the cases where
   the matrix is unassembled and/or a Schur complement is requested and/or
   a maximum transversal is requested on the matrix.

  -40 The matrix was indicated to be positive definite (SYM=1) by the user
   but a negative or null pivot was encountered during the processing of
   the root by ScaLAPACK. SYM=2 should be used.

  -41 Incompatible value of LWK USER between factorization and solution
   phases. This error may only occur when the factorization is in-core
   (ICNTL(22)=1), in which case both the contents of WK USER and LWK USER
   should be passed unchanged between the factorization (JOB=2) and
   solution (JOB=3) phases.

  -42 ICNTL(32) was set to 1 (forward during factorization), but the value
   of NRHS on the host processor is incorrect: either the value of NRHS
   provided at analysis is negative or zero, or the value provided at
   factorization or solve is different from the value provided at
   analysis. INFO(2) holds the value of id%NRHS that was provided at
   analysis.

  -43 Incompatible values of ICNTL(32) and ICNTL(xx). The index xx is
   stored in INFO(2).

  -44 The solve phase (JOB=3) cannot be performed because the factors or
   part of the factors are not available. INFO(2) contains the value of
   ICNTL(31).

  -45 NRHS ≤ 0. INFO(2) contains the value of NRHS.

  -46 NZ RHS ≤ 0. This is currently not allowed in case of reduced
   right-hand-side (ICNTL(26)=1) and in case entries of A inverse are
   requested (ICNTL(30)=1). INFO(2) contains the value of NZ RHS.

  -47 Entries of A inverse were requested during the solve phase (JOB=3,
   ICNTL(30)=1) but the constraint NRHS=N is not respected. The value of
   NRHS is provided in INFO(2).

  -48 A inverse Incompatible values of ICNTL(30) and ICNTL(xx). xx is
   stored in INFO(2).

  -49 SIZE SCHUR has an incorrect value: SIZE SCHUR < 0 or SIZE SCHUR ≥N,
   or SIZE SCHUR was modified on the host since the analysis phase. The
   value of SIZE SCHUR is provided in INFO(2).

  -50 An error occurred while computing the fill-reducing ordering during
   the analysis phase. This commonly happens when an (external) ordering
   tool returns an error code or a wrong result.

  -51 An external ordering (Metis/ParMetis, SCOTCH/PT-SCOTCH, PORD), with
   32-bit default integers, is invoked to processing a graph of size larger
   than 2^31 − 1. INFO(2) holds the size required to store the graph as a
   number of integer values; it is negative and its absolute value should
   be multiplied by 1 million.

  -52 When default Fortran integers are 64 bit (e.g. Fortran compiler flag
   -i8 -fdefault-integer-8 or something equivalent depending on your
   compiler) then external ordering libraries (Metis/ParMetis,
   SCOTCH/PT-SCOTCH, PORD) should also have 64-bit default
   integers. INFO(2) = 1, 2, 3 means that respectively Metis/ParMetis,
   SCOTCH/PT-SCOTCH or PORD were invoked and were not generated with 64-bit
   default integers.

  -53 Internal error that could be due to inconsistent input data between
   two consecutive calls.

  -54 The analysis phase (JOB=1) was called with ICNTL(35)=0 but the
   factorization phase was called with ICNTL(35)=1, 2 or 3. In order to
   perform the factorization with BLR compression, please perform the
   analysis phase again using ICNTL(35)=1, 2 or 3 (see the documentation of
   ICNTL(35)).

  -55 During a call to MUMPS including the solve phase with distributed
   right-hand side, LRHS loc was detected to be smaller than Nloc
   RHS. INFO(2)=LRHS loc.

  -56 During a call to MUMPS including the solve phase with distributed
   right-hand side and distributed solution, RHS loc and SOL loc point to
   the same workarray but LRHS loc < LSOL loc.  INFO(2)=LRHS loc.

  -57 During a call to MUMPS analysis phase with a block format (ICNTL(15)
   /= 0), an error in the interface provided by the user was
   detected. INFO(2) holds additional information about the issue:

   INFO(2) issue
     1     NBLK is incorrect (or not compatible with BLKPTR size),
            or -ICNTL(15) is not compatible with N
     2     BLKPTR is not provided or its content is incorrect
     3     BLKVAR if provided should be of size N
     4     BLKPTR is provided but ICNTL(15) < 0

  -70 During a call to MUMPS with JOB=7, the file specified to save the
   current instance, as derived from SAVE DIR and/or SAVE PREFIX, already
   exists. Before saving an instance into this file, it should be first
   suppressed (see JOB=-3). Otherwise, a different file should be specified
   by changing the values of SAVE DIR and/or SAVE PREFIX.

  -71 An error has occured during the creation of one of the files needed
   to save MUMPS data (JOB=7).

  -72 Error while saving data (JOB=7); a write operation did not succeed
   (e.g., disk full, I/O error, . . . ).  INFO(2) is the size that should
   have been written during that operation.  If INFO(2) is negative, then
   its absolute value should be multiplied by 1 million.

  -73 During a call to MUMPS with JOB=8, one parameter of the current
   instance is not compatible with the corresponding one in the saved
   INFO(2) points to the incorrect parameter in the table below:
   INFO(2) parameter
     1     fortran version (after/before 2003)
     2     integer size(32/64 bit)
     3     saved instance not compatible over MPI processes
     4     number of MPI processes
     5     arithmetic
     6     SYM
     7     PAR

  -74 The file resulting from the setting of SAVE DIR and SAVE PREFIX
   could not be opened for restoring data (JOB=8). INFO(2) is the rank of
   the process (in the communicator COMM) on which the error was detected.

  -75 Error while restoring data (JOB=8); a read operation did not succeed
   (e.g., end of file reached, I/O error, . . . ). INFO(2) is the size
   still to be read. If INFO(2) is negative, then the size that the package
   requested is obtained by multiplying the absolute value of INFO(2) by 1
   million.

  -76 Error while deleting the files (JOB=-3); some files to be erased
   were not found or could not be suppressed. INFO(2) is the rank of the
   process (in the communicator COMM) on which the error was detected.

  -77 Neither SAVE DIR nor the environment variable MUMPS SAVE DIR are
   defined.

  -78 Problem of workspace allocation during the restore step. The size
   still to be allocated is available in INFO(2). If INFO(2) is negative,
   then the size that the package requested is obtained by multiplying the
   absolute value of INFO(2) by 1 million.

  -79 MUMPS could not find a Fortran file unit to perform I/O’s. INFO(2)
   provides additional information on the error:
   • INFO(2)=1: the problem  occurs in the analysis phase, when attempting
     to find a free Fortran  unit for the WRITE PROBLEM feature
   • INFO(2)=2: the problem occurs during a call to MUMPS with JOB=7 or 8
     (save-restore feature).

  -90 Error in out-of-core management. See the error message returned on
   output unit ICNTL(1) for more information.

  -800 Temporary error associated to the current MUMPS release, subject to
   change or disappearance in the future. If INFO(2)=5, then this error is
   due to the fact that the elemental matrix format (ICNTL(5)=1) is
   currently incompatible with a BLR factorization (ICNTL(35)6=0).

  +1 Index (in IRN or JCN) out of range. Action taken by subroutine is to
   ignore any such entries and continue. INFO(2) is set to the number of
   faulty entries. Details of the first ten are printed on unit ICNTL(2).

  +2 During error analysis the max-norm of the computed solution is close
   to zero. In some cases, this could cause difficulties in the computation
   of RINFOG(6).

  +4 not used in current version.  +8 Warning return from the iterative
   refinement routine. More than ICNTL(10) iterations are required.

  + Combinations of the above warnings will correspond to summing the
   constituent warnings.

INFOG(2) holds additional information about the error or the warning.
 The difference between INFOG(1:2) and INFO(1:2) is that INFOG(1:2) is
 identical on all processors. It has the value of INFO(1:2) of the
 processor which returned with the most negative INFO(1) value. For
 example, if processor p returns with INFO(1)=-13, and INFO(2)=10000,
 then all other processors will return with INFOG(1)=-13 and
 INFOG(2)=10000, and with INFO(1)=-1 and INFO(2)=p.

INFOG(3) - after analysis: total (sum over all processors) estimated
 real/complex workspace to store the factors, assuming the factors are
 stored in full-rank format (ICNTL(35)=0 or 3). If INFOG(3) is negative,
 then its absolute value corresponds to millions of real/complex entries
 used to store the factor matrices. Assuming that the factors will be
 stored in full-rank format during the factorization (ICNTL(35)=0 or 3),
 a rough estimation of the size of the disk space in bytes of the files
 written all processors can be obtained by multiplying INFOG(3) (or its
 absolute value multiplied by 1 million when negative) by 4, 8, 8, or 16
 for single precision, double precision, single complex, and double
 complex arithmetics, respectively. See also RINFOG(15).  The effective
 size of the real/complex space needed will be returned in INFOG(9) (see
 below), but only after the factorization. Furthermore, after an
 out-of-core factorization, the size of the disk space for the files
 written by all processors is returned in RINFOG(16).

INFOG(4) - after analysis: total (sum over all processors) estimated
 integer workspace to store the factor matrices (assuming a full-rank
 storage of the factors).

INFOG(5) - after analysis: estimated maximum front size in the complete tree.

INFOG(6) - after analysis: number of nodes in the complete tree.

INFOG(7) - after analysis: the ordering method actually used. The
 returned value will depend on the type of analysis performed,
 e.g. sequential or parallel (see INFOG(32)). Please refer to ICNTL(7)
 and ICNTL(29) for more details on the ordering methods available in
 sequential and parallel analysis respectively.

INFOG(8) - after analysis: structural symmetry in percent (100 :
 symmetric, 0 : fully unsymmetric) of the (permuted) matrix, -1 indicates
 that the structural symmetry was not computed (which will be the case if
 the input matrix is in elemental form or if analysis by block was
 performed (ICNTL(15))).

INFOG(9) - after factorization: total (sum over all processors)
 real/complex workspace to store the factor matrices, possibly including
 low-rank factor matrices (ICNTL(35)=2). If negative, then the absolute
 value corresponds to the size in millions of real/complex entries used
 to store the factor matrices.

INFOG(10) - after factorization: total (sum over all processors) integer
 workspace to store the factor matrices. If negative the absolute value
 corresponds to millions of integer entries in the integer workspace to
 store the factor matrices.

INFOG(11) - after factorization: order of largest frontal matrix.

INFOG(12) - after factorization: total number of off-diagonal pivots if
 SYM=0 or total number of negative pivots (real arithmetic) if SYM=1 or
 2. If ICNTL(13)=0 (the default) this excludes pivots from the parallel
 root node treated by ScaLAPACK. (This means that the user should set
 ICNTL(13) to a positive value, say 1, or use a single processor in order
 to get the exact number of off-diagonal or negative pivots rather than a
 lower bound.) Furthermore, when ICNTL(24) is set to 1 and SYM=1 or 2,
 INFOG(12) excludes the null13 pivots, even if their sign is negative. In
 other words, a pivot cannot be both null and negative.  Note that if
 SYM=1 or 2, INFOG(12) will be 0 for complex symmetric matrices.

INFOG(13) - after factorization: total number of delayed pivots. A
 variable of the original matrix may be delayed several times between
 successive frontal matrices. In that case, it accounts for several
 delayed pivots. A large number (more that 10% of the order of the
 matrix) indicates numerical problems. Settings related to numerical
 preprocessing (ICNTL(6),ICNTL(8), ICNTL(12)) might then be modified by
 the user.

INFOG(14) - after factorization: total number of memory compresses.

INFOG(15) - after solution: number of steps of iterative refinement.

INFOG(16) and INFOG(17) - after analysis: estimated size (in million of
 bytes) of all MUMPS internal data for running full-rank factorization
 in-core for a given value of ICNTL(14).
 • —– (16) : max over all processors
 i.e., whose magnitude is smaller than the tolerance defined by CNTL(3).
 • —– (17) : sum over all processors.

INFOG(18) and INFOG(19) - after factorization: size in millions of bytes
 of all MUMPS internal data allocated during factorization.
 • —– (18) : max over all processors
 • —– (19) : sum over all processors.
 Note that in the case where WK USER is provided, the memory allocated by the
 user for the local arrays WK USER is not counted in INFOG(18) and INFOG(19).

INFOG(20) - after analysis: estimated number of entries in the factors
 assuming full-rank factorization.  If negative the absolute value
 corresponds to millions of entries in the factors. Note that in the
 unsymmetric case, INFOG(20)=INFOG(3). In the symmetric case, however,
 INFOG(20) < INFOG(3).

INFOG(21) and INFOG(22) - after factorization: size in millions of bytes
 of memory effectively used during factorization.
 • —– (21) : max over all processors
 • —– (22) : sum over all processors.
 This includes the memory effectively used in the local workarrays WK
 USER, in the case where the arrays WK USER are provided.

INFOG(23) - after analysis: value of ICNTL(6) effectively used.

INFOG(24) - after analysis: value of ICNTL(12) effectively used.

INFOG(25) - after factorization: number of tiny pivots (number of pivots
 modified by static pivoting)

INFOG(26) and INFOG(27) - after analysis: estimated size (in millions of
 bytes) of all MUMPS internal data for running full-rank factorization
 out-of-core (ICNTL(22)6= 0) for a given value of ICNTL(14).
 • —– (26) : max over all processors
 • —– (27) : sum over all processors

INFOG(28) - after factorization: number of null pivot rows
 encountered. See ICNTL(24) and CNTL(3) for the definition of a null
 pivot row.

INFOG(29) - after factorization: effective number of entries in the
 factor matrices (sum over all processors) assuming that full-rank
 factorization has been performed. If negative, then the absolute value
 corresponds to millions of entries in the factors. Note that in case the
 factor matrices are stored full-rank (ICNTL(35)=0 or 3), we have
 INFOG(29)=INFOG(9) in the unsymmetric case and INFOG(29) ≤ INFOG(9) in
 the symmetric case.

INFOG(30) and INFOG(31) - after solution: size in millions of bytes of
 memory effectively used during solution phase:
 • —– (30) : max over all processors
 • —– (31) : sum over all processors

INFOG(32) - after analysis: the type of analysis actually done (see
 ICNTL(28)). INFOG(32) has value 1 if sequential analysis was performed,
 in which case INFOG(7) returns the sequential ordering option used, as
 defined by ICNTL(7). INFOG(32) has value 2 if parallel analysis was
 performed, in which case INFOG(7) returns the parallel ordering used, as
 defined by ICNTL(29).

INFOG(33): effective value used for ICNTL(8). It is set both after the
 analysis and the factorization phases. If ICNTL(8)=77 on entry to the
 analysis and INFOG(33) has value 77 on exit from the analysis, then no
 scaling was computed during the analysis and the automatic decision will
 only be done during factorization (except if the user modifies ICNTL(8)
 to set a specific option on entry to the factorization).

INFOG(34): if the computation of the determinant was requested (see
 ICNTL(33)), INFOG(34) contains the exponent of the determinant. See also
 RINFOG(12) and RINFOG(13): the determinant is obtained by multiplying
 (RINFOG(12), RINFOG(13)) by 2 to the power INFOG(34).

INFOG(35) - after factorization: effective number of entries in the
 factors (sum over all processors) taking into account BLR factor
 compression. If negative, then the absolute value corresponds to
 millions of entries in the factors. It is equal to INFOG(29) when BLR
 functionality (see ICNTL(35)) is not activated or leads to nocompression.

INFOG(36), INFOG(37), INFOG(38), and INFOG(39) - after analysis:
 estimated size (in million of bytes) of all MUMPS internal data for
 running low-rank factorization with low-rank factors for a given value
 of ICNTL(14) and ICNTL(38).
 • in-core
 . —– (36) : max over all processors
 . —– (37) : sum over all processors.
 • out-of-core
 . —– (38) : max over all processors
 . —– (39) : sum over all processors.

INFOG(40) - INFOG(47) are not used in the current version.
INFOG(48) - INFOG(49) are reserved.
INFOG(50) - INFOG(80) are not used in the current version.

------
RINFOG:
------

RINFOG(1) - after analysis: the estimated number of floating-point
 operations (on all processors) for the elimination process.  91

RINFOG(2) - after factorization: the total number of floating-point
 operations (on all processors) for the assembly process.

RINFOG(3) - after factorization: the total number of floating-point
 operations (on all processors) for the elimination process. In case the
 BLR feature is activated (ICNTL(35)=1, 2 or 3), RINFOG(3) represents the
 theoretical number of operations for the standard full-rank
 factorization.

RINFOG(4) to RINFOG(8) - after solve with error analysis: Only returned
 if ICNTL(11) = 1 or 2.

RINFOG(9) to RINFOG(11) - after solve with error analysis: Only returned
 if ICNTL(11) = 2.

RINFOG(12) - after factorization: if the computation of the determinant
 was requested (see ICNTL(33)), RINFOG(12) contains the real part of the
 determinant. The determinant may contain an imaginary part in case of
 complex arithmetic (see RINFOG(13)). It is obtained by multiplying
 (RINFOG(12), RINFOG(13)) by 2 to the power INFOG(34).

RINFOG(13) - after factorization: if the computation of the determinant
 was requested (see ICNTL(33)), RINFOG(13) contains the imaginary part of
 the determinant. The determinant is then obtained by multiplying
 (RINFOG(12), RINFOG(13)) by 2 to the power INFOG(34).

RINFOG(14) - after factorization: the total effective number of
 floating-point operations (on all processors) for the elimination
 process. It is equal to RINFOG(3) when the BLR feature is not activated
 (ICNTL(35)=0) and will typically be smaller than RINFOG(3) when the BLR
 functionality is activated and leads to compression.

RINFOG(15) - after analysis: if the user decides to perform an
 out-of-core factorization (ICNTL(22)=1), then a rough estimation of the
 total size of the disk space in MegaBytes of the files written by all
 processors is provided in RINFOG(15). If the analysis is full-rank
 (ICNTL(35)=0 for the analysis step), then the factorization is
 necessarily full-rank so that RINFOG(15) is computed for a full-rank
 factorization (ICNTL(35)=0 also for the factorization).  If ICNTL(35)=1,
 2 or 3 at analysis, then RINFOG(15) is computed assuming a low-rank
 (incore) storage of the factors of the BLR fronts during the
 factorization (ICNTL(35)=2 during factorization). In case ICNTL(35)=1, 2
 or 3 for the analysis and the factors will be stored in fullrank format
 (ICNTL(35)=0 or 3 for the factorization), we refer the user to INFOG(3)
 in order to obtain a rough estimate of the necessary disk space for all
 processors.  The effective size in Megabytes of the files written by all
 processors will be returned in

RINFOG(16), but only after the factorization.

RINFOG(16) - after factorization: in the case of an out-of-core
 execution (ICNTL(22)=1), the total size in MegaBytes of the disk space
 used by the files written by all processors is provided.

RINFOG(17) - after each job: sum over all processors of the sizes (in
 MegaBytes) of the files used to save the instance.

RINFOG(18) - after each job: sum over all processors of the sizes (in
 MegaBytes) of the MUMPS structures.

RINFOG(19) - RINFOG(40) are not used in the current version.

----------------------------- per processor information ------------------------

----
INFO:
----

INFO(1) is 0 if the call to MUMPS was successful, negative if an error
 occurred, or positive if a warning is returned. In
 particular, after successfully saving or restoring an instance (call to
 MUMPS with JOB=7 or JOB=8), INFO(1) will be 0 even if INFO(1) was
 different from 0 at the moment of saving the MUMPS instance to disk.
INFO(2) holds additional information about the error or the warning. If
 INFO(1) = -1, INFO(2) is the processor number (in communicator COMM) on
 which the error was detected.

 See INFOG(1) for possible values.

INFO(3) - after analysis: Estimated size of the real/complex space
 needed on the processor to store the factors, assuming the factors are
 stored in full-rank format (ICNTL(35)=0 or 3 during factorization). If
 INFO(3) is negative, then its absolute value corresponds to millions of
 real/complex entries used to store the factor matrices. Assuming that
 the factors will be stored in full-rank format during the factorization
 (ICNTL(35)=0 or 3), a rough estimation of the size of the disk space in
 bytes of the files written by the concerned processor can be obtained by
 multiplying INFO(3) (or its absolute value multiplied by 1 million when
 negative) by 4, 8, 8, or 16 for single precision, double precision,
 single complex, and double complex arithmetics, respectively. See also
 RINFO(5).  The effective size of the real/complex space needed to store
 the factors will be returned in INFO(9) (see below), but only after the
 factorization. Furthermore, after an out-of-core factorization
 (ICNTL(22)=1), the size of the disk space for the files written by the
 local processor is returned in RINFO(6). Finally, the total estimated
 size of the full-rank factors for all processors (sum of the INFO(3)
 values over all processors) is returned in INFOG(3).

INFO(4) - after analysis: Estimated integer space needed on the
 processor for factors (assuming a full-rank storage for the factors)

INFO(5) - after analysis: Estimated maximum front size on the processor.

INFO(6) - after analysis: Number of nodes in the complete tree. The same
 value is returned on all processors.

INFO(7) - after analysis: Minimum estimated size of the main internal
 integer workarray IS to run the numerical factorization in-core .

INFO(8) - after analysis: Minimum estimated size of the main internal
 real/complex workarray S to run the numerical factorization in-core when
 factors are stored full-rank (ICNTL(35)=0 or 3).  If negative, then the
 absolute value corresponds to millions of real/complex entries needed in
 this workarray. It is also the estimated minimum size of LWK USER in
 that case, if the user provides WK USER.

INFO(9) - after factorization: Size of the real/complex space used on
 the processor to store the factor matrices, possibly including low-rank
 factor matrices (ICNTL(35)=1 or 2). If negative, then the absolute value
 corresponds to millions of real/complex entries used to store the factor
 matrices.  Finally, the total size of the factor matrices for all
 processors (sum of the INFO(9) values over all processors) is returned
 in INFOG(9).

INFO(10) - after factorization: Size of the integer space used on the
 processor to store the factor matrices.

INFO(11) - after factorization: Order of the largest frontal matrix
 processed on the processor.

INFO(12) - after factorization: Number of off-diagonal pivots selected
 on the processor if SYM=0 or number of negative pivots on the processor
 if SYM=1 or 2. If ICNTL(13)=0 (the default), this excludes pivots from
 the parallel root node treated by ScaLAPACK. (This means that the user
 should set ICNTL(13)=1 or use a single processor in order to get the
 exact number of off-diagonal or negative pivots rather than a lower
 bound.) Furthermore, when ICNTL(24) is set to 1 and SYM=1 or 2,
 INFOG(12) excludes the null12 pivots, even if their sign is negative. In
 other words, a pivot cannot be both null and negative. Note that for
 complex symmetric matrices (SYM=1 or 2), INFO(12) will be 0. See also
 INFOG(12), which provides the total number of off-diagonal or negative
 pivots over all processors.

INFO(13) - after factorization: The number of postponed elimination
 because of numerical issues.

INFO(14) - after factorization: Number of memory compresses.

INFO(15) - after analysis: estimated size in MegaBytes (millions of
 bytes) of all working space to perform full-rank numerical phases
 (factorization/solve) in-core (ICNTL(22)=0 for the factorization). The
 maximum and sum over all processors are returned respectively in
 INFOG(16) and INFOG(17). See also INFO(22) which provides the actual
 memory that was needed but only after factorization.

INFO(16) - after factorization: total size (in millions of bytes) of all
 MUMPS internal data allocated during the numerical factorization. This
 excludes the memory for WK USER, in the case where WK USER is
 provided. The maximum and sum over all processors are returned
 respectively in INFOG(18) and INFOG(19).

INFO(17) - after analysis: estimated size in MegaBytes (millions of
 bytes) of all working space to run the numerical phases out-of-core
 (ICNTL(22)6=0) with the default strategy. The maximum and sum over all
 processors are returned respectively in INFOG(26) and INFOG(27). See
 also INFO(22) which provides the actual memory that was needed but only
 after factorization.

INFO(18) - after factorization: local number of null pivot rows detected
 when ICNTL(24)=1.

INFO(19) - after analysis: Estimated size of the main internal integer
 workarray IS to run the numerical factorization out-of-core .

INFO(20) - after analysis: Estimated size of the main internal
 real/complex workarray S to run the numerical factorization out-of-core.
 If negative, then the absolute value corresponds to millions of
 real/complex entries needed in this workarray. It is also the estimated
 minimum size of LWK USER in that case, if the user provides WK USER.

INFO(21) - after factorization: Effective space used in the main
 real/complex workarray S– or in the workarray WK USER, in the case where
 WK USER is provided. If negative, then the absolute value corresponds to
 millions of real/complex entries needed in this workarray.

INFO(22) - after factorization: Size in millions of bytes of memory
 effectively used during factorization. This includes the part of the
 memory effectively used from the workarray WK USER, in the case where WK
 USER is provided. The maximum and sum over all processors are returned
 respectively in INFOG(21) and INFOG(22). The difference between
 estimated and effective memory may results from numerical pivoting
 difficulties, parallelism and BLR effective compression rates.  12i.e.,
 whose magnitude is smaller than the tolerance defined by CNTL(3).  90

INFO(23) - after factorization: total number of pivots eliminated on the
 processor. In the case of a distributed solution (see ICNTL(21)), this
 should be used by the user to allocate solution vectors ISOL loc and SOL
 loc of appropriate dimensions (ISOL loc of size INFO(23), SOL loc of
 size LSOL loc × NRHS where LSOL loc ≥ INFO(23)) on that processor,
 between the factorization and solve steps.

INFO(24) - after analysis: estimated number of entries in the factor
 matrices on the processor. If negative, then the absolute value
 corresponds to millions of entries in the factors. Note that in the
 unsymmetric case, INFO(24)=INFO(3). In the symmetric case, however,
 INFO(24) < INFO(3).  The total number of entries in the factor matrices
 for all processors (sum of the INFO(24) values over all processors) is
 returned in INFOG(20)

INFO(25) - after factorization: number of tiny pivots (number of pivots
 modified by static pivoting) detected on the processor (see INFOG(25)
 for the the total number of tiny pivots).

INFO(26) - after solution: effective size in MegaBytes (millions of
 bytes) of all working space to run the solution phase. (The maximum and
 sum over all processors are returned in INFOG(30) and INFOG(31),
 respectively).

INFO(27) - after factorization: effective number of entries in factor
 matrices assuming full-rank factorization has been performed. If
 negative, then the absolute value corresponds to millions of entries in
 the factors. Note that in case full-rank storage of factors (ICNTL(35)=0
 or 3), we have INFO(27)=INFO(9) in the unsymmetric case and INFO(27) ≤
 INFO(9) in the symmetric case.  The sum of INFO(27) over all processors
 is available in INFOG(29).

INFO(28) - after factorization: effective number of entries in factors
 on the processor taking into account BLR compression. If negative, then
 the absolute value corresponds to millions of entries in the factors. It
 is equal to INFO(27) when BLR functionality (see ICNTL(35)) is not
 activated or leads to no compression.

INFO(29) - after analysis: minimum estimated size of the main internal
 real/complex workarray S to run the numerical factorization in-core when
 factors are stored low-rank (ICNTL(35)=1,2).  If negative, then the
 absolute value corresponds to millions of real/complex entries needed in
 this workarray. It is also the estimated minimum size of LWK USER in
 that case, if the user provides WK USER.

INFO(30) and INFO(31) - after analysis: estimated size in MegaBytes
 (millions of bytes) of all working space to perform low-rank numerical
 phases (factorization/solve) with low-rank factors (ICNTL(35)=1,2) and
 estimated compression rate given by ICNTL(38).  • —– (30) in-core
 factorization and solve The maximum and sum over all processors are
 returned respectively in INFOG(36) and INFOG(37).  • —– (31) out-of-core
 factorization and solve The maximum and sum over all processors are
 returned respectively in INFOG(38) and INFOG(39).  See also INFO(22)
 which provides the actual memory that was needed but only after
 factorization. Numerical pivoting difficulties and the effective
 compression of the factors (in case ICNTL(35)=1,2) typically impact the
 difference between estimated and effective memory.

INFO(32-38) are not used in the current version.

INFO(39) - after factorization: effective size of the main internal
 real/complex workarray S (allocated internally or by the user when WK
 USER is provided) to run the numerical factorization. If negative, then
 the absolute value corresponds to millions of real/complex entries
 needed in this workarray.

INFO(40-80) are not used in the current version.


-----
RINFO:
-----

RINFO(1) - after analysis: The estimated number of floating-point
 operations on the processor for the elimination process.

RINFO(2) - after factorization: The number of floating-point operations
 on the processor for the assembly process.

RINFO(3) - after factorization: The number of floating-point operations
 on the processor for the elimination process. In case the BLR feature is
 activated (ICNTL(35)=1, 2 or 3), RINFO(3) represents the theoretical
 number of operations for the standard full-rank factorization.

RINFO(4) - after factorization: The effective number of floating-point
 operations on the processor for the elimination process. It is equal to
 RINFO(3) when the BLR feature is not activated (ICNTL(35)=0) and will
 typically be smaller than RINFO(3) when the BLR feature is activated and
 leads to compression.

RINFO(5) - after analysis: if the user decides to perform an out-of-core
 factorization (ICNTL(22)=1), then a rough estimation of the size of the
 disk space in MegaBytes of the 88 files written by the concerned
 processor is provided in RINFO(5). If the analysis is fullrank
 (ICNTL(35)=0 for the analysis step), then the factorization is
 necessarily full-rank so that RINFO(5) is computed for a full-rank
 factorization (ICNTL(35)=0 also for the factorization).  If ICNTL(35)=1,
 2 or 3 at analysis, then RINFO(5) is computed assuming a low-rank
 (incore) storage of the factors of the BLR fronts during the
 factorization (ICNTL(35)=1 or 2 during factorization). In case
 ICNTL(35)=1, 2 or 3 at analysis and the factors are stored in full-rank
 format (ICNTL(35)=0 or 3 for the factorization), we refer the user to
 INFO(3) in order to obtain a rough estimate of the necessary disk space
 for the concerned processor.  The effective size in MegaBytes of the
 files written by the current processor will be returned in

RINFO(6), but only after the factorization. The total estimated disk
 space (sum of the values of RINFO(5) over all processors) is returned in
 RINFOG(15).

RINFO(6) - after factorization: in the case of an out-of-core execution
 (ICNTL(22)=1), the size in MegaBytes of the disk space used by the files
 written by the concerned processor is provided. The total disk space
 (for all processors) is returned in RINFOG(16).

RINFO(7) - after each job: The size (in MegaBytes) of the file used to
 save the data on the processor

RINFO(8) - after each job: The size (in MegaBytes) of the MUMPS strucuture.

RINFO(9) - RINFO(40) are not used in the current version.


