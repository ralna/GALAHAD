SCU
===

.. module:: galahad.scu

The ``scu`` package computes the **solution to an extended system of $n + m$
sparse real linear equations in $n + m$ unknowns,**
$$\begin{pmatrix}A & B \\ C  & D\end{pmatrix}
\begin{pmatrix}x_1 \\ x_2\end{pmatrix} = 
\begin{pmatrix}b_1 \\ b_2\end{pmatrix}$$
in the case where the $n$ by $n$ matrix $A$ is nonsingular
and solutions to the systems
$$A x  =  b \;\;\mbox{and}\;\; A^T y  =  c$$
may be obtained from an external source, such as an existing
factorization.  The subroutine uses reverse communication to obtain
the solution to such smaller systems.  The method makes use of
the Schur complement matrix
$$S = D - C A^{-1} B.$$
The Schur complement is stored and factorized as a dense matrix
and the subroutine is thus appropriate only if there is
sufficient storage for this matrix. Special advantage is taken
of symmetry and definiteness in the coefficient matrices.
Provision is made for introducing additional rows and columns
to, and removing existing rows and columns from, the extended.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/scu.pdf for additional details.

method
------

The function ``scu_factorize`` forms the Schur complement 
$S$ of $A$ in the extended matrix by repeated 
reverse communication to obtain the columns of  $A^{-1} B$. 
The Schur complement or its negative is then factorized 
into its *QR* or, if possible, Cholesky factors. 

The function ``scu_solve`` solves the extended system using 
the following well-known scheme: 

* **(i)** Compute the solution to $A u = b_1$; 

* **(ii)** Compute $x_2$ from $S x_2 = b_2 - C u$; 

* **(iii)** Compute the solution to $A v = B x_2$; and 

* **(iv)** Compute $x_1 = u - v$. 

The functions ``scu__append`` and ``scu_delete`` compute the
factorization of the Schur complement after a row and column have been
appended to, and removed from, the extended matrix, respectively.
The existing factorization is updated to obtain the new one; this is
normally more efficient than forming the factorization from scratch.

functions
---------

   .. function:: scu.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options. Currently empty,
        but reserved for future use.

   .. function:: [optional] scu.information()

      Provide optional output information.

      **Returns:**

      status : int
         the return status.  Possible values are:

         * **-1**

           One or more of the stated restrictions on the components 
           $1 \leq$  class $\leq 4$, 
           n $\geq 0$,
           $0 \leq$ m $\leq$ m_max,
           ($0 \leq$ m $\leq$ m_max-1 in scu_append ) 
           $1 \leq$ col_del $\leq m$ and 
           $1 \leq$ row_del $\leq m$ has been violated.  

         * **-2**

           The subroutine has been called with an initial value status
           $\leq 0$. 

         * **-3**

           The factors of $S$ have not yet been formed in data. 
           This indicates that either  
           scu_factorize has not yet been called, or that 
           the last call to scu_factorize, 
           scu_append or scu_delete 
           ended in a failure.  

         * **-4**

           One or more of the arrays BD_val, BD_row and 
           BD_col_start has not been allocated. 

         * **-5**

           When the extended matrix is unsymmetric, one or more of the arrays  
           CD_val, CD_col and CD_row_start has not been allocated. 

         * **-6**

           One or more of the arrays BD_val, BD_row and 
           BD_col_start is not large enough. Check that the dimension 
           of BD_col_start is no smaller than m+1 
           (m+2 for scu_append), and that those of  
           BD_val and BD_row are no smaller than 
           BD_col_start(m+1)-1, and re-enter. 
           (BD_col_start(m+2)-1 for scu_append} 
           and BD_col_start(m+1)+ |col_delmatrixrow_del|-1} for scu_delete ). 

         * **-7**

           When the extended matrix is unsymmetric, one or more of the arrays  
           CD_val, CD_col and CD_row_start is not large enough. Check that the 
           dimension of CD_row_start is no smaller than m+1 
           (m+2 for scu_append), and that those of  
           CD_val and CD_col are no smaller than
           CD_row_start(m+1)-1 
           CD_row_start(m+2)-1 for scu_append
           and CD_row_start(m+1)+ 
           (|col_del-matrix.row_del|-1} for scu__delete ). 

         * **-8**

           The value recorded in does not correspond to the 
           dimension of $D$.  

         * **-9**

           The Schur complement matrix is singular; this has been 
           detected during the QR factorization of $S$. 

         * **-10**

           The Schur complement matrix is expected to be positive definite, 
           but this has been found not to be the case  
           during the Cholesky factorization of $S$. 

         * **-11**

           The Schur complement matrix is expected to be negative definite, 
           but this has been found not to be the case  
           during the Cholesky factorization of $-S$. 

         * **-12**

           An internal array allocation or deallocation failed.  
           See info['alloc_status'] for further details. 
      info : dict
         dictionary containing output information:
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          inertia : int
             the inertia of $S$ when the extended matrix is symmetric.
             Specifically, inertia(i), i=0,1,2 give the number of
             positive, negative and zero eigenvalues of $S$
             respectively.

   .. function:: scu.finalize()

     Deallocate all internal private storage.
