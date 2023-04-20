SCU
===

.. module:: galahad.scu

The ``scu`` package computes the **solution to an extended system of $n + m$
sparse real linear equations in $n + m\ unknowns,**
$$\left(\begin{matrix}A & B \\ C  & D\end{matrix}\right) \left(\begin{matrix}x_1 \\ x_2\end{matrix}\right)= \left(\begin{matrix}b_1 \\ b2\end{matrix}\right)$$
in the case where the $n$ by $n$ matrix $A$ is nonsingular
and solutions to the systems
$$A x  =  b \;\mbox{and}\; A^T y  =  c$$
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
to, and removing existing rows and columns from, the extended

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/scu.pdf for a brief description of the
method employed and other details.

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

      inform : dict
         dictionary containing output information:
          status : int
             return status. A non-zero value indicates an error or a
             request for further information. See SCU_solve for
             details.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          inertia : int
             the inertia of $S$ when the extended matrix is symmetric.
             Specifically, inertia(i), i=0,1,2 give the number of
             positive, negative and zero eigenvalues of $S$
             respectively.

   .. function:: scu.finalize()

     Deallocate all internal private storage.
