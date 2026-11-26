RPD
===

.. module:: galahad.rpd

.. include:: rpd_intro.rst

functions
---------

   .. function:: rpd.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          qplib : int
             QPLIB file input stream number.
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output,

             * **>=1**

               gives increasingly verbose (debugging) output.

          space_critical : bool
             if ``space_critical`` True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.

   .. function:: rpd.get_stats(qplib_file, options=None)

      Read the data for a problem from a file in QPLIB format, and
      return its type and dimensions. This must be called before any of
      the other ``rpd.get_*`` functions.

      **Parameters:**

      qplib_file : str
          holds the name of the QPLIB file.
      options : dict, optional
          dictionary of control options (see ``rpd.initialize``).

      **Returns:**

      p_type : str
          holds the three-character QPLIB problem type: the type of the
          objective (L, D, C or Q), of the variables (C, B, M, I or G) and
          of the constraints (N, B, L, D, C or Q).
      n : int
          holds the number of variables.
      m : int
          holds the number of general constraints.
      h_ne : int
          holds the number of entries in the lower triangle of the Hessian
          $H$ of the objective.
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      h_c_ne : int
          holds the number of entries in the lower triangles of the
          Hessians $H_c$ of the constraints.

   .. function:: rpd.get_g(n)

      Recover the gradient $g$ of the linear term of the objective.

      **Parameters:**

      n : int
          holds the number of variables, $n$.

      **Returns:**

      g : ndarray(n)
          holds the gradient $g$ of the linear term of the objective.

   .. function:: rpd.get_f()

      Recover the constant term $f$ of the objective.

      **Returns:**

      f : float
          holds $f$.

   .. function:: rpd.get_xlu(n)

      Recover the lower and upper bounds on the variables.

      **Parameters:**

      n : int
          holds the number of variables, $n$.

      **Returns:**

      x_l : ndarray(n)
          holds the lower bounds $x^l$.
      x_u : ndarray(n)
          holds the upper bounds $x^u$.

   .. function:: rpd.get_clu(m)

      Recover the lower and upper bounds on the general constraints.

      **Parameters:**

      m : int
          holds the number of general constraints, $m$.

      **Returns:**

      c_l : ndarray(m)
          holds the lower bounds $c^l$.
      c_u : ndarray(m)
          holds the upper bounds $c^u$.

   .. function:: rpd.get_h(ne)

      Recover the lower triangle of the Hessian $H$ of the objective in the sparse co-ordinate storage scheme.

      **Parameters:**

      ne : int
          holds the number of entries in the lower triangle of the Hessian $H$ of the objective.

      **Returns:**

      h_row : ndarray(ne)
          holds the row indices (starting from 0) of the entries.
      h_col : ndarray(ne)
          holds the column indices (starting from 0) of the entries.
      h_val : ndarray(ne)
          holds the values of the entries.

   .. function:: rpd.get_a(ne)

      Recover the constraint Jacobian $A$ in the sparse co-ordinate storage scheme.

      **Parameters:**

      ne : int
          holds the number of entries in the constraint Jacobian $A$.

      **Returns:**

      a_row : ndarray(ne)
          holds the row indices (starting from 0) of the entries.
      a_col : ndarray(ne)
          holds the column indices (starting from 0) of the entries.
      a_val : ndarray(ne)
          holds the values of the entries.

   .. function:: rpd.get_h_c(ne)

      Recover the lower triangles of the Hessians $H_c$ of the constraints
      in the joint sparse co-ordinate storage scheme.

      **Parameters:**

      ne : int
          holds the number of entries in the lower triangles of the
          Hessians.

      **Returns:**

      h_c_ptr : ndarray(ne)
          holds the constraint indices (starting from 0) of the entries.
      h_c_row : ndarray(ne)
          holds the row indices (starting from 0) of the entries.
      h_c_col : ndarray(ne)
          holds the column indices (starting from 0) of the entries.
      h_c_val : ndarray(ne)
          holds the values of the entries.

   .. function:: rpd.get_x_type(n)

      Recover the types of the variables.

      **Parameters:**

      n : int
          holds the number of variables, $n$.

      **Returns:**

      x_type : ndarray(n)
          holds the type of each variable: 0 if it is continuous, 1 if it
          is integer, and 2 if it is binary.

   .. function:: rpd.get_x(n)

      Recover the initial values of the variables $x$.

      **Parameters:**

      n : int
          holds the number of variables, $n$.

      **Returns:**

      x : ndarray(n)
          holds the initial values of the variables $x$.

   .. function:: rpd.get_y(m)

      Recover the initial values of the Lagrange multipliers $y$ of the general constraints.

      **Parameters:**

      m : int
          holds the number of general constraints, $m$.

      **Returns:**

      y : ndarray(m)
          holds the initial values of the Lagrange multipliers $y$ of the general constraints.

   .. function:: rpd.get_z(n)

      Recover the initial values of the dual variables $z$.

      **Parameters:**

      n : int
          holds the number of variables, $n$.

      **Returns:**

      z : ndarray(n)
          holds the initial values of the dual variables $z$.

   .. function:: [optional] rpd.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             return status. Possible values are

             * **0**

               The call was successful.

             * **-1**

               An allocation error occurred. A message indicating the
               offending array is written on unit options['error'], and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-2**

               A deallocation error occurred.  A message indicating the
               offending array is written on unit options['error'] and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-22**

               An input/outpur error occurred.

             * **-25**

               The end of the input file was reached prematurely.

             * **-5**

               The problem type was not recognised.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          io_status : int
             status from last read attempt.
          line : int
             number of last line read from i/o file.
          p_type : str
             problem type.

   .. function:: rpd.terminate()

     Deallocate all internal private storage.
