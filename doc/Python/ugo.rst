UGO
===

.. module:: galahad.ugo

.. include:: ugo_intro.rst

functions
---------

   .. function:: ugo.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
          dictionary containing default control options:
           error : int
             error and warning diagnostics occur on stream error.
           out :  int
             general output occurs on stream out.
           print_level : int
             the level of output required. Possible values are:

             * **<= 0**

               no output
             * **1**

               a one-line summary for every improvement
             * **2**

               a summary of each iteration
             * **>= 3**

               increasingly verbose (debugging) output.

           start_print : int
             any printing will start on this iteration.
           stop_print : int
             any printing will stop on this iteration.
           print_gap : int
             the number of iterations between printing.
           maxit : int
             the maximum number of iterations allowed.
           initial_point : int
             the number of initial (uniformly-spaced) evaluation points
             (<2 reset to 2).
           storage_increment : int
             incremenets of storage allocated (less that 1000 will be
             reset to 1000).
           buffer : int
             unit for any out-of-core writing when expanding arrays.
           lipschitz_estimate_used : int
             what sort of Lipschitz constant estimate will be used:

             * **1**

               the global contant provided

             * **2**

               an estimated global contant

             * **3**

               estimated local costants.

           next_interval_selection : int
             how is the next interval for examination chosen:

             * **1**

               traditional
             * **2**

               local_improvement.

           refine_with_newton : int
             try refine_with_newton Newton steps from the vacinity of
             the global minimizer to try to improve the estimate.
           alive_unit : int
             removal of the file alive_file from unit alive_unit
             terminates execution.
           alive_file : str
             see alive_unit.
           stop_length : float
             overall convergence tolerances. The iteration will terminate
             when the step is less than ``stop_length``.
           small_g_for_newton : float
             if the absolute value of the gradient is smaller than
             small_g_for_newton, the next evaluation point may be at a
             Newton estimate of a local minimizer.
           small_g : float
             if the absolute value of the gradient at the end of the interval
             search is smaller than small_g, no Newton search is necessary.
           obj_sufficient : float
             stop if the objective function is smaller than a specified value.
           global_lipschitz_constant : float
             the global Lipschitz constant for the gradient
             (-ve means unknown).
           reliability_parameter : float
             the reliability parameter that is used to boost insufficiently
             large estimates of the Lipschitz constant (-ve means that
             default values will be chosen depending on whether second
             derivatives are provided or not).
           lipschitz_lower_bound : float
             a lower bound on the Lipschitz constant for the gradient
             (not zero unless the function is constant).
           cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
           clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means infinite).
           second_derivative_available : bool
             if ``second_derivative_available`` is True, the user must provide
             them when requested. The package is generally more effective
             if second derivatives are available.
           space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
           deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
           prefix : str
             all output lines will be prefixed by the string contained
             in quotes within ``prefix``, e.g. 'word' (note the qutoes)
             will result in the prefix word.

   .. function:: ugo.load(x_l, x_u, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      x_l : double
          holds the value $x^l$ of the lower bound on the optimization
          variable $x$.
      x_u : double
          holds the value $x^u$ of the upper bound on the optimization
          variable $x$.
      options : dict, optional
          dictionary of control options (see ugo.initialize).

   .. function:: ugo.solve(eval_fgh)

      Find an approximation to the global minimizer of a given univariate
      function with a Lipschitz gradient in an interval.

      **Parameters:**

      eval_fgh : callable
          a user-defined function that must have the signature:

           ``f, g, h = eval_fgh(x)``

          The value of the objective function $f(x)$ and its first
          derivative $f'(x)$ evaluated at $x$ must be assigned
          to ``f`` and ``g`` respectively. In addition, if
          options['second_derivatives_available'] has been set to True
          when calling ``ugo.load``, the user must also assign the value of
          the second derivative $f''(x)$ to ``h``; it need not be
          assigned otherwise.

      **Returns:**

      x : double
          holds the value of the approximate global minimizer $x$
          after a successful call.
      f : double
          holds the value of the objective function $f(x)$ at the
          approximate global minimizer $x$ after a successful call.
      g : double
          holds the value of the gradient of the objective function
          $f'(x)$ at the approximate global minimizer $x$
          after a successful call.
      h : double
          holds the value of the second derivative of the objective function
          $f''(x)$ at the approximate global minimizer $x$ after
          a successful call.

   .. function:: [optional] ugo.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
           return status. Possible values are:

           * **0**

             The run was successful.

           * **-1**

             An allocation error occurred. A message indicating the
             offending array is written on unit options['error'], and the
             returned allocation status and a string containing the name
             of the offending array are held in inform['alloc_status']
             and inform['bad_alloc'] respectively.

           * **-2**

             A deallocation error occurred.  A message indicating the
             offending array is written on unit options['error'] and
             the returned allocation status and a string containing
             the name of the offending array are held in
             inform['alloc_status'] and inform['bad_alloc'] respectively.

           * **-7**

             The objective function appears to be unbounded from below.

           * **-18**

             Too many iterations have been performed. This may happen if
             options['maxit'] is too small, but may also be symptomatic
             of a badly scaled problem.

           * **-19**

             The CPU time limit has been reached. This may happen if
             options['cpu_time_limit'] is too small, but may also be
             symptomatic of a badly scaled problem.

           * **-40**

             The user has forced termination of the solver by removing
             the file named options['alive_file'] from unit
             options['alive_unit'].

          alloc_status : int
            the status of the last attempted internal array.
            allocation/deallocation
          bad_alloc : str
            the name of the array for which an internal array
            allocation/deallocation error occurred.
          iter : int
            the total number of iterations performed
          f_eval : int
            the total number of evaluations of the objective function.
          g_eval : int
            the total number of evaluations of the gradient of the objective
            function.
          h_eval : int
            the total number of evaluations of the Hessian of the objective
            function.
          time : dict
            dictionary containing tim information:
             total : float
               the total CPU time spent in the package.
             clock_total : float
               the total clock time spent in the package.

   .. function:: ugo.terminate()

      Deallocate all internal private storage.

example code
------------

.. include:: ../../src/ugo/Python/test_ugo.py
   :code: python

This example code is available in $GALAHAD/src/ugo/Python/test_ugo.py .
