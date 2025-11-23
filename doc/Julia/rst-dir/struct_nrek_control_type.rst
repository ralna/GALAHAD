.. index:: pair: table; nrek_control_type
.. _doxid-structnrek__control__type:

nrek_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct nrek_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          eks_max::INT
          it_max::INT
          f::T
          increase::T
          stop_residual::T
          reorthogonalize::Bool
          s_version_52::Bool
          perturb_c::Bool
          stop_check_all_orders::Bool
          new_weight::Bool
          new_values::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          linear_solver::NTuple{31,Cchar}
          linear_solver_for_s::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
          sls_s_control::sls_control_type{T,INT}
          rqs_control::ir_control_type{T,INT}

.. _details-structnrek__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structnrek__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structnrek__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structnrek__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structnrek__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; eks_max
.. _doxid-structnrek__control__type_eks_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT eks_max

maximum dimension of the extended Krylov space employed. If a negative value is given, the value 100 will be used instead

.. index:: pair: variable; it_max
.. _doxid-structnrek__control__type_it_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT it_max

the maximum number of iterations allowed.  If a negative value is given, the value 100 will be used instead


.. index:: pair: variable; f
.. _doxid-structnrek__control__type_f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T f

the value of $f$ in the objective function. This value has no effect on the computed $x$, and takes the value 0.0 by default

.. index:: pair: variable; increase
.. _doxid-structnrek__control__type_increase:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T increase

the value of the increase factor for a suggested subsequent regularization weight, see control['next_weight']. The suggested weight will be ``increase`` times the current weight

.. index:: pair: variable; stop_residual
.. _doxid-structnrek__control__type_stop_residual:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_residual
          
the value of the stopping tolerance used by the algorithm. The iteration stops as soon as $x$ and $\lambda$ are found to satisfy $\| ( H + \lambda S ) x + c \| <$ ``stop_residual`` $\times \max( 1, \|c\| )$

.. index:: pair: variable; reorthogonalize
.. _doxid-structnrek__control__type_reorthogonalize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool reorthogonalize
          
should be set to true if the generated basis of the extended-Krylov subspace is to be reorthogonalized at every iteration. This can be very expensive, and is generally not warranted

.. index:: pair: variable; s_version_52
.. _doxid-structnrek__control__type_s_version_52:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool s_version_52
          
should be set to true if Algorithm 5.2 in the paper is used to generate the extended Krylov space recurrences when a non-unit $S$ is given, and false if those from Algorithm B.3 ares used instead. In practice, there is very little difference in performance and accuracy

.. index:: pair: variable; perturb_c
.. _doxid-structnrek__control__type_perturb_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perturb_c
          
should be set to true if the user wishes to make a tiny pseudo-random perturbations to the components of the term $c$ to try to protect from the so-called (probability zero) "hard" case. Perturbations are generally not needed, and should only be used in very exceptional cases

.. index:: pair: variable; stop_check_all_orders
.. _doxid-structnrek__control__type_stop_check_all_orders:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool stop_check_all_orders

should be set to true if the algorithm checks for termination for each new member of the extended Krylov space. Such checks incur some extra cost, and experience shows that testing every second member is sufficient

.. index:: pair: variable; new_weight
.. _doxid-structnrek__control__type_new_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool new_weight
          
should be set to true if the call retains the previous $H$, $S$ and $c$, but with a new, smaller weight

.. index:: pair: variable; new_values
.. _doxid-structnrek__control__type_new_values:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool new_values

should be set to true if the any of the values of $H$, $S$ and $c$ has changed since a previous call

.. index:: pair: variable; space_critical
.. _doxid-structnrek__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structnrek__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; linear_solver
.. _doxid-structnrek__control__type_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char linear_solver[31]

the name of the linear equation solver used to solve any symmetric positive-definite linear system involving $H$ that might arise. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', ssids, 'pardiso', 'wsmp', 'sytr', 'potr' and 'pbtr' although only 'sytr', 'potr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; linear_solver_for_s
.. _doxid-structnrek__control__type_linear_solver_for_s:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char linear_solver_for_s[31]

the name of the linear equation solver used to solve any symmetric positive-definite linear system involving the optional $S$ that might arise. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', ssids, 'pardiso', 'wsmp', 'sytr', 'potr' and 'pbtr' although only 'sytr', 'potr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structnrek__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structnrek__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the Cholesky factorization and solution 
involving $H$ (see sls_c documentation)

.. index:: pair: variable; sls_s_control
.. _doxid-structnrek__control__type_sls_s_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_s_control

control parameters for the Cholesky factorization and solution 
involving $S$ (see sls_c documentation)

.. index:: pair: variable; rqs_control
.. _doxid-structnrek__control__type_rqs_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_control_type<doxid-structir__control__type>` rqs_control

control parameters for the solution of diagonal norm-regularization subproblems (see rqs_c documentation)

