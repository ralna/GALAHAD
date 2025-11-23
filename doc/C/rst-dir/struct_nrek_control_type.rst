.. index:: pair: table; nrek_control_type
.. _doxid-structnrek__control__type:

nrek_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_nrek.h>
	
	struct nrek_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structnrek__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structnrek__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structnrek__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structnrek__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`eks_max<doxid-structnrek__control__type_eks_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`it_max<doxid-structnrek__control__type_it_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`f<doxid-structnrek__control__type_f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`increase<doxid-structnrek__control__type_increase>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_residual<doxid-structnrek__control__type_stop_residual>`;
		bool :ref:`reorthogonalize<doxid-structnrek__control__type_reorthogonalize>`;
		bool :ref:`s_version_52<doxid-structnrek__control__type_s_version_52>`;
		bool :ref:`perturb_c<doxid-structnrek__control__type_perturb_c>`;
		bool :ref:`stop_check_all_orders<doxid-structnrek__control__type_stop_check_all_orders>`;
		bool :ref:`new_weight<doxid-structnrek__control__type_new_weight>`;
		bool :ref:`new_values<doxid-structnrek__control__type_new_values>`;
		bool :ref:`space_critical<doxid-structnrek__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structnrek__control__type_deallocate_error_fatal>`;
		char :ref:`linear_solver<doxid-structnrek__control__type_linear_solver>`[31];
		char :ref:`linear_solver_for_s<doxid-structnrek__control__type_linear_solver_for_s>`[31];
		char :ref:`prefix<doxid-structnrek__control__type_prefix>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structnrek__control__type_sls_control>`;
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_s_control<doxid-structnrek__control__type_sls_s_control>`;
		struct :ref:`rqs_control_type<doxid-structrqs__control__type>` :ref:`rqs_control<doxid-structnrek__control__type_rqs_control>`;
	};
.. _details-structnrek__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structnrek__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structnrek__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structnrek__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structnrek__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; eks_max
.. _doxid-structnrek__control__type_eks_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` eks_max

maximum dimension of the extended Krylov space employed. If a negative value is given, the value 100 will be used instead

.. index:: pair: variable; it_max
.. _doxid-structnrek__control__type_it_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` it_max

the maximum number of iterations allowed.  If a negative value is given, the value 100 will be used instead

.. index:: pair: variable; f
.. _doxid-structnrek__control__type_f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` f

the value of $f$ in the objective function. This value has no effect on the computed $x$, and takes the value 0.0 by default

.. index:: pair: variable; increase
.. _doxid-structnrek__control__type_increase:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` increase

the value of the increase factor for a suggested subsequent regularization weight, see control['next_weight']. The suggested weight will be ``increase`` times thecurrent weight.

.. index:: pair: variable; stop_residual
.. _doxid-structnrek__control__type_stop_residual:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_residual
          
the value of the stopping tolerance used by the algorithm. The iteration stops as soon as $x$ and $\lambda$ are found to satisfy $\| ( H + \lambda S ) x + c \| <$ ``stop_residual`` $\times \max( 1, \|c\| )$.

.. index:: pair: variable; reorthogonalize
.. _doxid-structnrek__control__type_reorthogonalize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool reorthogonalize
          
should be set to true if the generated basis of the extended-Krylov subspace is to be reorthogonalized at every iteration. This can be very expensive, and is generally not warranted

.. index:: pair: variable; s_version_52
.. _doxid-structnrek__control__type_s_version_52:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool s_version_52
          
should be set to true if Algorithm 5.2 in the paper is used to generate the extended Krylov space recurrences when a non-unit $S$ is given, and false if those from Algorithm B.3 ares used instead. In practice, there is very little difference in performance and accuracy

.. index:: pair: variable; perturb_c
.. _doxid-structnrek__control__type_perturb_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool perturb_c
          
should be set to true if the user wishes to make a tiny pseudo-random perturbations to the components of the term $c$ to try to protect from the so-called (probability zero) "hard" case. Perturbations are generally not needed, and should only be used in very exceptional cases

.. index:: pair: variable; stop_check_all_orders
.. _doxid-structnrek__control__type_stop_check_all_orders:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool stop_check_all_orders

should be set to true if the algorithm checks for termination for each new member of the extended Krylov space. Such checks incur some extra cost, and experience shows that testing every second member is sufficient

.. index:: pair: variable; new_weight
.. _doxid-structnrek__control__type_new_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool new_weight
          
should be set to true if the call retains the previous $H$, $S$ and $c$, but with a new, larger weight

.. index:: pair: variable; new_values
.. _doxid-structnrek__control__type_new_values:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool new_values

should be set to true if the any of the values of $H$, $S$ and $c$ has changed since a previous call

.. index:: pair: variable; space_critical
.. _doxid-structnrek__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structnrek__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; linear_solver
.. _doxid-structnrek__control__type_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char linear_solver[31]

the name of the linear equation solver used to solve any symmetric positive-definite linear system involving $H$ that might arise. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', ssids, 'pardiso', 'wsmp', 'sytr', 'potr' and 'pbtr' although only 'sytr', 'potr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; definite_linear_solver_for_s
.. _doxid-structnrek__control__type_linear_solver_for_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char linear_solver_for_s[31]

the name of the linear equation solver used to solve any symmetric positive-definite linear system involving the optional $S$ that might arise. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', ssids, 'pardiso', 'wsmp', 'sytr', 'potr' and 'pbtr' although only 'sytr', 'potr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structnrek__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structnrek__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the Cholesky factorization and solution (see sls_c documentation)

.. index:: pair: variable; sls_s_control
.. _doxid-structnrek__control__type_sls_s_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_s_control

control parameters for the Cholesky factorization and solution when applied to $S$ (see sls_c documentation)

.. index:: pair: variable; rqs_control
.. _doxid-structnrek__control__type_rqs_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`rqs_control_type<doxid-structrqs__control__type>` rqs_control

control parameters for the diagonal subproblem solve (see rqs_c documentation)

