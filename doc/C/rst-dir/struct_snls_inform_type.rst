.. index:: pair: struct; snls_inform_type
.. _doxid-structsnls__inform__type:

snls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_snls.h>
	
	struct snls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structsnls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structsnls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structsnls__inform__type_bad_alloc>`[81];
		char :ref:`bad_eval<doxid-structsnls__inform__type_bad_eval>`[13];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structsnls__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`inner_iter<doxid-structsnls__inform__type_inner_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`r_eval<doxid-structsnls__inform__type_r_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`jr_eval<doxid-structsnls__inform__type_jr_eval>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structsnls__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_r<doxid-structsnls__inform__type_norm_r>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_g<doxid-structsnls__inform__type_norm_g>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structsnls__inform__type_norm_pg>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structsnls__inform__type_weight>`;
		struct :ref:`snls_time_type<doxid-structsnls__time__type>` :ref:`time<doxid-structsnls__inform__type_time>`;
		struct :ref:`slls_inform_type<doxid-structslls__inform__type>` :ref:`slls_inform<doxid-structsnls__inform__type_slls_inform>`;
		struct :ref:`sllsb_inform_type<doxid-structsllsb__inform__type>` :ref:`sllsb_inform<doxid-structsnls__inform__type_sllsb_inform>`;
	};
.. _details-structsnls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsnls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See SNLS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structsnls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structsnls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structsnls__inform__type_bad_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structsnls__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; inner_iter
.. _doxid-structsnls__inform__type_inner_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` inner_iter

the total number of inner (projected gradient and/or interior-point) iterations performed

.. index:: pair: variable; r_eval
.. _doxid-structsnls__inform__type_r_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` r_eval

the total number of evaluations of the residual function $r(x)$

.. index:: pair: variable; jr_eval
.. _doxid-structsnls__inform__type_jr_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` jr_eval

the total number of evaluations of the Jacobian $J(x)$ of $r(x)$

.. index:: pair: variable; obj
.. _doxid-structsnls__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function $\frac{1}{2}\|r(x)\|^2_W$ at the best estimate the solution, $x$, determined by SNLS_solve

.. index:: pair: variable; norm_r
.. _doxid-structsnls__inform__type_norm_r:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_r

the norm of the residual $\|r(x)\|_W$ at the best estimate of the solution $x$, determined by SNLS_solve

.. index:: pair: variable; norm_g
.. _doxid-structsnls__inform__type_norm_g:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_g

the norm of the gradient of $\|r(x)\|_W$ of the objective function at the best estimate, $x$, of the solution determined by SNLS_solve

.. index:: pair: variable; norm_pg
.. _doxid-structsnls__inform__type_norm_pg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

the norm of the projected gradient $\|P[x - J_r^T(x) W r(x)] - x\|_2$ of the residual function at the best estimate, x, of the solution determined by SNLS_solve

.. index:: pair: variable; weight
.. _doxid-structsnls__inform__type_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the final regularization weight used

.. index:: pair: variable; time
.. _doxid-structsnls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`snls_time_type<doxid-structsnls__time__type>` time

timings (see above)

.. index:: pair: variable; slls_inform
.. _doxid-structsnls__inform__type_slls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`slls_inform_type<doxid-structslls__inform__type>` slls_inform

inform parameters for SLLS

.. index:: pair: variable; sllsb_inform
.. _doxid-structsnls__inform__type_sllsb_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sllsb_inform_type<doxid-structsllsb__inform__type>` sllsb_inform

inform parameters for SLLSB

