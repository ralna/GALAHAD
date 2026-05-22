.. index:: pair: struct; bnls_inform_type
.. _doxid-structbnls__inform__type:

bnls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bnls.h>
	
	struct bnls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structbnls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structbnls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structbnls__inform__type_bad_alloc>`[81];
		char :ref:`bad_eval<doxid-structbnls__inform__type_bad_eval>`[13];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structbnls__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`inner_iter<doxid-structbnls__inform__type_inner_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`r_eval<doxid-structbnls__inform__type_r_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`jr_eval<doxid-structbnls__inform__type_jr_eval>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structbnls__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_r<doxid-structbnls__inform__type_norm_r>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_g<doxid-structbnls__inform__type_norm_g>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structbnls__inform__type_norm_pg>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structbnls__inform__type_weight>`;
		struct :ref:`bnls_time_type<doxid-structbnls__time__type>` :ref:`time<doxid-structbnls__inform__type_time>`;
		struct :ref:`blls_inform_type<doxid-structblls__inform__type>` :ref:`blls_inform<doxid-structbnls__inform__type_blls_inform>`;
		struct :ref:`bllsb_inform_type<doxid-structbllsb__inform__type>` :ref:`bllsb_inform<doxid-structbnls__inform__type_bllsb_inform>`;
	};
.. _details-structbnls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbnls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See BNLS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structbnls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbnls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structbnls__inform__type_bad_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structbnls__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; inner_iter
.. _doxid-structbnls__inform__type_inner_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` inner_iter

the total number of inner (projected gradient and/or interior-point) iterations performed

.. index:: pair: variable; r_eval
.. _doxid-structbnls__inform__type_r_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` r_eval

the total number of evaluations of the residual function $r(x)$

.. index:: pair: variable; jr_eval
.. _doxid-structbnls__inform__type_jr_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` jr_eval

the total number of evaluations of the Jacobian $J_r(x)$ of $r(x)$

.. index:: pair: variable; obj
.. _doxid-structbnls__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function $\frac{1}{2}\|r(x)\|^2_W$ at the best estimate the solution, $x$, determined by BNLS_solve

.. index:: pair: variable; norm_r
.. _doxid-structbnls__inform__type_norm_r:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_r

the norm of the residual $\|r(x)\|_W$ at the best estimate of the solution $x$, determined by BNLS_solve

.. index:: pair: variable; norm_g
.. _doxid-structbnls__inform__type_norm_g:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_g

the norm of the gradient of $\|r(x)\|_W$ of the objective function at the best estimate, $x$, of the solution determined by BNLS_solve

.. index:: pair: variable; norm_pg
.. _doxid-structbnls__inform__type_norm_pg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

the norm of the projected gradient $\|P[x - J_r^T(x) W r(x)] - x\|_2$ of the residual function at the best estimate, x, of the solution determined by BNLS_solve

.. index:: pair: variable; weight
.. _doxid-structbnls__inform__type_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the final regularization weight used

.. index:: pair: variable; time
.. _doxid-structbnls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bnls_time_type<doxid-structbnls__time__type>` time

timings (see above)

.. index:: pair: variable; blls_inform
.. _doxid-structbnls__inform__type_blls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`blls_inform_type<doxid-structblls__inform__type>` blls_inform

inform parameters for BLLS

.. index:: pair: variable; bllsb_inform
.. _doxid-structbnls__inform__type_bllsb_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bllsb_inform_type<doxid-structbllsb__inform__type>` bllsb_inform

inform parameters for BLLSB

