.. index:: pair: struct; trb_inform_type
.. _doxid-structtrb__inform__type:

trb_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trb.h>
	
	struct trb_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structtrb__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structtrb__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structtrb__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structtrb__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structtrb__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structtrb__inform__type_1a7a1029142a22f3e2a1963c3428276849>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`f_eval<doxid-structtrb__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`g_eval<doxid-structtrb__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`h_eval<doxid-structtrb__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`n_free<doxid-structtrb__inform__type_1a1d6107630beebe9a594b0588ac88016f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_max<doxid-structtrb__inform__type_1a97dadabf3b7bdf921c4dcd1f43129f05>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structtrb__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		int64_t :ref:`max_entries_factors<doxid-structtrb__inform__type_1a177e429e737cfa2cd3df051a65fcfb68>`;
		int64_t :ref:`factorization_integer<doxid-structtrb__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`;
		int64_t :ref:`factorization_real<doxid-structtrb__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structtrb__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structtrb__inform__type_1acb02a4d1ae275a55874bb9897262b1fe>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`radius<doxid-structtrb__inform__type_1a72757b6410f755f008e2fb6d711b61be>`;
		struct :ref:`trb_time_type<doxid-structtrb__time__type>` :ref:`time<doxid-structtrb__inform__type_1ad1c7da271c9b5bab69d9e9cc52d2cf5b>`;
		struct :ref:`trs_inform_type<doxid-structtrs__inform__type>` :ref:`trs_inform<doxid-structtrb__inform__type_1aa7996c925462c655f2b3dd5a5da22c21>`;
		struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` :ref:`gltr_inform<doxid-structtrb__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46>`;
		struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` :ref:`psls_inform<doxid-structtrb__inform__type_1a57ca5ed37882eb917736f845d3cdb8ee>`;
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>` :ref:`lms_inform<doxid-structtrb__inform__type_1a6428cf213f8c899aa1bfb1fc3d24f37d>`;
		struct :ref:`lms_inform_type<doxid-structlms__inform__type>` :ref:`lms_inform_prec<doxid-structtrb__inform__type_1a2040147e726e4ad18ef6d81d8339644e>`;
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>` :ref:`sha_inform<doxid-structtrb__inform__type_1a196d9da91c7ed4a67aa6e009e336e101>`;
	};
.. _details-structtrb__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structtrb__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See TRB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structtrb__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structtrb__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structtrb__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structtrb__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

the total number of CG iterations performed

.. index:: pair: variable; cg_maxit
.. _doxid-structtrb__inform__type_1a7a1029142a22f3e2a1963c3428276849:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

the maximum number of CG iterations allowed per iteration

.. index:: pair: variable; f_eval
.. _doxid-structtrb__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structtrb__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structtrb__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; n_free
.. _doxid-structtrb__inform__type_1a1d6107630beebe9a594b0588ac88016f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` n_free

the number of variables that are free from their bounds

.. index:: pair: variable; factorization_max
.. _doxid-structtrb__inform__type_1a97dadabf3b7bdf921c4dcd1f43129f05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structtrb__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structtrb__inform__type_1a177e429e737cfa2cd3df051a65fcfb68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structtrb__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structtrb__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; obj
.. _doxid-structtrb__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function at the best estimate of the solution determined by TRB_solve

.. index:: pair: variable; norm_pg
.. _doxid-structtrb__inform__type_1acb02a4d1ae275a55874bb9897262b1fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by TRB_solve

.. index:: pair: variable; radius
.. _doxid-structtrb__inform__type_1a72757b6410f755f008e2fb6d711b61be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` radius

the current value of the trust-region radius

.. index:: pair: variable; time
.. _doxid-structtrb__inform__type_1ad1c7da271c9b5bab69d9e9cc52d2cf5b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trb_time_type<doxid-structtrb__time__type>` time

timings (see above)

.. index:: pair: variable; trs_inform
.. _doxid-structtrb__inform__type_1aa7996c925462c655f2b3dd5a5da22c21:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trs_inform_type<doxid-structtrs__inform__type>` trs_inform

inform parameters for TRS

.. index:: pair: variable; gltr_inform
.. _doxid-structtrb__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

inform parameters for GLTR

.. index:: pair: variable; psls_inform
.. _doxid-structtrb__inform__type_1a57ca5ed37882eb917736f845d3cdb8ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; lms_inform
.. _doxid-structtrb__inform__type_1a6428cf213f8c899aa1bfb1fc3d24f37d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform

inform parameters for LMS

.. index:: pair: variable; lms_inform_prec
.. _doxid-structtrb__inform__type_1a2040147e726e4ad18ef6d81d8339644e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform_prec

inform parameters for LMS used for preconditioning

.. index:: pair: variable; sha_inform
.. _doxid-structtrb__inform__type_1a196d9da91c7ed4a67aa6e009e336e101:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sha_inform_type<doxid-structsha__inform__type>` sha_inform

inform parameters for SHA

