.. index:: pair: table; l2rt_inform_type
.. _doxid-structl2rt__inform__type:

l2rt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_l2rt.h>
	
	struct l2rt_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structl2rt__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structl2rt__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structl2rt__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structl2rt__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter_pass2<doxid-structl2rt__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biters<doxid-structl2rt__inform__type_1a0c5347be8391fbb23d728cebe0f3a5a8>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biter_min<doxid-structl2rt__inform__type_1a6fe473492218a28f33e53f014c741e81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`biter_max<doxid-structl2rt__inform__type_1aaa032644e73bb5bbc6092733db7f013b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structl2rt__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structl2rt__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structl2rt__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`r_norm<doxid-structl2rt__inform__type_1ae908410fabf891cfd89626c3605c38ca>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`Atr_norm<doxid-structl2rt__inform__type_1a0dc3a69b13123a76ec6ee7dd031eadff>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`biter_mean<doxid-structl2rt__inform__type_1a0c9f077f6c3bc52c519c2045c0578b22>`;
	};
.. _details-structl2rt__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structl2rt__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See :ref:`l2rt_solve_problem <doxid-galahad__l2rt_8h_1a53042b19cef3a62c34631b00111ce754>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structl2rt__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structl2rt__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structl2rt__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structl2rt__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; biters
.. _doxid-structl2rt__inform__type_1a0c5347be8391fbb23d728cebe0f3a5a8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biters

the total number of inner iterations performed

.. index:: pair: variable; biter_min
.. _doxid-structl2rt__inform__type_1a6fe473492218a28f33e53f014c741e81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biter_min

the smallest number of inner iterations performed during an outer iteration

.. index:: pair: variable; biter_max
.. _doxid-structl2rt__inform__type_1aaa032644e73bb5bbc6092733db7f013b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` biter_max

the largest number of inner iterations performed during an outer iteration

.. index:: pair: variable; obj
.. _doxid-structl2rt__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function

.. index:: pair: variable; multiplier
.. _doxid-structl2rt__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the multiplier, $\lambda = \mu + \sigma \|x\|^{p-2} * \sqrt{\|Ax-b\|^2 + \mu \|x\|^2}$

.. index:: pair: variable; x_norm
.. _doxid-structl2rt__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the Euclidean norm of $x$

.. index:: pair: variable; r_norm
.. _doxid-structl2rt__inform__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` r_norm

the Euclidean norm of $Ax-b$

.. index:: pair: variable; Atr_norm
.. _doxid-structl2rt__inform__type_1a0dc3a69b13123a76ec6ee7dd031eadff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` Atr_norm

the Euclidean norm of $A^T (Ax-b) + \lambda x$

.. index:: pair: variable; biter_mean
.. _doxid-structl2rt__inform__type_1a0c9f077f6c3bc52c519c2045c0578b22:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` biter_mean

the average number of inner iterations performed during an outer iteration

