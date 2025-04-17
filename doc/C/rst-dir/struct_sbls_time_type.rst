.. index:: pair: table; sbls_time_type
.. _doxid-structsbls__time__type:

sbls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sbls.h>
	
	struct sbls_time_type {
		// fields
	
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`total<doxid-structsbls__time__type_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`form<doxid-structsbls__time__type_form>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`factorize<doxid-structsbls__time__type_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`apply<doxid-structsbls__time__type_apply>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structsbls__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_form<doxid-structsbls__time__type_clock_form>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structsbls__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_apply<doxid-structsbls__time__type_clock_apply>`;
	};
.. _details-structsbls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structsbls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` total

total cpu time spent in the package

.. index:: pair: variable; form
.. _doxid-structsbls__time__type_form:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` form

cpu time spent forming the preconditioner $K_G$

.. index:: pair: variable; factorize
.. _doxid-structsbls__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` factorize

cpu time spent factorizing $K_G$

.. index:: pair: variable; apply
.. _doxid-structsbls__time__type_apply:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` apply

cpu time spent solving linear systems inolving $K_G$

.. index:: pair: variable; clock_total
.. _doxid-structsbls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_form
.. _doxid-structsbls__time__type_clock_form:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_form

clock time spent forming the preconditioner $K_G$

.. index:: pair: variable; clock_factorize
.. _doxid-structsbls__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

clock time spent factorizing $K_G$

.. index:: pair: variable; clock_apply
.. _doxid-structsbls__time__type_clock_apply:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_apply

clock time spent solving linear systems inolving $K_G$

