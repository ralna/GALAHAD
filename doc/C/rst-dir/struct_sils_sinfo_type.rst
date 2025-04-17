.. index:: pair: struct; sils_sinfo_type
.. _doxid-structsils__sinfo__type:

sils_sinfo_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sils.h>
	
	struct sils_sinfo_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`flag<doxid-structsils__sinfo__type_flag>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stat<doxid-structsils__sinfo__type_stat>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cond<doxid-structsils__sinfo__type_cond>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cond2<doxid-structsils__sinfo__type_cond2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`berr<doxid-structsils__sinfo__type_berr>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`berr2<doxid-structsils__sinfo__type_berr2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`error<doxid-structsils__sinfo__type_error>`;
	};
.. _details-structsils__sinfo__type:

detailed documentation
----------------------

sinfo derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__sinfo__type_flag:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` flag

Flags success or failure case.

.. index:: pair: variable; stat
.. _doxid-structsils__sinfo__type_stat:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stat

STAT value after allocate failure.

.. index:: pair: variable; cond
.. _doxid-structsils__sinfo__type_cond:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cond

Condition number of matrix (category 1 eqs)

.. index:: pair: variable; cond2
.. _doxid-structsils__sinfo__type_cond2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cond2

Condition number of matrix (category 2 eqs)

.. index:: pair: variable; berr
.. _doxid-structsils__sinfo__type_berr:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` berr

Backward error for the system (category 1 eqs)

.. index:: pair: variable; berr2
.. _doxid-structsils__sinfo__type_berr2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` berr2

Backward error for the system (category 2 eqs)

.. index:: pair: variable; error
.. _doxid-structsils__sinfo__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` error

Estimate of forward error.

