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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`flag<doxid-structsils__sinfo__type_1adf916204820072417ed73a32de1cefcf>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stat<doxid-structsils__sinfo__type_1a7d6f8a25e94209bd3ba29b2051ca4f08>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cond<doxid-structsils__sinfo__type_1a006d728493fbea61aabf1e6229e34185>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cond2<doxid-structsils__sinfo__type_1ae6b598341b9634df4e446be3de0ed839>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`berr<doxid-structsils__sinfo__type_1ad2150d4466031c9e63281a146e5ccd03>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`berr2<doxid-structsils__sinfo__type_1ade02e126e145400e9ead3c3f3bc06dab>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`error<doxid-structsils__sinfo__type_1a2b7e3bae2c2111a08302ba1dc7f14cef>`;
	};
.. _details-structsils__sinfo__type:

detailed documentation
----------------------

sinfo derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__sinfo__type_1adf916204820072417ed73a32de1cefcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` flag

Flags success or failure case.

.. index:: pair: variable; stat
.. _doxid-structsils__sinfo__type_1a7d6f8a25e94209bd3ba29b2051ca4f08:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stat

STAT value after allocate failure.

.. index:: pair: variable; cond
.. _doxid-structsils__sinfo__type_1a006d728493fbea61aabf1e6229e34185:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cond

Condition number of matrix (category 1 eqs)

.. index:: pair: variable; cond2
.. _doxid-structsils__sinfo__type_1ae6b598341b9634df4e446be3de0ed839:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cond2

Condition number of matrix (category 2 eqs)

.. index:: pair: variable; berr
.. _doxid-structsils__sinfo__type_1ad2150d4466031c9e63281a146e5ccd03:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` berr

Backward error for the system (category 1 eqs)

.. index:: pair: variable; berr2
.. _doxid-structsils__sinfo__type_1ade02e126e145400e9ead3c3f3bc06dab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` berr2

Backward error for the system (category 2 eqs)

.. index:: pair: variable; error
.. _doxid-structsils__sinfo__type_1a2b7e3bae2c2111a08302ba1dc7f14cef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` error

Estimate of forward error.

