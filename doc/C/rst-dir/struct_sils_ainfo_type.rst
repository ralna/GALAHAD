.. index:: pair: struct; sils_ainfo_type
.. _doxid-structsils__ainfo__type:

sils_ainfo_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sils.h>
	
	struct sils_ainfo_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`flag<doxid-structsils__ainfo__type_flag>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more<doxid-structsils__ainfo__type_more>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nsteps<doxid-structsils__ainfo__type_nsteps>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrltot<doxid-structsils__ainfo__type_nrltot>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirtot<doxid-structsils__ainfo__type_nirtot>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrlnec<doxid-structsils__ainfo__type_nrlnec>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirnec<doxid-structsils__ainfo__type_nirnec>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrladu<doxid-structsils__ainfo__type_nrladu>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`niradu<doxid-structsils__ainfo__type_niradu>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ncmpa<doxid-structsils__ainfo__type_ncmpa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`oor<doxid-structsils__ainfo__type_oor>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dup<doxid-structsils__ainfo__type_dup>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxfrt<doxid-structsils__ainfo__type_maxfrt>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stat<doxid-structsils__ainfo__type_stat>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`faulty<doxid-structsils__ainfo__type_faulty>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opsa<doxid-structsils__ainfo__type_opsa>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opse<doxid-structsils__ainfo__type_opse>`;
	};
.. _details-structsils__ainfo__type:

detailed documentation
----------------------

ainfo derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__ainfo__type_flag:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structsils__ainfo__type_more:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more

More information on failure.

.. index:: pair: variable; nsteps
.. _doxid-structsils__ainfo__type_nsteps:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nsteps

Number of elimination steps.

.. index:: pair: variable; nrltot
.. _doxid-structsils__ainfo__type_nrltot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrltot

Size for a without compression.

.. index:: pair: variable; nirtot
.. _doxid-structsils__ainfo__type_nirtot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirtot

Size for iw without compression.

.. index:: pair: variable; nrlnec
.. _doxid-structsils__ainfo__type_nrlnec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrlnec

Size for a with compression.

.. index:: pair: variable; nirnec
.. _doxid-structsils__ainfo__type_nirnec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirnec

Size for iw with compression.

.. index:: pair: variable; nrladu
.. _doxid-structsils__ainfo__type_nrladu:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrladu

Number of reals to hold factors.

.. index:: pair: variable; niradu
.. _doxid-structsils__ainfo__type_niradu:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` niradu

Number of integers to hold factors.

.. index:: pair: variable; ncmpa
.. _doxid-structsils__ainfo__type_ncmpa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ncmpa

Number of compresses.

.. index:: pair: variable; oor
.. _doxid-structsils__ainfo__type_oor:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` oor

Number of indices out-of-range.

.. index:: pair: variable; dup
.. _doxid-structsils__ainfo__type_dup:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dup

Number of duplicates.

.. index:: pair: variable; maxfrt
.. _doxid-structsils__ainfo__type_maxfrt:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxfrt

Forecast maximum front size.

.. index:: pair: variable; stat
.. _doxid-structsils__ainfo__type_stat:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stat

STAT value after allocate failure.

.. index:: pair: variable; faulty
.. _doxid-structsils__ainfo__type_faulty:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` faulty

legacy component, now not used

.. index:: pair: variable; opsa
.. _doxid-structsils__ainfo__type_opsa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opsa

Anticipated number of operations in assembly.

.. index:: pair: variable; opse
.. _doxid-structsils__ainfo__type_opse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opse

Anticipated number of operations in elimination.

