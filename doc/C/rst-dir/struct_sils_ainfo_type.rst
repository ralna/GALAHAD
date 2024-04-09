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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`flag<doxid-structsils__ainfo__type_1adf916204820072417ed73a32de1cefcf>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more<doxid-structsils__ainfo__type_1a4628f2fb17af64608416810cc4e5a9d0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nsteps<doxid-structsils__ainfo__type_1aa2414080b021dbb9b56eeaeedec0ffa2>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrltot<doxid-structsils__ainfo__type_1a87d1fb870bb3dc94213c3adf550a7a3a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirtot<doxid-structsils__ainfo__type_1af19131d647173a929ed0594f33c46999>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrlnec<doxid-structsils__ainfo__type_1a812f6ee1b4bd452745b3876b084cd4af>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirnec<doxid-structsils__ainfo__type_1a2db255a81b52a84b3fec32d8c1a4ec17>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrladu<doxid-structsils__ainfo__type_1a881410b19c211fe38617f51dcb6709d7>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`niradu<doxid-structsils__ainfo__type_1a900738bbb8814829eb8e674c53a68ae3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ncmpa<doxid-structsils__ainfo__type_1a5e8b1d7aba05cfece368396811da24a3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`oor<doxid-structsils__ainfo__type_1afa0283d540a2a393c746b5eb05c3431c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dup<doxid-structsils__ainfo__type_1adc4cf3f551bb367858644559d69cfdf5>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxfrt<doxid-structsils__ainfo__type_1a1a03f6fb5030d0b1d6bf3703d8c24cc4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stat<doxid-structsils__ainfo__type_1a7d6f8a25e94209bd3ba29b2051ca4f08>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`faulty<doxid-structsils__ainfo__type_1a6dfcf1d95eca288e459a114d6418d6f1>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opsa<doxid-structsils__ainfo__type_1a5514c74cf880eb762c480fe9c94b45dd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opse<doxid-structsils__ainfo__type_1ae4d20d784bebe9b7f04a435f016e20d6>`;
	};
.. _details-structsils__ainfo__type:

detailed documentation
----------------------

ainfo derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__ainfo__type_1adf916204820072417ed73a32de1cefcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structsils__ainfo__type_1a4628f2fb17af64608416810cc4e5a9d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more

More information on failure.

.. index:: pair: variable; nsteps
.. _doxid-structsils__ainfo__type_1aa2414080b021dbb9b56eeaeedec0ffa2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nsteps

Number of elimination steps.

.. index:: pair: variable; nrltot
.. _doxid-structsils__ainfo__type_1a87d1fb870bb3dc94213c3adf550a7a3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrltot

Size for a without compression.

.. index:: pair: variable; nirtot
.. _doxid-structsils__ainfo__type_1af19131d647173a929ed0594f33c46999:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirtot

Size for iw without compression.

.. index:: pair: variable; nrlnec
.. _doxid-structsils__ainfo__type_1a812f6ee1b4bd452745b3876b084cd4af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrlnec

Size for a with compression.

.. index:: pair: variable; nirnec
.. _doxid-structsils__ainfo__type_1a2db255a81b52a84b3fec32d8c1a4ec17:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirnec

Size for iw with compression.

.. index:: pair: variable; nrladu
.. _doxid-structsils__ainfo__type_1a881410b19c211fe38617f51dcb6709d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrladu

Number of reals to hold factors.

.. index:: pair: variable; niradu
.. _doxid-structsils__ainfo__type_1a900738bbb8814829eb8e674c53a68ae3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` niradu

Number of integers to hold factors.

.. index:: pair: variable; ncmpa
.. _doxid-structsils__ainfo__type_1a5e8b1d7aba05cfece368396811da24a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ncmpa

Number of compresses.

.. index:: pair: variable; oor
.. _doxid-structsils__ainfo__type_1afa0283d540a2a393c746b5eb05c3431c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` oor

Number of indices out-of-range.

.. index:: pair: variable; dup
.. _doxid-structsils__ainfo__type_1adc4cf3f551bb367858644559d69cfdf5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dup

Number of duplicates.

.. index:: pair: variable; maxfrt
.. _doxid-structsils__ainfo__type_1a1a03f6fb5030d0b1d6bf3703d8c24cc4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxfrt

Forecast maximum front size.

.. index:: pair: variable; stat
.. _doxid-structsils__ainfo__type_1a7d6f8a25e94209bd3ba29b2051ca4f08:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stat

STAT value after allocate failure.

.. index:: pair: variable; faulty
.. _doxid-structsils__ainfo__type_1a6dfcf1d95eca288e459a114d6418d6f1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` faulty

legacy component, now not used

.. index:: pair: variable; opsa
.. _doxid-structsils__ainfo__type_1a5514c74cf880eb762c480fe9c94b45dd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opsa

Anticipated number of operations in assembly.

.. index:: pair: variable; opse
.. _doxid-structsils__ainfo__type_1ae4d20d784bebe9b7f04a435f016e20d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opse

Anticipated number of operations in elimination.

