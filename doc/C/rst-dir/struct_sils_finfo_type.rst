.. index:: pair: struct; sils_finfo_type
.. _doxid-structsils__finfo__type:

sils_finfo_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sils.h>
	
	struct sils_finfo_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`flag<doxid-structsils__finfo__type_flag>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more<doxid-structsils__finfo__type_more>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxfrt<doxid-structsils__finfo__type_maxfrt>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nebdu<doxid-structsils__finfo__type_nebdu>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrlbdu<doxid-structsils__finfo__type_nrlbdu>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirbdu<doxid-structsils__finfo__type_nirbdu>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrltot<doxid-structsils__finfo__type_nrltot>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirtot<doxid-structsils__finfo__type_nirtot>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nrlnec<doxid-structsils__finfo__type_nrlnec>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nirnec<doxid-structsils__finfo__type_nirnec>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ncmpbr<doxid-structsils__finfo__type_ncmpbr>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ncmpbi<doxid-structsils__finfo__type_ncmpbi>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ntwo<doxid-structsils__finfo__type_ntwo>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`neig<doxid-structsils__finfo__type_neig>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`delay<doxid-structsils__finfo__type_delay>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`signc<doxid-structsils__finfo__type_signc>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nstatic<doxid-structsils__finfo__type_nstatic>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`modstep<doxid-structsils__finfo__type_modstep>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`rank<doxid-structsils__finfo__type_rank>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stat<doxid-structsils__finfo__type_stat>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`faulty<doxid-structsils__finfo__type_faulty>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`step<doxid-structsils__finfo__type_step>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opsa<doxid-structsils__finfo__type_opsa>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opse<doxid-structsils__finfo__type_opse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`opsb<doxid-structsils__finfo__type_opsb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`maxchange<doxid-structsils__finfo__type_maxchange>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`smin<doxid-structsils__finfo__type_smin>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`smax<doxid-structsils__finfo__type_smax>`;
	};
.. _details-structsils__finfo__type:

detailed documentation
----------------------

finfo derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structsils__finfo__type_flag:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structsils__finfo__type_more:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more

More information on failure.

.. index:: pair: variable; maxfrt
.. _doxid-structsils__finfo__type_maxfrt:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxfrt

Largest front size.

.. index:: pair: variable; nebdu
.. _doxid-structsils__finfo__type_nebdu:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nebdu

Number of entries in factors.

.. index:: pair: variable; nrlbdu
.. _doxid-structsils__finfo__type_nrlbdu:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrlbdu

Number of reals that hold factors.

.. index:: pair: variable; nirbdu
.. _doxid-structsils__finfo__type_nirbdu:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirbdu

Number of integers that hold factors.

.. index:: pair: variable; nrltot
.. _doxid-structsils__finfo__type_nrltot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrltot

Size for a without compression.

.. index:: pair: variable; nirtot
.. _doxid-structsils__finfo__type_nirtot:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirtot

Size for iw without compression.

.. index:: pair: variable; nrlnec
.. _doxid-structsils__finfo__type_nrlnec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nrlnec

Size for a with compression.

.. index:: pair: variable; nirnec
.. _doxid-structsils__finfo__type_nirnec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nirnec

Size for iw with compression.

.. index:: pair: variable; ncmpbr
.. _doxid-structsils__finfo__type_ncmpbr:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ncmpbr

Number of compresses of real data.

.. index:: pair: variable; ncmpbi
.. _doxid-structsils__finfo__type_ncmpbi:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ncmpbi

Number of compresses of integer data.

.. index:: pair: variable; ntwo
.. _doxid-structsils__finfo__type_ntwo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ntwo

Number of 2x2 pivots.

.. index:: pair: variable; neig
.. _doxid-structsils__finfo__type_neig:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` neig

Number of negative eigenvalues.

.. index:: pair: variable; delay
.. _doxid-structsils__finfo__type_delay:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` delay

Number of delayed pivots (total)

.. index:: pair: variable; signc
.. _doxid-structsils__finfo__type_signc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` signc

Number of pivot sign changes when control.pivoting=3.

.. index:: pair: variable; nstatic
.. _doxid-structsils__finfo__type_nstatic:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nstatic

Number of static pivots chosen.

.. index:: pair: variable; modstep
.. _doxid-structsils__finfo__type_modstep:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` modstep

First pivot modification when control.pivoting=4.

.. index:: pair: variable; rank
.. _doxid-structsils__finfo__type_rank:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` rank

Rank of original factorization.

.. index:: pair: variable; stat
.. _doxid-structsils__finfo__type_stat:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stat

STAT value after allocate failure.

.. index:: pair: variable; faulty
.. _doxid-structsils__finfo__type_faulty:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` faulty

legacy component, now not used

.. index:: pair: variable; step
.. _doxid-structsils__finfo__type_step:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` step

legacy component, now not used

.. index:: pair: variable; opsa
.. _doxid-structsils__finfo__type_opsa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opsa



.. _doxid-galahad__sils_8h_1autotoc_md0:

operations in assembly
~~~~~~~~~~~~~~~~~~~~~~

.. index:: pair: variable; opse
.. _doxid-structsils__finfo__type_opse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opse

number of operations in elimination

.. index:: pair: variable; opsb
.. _doxid-structsils__finfo__type_opsb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` opsb

Additional number of operations for BLAS.

.. index:: pair: variable; maxchange
.. _doxid-structsils__finfo__type_maxchange:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` maxchange

Largest control.pivoting=4 modification.

.. index:: pair: variable; smin
.. _doxid-structsils__finfo__type_smin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` smin

Minimum scaling factor.

.. index:: pair: variable; smax
.. _doxid-structsils__finfo__type_smax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` smax

Maximum scaling factor.

