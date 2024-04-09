.. index:: pair: struct; gls_finfo
.. _doxid-structgls__finfo:

gls_finfo structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_gls.h>
	
	struct gls_finfo {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`flag<doxid-structgls__finfo_1adf916204820072417ed73a32de1cefcf>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more<doxid-structgls__finfo_1a4628f2fb17af64608416810cc4e5a9d0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`size_factor<doxid-structgls__finfo_1a79b3c4b1d5426fdbd7eaf1b346be2c49>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`len_factorize<doxid-structgls__finfo_1a41ce64037dc7aae66b5a4ac582a2985a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`drop<doxid-structgls__finfo_1a56da9169fe581b834e971dee4997ecfd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`rank<doxid-structgls__finfo_1a6cfd95afd0afebd625b889fb6e58371c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stat<doxid-structgls__finfo_1a7d6f8a25e94209bd3ba29b2051ca4f08>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`ops<doxid-structgls__finfo_1af0a337c9f4d03e088123ec071639aad7>`;
	};
.. _details-structgls__finfo:

detailed documentation
----------------------

finfo derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structgls__finfo_1adf916204820072417ed73a32de1cefcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structgls__finfo_1a4628f2fb17af64608416810cc4e5a9d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more

More information on failure.

.. index:: pair: variable; size_factor
.. _doxid-structgls__finfo_1a79b3c4b1d5426fdbd7eaf1b346be2c49:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` size_factor

Number of words to hold factors.

.. index:: pair: variable; len_factorize
.. _doxid-structgls__finfo_1a41ce64037dc7aae66b5a4ac582a2985a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` len_factorize

Size for subsequent factorization.

.. index:: pair: variable; drop
.. _doxid-structgls__finfo_1a56da9169fe581b834e971dee4997ecfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` drop

Number of entries dropped.

.. index:: pair: variable; rank
.. _doxid-structgls__finfo_1a6cfd95afd0afebd625b889fb6e58371c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` rank

Estimated rank.

.. index:: pair: variable; stat
.. _doxid-structgls__finfo_1a7d6f8a25e94209bd3ba29b2051ca4f08:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stat

Status value after allocate failure.

.. index:: pair: variable; ops
.. _doxid-structgls__finfo_1af0a337c9f4d03e088123ec071639aad7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` ops

Number of operations in elimination.

