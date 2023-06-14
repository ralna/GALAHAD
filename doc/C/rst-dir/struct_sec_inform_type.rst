.. index:: pair: table; sec_inform_type
.. _doxid-structsec__inform__type:

sec_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. _details-structsec__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: struct; sec_inform_type
.. _doxid-structsec__inform__type:

struct sec_inform_type
======================

.. toctree::
	:hidden:

Overview
~~~~~~~~

inform derived type as a C struct :ref:`More...<details-structsec__inform__type>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sec.h>
	
	struct sec_inform_type {
		// fields
	
		int :ref:`status<doxid-structsec__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
	};
.. _details-structsec__inform__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

inform derived type as a C struct

Fields
------

.. index:: pair: variable; status
.. _doxid-structsec__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int status

return status. Possible valuesa are:

* 0 successful return

* -85 an update is inappropriate and has been skipped

