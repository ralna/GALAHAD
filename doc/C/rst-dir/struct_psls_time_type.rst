.. index:: pair: table; psls_time_type
.. _doxid-structpsls__time__type:

psls_time_type structure
------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_psls.h>
	
	struct psls_time_type {
		// fields
	
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`total<doxid-structpsls__time__type_total>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`assemble<doxid-structpsls__time__type_assemble>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`analyse<doxid-structpsls__time__type_analyse>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`factorize<doxid-structpsls__time__type_factorize>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`solve<doxid-structpsls__time__type_solve>`;
		:ref:`spc_<doxid-galahad__spc_8h_>` :ref:`update<doxid-structpsls__time__type_update>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_total<doxid-structpsls__time__type_clock_total>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_assemble<doxid-structpsls__time__type_clock_assemble>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_analyse<doxid-structpsls__time__type_clock_analyse>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_factorize<doxid-structpsls__time__type_clock_factorize>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_solve<doxid-structpsls__time__type_clock_solve>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_update<doxid-structpsls__time__type_clock_update>`;
	};
.. _details-structpsls__time__type:

detailed documentation
----------------------

time derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; total
.. _doxid-structpsls__time__type_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` total

total time

.. index:: pair: variable; assemble
.. _doxid-structpsls__time__type_assemble:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` assemble

time to assemble the preconditioner prior to factorization

.. index:: pair: variable; analyse
.. _doxid-structpsls__time__type_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` analyse

time for the analysis phase

.. index:: pair: variable; factorize
.. _doxid-structpsls__time__type_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` factorize

time for the factorization phase

.. index:: pair: variable; solve
.. _doxid-structpsls__time__type_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` solve

time for the linear solution phase

.. index:: pair: variable; update
.. _doxid-structpsls__time__type_update:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`spc_<doxid-galahad__spc_8h_>` update

time to update the factorization

.. index:: pair: variable; clock_total
.. _doxid-structpsls__time__type_clock_total:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_total

total clock time spent in the package

.. index:: pair: variable; clock_assemble
.. _doxid-structpsls__time__type_clock_assemble:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_assemble

clock time to assemble the preconditioner prior to factorization

.. index:: pair: variable; clock_analyse
.. _doxid-structpsls__time__type_clock_analyse:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_analyse

clock time for the analysis phase

.. index:: pair: variable; clock_factorize
.. _doxid-structpsls__time__type_clock_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_factorize

clock time for the factorization phase

.. index:: pair: variable; clock_solve
.. _doxid-structpsls__time__type_clock_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_solve

clock time for the linear solution phase

.. index:: pair: variable; clock_update
.. _doxid-structpsls__time__type_clock_update:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_update

clock time to update the factorization

