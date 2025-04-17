.. index:: pair: struct; gls_sinfo
.. _doxid-structgls__sinfo:

gls_sinfo structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct gls_sinfo_type{INT}
          flag::INT
          more::INT
          stat::INT

.. _details-structgls__sinfo:

detailed documentation
----------------------

sinfo derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; flag
.. _doxid-structgls__sinfo_flag:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT flag

Flags success or failure case.

.. index:: pair: variable; more
.. _doxid-structgls__sinfo_more:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more

More information on failure.

.. index:: pair: variable; stat
.. _doxid-structgls__sinfo_stat:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stat

Status value after allocate failure.

