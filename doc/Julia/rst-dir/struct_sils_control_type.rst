.. index:: pair: table; sils_control_type
.. _doxid-structsils__control__type:

sils_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sils_control_type{T,INT}
          f_indexing::Bool
          ICNTL::NTuple{30,INT}
          lp::INT
          wp::INT
          mp::INT
          sp::INT
          ldiag::INT
          la::INT
          liw::INT
          maxla::INT
          maxliw::INT
          pivoting::INT
          nemin::INT
          factorblocking::INT
          solveblocking::INT
          thresh::INT
          ordering::INT
          scaling::INT
          CNTL::NTuple{5,T}
          multiplier::T
          reduce::T
          u::T
          static_tolerance::T
          static_level::T
          tolerance::T
          convergence::T

.. _details-structsils__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsils__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; ICNTL
.. _doxid-structsils__control__type_1a7d01102f1deac390a1e4e22a7bb59ea0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ICNTL[30]

MA27 internal integer controls.

.. index:: pair: variable; lp
.. _doxid-structsils__control__type_1a3eec33a4e6d8295c25d117bb25dd1b9b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lp

Unit for error messages.

.. index:: pair: variable; wp
.. _doxid-structsils__control__type_1af203f5ddbac4a47afed1a07b97e7e477:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT wp

Unit for warning messages.

.. index:: pair: variable; mp
.. _doxid-structsils__control__type_1a4b5efeeced2b749191f71afc3bc8bebd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mp

Unit for monitor output.

.. index:: pair: variable; sp
.. _doxid-structsils__control__type_1a024603cda52d9847e8d8df3c2e884b8c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sp

Unit for statistical output.

.. index:: pair: variable; ldiag
.. _doxid-structsils__control__type_1ad0e905fe282a7125424a53219afc0791:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ldiag

Controls level of diagnostic output.

.. index:: pair: variable; la
.. _doxid-structsils__control__type_1a8e48dcc59e4b8bbe40fe5b58321e4e72:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT la

Initial size for real array for the factors. If less than nrlnec, default size used.

.. index:: pair: variable; liw
.. _doxid-structsils__control__type_1a534631c6f6077b8813cb930ea43f3bfc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT liw

Initial size for integer array for the factors. If less than nirnec, default size used.

.. index:: pair: variable; maxla
.. _doxid-structsils__control__type_1a6437ead17fd48daf197640949e8d4ff3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxla

Max. size for real array for the factors.

.. index:: pair: variable; maxliw
.. _doxid-structsils__control__type_1a6f46a87af5a04d87697736b7d789ea33:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxliw

Max. size for integer array for the factors.

.. index:: pair: variable; pivoting
.. _doxid-structsils__control__type_1aa4d2d93f87da5df80b8aa2bce688c030:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivoting

Controls pivoting. Possible values are:

* 1 Numerical pivoting will be performed.

* 2 No pivoting will be performed and an error exit will occur immediately a pivot sign change is detected.

* 3 No pivoting will be performed and an error exit will occur if a zero pivot is detected.

* 4 No pivoting is performed but pivots are changed to all be positive.

.. index:: pair: variable; nemin
.. _doxid-structsils__control__type_1a2683299c9b72fbc4bf7d0b3078cd75ca:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nemin

Minimum number of eliminations in a step (unused)

.. index:: pair: variable; factorblocking
.. _doxid-structsils__control__type_1a32bd98cbf6436700c5a8848d77ce3917:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorblocking

Level 3 blocking in factorize (unused)

.. index:: pair: variable; solveblocking
.. _doxid-structsils__control__type_1ae3a41a26d065502707d23c2936aaed6e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT solveblocking

Level 2 and 3 blocking in solve.

.. index:: pair: variable; thresh
.. _doxid-structsils__control__type_1a6cf8bf160a02413bc3d5d18b0294b581:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT thresh

Controls threshold for detecting full rows in analyse, registered as percentage of N, 100 Only fully dense rows detected (default)

.. index:: pair: variable; ordering
.. _doxid-structsils__control__type_1a4175ebe476addcfc3433fc97c19e0708:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ordering

Controls ordering: Possible values are:

* 0 AMD using HSL's MC47

* 1 User defined

* 2 AMD using HSL's MC50

* 3 Min deg as in HSL's MA57

* 4 Metis_nodend ordering

* 5 Ordering chosen depending on matrix characteristics. At the moment choices are HSL's MC50 or Metis_nodend

* >5 Presently equivalent to 5 but may chnage

.. index:: pair: variable; scaling
.. _doxid-structsils__control__type_1a26f0572eeeaa419eabb09dc89c00b89d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT scaling

Controls scaling: Possible values are:

* 0 No scaling

* >0 Scaling using HSL's MC64 but may change for > 1

.. index:: pair: variable; CNTL
.. _doxid-structsils__control__type_1a9dd8bf3b6f412e66450dab7b2261846e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T CNTL[5]

MA27 internal real controls.

.. index:: pair: variable; multiplier
.. _doxid-structsils__control__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

Factor by which arrays sizes are to be increased if they are too small.

.. index:: pair: variable; reduce
.. _doxid-structsils__control__type_1a595df8d359282d27f49ac529283c509a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reduce

If previously allocated internal workspace arrays are greater than reduce times the currently required sizes, they are reset to current requirment.

.. index:: pair: variable; u
.. _doxid-structsils__control__type_1abb669b70ee8fb00689add7fad23ce00f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T u

Pivot threshold.

.. index:: pair: variable; static_tolerance
.. _doxid-structsils__control__type_1a3ce9c9cc8dd0f7c4684ea0bd80cc5946:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T static_tolerance

used for setting static pivot level

.. index:: pair: variable; static_level
.. _doxid-structsils__control__type_1ae7faebd3367553993434f6a03e65502d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T static_level

used for switch to static

.. index:: pair: variable; tolerance
.. _doxid-structsils__control__type_1ad0dcb73e98bad740852a54d6b7d1f6c2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T tolerance

Anything less than this is considered zero.

.. index:: pair: variable; convergence
.. _doxid-structsils__control__type_1ae7c621b1f1fcf3364b6c47d62406e82f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T convergence

used to monitor convergence in iterative refinement

