.. _global:

.. _details-spec_file:

control specification files
---------------------------

It is possible to set components of the structure ``control``, of 
type |package| _control_type, by reading an appropriate 
**data specification file** 
using the |package| _read_specfile function; here |package| refers
to the name of the package under consideration. This facility is
useful as it allows one to change |package| control parameters
without editing and recompiling programs that call |package| .

A specification file, or specfile, is a data file containing a number of
"specification commands". Each command occurs on a separate line, and
comprises a "keyword", which is a string (in a close-to-natural
language) used to identify a control parameter, and an (optional)
"value", which defines the value to be assigned to the given control
parameter. All keywords and values are case insensitive, keywords may be
preceded by one or more blanks but values must not contain blanks, and
each value must be separated from its keyword by at least one blank.
Values must not contain more than 30 characters, and each line of the
specfile is limited to 80 characters, including the blanks separating
keyword and value.

The portion of the specification file used by |package| _read_specfile
must start with a "``BEGIN |package|``" command and end with an
"``END``" command.  The syntax of the specfile is thus defined as follows:

.. ref-code-block:: 
	:class: doxyrest-title-code-block

        ( .. lines ignored by |package|_read_specfile .. )
          BEGIN |package|
             keyword    value
             .......    .....
             keyword    value
          END
        ( .. lines ignored by package|_read_specfile .. )

where keyword and value are two strings separated by (at least) one blank.
The ``BEGIN |package|`` and ``END`` delimiter command lines
may contain additional (trailing) strings so long as such strings are
separated by one or more blanks, so that lines such as

.. ref-code-block:: 
	:class: doxyrest-title-code-block

         BEGIN |package| SPECIFICATION

and

.. ref-code-block:: 
	:class: doxyrest-title-code-block

        END |package| SPECIFICATION

are acceptable. Furthermore,
between the
``BEGIN |package|`` and ``END`` delimiters,
specification commands may occur in any order.  Blank lines and
lines whose first non-blank character is ``!`` or ``*`` are ignored.
The content of a line after a ``!`` or ``*`` character is also ignored
(as is the ``!`` or ``*`` character itself). This provides an easy
manner to "comment out" some specification commands, or to comment
specific values of certain control parameters.

The value of a control parameters may be of three different types, namely
integer, logical or real (of the appropriate kinds).
Integer and real values may be expressed in any relevant Fortran integer and
floating-point formats (respectively). Permitted values for logical/boolean
parameters are 
``ON``, ``TRUE``, ``.TRUE.``, ``T``, ``YES``, ``Y``, or ``OFF``, ``NO``,
``N``, ``FALSE``, ``.FALSE.`` and ``F``.
Empty values are also allowed for logical control parameters, 
and are interpreted as ``TRUE``.

If a package uses another, and if the control structure 
for the former has a component that corresponds to a control structure
for the latter, the specfile may contain more than one ``BEGIN`` -- ``END`` 
block.

Thus, for example if package a uses package b, and a has a control structure

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct a_control_type{T}
          ( components of structure a )
          b_control::b_control_type{T}

a suitable spec file would be

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

          BEGIN a
             ( keywords for a )   values
          END

          BEGIN b
             ( keywords for b )   values
          END
