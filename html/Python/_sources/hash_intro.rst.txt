purpose
-------

The ``hash`` package **sets up, inserts into, removes from and searches**
a chained scatter table (Williams, 1959).

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/hash.pdf for additional details.

method
------

To insert a word in the table, the word is first mapped onto an
integer value, the entry integer.  This mapping is often called
hashing. As many words may be mapped to the same value (a collision),
a chain of used locations starting from the entry integer is searched
until an empty location is found. The word is inserted in the table at
this point and the chain extended to the next unoccupied entry. The
hashing routine is intended to reduce the number of collisions.  Words
are located and flagged as deleted from the table in exactly the same
way; the word is hashed and the resulting chain searched until the
word is matched or the end of the chain reached.  Provided there is
sufficient space in the table, the expected number of operations
needed to perform an insertion, search or removal is $O(1)$.

reference
---------

The chained scatter table search and insertion method is due to 

  F. A.  Williams (1959),
  ``Handling identifies as internal symbols in language processors'',
  *Communications of the ACM* **2(6)** (1959) 21-24.
