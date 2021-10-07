.. default-domain:: c

Dataset
-------------------------------------------------------------------------------

.. struct:: DataSet

   .. code-block:: c

      typedef struct DataSet {
         scalar *data;
         size_t size;
         bool err;
      } DataSet;


   .. member:: scalar *data

      Object pointer that stores the raw data.

   .. member:: size_t size

      Size of the data, that is, the number of elements ``data`` can hold.

   .. member::  bool err

      Error indicator.


Object creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: DataSet *ds_NewEmpty (void)

   Create a new, uninitialized ``DataSet`` object and return an pointer to it.
   
.. function:: DataSet *ds_New (size_t n_elem)

   Create a new data set for ``n_elem`` elements and return a pointer to it.

.. function:: DataSet *ds_NewFromFile (const char *path)

   Create a new data set, set it with the values read from ``path`` and return
   a pointer it.


Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: void ds_set (DataSet *restrict pds, size_t idx, scalar val)

   Set the element with index ``idx`` of ``pds`` to ``val``. Perform bounds
   checking.


.. function:: void ds_get (DataSet *restrict pds, size_t idx, scalar *out)

   Write the value of the ``pds`` element with index ``idx`` to ``out``.
   Perform bounds checking.


Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: void ds_print (DataSet *pds)

   Print the data set nicely formatted to the standard error stream.
