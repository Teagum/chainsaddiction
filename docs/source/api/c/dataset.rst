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

.. function:: DataSet *DataSet_NewEmpty (void)

   Create a new, uninitialized ``DataSet`` object and return an pointer to it.
   
.. function:: DataSet *DataSet_New (size_t n_elem)

   Create a new data set for ``n_elem`` elements and return a pointer to it.

.. function:: DataSet *DataSet_NewFromFile (const char *path)

   Create a new data set, set it with the values read from ``path`` and return
   a pointer it.


Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: void DataSet_SetValue (DataSet *restrict this, size_t idx, scalar val)

   Set the element with index ``idx`` of ``this`` to ``val``.  Perform bounds
   checking. Set :member:`DataSet.err` to ``true`` on failure.


.. function:: void DataSet_GetValue (DataSet *restrict this, size_t idx, scalar *out)

   Write the value of the ``this`` element with index ``idx`` to ``out``.
   Perform bounds checking. Set :member:`DataSet.err` to ``true`` on failure.


Utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: void DataSet_Print (DataSet *this)

   Print the data set nicely formatted to the standard error stream.
