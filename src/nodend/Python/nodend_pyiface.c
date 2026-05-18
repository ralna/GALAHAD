//* \file nodend_pyiface.c */

/*
 * THIS VERSION: GALAHAD 5.2 - 2025-04-05 AT 16:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_NODEND PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Fowkes/Gould/Montoison/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 5.2. March 26th 3rd 2025
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_nodend.h"

/* Module global variables */
static void *data;                       // private internal data
static struct nodend_control_type control;  // control struct
static struct nodend_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SLS Python interface
bool nodend_update_control(struct nodend_control_type *control,
                        PyObject *py_options){

    // Use C defaults if Python options not passed
    if(!py_options) return true;

    PyObject *key, *value;
    Py_ssize_t pos = 0;
    const char* key_name;

    // Iterate over Python options dictionary
    while(PyDict_Next(py_options, &pos, &key, &value)) {

        // Parse options key
        if(!parse_options_key(key, &key_name))
            return false;

        // Parse each option
        if(strcmp(key_name, "version") == 0){
            if(!parse_char_option(value, "version",
                                  control->version,
                                  sizeof(control->version)))
                return false;
            continue;
        }
        if(strcmp(key_name, "error") == 0){
            if(!parse_int_option(value, "error",
                                  &control->error))
                return false;
            continue;
        }
        if(strcmp(key_name, "out") == 0){
            if(!parse_int_option(value, "out",
                                  &control->out))
                return false;
            continue;
        }
        if(strcmp(key_name, "print_level") == 0){
            if(!parse_int_option(value, "print_level",
                                  &control->print_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "no_metis_4_use_5_instead") == 0){
            if(!parse_bool_option(value, "no_metis_4_use_5_instead",
                                  &control->no_metis_4_use_5_instead))
                return false;
            continue;
        }
        if(strcmp(key_name, "prefix") == 0){
            if(!parse_char_option(value, "prefix",
                                  control->prefix,
                                  sizeof(control->prefix)))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_ptype") == 0){
            if(!parse_int_option(value, "metis4_ptype",
                                  &control->metis4_ptype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_ctype") == 0){
            if(!parse_int_option(value, "metis4_ctype",
                                  &control->metis4_ctype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_itype") == 0){
            if(!parse_int_option(value, "metis4_itype",
                                  &control->metis4_itype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_rtype") == 0){
            if(!parse_int_option(value, "metis4_rtype",
                                  &control->metis4_rtype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_dbglvl") == 0){
            if(!parse_int_option(value, "metis4_dbglvl",
                                  &control->metis4_dbglvl))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_oflags") == 0){
            if(!parse_int_option(value, "metis4_oflags",
                                  &control->metis4_oflags))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_pfactor") == 0){
            if(!parse_int_option(value, "metis4_pfactor",
                                  &control->metis4_pfactor))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis4_nseps") == 0){
            if(!parse_int_option(value, "metis4_nseps",
                                  &control->metis4_nseps))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_ptype") == 0){
            if(!parse_int_option(value, "metis5_ptype",
                                  &control->metis5_ptype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_objtype") == 0){
            if(!parse_int_option(value, "metis5_objtype",
                                  &control->metis5_objtype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_ctype") == 0){
            if(!parse_int_option(value, "metis5_ctype",
                                  &control->metis5_ctype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_iptype") == 0){
            if(!parse_int_option(value, "metis5_iptype",
                                  &control->metis5_iptype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_rtype") == 0){
            if(!parse_int_option(value, "metis5_rtype",
                                  &control->metis5_rtype))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_dbglvl") == 0){
            if(!parse_int_option(value, "metis5_dbglvl",
                                  &control->metis5_dbglvl))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_niter") == 0){
            if(!parse_int_option(value, "metis5_niter",
                                  &control->metis5_niter))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_ncuts") == 0){
            if(!parse_int_option(value, "metis5_ncuts",
                                  &control->metis5_ncuts))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_seed") == 0){
            if(!parse_int_option(value, "metis5_seed",
                                  &control->metis5_seed))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_no2hop") == 0){
            if(!parse_int_option(value, "metis5_no2hop",
                                  &control->metis5_no2hop))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_minconn") == 0){
            if(!parse_int_option(value, "metis5_minconn",
                                  &control->metis5_minconn))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_contig") == 0){
            if(!parse_int_option(value, "metis5_contig",
                                  &control->metis5_contig))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_compress") == 0){
            if(!parse_int_option(value, "metis5_compress",
                                  &control->metis5_compress))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_ccorder") == 0){
            if(!parse_int_option(value, "metis5_ccorder",
                                  &control->metis5_ccorder))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_pfactor") == 0){
            if(!parse_int_option(value, "metis5_pfactor",
                                  &control->metis5_pfactor))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_nseps") == 0){
            if(!parse_int_option(value, "metis5_nseps",
                                  &control->metis5_nseps))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_ufactor") == 0){
            if(!parse_int_option(value, "metis5_ufactor",
                                  &control->metis5_ufactor))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_niparts") == 0){
            if(!parse_int_option(value, "metis5_niparts",
                                  &control->metis5_niparts))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_ondisk") == 0){
            if(!parse_int_option(value, "metis5_ondisk",
                                  &control->metis5_ondisk))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_dropedges") == 0){
            if(!parse_int_option(value, "metis5_dropedges",
                                  &control->metis5_dropedges))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_twohop") == 0){
            if(!parse_int_option(value, "metis5_twohop",
                                  &control->metis5_twohop))
                return false;
            continue;
        }
        if(strcmp(key_name, "metis5_fast") == 0){
            if(!parse_int_option(value, "metis5_fast",
                                  &control->metis5_fast))
                return false;
            continue;
        }

        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within SLS Python interface
PyObject* nodend_make_options_dict(const struct nodend_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "version",
                         PyUnicode_FromString(control->version));
    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "no_metis_4_use_5_instead",
                         PyBool_FromLong(control->no_metis_4_use_5_instead));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "metis4_ptype",
                         PyLong_FromLong(control->metis4_ptype));
    PyDict_SetItemString(py_options, "metis4_ctype",
                         PyLong_FromLong(control->metis4_ctype));
    PyDict_SetItemString(py_options, "metis4_itype",
                         PyLong_FromLong(control->metis4_itype));
    PyDict_SetItemString(py_options, "metis4_rtype",
                         PyLong_FromLong(control->metis4_rtype));
    PyDict_SetItemString(py_options, "metis4_dbglvl",
                         PyLong_FromLong(control->metis4_dbglvl));
    PyDict_SetItemString(py_options, "metis4_oflags",
                         PyLong_FromLong(control->metis4_oflags));
    PyDict_SetItemString(py_options, "metis4_pfactor",
                         PyLong_FromLong(control->metis4_pfactor));
    PyDict_SetItemString(py_options, "metis4_nseps",
                         PyLong_FromLong(control->metis4_nseps));
    PyDict_SetItemString(py_options, "metis5_ptype",
                         PyLong_FromLong(control->metis5_ptype));
    PyDict_SetItemString(py_options, "metis5_objtype",
                         PyLong_FromLong(control->metis5_objtype));
    PyDict_SetItemString(py_options, "metis5_ctype",
                         PyLong_FromLong(control->metis5_ctype));
    PyDict_SetItemString(py_options, "metis5_iptype",
                         PyLong_FromLong(control->metis5_iptype));
    PyDict_SetItemString(py_options, "metis5_rtype",
                         PyLong_FromLong(control->metis5_rtype));
    PyDict_SetItemString(py_options, "metis5_dbglvl",
                         PyLong_FromLong(control->metis5_dbglvl));
    PyDict_SetItemString(py_options, "metis5_niter",
                         PyLong_FromLong(control->metis5_niter));
    PyDict_SetItemString(py_options, "metis5_ncuts",
                         PyLong_FromLong(control->metis5_ncuts));
    PyDict_SetItemString(py_options, "metis5_seed",
                         PyLong_FromLong(control->metis5_seed));
    PyDict_SetItemString(py_options, "metis5_no2hop",
                         PyLong_FromLong(control->metis5_no2hop));
    PyDict_SetItemString(py_options, "metis5_minconn",
                         PyLong_FromLong(control->metis5_minconn));
    PyDict_SetItemString(py_options, "metis5_contig",
                         PyLong_FromLong(control->metis5_contig));
    PyDict_SetItemString(py_options, "metis5_compress",
                         PyLong_FromLong(control->metis5_compress));
    PyDict_SetItemString(py_options, "metis5_ccorder",
                         PyLong_FromLong(control->metis5_ccorder));
    PyDict_SetItemString(py_options, "metis5_pfactor",
                         PyLong_FromLong(control->metis5_pfactor));
    PyDict_SetItemString(py_options, "metis5_nseps",
                         PyLong_FromLong(control->metis5_nseps));
    PyDict_SetItemString(py_options, "metis5_ufactor",
                         PyLong_FromLong(control->metis5_ufactor));
    PyDict_SetItemString(py_options, "metis5_niparts",
                         PyLong_FromLong(control->metis5_niparts));
    PyDict_SetItemString(py_options, "metis5_ondisk",
                         PyLong_FromLong(control->metis5_ondisk));
    PyDict_SetItemString(py_options, "metis5_dropedges",
                         PyLong_FromLong(control->metis5_dropedges));
    PyDict_SetItemString(py_options, "metis5_twohop",
                         PyLong_FromLong(control->metis5_twohop));
    PyDict_SetItemString(py_options, "metis5_fast",
                         PyLong_FromLong(control->metis5_fast ));

   return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* nodend_make_time_dict(const struct nodend_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "metis",
                         PyFloat_FromDouble(time->metis));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_metis",
                         PyFloat_FromDouble(time->clock_metis));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* nodend_make_inform_dict(const struct nodend_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "version",
                         PyUnicode_FromString(inform->version));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         nodend_make_time_dict(&inform->time));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   NODEND_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_nodend_initialize(PyObject *self){

    // Call nodend_initialize
    nodend_initialize(&data, &control, &status);

    // Record that NODEND has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = nodend_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-*-*-   NODEND_ORDER    -*-*-*-*-*-*-*-*-*-*-*-*

static PyObject* py_nodend_order(PyObject *self, PyObject *args, PyObject *keywds){
    PyArrayObject *py_A_row, *py_A_col, *py_A_ptr;
    PyObject *py_options = NULL;
    int *A_row = NULL, *A_col = NULL, *A_ptr = NULL;
    const char *A_type;
    int n, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n","A_type","A_ne","A_row","A_col","A_ptr",
                             "options",NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "isiOOO|O",
                                    kwlist, &n, &A_type, &A_ne, &py_A_row,
                                    &py_A_col, &py_A_ptr,
                                    &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("A_row", py_A_row, A_ne) &&
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, n+1)
        ))
        return NULL;

    // Create NumPy output arrays
    npy_intp ndim[] = {n}; // size of perm
    PyArrayObject *py_perm =
      (PyArrayObject *) PyArray_SimpleNew(1, ndim, NPY_INT);
    int *perm = (int *) PyArray_DATA(py_perm);

    // Convert 64bit integer A_row array to 32bit
    if((PyObject *) py_A_row != Py_None){
        A_row = malloc(A_ne * sizeof(int));
        long int *A_row_long = (long int *) PyArray_DATA(py_A_row);
        for(int i = 0; i < A_ne; i++) A_row[i] = (int) A_row_long[i];
    }

    // Convert 64bit integer A_col array to 32bit
    if((PyObject *) py_A_col != Py_None){
        A_col = malloc(A_ne * sizeof(int));
        long int *A_col_long = (long int *) PyArray_DATA(py_A_col);
        for(int i = 0; i < A_ne; i++) A_col[i] = (int) A_col_long[i];
    }

    // Convert 64bit integer A_ptr array to 32bit
    if((PyObject *) py_A_ptr != Py_None){
        A_ptr = malloc((n+1) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < n+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Call nodend_import
    nodend_order(&control, &data, &status, n, perm,
               A_type, A_ne, A_row, A_col, A_ptr);

    // Free allocated memory
    if(A_row != NULL) free(A_row);
    if(A_col != NULL) free(A_col);
    if(A_ptr != NULL) free(A_ptr);

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // Return perm
    PyObject *solve_nodend_return;
    solve_nodend_return = Py_BuildValue("O", py_perm);
    Py_INCREF(solve_nodend_return);
    return solve_nodend_return;
}

//  *-*-*-*-*-*-*-*-*-*-   NODEND_INFORMATION   -*-*-*-*-*-*-*-*-

static PyObject* py_nodend_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call nodend_information
    nodend_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = nodend_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   NODEND_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_nodend_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call nodend_terminate
    nodend_terminate(&data);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE NODEND PYTHON MODULE    -*-*-*-*-*-*-*-*-

/* nodend python module method table */
static PyMethodDef nodend_module_methods[] = {
    {"initialize", (PyCFunction) py_nodend_initialize, METH_NOARGS,NULL},
    {"order", (PyCFunction) py_nodend_order, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_nodend_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_nodend_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* nodend python module documentation */

PyDoc_STRVAR(nodend_module_doc,

"The nodend package Find a symmetric row and column permutation P A P'\n"
"of a symmetric, sparse matrix A with the aim of limiting\n"
"the fill-in during subsequent Cholesky-like factorization\n" 
"The package is actually a wrapper to the METIS_NodeND\n"
"procedure from versions 4.0, 5.1 and 5.2 of the\n"
"METIS package from the Karypis Lab.\n"
"\n"
);

/* nodend python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "nodend",               /* name of module */
   nodend_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   nodend_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_nodend(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
