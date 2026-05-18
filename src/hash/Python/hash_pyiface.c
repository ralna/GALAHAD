//* \file hash_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:30 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_HASH PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. May 3rd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_hash.h"

/* Module global variables */
static void *data;                       // private internal data
static struct hash_control_type control;  // control struct
static struct hash_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within DGO Python interface
bool hash_update_control(struct hash_control_type *control,
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
        if(strcmp(key_name, "space_critical") == 0){
            if(!parse_bool_option(value, "space_critical",
                                  &control->space_critical))
                return false;
            continue;
        }
        if(strcmp(key_name, "deallocate_error_fatal") == 0){
            if(!parse_bool_option(value, "deallocate_error_fatal",
                                  &control->deallocate_error_fatal))
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
        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within DGO Python interface
PyObject* hash_make_options_dict(const struct hash_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within DGO Python interface
PyObject* hash_make_inform_dict(const struct hash_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   HASH_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

 static PyObject* py_hash_initialize(PyObject *self, PyObject *args, PyObject *keywds){
    int nchar, length;

    // Parse positional arguments
    static char *kwlist[] = {"nchar", "length", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "ii", kwlist, &nchar, &length))
        return NULL;

    // Call hash_initialize
    hash_initialize(nchar, length, &data, &control, &inform);

    // Record that HASH has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = hash_make_options_dict(&control);
    PyObject *py_inform = hash_make_inform_dict(&inform);
    return Py_BuildValue("OO", py_options, py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   HASH_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_hash_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call hash_information
    hash_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = hash_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   HASH_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_hash_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call hash_terminate
    hash_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE HASH PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* hash python module method table */
static PyMethodDef hash_module_methods[] = {
    {"initialize", (PyCFunction) py_hash_initialize, METH_VARARGS | METH_KEYWORDS, NULL},
    {"information", (PyCFunction) py_hash_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_hash_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};


/* hash python module documentation */

PyDoc_STRVAR(hash_module_doc,

"The hash package sets up, inserts into, removes from and searches \n"
"a chained scatter table  (Williams, CACM 2, 21-24, 1959)\n"
"\n"
"See $GALAHAD/html/Python/hash.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* hash python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "hash",               /* name of module */
   hash_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   hash_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_hash(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
