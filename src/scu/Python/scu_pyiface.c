//* \file scu_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-12 AT 08:40 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SCU PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
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
#include "galahad_scu.h"

/* Module global variables */
static void *data;                       // private internal data
static struct scu_control_type control;  // control struct
static struct scu_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within CRO Python interface
bool scu_update_control(struct scu_control_type *control,
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
        // None!
        // Otherwise unrecognised option
        PyErr_Format(PyExc_ValueError,
          "unrecognised option options['%s']\n", key_name);
        return false;
    }

    return true; // success
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE OPTIONS    -*-*-*-*-*-*-*-*-*-*

/* Take the control struct from C and turn it into a python options dict */
// NB not static as it is used for nested inform within CRO Python interface
PyObject* scu_make_options_dict(const struct scu_control_type *control){
    PyObject *py_options = PyDict_New();

    // None!
    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within CRO Python interface
PyObject* scu_make_inform_dict(const struct scu_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    npy_intp idim[] = {3};
    PyArrayObject *py_inertia =
      (PyArrayObject*) PyArray_SimpleNew(1, idim, NPY_INT);
    int *inertia = (int *) PyArray_DATA(py_inertia);
    for(int i=0; i<3; i++) inertia[i] = inform->inertia[i];
    PyDict_SetItemString(py_inform, "inertia", (PyObject *) py_inertia);

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   SCU_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_scu_information(PyObject *self){

    // Record that FIT has been initialised
    init_called = true;

    // Call scu_information
    scu_information(&data, &inform, &status);

    // Return status and inform Python dictionary
    PyObject *py_inform = scu_make_inform_dict(&inform);
    return Py_BuildValue("O", py_inform);
}

//  *-*-*-*-*-*-*-*-*-*-   SCU_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_scu_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call scu_terminate
    scu_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SCU PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* scu python module method table */
static PyMethodDef scu_module_methods[] = {
    {"information", (PyCFunction) py_scu_information, METH_NOARGS, NULL},
    {"terminate", (PyCFunction) py_scu_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* scu python module documentation */

PyDoc_STRVAR(scu_module_doc,

"The scu package computes the solution to an extended system of n + m \n"
"sparse real linear equations in n + m unknowns, \n"
" ( A B ) ( x_1 ) = ( b_1 ) \n"
" ( C D ) ( x_2 ) = ( b_2 ) \n"
"in the case where the n by n matrix A is nonsingular \n"
"and solutions to the systems \n"
"  A x  =  b  and A^T y  =  c \n"
"may be obtained from an external source, such as an existing \n"
"factorization.  The subroutine uses reverse communication to obtain \n"
"the solution to such smaller systems.  The method makes use of \n"
"the Schur complement matrix S = D - C A^{-1} B. \n"
"The Schur complement is stored and factorized as a dense matrix \n"
"and the subroutine is thus appropriate only if there is \n"
"sufficient storage for this matrix. Special advantage is taken \n"
"of symmetry and definiteness in the coefficient matrices. \n"
"Provision is made for introducing additional rows and columns \n"
"to, and removing existing rows and columns from, the extended.\n"
"\n"
"See $GALAHAD/html/Python/scu.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* scu python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "scu",               /* name of module */
   scu_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   scu_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_scu(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
