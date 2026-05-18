//* \file gls_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-04-03 AT 15:10 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_GLS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. March 24th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_gls.h"

/* Module global variables */
static void *data;                       // private internal data
static struct gls_control_type control;  // control struct
static struct gls_ainfo_type ainfo;      // ainfo struct
static struct gls_finfo_type finfo;      // ainfo struct
static struct gls_sinfo_type sinfo;      // ainfo struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SBLS Python interface
bool gls_update_control(struct gls_control_type *control,
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
        if(strcmp(key_name, "lp") == 0){
            if(!parse_int_option(value, "lp",
                                  &control->lp))
                return false;
            continue;
        }
        if(strcmp(key_name, "wp") == 0){
            if(!parse_int_option(value, "wp",
                                  &control->wp))
                return false;
            continue;
        }
        if(strcmp(key_name, "mp") == 0){
            if(!parse_int_option(value, "mp",
                                  &control->mp))
                return false;
            continue;
        }
        if(strcmp(key_name, "ldiag") == 0){
            if(!parse_int_option(value, "ldiag",
                                  &control->ldiag))
                return false;
            continue;
        }
        if(strcmp(key_name, "btf") == 0){
            if(!parse_int_option(value, "btf",
                                  &control->btf))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxit") == 0){
            if(!parse_int_option(value, "maxit",
                                  &control->maxit))
                return false;
            continue;
        }
        if(strcmp(key_name, "factor_blocking") == 0){
            if(!parse_int_option(value, "factor_blocking",
                                  &control->factor_blocking))
                return false;
            continue;
        }
        if(strcmp(key_name, "solve_blas") == 0){
            if(!parse_int_option(value, "solve_blas",
                                  &control->solve_blas))
                return false;
            continue;
        }
        if(strcmp(key_name, "la") == 0){
            if(!parse_int_option(value, "la",
                                  &control->la))
                return false;
            continue;
        }
        if(strcmp(key_name, "la_int") == 0){
            if(!parse_int_option(value, "la_int",
                                  &control->la_int))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxla") == 0){
            if(!parse_int_option(value, "maxla",
                                  &control->maxla))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivoting") == 0){
            if(!parse_int_option(value, "pivoting",
                                  &control->pivoting))
                return false;
            continue;
        }
        if(strcmp(key_name, "fill_in") == 0){
            if(!parse_int_option(value, "fill_in",
                                  &control->fill_in))
                return false;
            continue;
        }
        if(strcmp(key_name, "multiplier") == 0){
            if(!parse_double_option(value, "multiplier",
                                  &control->multiplier))
                return false;
            continue;
        }
        if(strcmp(key_name, "reduce") == 0){
            if(!parse_double_option(value, "reduce",
                                  &control->reduce))
                return false;
            continue;
        }
        if(strcmp(key_name, "u") == 0){
            if(!parse_double_option(value, "u",
                                  &control->u))
                return false;
            continue;
        }
        if(strcmp(key_name, "switch_full") == 0){
            if(!parse_double_option(value, "switch_full",
                                  &control->switch_full))
                return false;
            continue;
        }
        if(strcmp(key_name, "drop") == 0){
            if(!parse_double_option(value, "drop",
                                  &control->drop))
                return false;
            continue;
        }
        if(strcmp(key_name, "tolerance") == 0){
            if(!parse_double_option(value, "tolerance",
                                  &control->tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "cgce") == 0){
            if(!parse_double_option(value, "cgce",
                                  &control->cgce))
                return false;
            continue;
        }
        if(strcmp(key_name, "diagonal_pivoting") == 0){
            if(!parse_bool_option(value, "diagonal_pivoting",
                                  &control->diagonal_pivoting))
                return false;
            continue;
        }
        if(strcmp(key_name, "struct_abort") == 0){
            if(!parse_bool_option(value, "struct_abort",
                                  &control->struct_abort))
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
// NB not static as it is used for nested inform within SBLS Python interface
PyObject* gls_make_options_dict(const struct gls_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "lp",
                         PyLong_FromLong(control->lp));
    PyDict_SetItemString(py_options, "wp",
                         PyLong_FromLong(control->wp));
    PyDict_SetItemString(py_options, "mp",
                         PyLong_FromLong(control->mp));
    PyDict_SetItemString(py_options, "ldiag",
                         PyLong_FromLong(control->ldiag));
    PyDict_SetItemString(py_options, "btf",
                         PyLong_FromLong(control->btf));
    PyDict_SetItemString(py_options, "maxit",
                         PyLong_FromLong(control->maxit));
    PyDict_SetItemString(py_options, "factor_blocking",
                         PyLong_FromLong(control->factor_blocking));
    PyDict_SetItemString(py_options, "solve_blas",
                         PyLong_FromLong(control->solve_blas));
    PyDict_SetItemString(py_options, "la",
                         PyLong_FromLong(control->la));
    PyDict_SetItemString(py_options, "la_int",
                         PyLong_FromLong(control->la_int));
    PyDict_SetItemString(py_options, "maxla",
                         PyLong_FromLong(control->maxla));
    PyDict_SetItemString(py_options, "pivoting",
                         PyLong_FromLong(control->pivoting));
    PyDict_SetItemString(py_options, "fill_in",
                         PyLong_FromLong(control->fill_in));
    PyDict_SetItemString(py_options, "multiplier",
                         PyFloat_FromDouble(control->multiplier));
    PyDict_SetItemString(py_options, "reduce",
                         PyFloat_FromDouble(control->reduce));
    PyDict_SetItemString(py_options, "u",
                         PyFloat_FromDouble(control->u));
    PyDict_SetItemString(py_options, "switch_full",
                         PyFloat_FromDouble(control->switch_full));
    PyDict_SetItemString(py_options, "drop",
                         PyFloat_FromDouble(control->drop));
    PyDict_SetItemString(py_options, "tolerance",
                         PyFloat_FromDouble(control->tolerance));
    PyDict_SetItemString(py_options, "cgce",
                         PyFloat_FromDouble(control->cgce));
    PyDict_SetItemString(py_options, "diagonal_pivoting",
                         PyBool_FromLong(control->diagonal_pivoting));
    PyDict_SetItemString(py_options, "struct_abort",
                         PyBool_FromLong(control->struct_abort));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
//static PyObject* gls_make_time_dict(const struct gls_time_type *time){
//    PyObject *py_time = PyDict_New();

// Set float/double time entries
//    return py_time;
//}

//  *-*-*-*-*-*-*-*-*-*-   MAKE AINFO    -*-*-*-*-*-*-*-*-*-*

/* Take the ainfo struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* gls_make_ainfo_dict(const struct gls_ainfo_type *ainfo){
    PyObject *py_ainfo = PyDict_New();

    PyDict_SetItemString(py_ainfo, "flag",
                         PyLong_FromLong(ainfo->flag));
    PyDict_SetItemString(py_ainfo, "more",
                         PyLong_FromLong(ainfo->more));
    PyDict_SetItemString(py_ainfo, "len_analyse",
                         PyLong_FromLong(ainfo->len_analyse));
    PyDict_SetItemString(py_ainfo, "len_factorize",
                         PyLong_FromLong(ainfo->len_factorize));
    PyDict_SetItemString(py_ainfo, "ncmpa",
                         PyLong_FromLong(ainfo->ncmpa));
    PyDict_SetItemString(py_ainfo, "rank",
                         PyLong_FromLong(ainfo->rank));
    PyDict_SetItemString(py_ainfo, "drop",
                         PyLong_FromLong(ainfo->drop));
    PyDict_SetItemString(py_ainfo, "struc_rank",
                         PyLong_FromLong(ainfo->struc_rank));
    PyDict_SetItemString(py_ainfo, "oor",
                         PyLong_FromLong(ainfo->oor));
    PyDict_SetItemString(py_ainfo, "dup",
                         PyLong_FromLong(ainfo->dup));
    PyDict_SetItemString(py_ainfo, "stat",
                         PyLong_FromLong(ainfo->stat));
    PyDict_SetItemString(py_ainfo, "lblock",
                         PyLong_FromLong(ainfo->lblock));
    PyDict_SetItemString(py_ainfo, "sblock",
                         PyLong_FromLong(ainfo->sblock));
    PyDict_SetItemString(py_ainfo, "tblock",
                         PyLong_FromLong(ainfo->tblock));
    PyDict_SetItemString(py_ainfo, "ops",
                         PyFloat_FromDouble(ainfo->ops));

    return py_ainfo;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE FINFO    -*-*-*-*-*-*-*-*-*-*

/* Take the finfo struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* gls_make_finfo_dict(const struct gls_finfo_type *finfo){
    PyObject *py_finfo = PyDict_New();

    PyDict_SetItemString(py_finfo, "flag",
                         PyLong_FromLong(finfo->flag));
    PyDict_SetItemString(py_finfo, "more",
                         PyLong_FromLong(finfo->more));
    PyDict_SetItemString(py_finfo, "size_factor",
                         PyLong_FromLong(finfo->size_factor));
    PyDict_SetItemString(py_finfo, "len_factorize",
                         PyLong_FromLong(finfo->len_factorize));
    PyDict_SetItemString(py_finfo, "drop",
                         PyLong_FromLong(finfo->drop));
    PyDict_SetItemString(py_finfo, "rank",
                         PyLong_FromLong(finfo->rank));
    PyDict_SetItemString(py_finfo, "stat",
                         PyLong_FromLong(finfo->stat));
    PyDict_SetItemString(py_finfo, "ops",
                         PyFloat_FromDouble(finfo->ops));

    return py_finfo;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE SINFO    -*-*-*-*-*-*-*-*-*-*

/* Take the sinfo struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* gls_make_sinfo_dict(const struct gls_sinfo_type *sinfo){
    PyObject *py_sinfo = PyDict_New();

    PyDict_SetItemString(py_sinfo, "flag",
                         PyLong_FromLong(sinfo->flag));
    PyDict_SetItemString(py_sinfo, "more",
                         PyLong_FromLong(sinfo->more));
    PyDict_SetItemString(py_sinfo, "stat",
                         PyLong_FromLong(sinfo->stat));

    return py_sinfo;
}

//  *-*-*-*-*-*-*-*-*-*-   GLS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_gls_initialize(PyObject *self){

    // Call gls_initialize
    gls_initialize(&data, &control);

    // Record that GLS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = gls_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   GLS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_gls_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call gls_information
    gls_information(&data, &ainfo, &finfo, &sinfo, &status);

    // Return status and inform Python dictionary
    PyObject *py_ainfo = gls_make_ainfo_dict(&ainfo);
    PyObject *py_finfo = gls_make_finfo_dict(&finfo);
    PyObject *py_sinfo = gls_make_sinfo_dict(&sinfo);
    return Py_BuildValue("OOO", py_ainfo, py_finfo, py_sinfo);
}

//  *-*-*-*-*-*-*-*-*-*-   GLS_FINALIZE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_gls_finalize(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call gls_finalize
    gls_finalize(&data, &control, &status);

    // Record that GLS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE GLS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* gls python module method table */
static PyMethodDef gls_module_methods[] = {
    {"initialize", (PyCFunction) py_gls_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_gls_information, METH_NOARGS, NULL},
    {"finalize", (PyCFunction) py_gls_finalize, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* gls python module documentation */

PyDoc_STRVAR(gls_module_doc,

"The gls package solves sparse unsymmetric systems of linear equations \n"
"using a variant of Gaussian elimination.\n"
"Given a sparse matrix A = {a_ij}_mxn,  and an n-vector b,\n"
"this function solves the system Ax=b or its transpose A^Tx=b.\n"
"Both square (m=n) and rectangular (m/=n) matrices are handled;\n"
"one of an infinite class of solutions for consistent systems will\n"
"be returned whenever A is not of full rank.\n"
"gls is based upon a modern fortran interface to the HSL Archive \n"
"fortran 77 package MA28, which itself relies on MA33. \n"
"To obtain HSL Archive packages, see http://hsl.rl.ac.uk/archive/ .\n"
"\n"
"Currently only the options and info dictionaries are exposed; these are \n"
"provided and used by other GALAHAD packages with Python interfaces.\n"
"Extended functionality is available with the uls function.\n"
"\n"
"See $GALAHAD/html/Python/gls.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* gls python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "gls",               /* name of module */
   gls_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   gls_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_gls(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
