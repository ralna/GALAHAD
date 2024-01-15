//* \file sils_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-06-02 AT 13:00 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_SILS PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 3rd 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_sils.h"

/* Module global variables */
static void *data;                  // private internal data
static struct sils_control_type control;  // control struct
static struct sils_ainfo_type ainfo;      // ainfo struct
static struct sils_finfo_type finfo;      // finfo struct
static struct sils_sinfo_type sinfo;      // sinfo struct
static bool init_called = false;    // record if initialise was called
static int status = 0;              // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within SBLS Python interface
bool sils_update_control(struct sils_control_type *control,
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
        if(strcmp(key_name, "ICNTL") == 0){
            if(!parse_int_array_option((PyArrayObject*) value, "ICNTL",
                                       control->ICNTL, 30))
                return false;
            continue;
        }
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
        if(strcmp(key_name, "sp") == 0){
            if(!parse_int_option(value, "sp",
                                  &control->sp))
                return false;
            continue;
        }
        if(strcmp(key_name, "ldiag") == 0){
            if(!parse_int_option(value, "ldiag",
                                  &control->ldiag))
                return false;
            continue;
        }
        if(strcmp(key_name, "la") == 0){
            if(!parse_int_option(value, "la",
                                  &control->la))
                return false;
            continue;
        }
        if(strcmp(key_name, "liw") == 0){
            if(!parse_int_option(value, "liw",
                                  &control->liw))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxla") == 0){
            if(!parse_int_option(value, "maxla",
                                  &control->maxla))
                return false;
            continue;
        }
        if(strcmp(key_name, "maxliw") == 0){
            if(!parse_int_option(value, "maxliw",
                                  &control->maxliw))
                return false;
            continue;
        }
        if(strcmp(key_name, "pivoting") == 0){
            if(!parse_int_option(value, "pivoting",
                                  &control->pivoting))
                return false;
            continue;
        }
        if(strcmp(key_name, "nemin") == 0){
            if(!parse_int_option(value, "nemin",
                                  &control->nemin))
                return false;
            continue;
        }
        if(strcmp(key_name, "factorblocking") == 0){
            if(!parse_int_option(value, "factorblocking",
                                  &control->factorblocking))
                return false;
            continue;
        }
        if(strcmp(key_name, "solveblocking") == 0){
            if(!parse_int_option(value, "solveblocking",
                                  &control->solveblocking))
                return false;
            continue;
        }
        if(strcmp(key_name, "thresh") == 0){
            if(!parse_int_option(value, "thresh",
                                  &control->thresh))
                return false;
            continue;
        }
        if(strcmp(key_name, "ordering") == 0){
            if(!parse_int_option(value, "ordering",
                                  &control->ordering))
                return false;
            continue;
        }
        if(strcmp(key_name, "scaling") == 0){
            if(!parse_int_option(value, "scaling",
                                  &control->scaling))
                return false;
            continue;
        }
        if(strcmp(key_name, "CNTL") == 0){
            if(!parse_double_array_option((PyArrayObject*) value, "CNTL",
                                  control->CNTL, 5))
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
        if(strcmp(key_name, "static_tolerance") == 0){
            if(!parse_double_option(value, "static_tolerance",
                                  &control->static_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "static_level") == 0){
            if(!parse_double_option(value, "static_level",
                                  &control->static_level))
                return false;
            continue;
        }
        if(strcmp(key_name, "tolerance") == 0){
            if(!parse_double_option(value, "tolerance",
                                  &control->tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "convergence") == 0){
            if(!parse_double_option(value, "convergence",
                                  &control->convergence))
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
PyObject* sils_make_options_dict(const struct sils_control_type *control){
    PyObject *py_options = PyDict_New();

    //PyDict_SetItemString(py_options, "ICNTL",
    //                     PyLong_FromLong(control->ICNTL));
    PyDict_SetItemString(py_options, "lp",
                         PyLong_FromLong(control->lp));
    PyDict_SetItemString(py_options, "wp",
                         PyLong_FromLong(control->wp));
    PyDict_SetItemString(py_options, "mp",
                         PyLong_FromLong(control->mp));
    PyDict_SetItemString(py_options, "sp",
                         PyLong_FromLong(control->sp));
    PyDict_SetItemString(py_options, "ldiag",
                         PyLong_FromLong(control->ldiag));
    PyDict_SetItemString(py_options, "la",
                         PyLong_FromLong(control->la));
    PyDict_SetItemString(py_options, "liw",
                         PyLong_FromLong(control->liw));
    PyDict_SetItemString(py_options, "maxla",
                         PyLong_FromLong(control->maxla));
    PyDict_SetItemString(py_options, "maxliw",
                         PyLong_FromLong(control->maxliw));
    PyDict_SetItemString(py_options, "pivoting",
                         PyLong_FromLong(control->pivoting));
    PyDict_SetItemString(py_options, "nemin",
                         PyLong_FromLong(control->nemin));
    PyDict_SetItemString(py_options, "factorblocking",
                         PyLong_FromLong(control->factorblocking));
    PyDict_SetItemString(py_options, "solveblocking",
                         PyLong_FromLong(control->solveblocking));
    PyDict_SetItemString(py_options, "thresh",
                         PyLong_FromLong(control->thresh));
    PyDict_SetItemString(py_options, "ordering",
                         PyLong_FromLong(control->ordering));
    PyDict_SetItemString(py_options, "scaling",
                         PyLong_FromLong(control->scaling));
    //PyDict_SetItemString(py_options, "CNTL",
    //                     PyFloat_FromDouble(control->CNTL));
    PyDict_SetItemString(py_options, "multiplier",
                         PyFloat_FromDouble(control->multiplier));
    PyDict_SetItemString(py_options, "reduce",
                         PyFloat_FromDouble(control->reduce));
    PyDict_SetItemString(py_options, "u",
                         PyFloat_FromDouble(control->u));
    PyDict_SetItemString(py_options, "static_tolerance",
                         PyFloat_FromDouble(control->static_tolerance));
    PyDict_SetItemString(py_options, "static_level",
                         PyFloat_FromDouble(control->static_level));
    PyDict_SetItemString(py_options, "tolerance",
                         PyFloat_FromDouble(control->tolerance));
    PyDict_SetItemString(py_options, "convergence",
                         PyFloat_FromDouble(control->convergence));

    return py_options;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
//static PyObject* sils_make_time_dict(const struct sils_time_type *time){
//    PyObject *py_time = PyDict_New();

// Set float/double time entries
//    return py_time;
//}

//  *-*-*-*-*-*-*-*-*-*-   MAKE AINFO    -*-*-*-*-*-*-*-*-*-*

/* Take the ainfo struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* sils_make_ainfo_dict(const struct sils_ainfo_type *ainfo){
    PyObject *py_ainfo = PyDict_New();

    PyDict_SetItemString(py_ainfo, "flag",
                         PyLong_FromLong(ainfo->flag));
    PyDict_SetItemString(py_ainfo, "more",
                         PyLong_FromLong(ainfo->more));
    PyDict_SetItemString(py_ainfo, "nsteps",
                         PyLong_FromLong(ainfo->nsteps));
    PyDict_SetItemString(py_ainfo, "nrltot",
                         PyLong_FromLong(ainfo->nrltot));
    PyDict_SetItemString(py_ainfo, "nirtot",
                         PyLong_FromLong(ainfo->nirtot));
    PyDict_SetItemString(py_ainfo, "nrlnec",
                         PyLong_FromLong(ainfo->nrlnec));
    PyDict_SetItemString(py_ainfo, "nirnec",
                         PyLong_FromLong(ainfo->nirnec));
    PyDict_SetItemString(py_ainfo, "nrladu",
                         PyLong_FromLong(ainfo->nrladu));
    PyDict_SetItemString(py_ainfo, "niradu",
                         PyLong_FromLong(ainfo->niradu));
    PyDict_SetItemString(py_ainfo, "ncmpa",
                         PyLong_FromLong(ainfo->ncmpa));
    PyDict_SetItemString(py_ainfo, "oor",
                         PyLong_FromLong(ainfo->oor));
    PyDict_SetItemString(py_ainfo, "dup",
                         PyLong_FromLong(ainfo->dup));
    PyDict_SetItemString(py_ainfo, "maxfrt",
                         PyLong_FromLong(ainfo->maxfrt));
    PyDict_SetItemString(py_ainfo, "stat",
                         PyLong_FromLong(ainfo->stat));
    PyDict_SetItemString(py_ainfo, "faulty",
                         PyLong_FromLong(ainfo->faulty));
    PyDict_SetItemString(py_ainfo, "opsa",
                         PyFloat_FromDouble(ainfo->opsa));
    PyDict_SetItemString(py_ainfo, "opse",
                         PyFloat_FromDouble(ainfo->opse));

    return py_ainfo;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE FINFO    -*-*-*-*-*-*-*-*-*-*

/* Take the finfo struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* sils_make_finfo_dict(const struct sils_finfo_type *finfo){
    PyObject *py_finfo = PyDict_New();

    PyDict_SetItemString(py_finfo, "flag",
                         PyLong_FromLong(finfo->flag));
    PyDict_SetItemString(py_finfo, "more",
                         PyLong_FromLong(finfo->more));
    PyDict_SetItemString(py_finfo, "maxfrt",
                         PyLong_FromLong(finfo->maxfrt));
    PyDict_SetItemString(py_finfo, "nebdu",
                         PyLong_FromLong(finfo->nebdu));
    PyDict_SetItemString(py_finfo, "nrlbdu",
                         PyLong_FromLong(finfo->nrlbdu));
    PyDict_SetItemString(py_finfo, "nirbdu",
                         PyLong_FromLong(finfo->nirbdu));
    PyDict_SetItemString(py_finfo, "nrltot",
                         PyLong_FromLong(finfo->nrltot));
    PyDict_SetItemString(py_finfo, "nirtot",
                         PyLong_FromLong(finfo->nirtot));
    PyDict_SetItemString(py_finfo, "nrlnec",
                         PyLong_FromLong(finfo->nrlnec));
    PyDict_SetItemString(py_finfo, "nirnec",
                         PyLong_FromLong(finfo->nirnec));
    PyDict_SetItemString(py_finfo, "ncmpbr",
                         PyLong_FromLong(finfo->ncmpbr));
    PyDict_SetItemString(py_finfo, "ncmpbi",
                         PyLong_FromLong(finfo->ncmpbi));
    PyDict_SetItemString(py_finfo, "ntwo",
                         PyLong_FromLong(finfo->ntwo));
    PyDict_SetItemString(py_finfo, "neig",
                         PyLong_FromLong(finfo->neig));
    PyDict_SetItemString(py_finfo, "delay",
                         PyLong_FromLong(finfo->delay));
    PyDict_SetItemString(py_finfo, "signc",
                         PyLong_FromLong(finfo->signc));
    PyDict_SetItemString(py_finfo, "nstatic",
                         PyLong_FromLong(finfo->nstatic));
    PyDict_SetItemString(py_finfo, "modstep",
                         PyLong_FromLong(finfo->modstep));
    PyDict_SetItemString(py_finfo, "rank",
                         PyLong_FromLong(finfo->rank));
    PyDict_SetItemString(py_finfo, "stat",
                         PyLong_FromLong(finfo->stat));
    PyDict_SetItemString(py_finfo, "faulty",
                         PyLong_FromLong(finfo->faulty));
    PyDict_SetItemString(py_finfo, "step",
                         PyLong_FromLong(finfo->step));
    PyDict_SetItemString(py_finfo, "opsa",
                         PyFloat_FromDouble(finfo->opsa));
    PyDict_SetItemString(py_finfo, "opse",
                         PyFloat_FromDouble(finfo->opse));
    PyDict_SetItemString(py_finfo, "opsb",
                         PyFloat_FromDouble(finfo->opsb));
    PyDict_SetItemString(py_finfo, "maxchange",
                         PyFloat_FromDouble(finfo->maxchange));
    PyDict_SetItemString(py_finfo, "smin",
                         PyFloat_FromDouble(finfo->smin));
    PyDict_SetItemString(py_finfo, "smax",
                         PyFloat_FromDouble(finfo->smax));

    return py_finfo;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE SINFO    -*-*-*-*-*-*-*-*-*-*

/* Take the sinfo struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within SLS Python interface
PyObject* sils_make_sinfo_dict(const struct sils_sinfo_type *sinfo){
    PyObject *py_sinfo = PyDict_New();

    PyDict_SetItemString(py_sinfo, "flag",
                         PyLong_FromLong(sinfo->flag));
    PyDict_SetItemString(py_sinfo, "stat",
                         PyLong_FromLong(sinfo->stat));
    PyDict_SetItemString(py_sinfo, "cond",
                         PyFloat_FromDouble(sinfo->cond));
    PyDict_SetItemString(py_sinfo, "cond2",
                         PyFloat_FromDouble(sinfo->cond2));
    PyDict_SetItemString(py_sinfo, "berr",
                         PyFloat_FromDouble(sinfo->berr));
    PyDict_SetItemString(py_sinfo, "berr2",
                         PyFloat_FromDouble(sinfo->berr2));
    PyDict_SetItemString(py_sinfo, "error",
                         PyFloat_FromDouble(sinfo->error));

    return py_sinfo;
}

//  *-*-*-*-*-*-*-*-*-*-   SILS_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sils_initialize(PyObject *self){

    // Call sils_initialize
    sils_initialize(&data, &control, &status);

    // Record that SILS has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = sils_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}

//  *-*-*-*-*-*-*-*-*-*-   SILS_INFORMATION   -*-*-*-*-*-*-*-*

static PyObject* py_sils_information(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sils_information
    sils_information(&data, &ainfo, &finfo, &sinfo, &status);

    // Return status and inform Python dictionary
    PyObject *py_ainfo = sils_make_ainfo_dict(&ainfo);
    PyObject *py_finfo = sils_make_finfo_dict(&finfo);
    PyObject *py_sinfo = sils_make_sinfo_dict(&sinfo);
    return Py_BuildValue("OOO", py_ainfo, py_finfo, py_sinfo);
}

//  *-*-*-*-*-*-*-*-*-*-   SILS_FINALIZE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_sils_finalize(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call sils_finalize
    sils_finalize(&data, &control, &status);

    // Record that SILS must be reinitialised if called again
    init_called = false;

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}

//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE SILS PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* sils python module method table */
static PyMethodDef sils_module_methods[] = {
    {"initialize", (PyCFunction) py_sils_initialize, METH_NOARGS,NULL},
    {"information", (PyCFunction) py_sils_information, METH_NOARGS, NULL},
    {"finalize", (PyCFunction) py_sils_finalize, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* sils python module documentation */

PyDoc_STRVAR(sils_module_doc,

"The sils package solves sparse symmetric systems of linear equations \n"
"using a multifrontal variant of Gaussian elimination.\n"
"Given a sparse symmetric matrix A = {a_ij}_mxn,  and an n-vector b,\n"
"this function solves the system Ax=b. \n"
"sils is based upon a modern fortran interface to the HSL Archive \n"
"fortran 77 package MA27. \n"
"To obtain HSL Archive packages, see http://hsl.rl.ac.uk/archive/ .\n"
"\n"
"Currently only the options and info dictionaries are exposed; these are \n"
"provided and used by other GALAHAD packages with Python interfaces.\n"
"Extended functionality is available with the sls function.\n"
"\n"
"See $GALAHAD/html/Python/sils.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* sils python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "sils",               /* name of module */
   sils_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   sils_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_sils(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
