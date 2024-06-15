//* \file cro_pyiface.c */

/*
 * THIS VERSION: GALAHAD 4.1 - 2023-05-20 AT 10:20 GMT.
 *
 *-*-*-*-*-*-*-*-*-  GALAHAD_CRO PYTHON INTERFACE  *-*-*-*-*-*-*-*-*-*-
 *
 *  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
 *  Principal author: Jaroslav Fowkes & Nick Gould
 *
 *  History -
 *   originally released GALAHAD Version 4.1. April 16th 2023
 *
 *  For full documentation, see
 *   http://galahad.rl.ac.uk/galahad-www/specs.html
 */

#include "galahad_python.h"
#include "galahad_cro.h"

/* Nested SLS, SBLS, ULS, SCU and IR control and inform prototypes */
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* sls_make_options_dict(const struct sls_control_type *control);
PyObject* sls_make_inform_dict(const struct sls_inform_type *inform);
bool sbls_update_control(struct sbls_control_type *control,
                         PyObject *py_options);
PyObject* sbls_make_options_dict(const struct sbls_control_type *control);
PyObject* sbls_make_inform_dict(const struct sbls_inform_type *inform);
bool sls_update_control(struct sls_control_type *control,
                        PyObject *py_options);
PyObject* uls_make_options_dict(const struct uls_control_type *control);
PyObject* uls_make_inform_dict(const struct uls_inform_type *inform);
bool uls_update_control(struct uls_control_type *control,
                        PyObject *py_options);
bool ir_update_control(struct ir_control_type *control,
                        PyObject *py_options);
PyObject* ir_make_options_dict(const struct ir_control_type *control);
PyObject* ir_make_inform_dict(const struct ir_inform_type *inform);
PyObject* scu_make_inform_dict(const struct scu_inform_type *inform);

/* Module global variables */
static void *data;                       // private internal data
static struct cro_control_type control;  // control struct
static struct cro_inform_type inform;    // inform struct
static bool init_called = false;         // record if initialise was called
static int status = 0;                   // exit status

//  *-*-*-*-*-*-*-*-*-*-   UPDATE CONTROL    -*-*-*-*-*-*-*-*-*-*

/* Update the control options: use C defaults but update any passed via Python*/
// NB not static as it is used for nested control within QP Python interfaces
bool cro_update_control(struct cro_control_type *control,
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
        if(strcmp(key_name, "max_schur_complement") == 0){
            if(!parse_int_option(value, "max_schur_complement",
                                  &control->max_schur_complement))
                return false;
            continue;
        }
        if(strcmp(key_name, "infinity") == 0){
            if(!parse_double_option(value, "infinity",
                                  &control->infinity))
                return false;
            continue;
        }
        if(strcmp(key_name, "feasibility_tolerance") == 0){
            if(!parse_double_option(value, "feasibility_tolerance",
                                  &control->feasibility_tolerance))
                return false;
            continue;
        }
        if(strcmp(key_name, "check_io") == 0){
            if(!parse_bool_option(value, "check_io",
                                  &control->check_io))
                return false;
            continue;
        }
        if(strcmp(key_name, "refine_solution") == 0){
            if(!parse_bool_option(value, "refine_solution",
                                  &control->refine_solution))
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
        if(strcmp(key_name, "symmetric_linear_solver") == 0){
            if(!parse_char_option(value, "symmetric_linear_solver",
                                  control->symmetric_linear_solver,
                                  sizeof(control->symmetric_linear_solver)))
                return false;
            continue;
        }
        if(strcmp(key_name, "unsymmetric_linear_solver") == 0){
            if(!parse_char_option(value, "unsymmetric_linear_solver",
                                  control->unsymmetric_linear_solver,
                                  sizeof(control->unsymmetric_linear_solver)))
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
        if(strcmp(key_name, "sls_options") == 0){
            if(!sls_update_control(&control->sls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "sbls_options") == 0){
            if(!sbls_update_control(&control->sbls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "uls_options") == 0){
            if(!uls_update_control(&control->uls_control, value))
                return false;
            continue;
        }
        if(strcmp(key_name, "ir_options") == 0){
            if(!ir_update_control(&control->ir_control, value))
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
// NB not static as it is used for nested inform within QP Python interfaces
PyObject* cro_make_options_dict(const struct cro_control_type *control){
    PyObject *py_options = PyDict_New();

    PyDict_SetItemString(py_options, "error",
                         PyLong_FromLong(control->error));
    PyDict_SetItemString(py_options, "out",
                         PyLong_FromLong(control->out));
    PyDict_SetItemString(py_options, "print_level",
                         PyLong_FromLong(control->print_level));
    PyDict_SetItemString(py_options, "max_schur_complement",
                         PyLong_FromLong(control->max_schur_complement));
    PyDict_SetItemString(py_options, "infinity",
                         PyFloat_FromDouble(control->infinity));
    PyDict_SetItemString(py_options, "feasibility_tolerance",
                         PyFloat_FromDouble(control->feasibility_tolerance));
    PyDict_SetItemString(py_options, "check_io",
                         PyBool_FromLong(control->check_io));
    PyDict_SetItemString(py_options, "refine_solution",
                         PyBool_FromLong(control->refine_solution));
    PyDict_SetItemString(py_options, "space_critical",
                         PyBool_FromLong(control->space_critical));
    PyDict_SetItemString(py_options, "deallocate_error_fatal",
                         PyBool_FromLong(control->deallocate_error_fatal));
    PyDict_SetItemString(py_options, "symmetric_linear_solver",
                         PyUnicode_FromString(control->symmetric_linear_solver));
    PyDict_SetItemString(py_options, "unsymmetric_linear_solver",
                         PyUnicode_FromString(control->unsymmetric_linear_solver));
    PyDict_SetItemString(py_options, "prefix",
                         PyUnicode_FromString(control->prefix));
    PyDict_SetItemString(py_options, "sls_options",
                         sls_make_options_dict(&control->sls_control));
    PyDict_SetItemString(py_options, "sbls_options",
                         sbls_make_options_dict(&control->sbls_control));
    PyDict_SetItemString(py_options, "uls_options",
                         uls_make_options_dict(&control->uls_control));
    PyDict_SetItemString(py_options, "ir_options",
                         ir_make_options_dict(&control->ir_control));

    return py_options;
}


//  *-*-*-*-*-*-*-*-*-*-   MAKE TIME    -*-*-*-*-*-*-*-*-*-*

/* Take the time struct from C and turn it into a python dictionary */
static PyObject* cro_make_time_dict(const struct cro_time_type *time){
    PyObject *py_time = PyDict_New();

    // Set float/double time entries

    PyDict_SetItemString(py_time, "total",
                         PyFloat_FromDouble(time->total));
    PyDict_SetItemString(py_time, "analyse",
                         PyFloat_FromDouble(time->analyse));
    PyDict_SetItemString(py_time, "factorize",
                         PyFloat_FromDouble(time->factorize));
    PyDict_SetItemString(py_time, "solve",
                         PyFloat_FromDouble(time->solve));
    PyDict_SetItemString(py_time, "clock_total",
                         PyFloat_FromDouble(time->clock_total));
    PyDict_SetItemString(py_time, "clock_analyse",
                         PyFloat_FromDouble(time->clock_analyse));
    PyDict_SetItemString(py_time, "clock_factorize",
                         PyFloat_FromDouble(time->clock_factorize));
    PyDict_SetItemString(py_time, "clock_solve",
                         PyFloat_FromDouble(time->clock_solve));

    return py_time;
}

//  *-*-*-*-*-*-*-*-*-*-   MAKE INFORM    -*-*-*-*-*-*-*-*-*-*

/* Take the inform struct from C and turn it into a python dictionary */
// NB not static as it is used for nested control within QP Python interfaces
PyObject* cro_make_inform_dict(const struct cro_inform_type *inform){
    PyObject *py_inform = PyDict_New();

    PyDict_SetItemString(py_inform, "status",
                         PyLong_FromLong(inform->status));
    PyDict_SetItemString(py_inform, "alloc_status",
                         PyLong_FromLong(inform->alloc_status));
    PyDict_SetItemString(py_inform, "bad_alloc",
                         PyUnicode_FromString(inform->bad_alloc));
    PyDict_SetItemString(py_inform, "dependent",
                         PyLong_FromLong(inform->dependent));

    // Set time nested dictionary
    PyDict_SetItemString(py_inform, "time",
                         cro_make_time_dict(&inform->time));

    // Set dictionaries from subservient packages
    PyDict_SetItemString(py_inform, "sls_inform",
                         sls_make_inform_dict(&inform->sls_inform));
    PyDict_SetItemString(py_inform, "sbls_inform",
                         sbls_make_inform_dict(&inform->sbls_inform));
    PyDict_SetItemString(py_inform, "uls_inform",
                         uls_make_inform_dict(&inform->uls_inform));
    PyDict_SetItemString(py_inform, "scu_status",
                         PyLong_FromLong(inform->scu_status));
    //PyDict_SetItemString(py_inform, "scu_inform",
    //                     scu_make_inform_dict(&inform->scu_inform));
    PyDict_SetItemString(py_inform, "ir_inform",
                         ir_make_inform_dict(&inform->ir_inform));

    return py_inform;
}

//  *-*-*-*-*-*-*-*-*-*-   CRO_INITIALIZE    -*-*-*-*-*-*-*-*-*-*

static PyObject* py_cro_initialize(PyObject *self){

    // Call cro_initialize
    cro_initialize(&data, &control, &status);

    // Record that CRO has been initialised
    init_called = true;

    // Return options Python dictionary
    PyObject *py_options = cro_make_options_dict(&control);
    return Py_BuildValue("O", py_options);
}


//  *-*-*-*-*-*-*-*-*-*-   CRO_CROSSOVER_SOLUTION   -*-*-*-*-*-*-*-*

static PyObject* py_cro_crossover_solution(PyObject *self, PyObject *args,
                                           PyObject *keywds){
    PyArrayObject *py_H_val, *py_H_col, *py_H_ptr;
    PyArrayObject *py_A_val, *py_A_col, *py_A_ptr;
    PyArrayObject *py_g, *py_c_l, *py_c_u, *py_x_l, *py_x_u;
    PyArrayObject *py_x, *py_c, *py_y, *py_z;
    PyArrayObject *py_x_stat, *py_c_stat;
    PyObject *py_options = NULL;
    double *g, *H_val, *A_val, *c_l, *c_u, *x_l, *x_u, *x, *c, *y, *z;
    int *H_col = NULL, *H_ptr = NULL;
    int *A_col = NULL, *A_ptr = NULL;
    int *x_stat, *c_stat;
    int n, m, m_equal, H_ne, A_ne;

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Parse positional and keyword arguments
    static char *kwlist[] = {"n", "m", "m_equal", "g",
                             "H_ne", "H_val", "H_col", "H_ptr",
                             "A_ne", "A_val", "A_col", "A_ptr",
                             "c_l", "c_u", "x_l", "x_u",
                             "x", "c", "y", "z", "x_stat", "c_stat",
                             "options", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiiOiOOOiOOOOOOOOOOOOO|O",
                                    kwlist, &n, &m, &m_equal, &py_g,
                                    &H_ne, &py_H_val, &py_H_col, &py_H_ptr,
                                    &A_ne, &py_A_val, &py_A_col, &py_A_ptr,
                                    &py_c_l, &py_c_u, &py_x_l, &py_x_u,
                                    &py_x, &py_c, &py_y, &py_z,
                                    &py_x_stat, &py_c_stat, &py_options))
        return NULL;

    // Check that array inputs are of correct type, size, and shape

    if(!(
        check_array_int("H_col", py_H_col, H_ne) &&
        check_array_int("H_ptr", py_H_ptr, n+1)
        ))
        return NULL;
    if(!(
        check_array_int("A_col", py_A_col, A_ne) &&
        check_array_int("A_ptr", py_A_ptr, m+1)
        ))
        return NULL;

    // Convert 64bit integer H_col array to 32bit
    if((PyObject *) py_H_col != Py_None){
        H_col = malloc(H_ne * sizeof(int));
        long int *H_col_long = (long int *) PyArray_DATA(py_H_col);
        for(int i = 0; i < H_ne; i++) H_col[i] = (int) H_col_long[i];
    }

    // Convert 64bit integer H_ptr array to 32bit
    if((PyObject *) py_H_ptr != Py_None){
        H_ptr = malloc((n+1) * sizeof(int));
        long int *H_ptr_long = (long int *) PyArray_DATA(py_H_ptr);
        for(int i = 0; i < n+1; i++) H_ptr[i] = (int) H_ptr_long[i];
    }

    // Convert 64bit integer A_col array to 32bit
    if((PyObject *) py_A_col != Py_None){
        A_col = malloc(A_ne * sizeof(int));
        long int *A_col_long = (long int *) PyArray_DATA(py_A_col);
        for(int i = 0; i < A_ne; i++) A_col[i] = (int) A_col_long[i];
    }

    // Convert 64bit integer A_ptr array to 32bit
    if((PyObject *) py_A_ptr != Py_None){
        A_ptr = malloc((m+1) * sizeof(int));
        long int *A_ptr_long = (long int *) PyArray_DATA(py_A_ptr);
        for(int i = 0; i < m+1; i++) A_ptr[i] = (int) A_ptr_long[i];
    }

    // Check that array inputs are of correct type, size, and shape
    if(!check_array_double("g", py_g, n))
        return NULL;
    if(!check_array_double("H_val", py_H_val, H_ne))
        return NULL;
    if(!check_array_double("A_val", py_A_val, A_ne))
        return NULL;
    if(!check_array_double("c_l", py_c_l, m))
        return NULL;
    if(!check_array_double("c_u", py_c_u, m))
        return NULL;
    if(!check_array_double("x_l", py_x_l, n))
        return NULL;
    if(!check_array_double("x_u", py_x_u, n))
        return NULL;
    if(!check_array_double("x", py_x, n))
        return NULL;
    if(!check_array_double("c", py_c, m))
        return NULL;
    if(!check_array_double("y", py_y, m))
        return NULL;
    if(!check_array_double("z", py_z, n))
        return NULL;
    if(!check_array_int("x_stat", py_x_stat, n))
        return NULL;
    if(!check_array_int("c_stat", py_c_stat, m))
        return NULL;

    // Get array data pointer
    H_val = (double *) PyArray_DATA(py_H_val);
    A_val = (double *) PyArray_DATA(py_A_val);
    g = (double *) PyArray_DATA(py_g);
    c_l = (double *) PyArray_DATA(py_c_l);
    c_u = (double *) PyArray_DATA(py_c_u);
    x_l = (double *) PyArray_DATA(py_x_l);
    x_u = (double *) PyArray_DATA(py_x_u);
    x = (double *) PyArray_DATA(py_x);
    c = (double *) PyArray_DATA(py_c);
    y = (double *) PyArray_DATA(py_y);
    z = (double *) PyArray_DATA(py_z);

    x_stat = malloc(n * sizeof(int));
    long int *x_stat_long = (long int *) PyArray_DATA(py_x_stat);
    for(int i = 0; i < n; i++) x_stat[i] = (int) x_stat_long[i];
    c_stat = malloc(m * sizeof(int));
    long int *c_stat_long = (long int *) PyArray_DATA(py_c_stat);
    for(int i = 0; i < m; i++) c_stat[i] = (int) c_stat_long[i];

    // Update CRO control options
    if(!cro_update_control(&control, py_options))
        return NULL;

    // Call cro_crossover_solution
    cro_crossover_solution(&data, &control, &inform, n, m, m_equal,
                           H_ne, H_val, H_col, H_ptr,
                           A_ne, A_val, A_col, A_ptr,
                           g, c_l, c_u, x_l, x_u,
                           x, c, y, z, x_stat, c_stat);

    // for( int i = 0; i < n; i++) printf("x %f\n", x[i]);
    // for( int i = 0; i < m; i++) printf("c %f\n", c[i]);
    // for( int i = 0; i < n; i++) printf("x_stat %i\n", x_stat[i]);
    // for( int i = 0; i < m; i++) printf("c_stat %i\n", c_stat[i]);

    for(int i = 0; i < n; i++) x_stat_long[i] = x_stat[i];
    for(int i = 0; i < m; i++) c_stat_long[i] = c_stat[i];

    // Free allocated memory
    free(H_col);
    free(H_ptr);
    free(A_col);
    free(A_ptr);
    free(x_stat);
    free(c_stat);

    // Propagate any errors with the callback function
    if(PyErr_Occurred())
        return NULL;

    // Raise any status errors
    if(!check_error_codes(status))
        return NULL;

    // create inform Python dictionary
    PyObject *py_inform = cro_make_inform_dict(&inform);

    // Return x, c, y, z, x_stat and c_stat
    PyObject *crossover_solution_return;
    crossover_solution_return = Py_BuildValue("OOOOOOO", py_x, py_c, py_y,
                                              py_z, py_x_stat, py_c_stat,
                                              py_inform);
    Py_INCREF(crossover_solution_return);
    return crossover_solution_return;

}

//  *-*-*-*-*-*-*-*-*-*-   CRO_TERMINATE   -*-*-*-*-*-*-*-*-*-*

static PyObject* py_cro_terminate(PyObject *self){

    // Check that package has been initialised
    if(!check_init(init_called))
        return NULL;

    // Call cro_terminate
    cro_terminate(&data, &control, &inform);

    // Return None boilerplate
    Py_INCREF(Py_None);
    return Py_None;
}


//  *-*-*-*-*-*-*-*-*-*-   INITIALIZE CRO PYTHON MODULE    -*-*-*-*-*-*-*-*-*-*

/* cro python module method table */
static PyMethodDef cro_module_methods[] = {
    {"initialize", (PyCFunction) py_cro_initialize, METH_NOARGS, NULL},
    {"crossover_solution", (PyCFunction) py_cro_crossover_solution,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"terminate", (PyCFunction) py_cro_terminate, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/* cro python module documentation */

PyDoc_STRVAR(cro_module_doc,

"The cro package provides a crossover from a primal-dual interior-point\n"
"solution to given convex quadratic program to a basic one in which \n"
"the matrix of defining active constraints/variables is of full rank. \n"
"This applies to the problem of minimizing the quadratic objective function\n"
"q(x) = f + g^T x + 1/2 x^T H x \n"
"subject to the general linear constraints and simple bounds\n"
"c_l <= A x <= c_u and x_l <= x <= x_u,\n"
"where H and A are, respectively, given n by n symmetric \n"
"postive-semi-definite and m by n matrices,  \n"
"g is a vector, f is a scalar, and any of the components \n"
"of the vectors c_l, c_u, x_l or x_u may be infinite.\n"
"The method is most suitable for problems involving a large number of \n"
"unknowns x.\n"
"\n"
"See $GALAHAD/html/Python/cro.html for argument lists, call order\n"
"and other details.\n"
"\n"
);

/* cro python module definition */
static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "cro",               /* name of module */
   cro_module_doc,      /* module documentation, may be NULL */
   -1,                  /* size of per-interpreter state of the module,or -1
                           if the module keeps state in global variables */
   cro_module_methods   /* module methods */
};

/* Python module initialization */
PyMODINIT_FUNC PyInit_cro(void) { // must be same as module name above
    import_array();  // for NumPy arrays
    return PyModule_Create(&module);
}
