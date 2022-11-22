#define NPY_NO_DEPRECATED_API NPY_1_20_API_VERSION

#include <stdbool.h>
#include <stdint.h>

// include guard
#ifndef GALAHAD_PYTHON_H
#define GALAHAD_PYTHON_H

// Python and NumPy C APIs
#include <Python.h>
#include <numpy/arrayobject.h>


/*
 * Check and handle general status error codes
 */
static inline bool check_error_codes(int status){
    switch(status){
        case -1: // NB this is a RutimeError as opposed to a warning
            PyErr_SetString(PyExc_RuntimeError,
            "an allocation error occurred. A message indicating the offending"
            " array is written on unit options['error'], and the returned allocation"
            " status and a string containing the name of the offending array"
            " are held in inform['alloc_status'] and inform['bad_alloc'] respectively."
            );
            return false; // errors return false
        case -2: // NB this is a RutimeError as opposed to a warning
            PyErr_SetString(PyExc_RuntimeError,
            "a deallocation error occurred. A message indicating the offending"
            " array is written on unit options['error'] and the returned allocation"
            " status and a string containing the name of the offending array"
            " are held in inform['alloc_status'] and inform['bad_alloc'] respectively."
            );
            return false; // errors return false
        case -3: // NB this is a ValueError as n or H_type have the wrong values
            PyErr_SetString(PyExc_ValueError,
            "the restriction n > 0 or requirement that H_type contains"
            " its relevant string 'dense', 'coordinate', 'sparse_by_rows',"
            " 'diagonal' or 'absent' has been violated."
            );
            return false; // errors return false
        case -4:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -5:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -6:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -7:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "the objective function appears to be unbounded from below."
            ,2); // raise in module
            return true; // warnings return true
        case -8:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -9:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "the analysis phase of the factorization failed; the return status"
            " from the factorization package is given in inform['factor_status']."
            ,2); // raise in module
            return true; // warnings return true
        case -10:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "the factorization failed; the return status from the factorization"
            " package is given in inform['factor_status']."
            ,2); // raise in module
            return true; // warnings return true
        case -11:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "the solution of a set of linear equations using factors from the"
            " factorization package failed; the return status from the factorization"
            " package is given in inform['factor_status']."
            ,2); // raise in module
            return true; // warnings return true
        case -12:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -13:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -14:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -15:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -16:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "The problem is so ill-conditioned that further progress is impossible."
            ,2); // raise in module
            return true; // warnings return true
        case -17:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -18:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "too many iterations have been performed. This may happen if"
            " options['maxit'] is too small, but may also be symptomatic of"
            " a badly scaled problem."
            ,2); // raise in module
            return true; // warnings return true
        case -19:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "the CPU time limit has been reached. This may happen if"
            " options['cpu_time_limit'] is too small, but may also be symptomatic of"
            " a badly scaled problem."
            ,2); // raise in module
            return true; // warnings return true
        case -20:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "TODO"
            ,2); // raise in module
            return true; // warnings return true
        case -40:
            PyErr_WarnEx(PyExc_RuntimeWarning,
            "the user has forced termination of solver by removing the file"
            " named options['alive_file'] from unit options['alive_unit']."
            ,2); // raise in module
            return true; // warnings return true
        default:
            return true; // no error
    }
}


/*
 * General helper functions
 */

/* Check that package has been initialised */
static inline bool check_init(bool init_called){
    if(!init_called){
        PyErr_SetString(PyExc_Exception, "package has not been initialised");
        return false;
    }
    return true;
}

/* Check that argument is callable */
static inline bool check_callable(PyObject *arg){
    if(!PyCallable_Check(arg)){
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        return false;
    }
    return true;
}

/*
 * Check that NumPy ndarrays have the correct type, size, and shape
 */

/* Check that ndarray is 1D, double and of correct length */
static inline bool check_array_double(char *name, PyArrayObject *arr, int n){
    if(!(PyArray_Check(arr) && PyArray_ISFLOAT(arr) && PyArray_TYPE(arr)==NPY_DOUBLE
         && PyArray_NDIM(arr)==1 && PyArray_DIM(arr,0)==n)){
        PyErr_Format(PyExc_TypeError, "%s must be a 1D double array of length %i", name, n);
        return false;
    }
    return true;
}

/* Check that ndarray is 1D, int and of correct length */
static inline bool check_array_int(char *name, PyArrayObject *arr, int n){
    if((PyObject *) arr == Py_None) // allowed to be None
        return true;
    if(!(PyArray_Check(arr) && PyArray_ISINTEGER(arr) && PyArray_TYPE(arr)==NPY_INT64
         && PyArray_NDIM(arr)==1 && PyArray_DIM(arr,0)==n)){
        PyErr_Format(PyExc_TypeError, "%s must be a 1D int array of length %i", name, n);
        return false;
    }
    return true;
}

/*
 * General parsing helper functions
 */

/* Parse double from Python value to C out */
static inline bool parse_double(char *name, PyObject *value, double *out){
    *out = PyFloat_AsDouble(value);
    if(*out == -1.0 && PyErr_Occurred()){
        PyErr_Format(PyExc_TypeError, "%s must be a float", name);
        return false;
    }
    return true;
}

/*
 * Python options dictionary to C control struct parsing helper functions
 */

/* Parse options dictionary key from Python key to C key_name */
static inline bool parse_options_key(PyObject *key, const char **key_name){
    *key_name = PyUnicode_AsUTF8AndSize(key, NULL);
    if(!*key_name){
        PyErr_SetString(PyExc_TypeError, "the keys in the options dictionary must be strings");
        return false;
    }
    return true;
}

/* Parse int option from Python value to C out */
static inline bool parse_int_option(PyObject *value, char *option_name, int *out){
    *out = PyLong_AsLong(value);
    if(*out == -1 && PyErr_Occurred()){
        PyErr_Format(PyExc_TypeError, "options['%s'] must be an integer", option_name);
        return false;
    }
    return true;
}

/* Parse double option from Python value to C out */
static inline bool parse_double_option(PyObject *value, char *option_name, double *out){
    *out = PyFloat_AsDouble(value);
    if(*out == -1.0 && PyErr_Occurred()){
        PyErr_Format(PyExc_TypeError, "options['%s'] must be a float", option_name);
        return false;
    }
    return true;
}

/* Parse bool option from Python value to C out */
static inline bool parse_bool_option(PyObject *value, char *option_name, bool *out){
    if(!PyBool_Check(value)){
        PyErr_Format(PyExc_TypeError, "options['%s'] must be a bool", option_name);
        return false;
    }
    int vint = PyObject_IsTrue(value);
    if(vint == 1){
	    *out = true;
	}else if(vint == 0){
	    *out = false;
	}else{ // returns -1 on error, should never reach this
        PyErr_Format(PyExc_TypeError, "error parsing bool options['%s']", option_name);
        return false;
    }
    return true;
}

/* Parse char option from Python value to C out */
static inline bool parse_char_option(PyObject *value, char *option_name, const char out[]){
    out = PyUnicode_AsUTF8AndSize(value, NULL);
    if(!out){
        PyErr_Format(PyExc_TypeError, "options['%s'] must be a string", option_name);
        return false;
    }
    return true;
}

// end include guard
#endif
