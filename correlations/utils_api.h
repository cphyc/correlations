/* Generated by Cython 0.28.2 */

#ifndef __PYX_HAVE_API__correlations__utils
#define __PYX_HAVE_API__correlations__utils
#include "Python.h"
#include "utils.h"

static double (*__pyx_api_f_12correlations_5utils_integrand)(double, double, double, int, int, int, int, double, double, double, double, double, int __pyx_skip_dispatch) = 0;
#define integrand __pyx_api_f_12correlations_5utils_integrand
static double (*__pyx_api_f_12correlations_5utils_integrand_lambdaCDM)(double, double, int, int, int, int, double, double, double, double, double, int __pyx_skip_dispatch) = 0;
#define integrand_lambdaCDM __pyx_api_f_12correlations_5utils_integrand_lambdaCDM
#if !defined(__Pyx_PyIdentifier_FromString)
#if PY_MAJOR_VERSION < 3
  #define __Pyx_PyIdentifier_FromString(s) PyString_FromString(s)
#else
  #define __Pyx_PyIdentifier_FromString(s) PyUnicode_FromString(s)
#endif
#endif

#ifndef __PYX_HAVE_RT_ImportModule
#define __PYX_HAVE_RT_ImportModule
static PyObject *__Pyx_ImportModule(const char *name) {
    PyObject *py_name = 0;
    PyObject *py_module = 0;
    py_name = __Pyx_PyIdentifier_FromString(name);
    if (!py_name)
        goto bad;
    py_module = PyImport_Import(py_name);
    Py_DECREF(py_name);
    return py_module;
bad:
    Py_XDECREF(py_name);
    return 0;
}
#endif

#ifndef __PYX_HAVE_RT_ImportFunction
#define __PYX_HAVE_RT_ImportFunction
static int __Pyx_ImportFunction(PyObject *module, const char *funcname, void (**f)(void), const char *sig) {
    PyObject *d = 0;
    PyObject *cobj = 0;
    union {
        void (*fp)(void);
        void *p;
    } tmp;
    d = PyObject_GetAttrString(module, (char *)"__pyx_capi__");
    if (!d)
        goto bad;
    cobj = PyDict_GetItemString(d, funcname);
    if (!cobj) {
        PyErr_Format(PyExc_ImportError,
            "%.200s does not export expected C function %.200s",
                PyModule_GetName(module), funcname);
        goto bad;
    }
#if PY_VERSION_HEX >= 0x02070000
    if (!PyCapsule_IsValid(cobj, sig)) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, PyCapsule_GetName(cobj));
        goto bad;
    }
    tmp.p = PyCapsule_GetPointer(cobj, sig);
#else
    {const char *desc, *s1, *s2;
    desc = (const char *)PyCObject_GetDesc(cobj);
    if (!desc)
        goto bad;
    s1 = desc; s2 = sig;
    while (*s1 != '\0' && *s1 == *s2) { s1++; s2++; }
    if (*s1 != *s2) {
        PyErr_Format(PyExc_TypeError,
            "C function %.200s.%.200s has wrong signature (expected %.500s, got %.500s)",
             PyModule_GetName(module), funcname, sig, desc);
        goto bad;
    }
    tmp.p = PyCObject_AsVoidPtr(cobj);}
#endif
    *f = tmp.fp;
    if (!(*f))
        goto bad;
    Py_DECREF(d);
    return 0;
bad:
    Py_XDECREF(d);
    return -1;
}
#endif


static int import_correlations__utils(void) {
  PyObject *module = 0;
  module = __Pyx_ImportModule("correlations.utils");
  if (!module) goto bad;
  if (__Pyx_ImportFunction(module, "integrand", (void (**)(void))&__pyx_api_f_12correlations_5utils_integrand, "double (double, double, double, int, int, int, int, double, double, double, double, double, int __pyx_skip_dispatch)") < 0) goto bad;
  if (__Pyx_ImportFunction(module, "integrand_lambdaCDM", (void (**)(void))&__pyx_api_f_12correlations_5utils_integrand_lambdaCDM, "double (double, double, int, int, int, int, double, double, double, double, double, int __pyx_skip_dispatch)") < 0) goto bad;
  Py_DECREF(module); module = 0;
  return 0;
  bad:
  Py_XDECREF(module);
  return -1;
}

#endif /* !__PYX_HAVE_API__correlations__utils */
