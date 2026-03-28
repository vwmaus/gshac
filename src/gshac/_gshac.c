/*
 * _gshac.c — C extension module for GSHAC
 *
 * Provides optimised C implementations of performance-critical GSHAC
 * functions, callable from Python via the NumPy C API.
 *
 * Functions:
 *   linkage_from_mst(rows, cols, weights, size) -> Z
 *   haversine_edges(lon1, lat1, lon2, lat2, R) -> dists
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Union-Find helpers                                                  */
/* ------------------------------------------------------------------ */

static inline npy_intp uf_find(npy_intp *parent, npy_intp x) {
    /* Path-halving find — iterative, no recursion */
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/* ------------------------------------------------------------------ */
/* linkage_from_mst                                                    */
/*                                                                     */
/* Build a scipy-format linkage matrix Z from pre-sorted MST edges     */
/* using union-find with union-by-size and path compression.           */
/*                                                                     */
/* Parameters:                                                         */
/*   rows    : int64 array (n-1,) — MST edge source indices            */
/*   cols    : int64 array (n-1,) — MST edge target indices            */
/*   weights : float64 array (n-1,) — edge weights (sorted ascending)  */
/*   size    : int — number of nodes in the component                  */
/*                                                                     */
/* Returns:                                                            */
/*   Z : float64 array (n-1, 4) — scipy linkage matrix                */
/* ------------------------------------------------------------------ */

static PyObject *
gshac_linkage_from_mst(PyObject *self, PyObject *args)
{
    PyArrayObject *py_rows, *py_cols, *py_weights;
    int size;

    if (!PyArg_ParseTuple(args, "O!O!O!i",
                          &PyArray_Type, &py_rows,
                          &PyArray_Type, &py_cols,
                          &PyArray_Type, &py_weights,
                          &size))
        return NULL;

    /* Validate inputs */
    if (size < 2) {
        PyErr_SetString(PyExc_ValueError, "size must be >= 2");
        return NULL;
    }

    npy_intp n_edges = size - 1;

    /* Ensure contiguous int64/float64 arrays */
    PyArrayObject *rows = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_rows, NPY_INT64));
    PyArrayObject *cols = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_cols, NPY_INT64));
    PyArrayObject *weights = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_weights, NPY_DOUBLE));

    if (!rows || !cols || !weights) {
        Py_XDECREF(rows);
        Py_XDECREF(cols);
        Py_XDECREF(weights);
        return NULL;
    }

    int64_t *r_data = (int64_t *)PyArray_DATA(rows);
    int64_t *c_data = (int64_t *)PyArray_DATA(cols);
    double  *w_data = (double  *)PyArray_DATA(weights);

    /* Allocate output Z matrix */
    npy_intp z_dims[2] = {n_edges, 4};
    PyArrayObject *Z = (PyArrayObject *)PyArray_SimpleNew(2, z_dims, NPY_DOUBLE);
    if (!Z) {
        Py_DECREF(rows); Py_DECREF(cols); Py_DECREF(weights);
        return NULL;
    }
    double *z_data = (double *)PyArray_DATA(Z);

    /* Allocate union-find state */
    npy_intp *parent    = (npy_intp *)malloc(size * sizeof(npy_intp));
    npy_intp *sz        = (npy_intp *)malloc(size * sizeof(npy_intp));
    double   *sci_label = (double   *)malloc(size * sizeof(double));

    if (!parent || !sz || !sci_label) {
        free(parent); free(sz); free(sci_label);
        Py_DECREF(rows); Py_DECREF(cols); Py_DECREF(weights);
        Py_DECREF(Z);
        return PyErr_NoMemory();
    }

    /* Initialise union-find */
    for (npy_intp i = 0; i < size; i++) {
        parent[i]    = i;
        sz[i]        = 1;
        sci_label[i] = (double)i;
    }
    double next_lbl = (double)size;

    /* Main loop: merge edges in weight order */
    Py_BEGIN_ALLOW_THREADS  /* Release GIL */

    for (npy_intp k = 0; k < n_edges; k++) {
        npy_intp i = (npy_intp)r_data[k];
        npy_intp j = (npy_intp)c_data[k];
        double   w = w_data[k];

        npy_intp ri = uf_find(parent, i);
        npy_intp rj = uf_find(parent, j);

        /* Fill Z row: (label_i, label_j, distance, new_size) */
        z_data[k * 4 + 0] = sci_label[ri];
        z_data[k * 4 + 1] = sci_label[rj];
        z_data[k * 4 + 2] = w;
        z_data[k * 4 + 3] = (double)(sz[ri] + sz[rj]);

        /* Union by size */
        if (sz[ri] < sz[rj]) {
            npy_intp tmp = ri;
            ri = rj;
            rj = tmp;
        }
        parent[rj]    = ri;
        sz[ri]       += sz[rj];
        sci_label[ri] = next_lbl;
        next_lbl     += 1.0;
    }

    Py_END_ALLOW_THREADS

    free(parent);
    free(sz);
    free(sci_label);
    Py_DECREF(rows);
    Py_DECREF(cols);
    Py_DECREF(weights);

    return (PyObject *)Z;
}


/* ------------------------------------------------------------------ */
/* haversine_edges                                                     */
/*                                                                     */
/* Compute haversine distances for m pairs of (lon, lat) in radians.   */
/*                                                                     */
/* Parameters:                                                         */
/*   lon1, lat1, lon2, lat2 : float64 arrays (m,) — coords in radians */
/*   R : float — Earth radius in metres (default 6_371_000)            */
/*                                                                     */
/* Returns:                                                            */
/*   dists : float64 array (m,) — haversine distances in metres        */
/* ------------------------------------------------------------------ */

static PyObject *
gshac_haversine_edges(PyObject *self, PyObject *args)
{
    PyArrayObject *py_lon1, *py_lat1, *py_lon2, *py_lat2;
    double R;

    if (!PyArg_ParseTuple(args, "O!O!O!O!d",
                          &PyArray_Type, &py_lon1,
                          &PyArray_Type, &py_lat1,
                          &PyArray_Type, &py_lon2,
                          &PyArray_Type, &py_lat2,
                          &R))
        return NULL;

    /* Ensure contiguous float64 */
    PyArrayObject *lon1 = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_lon1, NPY_DOUBLE));
    PyArrayObject *lat1 = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_lat1, NPY_DOUBLE));
    PyArrayObject *lon2 = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_lon2, NPY_DOUBLE));
    PyArrayObject *lat2 = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_lat2, NPY_DOUBLE));

    if (!lon1 || !lat1 || !lon2 || !lat2) {
        Py_XDECREF(lon1); Py_XDECREF(lat1);
        Py_XDECREF(lon2); Py_XDECREF(lat2);
        return NULL;
    }

    npy_intp m = PyArray_SIZE(lon1);

    double *d_lon1 = (double *)PyArray_DATA(lon1);
    double *d_lat1 = (double *)PyArray_DATA(lat1);
    double *d_lon2 = (double *)PyArray_DATA(lon2);
    double *d_lat2 = (double *)PyArray_DATA(lat2);

    /* Allocate output */
    npy_intp dims[1] = {m};
    PyArrayObject *dists = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!dists) {
        Py_DECREF(lon1); Py_DECREF(lat1);
        Py_DECREF(lon2); Py_DECREF(lat2);
        return NULL;
    }
    double *d_out = (double *)PyArray_DATA(dists);

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp i = 0; i < m; i++) {
        double dlat = d_lat2[i] - d_lat1[i];
        double dlon = d_lon2[i] - d_lon1[i];
        double slat = sin(dlat * 0.5);
        double slon = sin(dlon * 0.5);
        double a = slat * slat + cos(d_lat1[i]) * cos(d_lat2[i]) * slon * slon;
        double c = 2.0 * asin(sqrt(a));
        d_out[i] = R * c;
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(lon1); Py_DECREF(lat1);
    Py_DECREF(lon2); Py_DECREF(lat2);

    return (PyObject *)dists;
}


/* ------------------------------------------------------------------ */
/* fcluster_batch                                                      */
/*                                                                     */
/* Cut a linkage matrix Z at multiple thresholds in one pass.          */
/* Equivalent to calling scipy.cluster.hierarchy.fcluster(Z, t,        */
/* criterion='distance') for each t, but avoids per-call overhead.     */
/*                                                                     */
/* Parameters:                                                         */
/*   Z      : float64 array (n-1, 4) — scipy linkage matrix           */
/*   h_cuts : float64 array (k,) — cut thresholds (sorted ascending)  */
/*   size   : int — number of original points (= n)                   */
/*                                                                     */
/* Returns:                                                            */
/*   labels : int64 array (k, n) — cluster labels per cut             */
/* ------------------------------------------------------------------ */

static PyObject *
gshac_fcluster_batch(PyObject *self, PyObject *args)
{
    PyArrayObject *py_Z, *py_hcuts;
    int size;

    if (!PyArg_ParseTuple(args, "O!O!i",
                          &PyArray_Type, &py_Z,
                          &PyArray_Type, &py_hcuts,
                          &size))
        return NULL;

    PyArrayObject *Z = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_Z, NPY_DOUBLE));
    PyArrayObject *hcuts = (PyArrayObject *)PyArray_GETCONTIGUOUS(
        (PyArrayObject *)PyArray_Cast(py_hcuts, NPY_DOUBLE));

    if (!Z || !hcuts) {
        Py_XDECREF(Z);
        Py_XDECREF(hcuts);
        return NULL;
    }

    npy_intp n_merges = size - 1;
    npy_intp n_cuts = PyArray_SIZE(hcuts);
    double *z_data = (double *)PyArray_DATA(Z);
    double *h_data = (double *)PyArray_DATA(hcuts);

    /* Allocate output labels (n_cuts x size) */
    npy_intp out_dims[2] = {n_cuts, size};
    PyArrayObject *labels = (PyArrayObject *)PyArray_SimpleNew(
        2, out_dims, NPY_INT64);
    if (!labels) {
        Py_DECREF(Z); Py_DECREF(hcuts);
        return NULL;
    }
    int64_t *lab_data = (int64_t *)PyArray_DATA(labels);

    /* Allocate cluster-membership array: which cluster does each
       original point / merged cluster belong to? */
    int64_t *membership = (int64_t *)malloc((2 * size - 1) * sizeof(int64_t));
    if (!membership) {
        Py_DECREF(Z); Py_DECREF(hcuts); Py_DECREF(labels);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp c = 0; c < n_cuts; c++) {
        double threshold = h_data[c];
        int64_t next_cluster = 1;

        /* Initially each original point is its own cluster */
        for (npy_intp i = 0; i < 2 * size - 1; i++)
            membership[i] = 0;  /* 0 = unassigned */

        /* Walk merges; if merge distance > threshold, stop merging */
        for (npy_intp k = 0; k < n_merges; k++) {
            double dist = z_data[k * 4 + 2];
            if (dist > threshold) {
                /* Remaining merges are above threshold — all unmerged
                   points/clusters get their own labels */
                break;
            }
            /* Mark this merged cluster (index = size + k) as a valid merge */
            membership[size + k] = -1;  /* -1 = merged, label assigned later */
        }

        /* Assign labels: walk the tree top-down.
           For each leaf or unmerged cluster, assign a unique label.
           For merged clusters, propagate the merge label down. */

        /* Process in reverse merge order (top-down) */
        for (npy_intp k = n_merges - 1; k >= 0; k--) {
            npy_intp node = size + k;
            int64_t left  = (int64_t)z_data[k * 4 + 0];
            int64_t right = (int64_t)z_data[k * 4 + 1];
            double  dist  = z_data[k * 4 + 2];

            if (dist > threshold) {
                /* This merge didn't happen — left and right are separate */
                continue;
            }

            if (membership[node] == 0 || membership[node] == -1) {
                /* This is a root of a merged subtree — assign a new label */
                membership[node] = next_cluster++;
            }

            /* Propagate label to children */
            if (left < 2 * size - 1)
                membership[left] = membership[node];
            if (right < 2 * size - 1)
                membership[right] = membership[node];
        }

        /* Assign labels to original points */
        for (npy_intp i = 0; i < size; i++) {
            if (membership[i] <= 0) {
                /* Point was never merged — its own cluster */
                membership[i] = next_cluster++;
            }
            lab_data[c * size + i] = membership[i];
        }
    }

    Py_END_ALLOW_THREADS

    free(membership);
    Py_DECREF(Z);
    Py_DECREF(hcuts);

    return (PyObject *)labels;
}


/* ------------------------------------------------------------------ */
/* Module definition                                                   */
/* ------------------------------------------------------------------ */

static PyMethodDef gshac_methods[] = {
    {"linkage_from_mst", gshac_linkage_from_mst, METH_VARARGS,
     "Build scipy linkage matrix from sorted MST edges via union-find.\n\n"
     "Parameters\n"
     "----------\n"
     "rows, cols : int64 arrays (n-1,)\n"
     "    MST edge indices, sorted by weight.\n"
     "weights : float64 array (n-1,)\n"
     "    Edge weights (ascending order).\n"
     "size : int\n"
     "    Number of nodes in the component.\n\n"
     "Returns\n"
     "-------\n"
     "Z : float64 array (n-1, 4)\n"
     "    Scipy-format linkage matrix.\n"},

    {"fcluster_batch", gshac_fcluster_batch, METH_VARARGS,
     "Cut linkage matrix at multiple thresholds in one call.\n\n"
     "Parameters\n"
     "----------\n"
     "Z : float64 array (n-1, 4)\n"
     "    Scipy-format linkage matrix.\n"
     "h_cuts : float64 array (k,)\n"
     "    Cut thresholds (sorted ascending).\n"
     "size : int\n"
     "    Number of original points.\n\n"
     "Returns\n"
     "-------\n"
     "labels : int64 array (k, n)\n"
     "    Cluster labels for each cut height (1-indexed).\n"},

    {"haversine_edges", gshac_haversine_edges, METH_VARARGS,
     "Compute haversine distances for pairs of (lon, lat) in radians.\n\n"
     "Parameters\n"
     "----------\n"
     "lon1, lat1, lon2, lat2 : float64 arrays (m,)\n"
     "    Coordinates in radians.\n"
     "R : float\n"
     "    Earth radius in metres.\n\n"
     "Returns\n"
     "-------\n"
     "dists : float64 array (m,)\n"
     "    Great-circle distances in metres.\n"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef gshac_module = {
    PyModuleDef_HEAD_INIT,
    "_gshac",
    "C extension for GSHAC (Sparse Geographic HAC).\n"
    "Provides optimised union-find linkage and haversine distance computation.",
    -1,
    gshac_methods
};

PyMODINIT_FUNC
PyInit__gshac(void)
{
    import_array();  /* Initialise NumPy C API */
    return PyModule_Create(&gshac_module);
}
