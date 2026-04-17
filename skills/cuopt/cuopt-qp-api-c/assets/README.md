# Assets — QP C API reference

QP uses the same cuOpt C library as LP/MILP; the API extends to quadratic objectives.

**Build and run:** Use the same include/lib paths and link steps as for LP/MILP C (see repository documentation for build and examples). Then use the QP-specific creation and solve calls from the cuOpt C headers.

**Repo docs:** `docs/cuopt/source/cuopt-c/lp-qp-milp/` for QP C API and examples; parameter constants and CSR format are in the same doc tree.

No standalone QP C source files are included in this skill; adapt the LP/MILP C build pattern for quadratic objective APIs from the headers.
