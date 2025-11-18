# Dependency installation notes (fwildclusterboot, Julia, R)

- Installed R 4.3.3 via `apt-get install r-base r-base-dev`.
- Installed Julia 1.10.4 from upstream tarball and symlinked `/usr/local/bin/julia`.
- Successfully installed `JuliaConnectoR`, `collapse`, `summclust` (GitHub: `s3alfisc/summclust`), and `fwildclusterboot` (GitHub: `s3alfisc/fwildclusterboot`) after compiling the full dependency stack (`Rcpp`, `RcppArmadillo`, `RcppEigen`, `cli`, `dqrng`, `dreamerr`, etc.).
- `fwildclusterboot`/`summclust` installs required long C++ builds; avoid interrupts. Installation logs use Student-t MDE formula downstream in the tuner.
- Updated Mega handling: use `megadl` when the pinned TLS key matches the current Mega cert, otherwise fall back to the patched `mega.py` shim (reintroduces `asyncio.coroutine` on Python 3.12+) to fetch Mega public links. If both fail, refresh the megatools pinned key (sha256 of Mega's SubjectPublicKeyInfo) and retry.
