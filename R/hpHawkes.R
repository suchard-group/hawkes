#' hpHawkes
#'
#' hpHawkes facilitates fast Bayesian inference for Hawkes processes through GPU, multi-core CPU, and SIMD vectorization powered implementations of the Hamiltonian Monte Carlo algorithm.
#' The package relies on Rcpp.
#'
#' @docType package
#' @name hpHawkes
#' @author Marc Suchard and Andrew Holbrook
#' @importFrom Rcpp evalCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats dist dnorm pnorm rWishart rnorm runif
#' @importFrom utils read.table
#' @importFrom mvtnorm dmvnorm
#' @useDynLib hpHawkes
NULL
