// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// rcpp_hello
List rcpp_hello();
RcppExport SEXP _hpHawkes_rcpp_hello() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpp_hello());
    return rcpp_result_gen;
END_RCPP
}
// createEngine
Rcpp::List createEngine(int embeddingDimension, int locationCount, int tbb, int simd, int gpu, bool single);
RcppExport SEXP _hpHawkes_createEngine(SEXP embeddingDimensionSEXP, SEXP locationCountSEXP, SEXP tbbSEXP, SEXP simdSEXP, SEXP gpuSEXP, SEXP singleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type embeddingDimension(embeddingDimensionSEXP);
    Rcpp::traits::input_parameter< int >::type locationCount(locationCountSEXP);
    Rcpp::traits::input_parameter< int >::type tbb(tbbSEXP);
    Rcpp::traits::input_parameter< int >::type simd(simdSEXP);
    Rcpp::traits::input_parameter< int >::type gpu(gpuSEXP);
    Rcpp::traits::input_parameter< bool >::type single(singleSEXP);
    rcpp_result_gen = Rcpp::wrap(createEngine(embeddingDimension, locationCount, tbb, simd, gpu, single));
    return rcpp_result_gen;
END_RCPP
}
// setTimesData
void setTimesData(SEXP sexp, std::vector<double>& data);
RcppExport SEXP _hpHawkes_setTimesData(SEXP sexpSEXP, SEXP dataSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type data(dataSEXP);
    setTimesData(sexp, data);
    return R_NilValue;
END_RCPP
}
// setRandomRates
void setRandomRates(SEXP sexp, std::vector<double>& randomRates);
RcppExport SEXP _hpHawkes_setRandomRates(SEXP sexpSEXP, SEXP randomRatesSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type randomRates(randomRatesSEXP);
    setRandomRates(sexp, randomRates);
    return R_NilValue;
END_RCPP
}
// setParameters
void setParameters(SEXP sexp, std::vector<double>& parameters);
RcppExport SEXP _hpHawkes_setParameters(SEXP sexpSEXP, SEXP parametersSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type parameters(parametersSEXP);
    setParameters(sexp, parameters);
    return R_NilValue;
END_RCPP
}
// getLogLikelihoodGradient
std::vector<double> getLogLikelihoodGradient(SEXP sexp, size_t len);
RcppExport SEXP _hpHawkes_getLogLikelihoodGradient(SEXP sexpSEXP, SEXP lenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    Rcpp::traits::input_parameter< size_t >::type len(lenSEXP);
    rcpp_result_gen = Rcpp::wrap(getLogLikelihoodGradient(sexp, len));
    return rcpp_result_gen;
END_RCPP
}
// getProbsSelfExcite
std::vector<double> getProbsSelfExcite(SEXP sexp, size_t len);
RcppExport SEXP _hpHawkes_getProbsSelfExcite(SEXP sexpSEXP, SEXP lenSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    Rcpp::traits::input_parameter< size_t >::type len(lenSEXP);
    rcpp_result_gen = Rcpp::wrap(getProbsSelfExcite(sexp, len));
    return rcpp_result_gen;
END_RCPP
}
// updateLocations
void updateLocations(SEXP sexp, std::vector<double>& locations);
RcppExport SEXP _hpHawkes_updateLocations(SEXP sexpSEXP, SEXP locationsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    Rcpp::traits::input_parameter< std::vector<double>& >::type locations(locationsSEXP);
    updateLocations(sexp, locations);
    return R_NilValue;
END_RCPP
}
// getSumOfLikContribs
double getSumOfLikContribs(SEXP sexp);
RcppExport SEXP _hpHawkes_getSumOfLikContribs(SEXP sexpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type sexp(sexpSEXP);
    rcpp_result_gen = Rcpp::wrap(getSumOfLikContribs(sexp));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_hpHawkes_rcpp_hello", (DL_FUNC) &_hpHawkes_rcpp_hello, 0},
    {"_hpHawkes_createEngine", (DL_FUNC) &_hpHawkes_createEngine, 6},
    {"_hpHawkes_setTimesData", (DL_FUNC) &_hpHawkes_setTimesData, 2},
    {"_hpHawkes_setRandomRates", (DL_FUNC) &_hpHawkes_setRandomRates, 2},
    {"_hpHawkes_setParameters", (DL_FUNC) &_hpHawkes_setParameters, 2},
    {"_hpHawkes_getLogLikelihoodGradient", (DL_FUNC) &_hpHawkes_getLogLikelihoodGradient, 2},
    {"_hpHawkes_getProbsSelfExcite", (DL_FUNC) &_hpHawkes_getProbsSelfExcite, 2},
    {"_hpHawkes_updateLocations", (DL_FUNC) &_hpHawkes_updateLocations, 2},
    {"_hpHawkes_getSumOfLikContribs", (DL_FUNC) &_hpHawkes_getSumOfLikContribs, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_hpHawkes(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
