
#include "AbstractHawkes.hpp"
#include "NewHawkes.hpp"

#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::depends(RcppParallel,BH,RcppXsimd)]]
#include <RcppParallel.h>

// [[Rcpp::export]]
List rcpp_hello() {
  CharacterVector x = CharacterVector::create("foo", "bar");
  NumericVector y   = NumericVector::create(0.0, 1.0);
  List z            = List::create(x, y);
  return z;
}

using HphSharedPtr = std::shared_ptr<hph::AbstractHawkes>;

class HphWrapper {
private:
  HphSharedPtr hph;

public:
  HphWrapper(HphSharedPtr hph) : hph(hph) { }

  HphSharedPtr& get() {
    return hph;
  }
};

using XPtrHphWrapper = Rcpp::XPtr<HphWrapper>;

HphSharedPtr& parsePtr(SEXP sexp) {
  XPtrHphWrapper ptr(sexp);
  if (!ptr) {
    Rcpp::stop("External pointer is uninitialized");
  }
  return ptr->get();
}

//' Create HPH engine object
//'
//' Helper function creates HPH engine object with given latent dimension, location count and various
//' implementation details. Called by \code{hpHawkes::engineInitial()}.
//'
//' @param embeddingDimension Dimension of latent locations.
//' @param locationCount Number of locations and size of distance matrix.
//' @param tbb Number of CPU cores to be used.
//' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
//' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
//' @param single Set \code{single=1} if your GPU does not accommodate doubles.
//' @return HPH engine object.
//'
//' @export
// [[Rcpp::export(createEngine)]]
Rcpp::List createEngine(int embeddingDimension, int locationCount, int tbb, int simd, int gpu, bool single) {

  long flags = 0L;

  int deviceNumber = -1;
  int threads = 0;
  if (gpu > 0) {
    Rcout << "Running on GPU" << std::endl;
    flags |= hph::Flags::OPENCL;
    deviceNumber = gpu;
    if(single){
      flags |= hph::Flags::FLOAT;
      Rcout << "Single precision" << std::endl;
    }
  } else {
    Rcout << "Running on CPU" << std::endl;

#if RCPP_PARALLEL_USE_TBB
  if (tbb > 0) {
    threads = tbb;
    flags |= hph::Flags::TBB;
    std::shared_ptr<tbb::task_scheduler_init> task{nullptr};
    task = std::make_shared<tbb::task_scheduler_init>(threads);
  }
#endif

  if (simd == 1) {
    flags |= hph::Flags::SSE;
  } else if (simd == 2) {
    flags |= hph::Flags::AVX;
  }

  }

  auto hph = new HphWrapper(hph::factory(embeddingDimension, locationCount,
                                         flags, deviceNumber, threads));
  XPtrHphWrapper engine(hph);

  Rcpp::List list = Rcpp::List::create(
    Rcpp::Named("engine") = engine,
    Rcpp::Named("embeddingDimension") = embeddingDimension,
    Rcpp::Named("locationCount") = locationCount,
    Rcpp::Named("timesInitialzied") = false,
    Rcpp::Named("timDiffsInitialized") = false,
    Rcpp::Named("locDistsInitialized") = false,
    Rcpp::Named("threads") = threads,
    Rcpp::Named("deviceNumber") = deviceNumber,
    Rcpp::Named("flags") = flags
  );

  return list;
}

// [[Rcpp::export(.setTimesData)]]
void setTimesData(SEXP sexp,
                     std::vector<double>& data) {
  auto ptr = parsePtr(sexp);
  ptr->setTimesData(&data[0], data.size());
}

// [[Rcpp::export(.setParameters)]]
void setParameters(SEXP sexp, std::vector<double>& parameters) {
  auto ptr = parsePtr(sexp);
  ptr->setParameters(&parameters[0], parameters.size());
}

// [[Rcpp::export(.getLogLikelihoodGradient)]]
std::vector<double> getLogLikelihoodGradient(SEXP sexp, size_t len) {
  auto ptr = parsePtr(sexp);
  std::vector<double> result(len);
  ptr->getLogLikelihoodGradient(&result[0], len);
  return result;
}

// [[Rcpp::export(.updateLocations)]]
void updateLocations(SEXP sexp,
                     std::vector<double>& locations) {
  auto ptr = parsePtr(sexp);
  ptr->updateLocations(-1, &locations[0], locations.size());
}

// [[Rcpp::export(.getSumOfLikContribs)]]
double getSumOfLikContribs(SEXP sexp) {
  auto ptr = parsePtr(sexp);
  return ptr->getSumOfLikContribs();
}
