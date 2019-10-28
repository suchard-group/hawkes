#ifndef _NEWHAWKES_HPP
#define _NEWHAWKES_HPP

#include <numeric>
#include <vector>

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include "tbb/global_control.h"

#ifdef RBUILD
#include <Rcpp.h>
#endif


//#define XSIMD_ENABLE_FALLBACK

#include "xsimd/xsimd.hpp"
#include "AbstractHawkes.hpp"
#include "Distance.hpp"


namespace hph {

	struct DoubleNoSimdTypeInfo {
		using BaseType = double;
		using SimdType = double;
		static const int SimdSize = 1;
	};

    struct FloatNoSimdTypeInfo {
        using BaseType = float;
        using SimdType = float;
        static const int SimdSize = 1;
    };

#ifdef USE_SIMD

#ifdef USE_SSE
    struct DoubleSseTypeInfo {
        using BaseType = double;
        using SimdType = xsimd::batch<double, 2>;
        static const int SimdSize = 2;
    };

    struct FloatSseTypeInfo {
        using BaseType = float;
        using SimdType = xsimd::batch<float, 4>;
        static const int SimdSize = 4;
    };
#endif

#ifdef USE_AVX
    struct DoubleAvxTypeInfo {
        using BaseType = double;
        using SimdType = xsimd::batch<double, 4>;
        static const int SimdSize = 4;
    };
#endif

#ifdef USE_AVX512
    struct DoubleAvx512TypeInfo {
        using BaseType = double;
        using SimdType = xsimd::batch<double, 8>;
        static const int SimdSize = 8;
    };
#endif

#endif

template <typename TypeInfo, typename ParallelType>
class NewHawkes : public AbstractHawkes {
public:

	using RealType = typename TypeInfo::BaseType;

    NewHawkes(int embeddingDimension, int locationCount, long flags, int threads)
        : AbstractHawkes(embeddingDimension, locationCount, flags),
          sigmaXprec(0.0), storedSigmaXprec(0.0),
          tauXprec(0.0), storedTauXprec(0.0),
          tauTprec(0.0), storedTauTprec(0.0),
          omega(0.0), storedOmega(0.0),
          theta(0.0), storedTheta(0.0),
          mu0(0.0), storedMu0(0.0),

          locDists(locationCount * locationCount),
          timDiffs(locationCount * locationCount),

          times(locationCount),

          sumOfLikContribs(0.0), storedSumOfLikContribs(0.0),

          likContribs(locationCount),
          storedLikContribs(locationCount),

          isStoredLikContribsEmpty(false),

          nThreads(threads)
    {



#ifdef USE_TBB
        if (flags & hph::Flags::TBB) {
    		if (nThreads <= 0) {
                nThreads = tbb::task_scheduler_init::default_num_threads();
    		}
#ifdef RBUILD
    		    Rcpp::Rcout << "Using " << nThreads << " threads" << std::endl;
#else
            std::cout << "Using " << nThreads << " threads" << std::endl;
#endif

            control = std::make_shared<tbb::global_control>(tbb::global_control::max_allowed_parallelism, nThreads);
    	}
#endif
    }



    virtual ~NewHawkes() { }

	int getInternalDimension() { return embeddingDimension; }

    double getSumOfLikContribs() {
        computeSumOfLikContribsGeneric<typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
    	return sumOfLikContribs;
 	}

    void storeState() {
    	storedSumOfLikContribs = sumOfLikContribs;
    	isStoredLikContribsEmpty = true;

        sigmaXprec = storedSigmaXprec;
        tauXprec = storedTauXprec;
        tauTprec = storedTauTprec;
        omega = storedOmega;
        theta = storedTheta;
        mu0 = storedMu0;
    }

    void acceptState() {
        if (!isStoredLikContribsEmpty) {
    		for (int j = 0; j < locationCount; ++j) {
    			likContribs[j * locationCount + updatedLocation] = likContribs[updatedLocation * locationCount + j];
    		}
    	}
    }

    void restoreState() {
    	sumOfLikContribs = storedSumOfLikContribs;

        sigmaXprec = storedSigmaXprec;
        tauXprec = storedTauXprec;
        tauTprec = storedTauTprec;
        omega = storedOmega;
        theta = storedTheta;
        mu0 = storedMu0;
    }

    void setLocDistsData(double* data, size_t length) {
		assert(length == locDists.size());
		mm::bufferedCopy(data, data + length, begin(locDists), buffer);
    }

    void setTimDiffsData(double* data, size_t length) {
        assert(length == timDiffs.size());
        mm::bufferedCopy(data, data + length, begin(timDiffs), buffer);
    }

    void setTimesData(double* data, size_t length) {
        assert(length == times.size());
        mm::bufferedCopy(data, data + length, begin(times), buffer);
    }

    void setParameters(double* data, size_t length) {
        assert(length == 6);
        sigmaXprec = data[0];
        tauXprec = data[1];
        tauTprec = data[2];
        omega = data[3];
        theta = data[4];
        mu0 = data[5];
    }

//	void getLogLikelihoodGradient(double* result, size_t length) {
//
//		assert (length == locationCount * embeddingDimension);
//
//        // TODO Cache values
//
//		if (isLeftTruncated) { // run-time dispatch to compile-time optimization
//			if (embeddingDimension == 2) {
//				computeLogLikelihoodGradientGeneric<true, typename TypeInfo::SimdType, TypeInfo::SimdSize, NonGeneric>();
//			} else {
//				computeLogLikelihoodGradientGeneric<true, typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
//			}
//		} else {
//			if (embeddingDimension == 2) {
//				computeLogLikelihoodGradientGeneric<false, typename TypeInfo::SimdType, TypeInfo::SimdSize, NonGeneric>();
//			} else {
//				computeLogLikelihoodGradientGeneric<false, typename TypeInfo::SimdType, TypeInfo::SimdSize, Generic>();
//			}
//		}
//
//		mm::bufferedCopy(std::begin(*gradientPtr), std::end(*gradientPtr), result, buffer);
//    }

//	template <bool withTruncation, typename SimdType, int SimdSize, typename Algorithm>
//    void computeLogLikelihoodGradientGeneric() {
//
//        const auto length = locationCount * embeddingDimension;
//        if (length != gradientPtr->size()) {
//            gradientPtr->resize(length);
//        }
//
//        std::fill(std::begin(*gradientPtr), std::end(*gradientPtr),
//                  static_cast<RealType>(0.0));
//
//        //const auto dim = embeddingDimension;
//        //RealType* gradient = gradientPtr->data();
//        const RealType scale = precision;
//
//        for_each(0, locationCount, [this, scale](const int i) { // [gradient,dim]
//
//			const int vectorCount = locationCount - locationCount % SimdSize;
//
//			DistanceDispatch<SimdType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);
//
//			innerGradientLoop<withTruncation, SimdType, SimdSize>(dispatch, scale, i, 0, vectorCount);
//
//			if (vectorCount < locationCount) { // Edge-cases
//
//				DistanceDispatch<RealType, RealType, Algorithm> dispatch(*locationsPtr, i, embeddingDimension);
//
//				innerGradientLoop<withTruncation, RealType, 1>(dispatch, scale, i, vectorCount, locationCount);
//			}
//        }, ParallelType());
//    }

#ifdef USE_SIMD

	template <typename T, size_t N>
	using SimdBatch = xsimd::batch<T, N>;

	template <typename T, size_t N>
	using SimdBatchBool = xsimd::batch_bool<T, N>;

	template <typename T, size_t N>
	bool any(SimdBatchBool<T, N> x) {
	    return xsimd::any(x);
	}

	template <typename T, size_t N>
	T reduce(SimdBatch<T, N> x) {
	    return xsimd::hadd(x);
	}

	template <typename T, size_t N>
	SimdBatch<T, N> mask(SimdBatchBool<T, N> flag, SimdBatch<T, N> x) {
	    return SimdBatch<T, N>(flag()) & x;
	}

	template <typename T, size_t N>
	SimdBatchBool<T, N> getMissing(int i, int j, SimdBatch<T, N> x) {
        return SimdBatch<T, N>(i) == (SimdBatch<T, N>(j) + getIota(SimdBatch<T, N>())) || xsimd::isnan(x);
	}
#endif

#ifdef USE_SSE
	using D2 = xsimd::batch<double, 2>;
	using D2Bool = xsimd::batch_bool<double, 2>;
	using S4 = xsimd::batch<float, 4>;
	using S4Bool = xsimd::batch_bool<float, 4>;

    D2 getIota(D2) {
        return D2(0, 1);
    }

    S4 getIota(S4) {
        return S4(0, 1, 2, 3);
    }
#endif

#ifdef USE_AVX
	using D4 = xsimd::batch<double, 4>;
	using D4Bool = xsimd::batch_bool<double, 4>;

	D4 getIota(D4) {
        return D4(0, 1, 2, 3);
    }
#endif

#ifdef USE_AVX512
    using D8 = xsimd::batch<double, 8>;
	using D8Bool = xsimd::batch_bool<double, 8>;

	D8 getIota(D8) {
	    return D8(0, 1, 2, 3, 4, 5, 6, 7);
	}

	D8 mask(D8Bool flag, D8 x) {
		return xsimd::select(flag, x, D8(0.0)); // bitwise & does not appear to work
    }
#endif

    template <typename T>
    bool getMissing(int i, int j, T x) {
        return i == j || std::isnan(x);
    }

    template <typename T>
    T mask(bool flag, T x) {
        return flag ? x : T(0.0);
    }

    template <typename T>
    T reduce(T x) {
        return x;
    }

	bool any(bool x) {
		return x;
	}

//	template <bool withTruncation, typename SimdType, int SimdSize, typename DispatchType>
//	void innerGradientLoop(const DispatchType& dispatch, const RealType scale, const int i,
//								 const int begin, const int end) {
//
//        const SimdType sqrtScale(std::sqrt(scale));
//
//		for (int j = begin; j < end; j += SimdSize) {
//
//			const auto distance = dispatch.calculate(j);
//			const auto observation = SimdHelper<SimdType, RealType>::get(&observations[i * locationCount + j]);
//			const auto notMissing = !getMissing(i, j, observation);
//
//			if (any(notMissing)) {
//
//				auto residual = mask(notMissing, observation - distance);
//
//				if (withTruncation) {
//
//					residual -= mask(notMissing, math::pdf_new( distance * sqrtScale ) /
//									  (xsimd::exp(math::phi_new(distance * sqrtScale)) *
//									   sqrtScale) );
//				}
//
//				auto dataContribution = mask(notMissing, residual * scale / distance);
//
//                for (int k = 0; k < SimdSize; ++k) {
//                    for (int d = 0; d < embeddingDimension; ++d) {
//
//                        const RealType something = getScalar(dataContribution, k);
//
//                        const RealType update = something *
//                                                ((*locationsPtr)[i * embeddingDimension + d] -
//                                                 (*locationsPtr)[(j + k) * embeddingDimension + d]);
//
//
//                        (*gradientPtr)[i * embeddingDimension + d] += update;
//                    }
//                }
//			}
//		}
//	}

	double getScalar(double x, int i) {
		return x;
	}

	float getScalar(float x, int i) {
		return x;
	}

#ifdef USE_SIMD
#ifdef USE_AVX
	double getScalar(D4 x, int i) {
		return x[i];
	}
#endif
#ifdef USE_SSE
	double getScalar(D2 x, int i) {
		return x[i];
	}

	float getScalar(S4 x, int i) {
		return x[i];
	}
#endif
#endif // USE_SIMD

#ifdef USE_AVX512
    double getScalar(D8 x, int i) {
	    return x[i];
	}
#endif

    template <typename SimdType, int SimdSize>
    RealType innerLikelihoodLoop(const int i, const int begin, const int end) {

        SimdType sum = SimdType(RealType(0));

        for (int j = begin; j < end; j += SimdSize) {

            const auto locDist = SimdHelper<SimdType, RealType>::get(&locDists[i * locationCount + j]);
            const auto timDiff = SimdHelper<SimdType, RealType>::get(&timDiffs[i * locationCount + j]);

            const auto rate = mu0 * tauXprec * tauTprec * math::pdf_new(locDist * tauXprec) *
                    math::pdf_new( timDiff*tauTprec ) +
                    mask(timDiff>0,
                         xsimd::exp(-omega*timDiff) * math::pdf_new(locDist*sigmaXprec));;

            SimdHelper<SimdType, RealType>::put(rate, &likContribs[i * locationCount + j]);
            sum += rate;
        }

        return reduce(sum);
    }

    template <typename SimdType, int SimdSize, typename Algorithm>
    void computeSumOfLikContribsGeneric() {

        RealType delta =
                accumulate(0, locationCount, RealType(0), [this](const int i) {

                    const int vectorCount = locationCount - locationCount % SimdSize;

					RealType sumOfRates = xsimd::log(innerLikelihoodLoop<SimdType, SimdSize>(i, 0, vectorCount)) +
					        theta/omega*(xsimd::exp(-omega*(times[locationCount]-times[i]))-1) -
					        mu0*(math::phi_new(tauTprec*(times[locationCount]-times[i])) -
                                    math::phi_new(tauTprec*(-times[i])));

                    if (vectorCount < locationCount) { // Edge-cases
                        sumOfRates += xsimd::log(innerLikelihoodLoop<RealType, 1>(i, vectorCount, locationCount)) +
                                theta/omega*(xsimd::exp(-omega*(times[locationCount]-times[i]))-1) -
                                mu0*(math::phi_new(tauTprec*(times[locationCount]-times[i])) -
                                     math::phi_new(tauTprec*(-times[i])));
                    }

                    return sumOfRates;

                }, ParallelType());

        sumOfLikContribs = delta;
    }


// Parallelization helper functions

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, Integer end, Function function, CpuAccumulate) {
	    for (; begin != end; ++begin) {
	        function(begin);
	    }
	}

	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function, CpuAccumulate) {
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}

#ifdef USE_C_ASYNC
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_thread(Integer begin, Integer end, Real sum, Function function) {
		std::vector<std::future<Real>> results;

		int chunkSize = (end - begin) / nThreads;
		int start = 0;

		for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
			results.emplace_back(std::async([=] {
				return accumulate(begin + start, begin + start + chunkSize, 0.0, function);
			}));
		}
		results.emplace_back(std::async([=] {
			return accumulate(begin + start, end, 0.0, function);
		}));

		for (auto&& result : results) {
			sum += result.get();
		}
		return sum;
	}
#endif

#ifdef USE_OMOP
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_omp(Integer begin, Integer end, Real sum, Function function) {
		#pragma omp
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}
#endif

#ifdef USE_THREAD_POOL
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate_thread_pool(Integer begin, Integer end, Real sum, Function function) {
		std::vector<std::future<Real>> results;

		int chunkSize = (end - begin) / nThreads;
		int start = 0;

		for (int i = 0; i < nThreads - 1; ++i, start += chunkSize) {
			results.emplace_back(
				pool.enqueue([=] {
					return accumulate(
						begin + start,
						begin + start + chunkSize,
						Real(0),
						function);
				})
			);
		}
		results.emplace_back(
			pool.enqueue([=] {
				return accumulate(
					begin + start,
					end,
					Real(0),
					function);
			})

		);

		Real total = static_cast<Real>(0);
		for (auto&& result : results) {
			total += result.get();
		}
		return total;
	}
#endif

#ifdef USE_TBB
	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function, TbbAccumulate) {

		return tbb::parallel_reduce(
 			tbb::blocked_range<size_t>(begin, end
 			//, 200
 			),
 			sum,
 			[function](const tbb::blocked_range<size_t>& r, Real sum) -> Real {
 				//return accumulate
 				const auto end = r.end();
 				for (auto i = r.begin(); i != end; ++i) {
 					sum += function(i);
 				}
 				return sum;
 			},
 			std::plus<Real>()
		);
	}

	template <typename Integer, typename Function>
	inline void for_each(Integer begin, const Integer end, Function function, TbbAccumulate) {

		tbb::parallel_for(
				tbb::blocked_range<size_t>(begin, end
				//, 200
				),
				[function](const tbb::blocked_range<size_t>& r) -> void {
					const auto end = r.end();
					for (auto i = r.begin(); i != end; ++i) {
						function(i);
					}
				}
		);
	}
#endif

private:
    double sigmaXprec;
    double storedSigmaXprec;
    double tauXprec;
    double storedTauXprec;
    double tauTprec;
    double storedTauTprec;
    double omega;
    double storedOmega;
    double theta;
    double storedTheta;
    double mu0;
    double storedMu0;

    double sumOfLikContribs;
    double storedSumOfLikContribs;

    mm::MemoryManager<RealType> locDists;
    mm::MemoryManager<RealType> timDiffs;

    mm::MemoryManager<RealType> times;


    mm::MemoryManager<RealType> likContribs;
    mm::MemoryManager<RealType> storedLikContribs;

//    mm::MemoryManager<RealType> gradient0;
//    mm::MemoryManager<RealType> gradient1;
//
//    mm::MemoryManager<RealType>* gradientPtr;
//    mm::MemoryManager<RealType>* storedGradientPtr;

    mm::MemoryManager<double> buffer;

    bool isStoredLikContribsEmpty;

    int nThreads;

#ifdef USE_TBB
    std::shared_ptr<tbb::global_control> control;
#endif

};

// factory
std::shared_ptr<AbstractHawkes>
constructNewHawkesDoubleNoParallelNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
	Rcpp::Rcout << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "DOUBLE, NO PARALLEL, NO SIMD" << std::endl;
#endif
	return std::make_shared<NewHawkes<DoubleNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractHawkes>
constructNewHawkesDoubleTbbNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
  Rcpp::Rcout << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "DOUBLE, TBB PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewHawkes<DoubleNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractHawkes>
constructNewHawkesFloatNoParallelNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
  Rcpp::Rcout << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "SINGLE, NO PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewHawkes<FloatNoSimdTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

std::shared_ptr<AbstractHawkes>
constructNewHawkesFloatTbbNoSimd(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
  Rcpp::Rcout << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
#else
  std::cerr << "SINGLE, TBB PARALLEL, NO SIMD" << std::endl;
#endif
  return std::make_shared<NewHawkes<FloatNoSimdTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
}

#ifdef USE_SIMD

#ifdef USE_AVX
    std::shared_ptr<AbstractHawkes>
    constructNewHawkesDoubleNoParallelAvx(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
        Rcpp::Rcout << "DOUBLE, NO PARALLEL, AVX" << std::endl;
#else
        std::cerr << "DOUBLE, NO PARALLEL, AVX" << std::endl;
#endif
        return std::make_shared<NewHawkes<DoubleAvxTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

    std::shared_ptr<AbstractHawkes>
    constructNewHawkesDoubleTbbAvx(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, AVX" << std::endl;
#endif
        return std::make_shared<NewHawkes<DoubleAvxTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }
#endif

#ifdef USE_AVX512
    std::shared_ptr<AbstractHawkes>
    constructNewHawkesDoubleNoParallelAvx512(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
#else
      std::cerr << "DOUBLE, NO PARALLEL, AVX512" << std::endl;
#endif
        return std::make_shared<NewHawkes<DoubleAvx512TypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

    std::shared_ptr<AbstractHawkes>
    constructNewHawkesDoubleTbbAvx512(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, AVX512" << std::endl;
#endif
        return std::make_shared<NewHawkes<DoubleAvx512TypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }
#endif

#ifdef USE_SSE
    std::shared_ptr<AbstractHawkes>
    constructNewHawkesDoubleNoParallelSse(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, NO PARALLEL, SSE" << std::endl;
#else
      std::cerr << "DOUBLE, NO PARALLEL, SSE" << std::endl;
#endif
      return std::make_shared<NewHawkes<DoubleSseTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

    std::shared_ptr<AbstractHawkes>
    constructNewHawkesDoubleTbbSse(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
      Rcpp::Rcout << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
#else
      std::cerr << "DOUBLE, TBB PARALLEL, SSE" << std::endl;
#endif
      return std::make_shared<NewHawkes<DoubleSseTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
    }

	std::shared_ptr<AbstractHawkes>
	constructNewHawkesFloatNoParallelSse(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
	  Rcpp::Rcout << "SINGLE, NO PARALLEL, SSE" << std::endl;
#else
	  std::cerr << "SINGLE, NO PARALLEL, SSE" << std::endl;
#endif
	  return std::make_shared<NewHawkes<FloatSseTypeInfo, CpuAccumulate>>(embeddingDimension, locationCount, flags, threads);
	}

	std::shared_ptr<AbstractHawkes>
	constructNewHawkesFloatTbbSse(int embeddingDimension, int locationCount, long flags, int threads) {
#ifdef RBUILD
	  Rcpp::Rcout << "SINGLE, TBB PARALLEL, SSE" << std::endl;
#else
	  std::cerr << "SINGLE, TBB PARALLEL, SSE" << std::endl;
#endif
	  return std::make_shared<NewHawkes<FloatSseTypeInfo, TbbAccumulate>>(embeddingDimension, locationCount, flags, threads);
	}
#endif

#endif

} // namespace hph

#endif // _NEWHAWKES_HPP
