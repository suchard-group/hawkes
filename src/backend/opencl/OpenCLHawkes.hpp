#ifndef _OPENCL_HAWKES_HPP
#define _OPENCL_HAWKES_HPP

#include <iostream>
#include <cmath>

#include "AbstractHawkes.hpp"

#include <boost/compute/algorithm/reduce.hpp>
#include "reduce_fast.hpp"

#ifdef RBUILD
#include <Rcpp.h>
#endif

#define SSE
//#undef SSE

#define USE_VECTORS

#define TILE_DIM 16

#define TILE_DIM_I  128
//#define TILE_DIM_J  128
#define TPB 128
#define DELTA 1;

#define USE_VECTOR

#include "OpenCLMemoryManagement.hpp"
#include "Reducer.hpp"

#include <boost/compute/algorithm/accumulate.hpp>

namespace hph {

template <typename OpenCLRealType>
class OpenCLHawkes : public AbstractHawkes {
public:

	typedef typename OpenCLRealType::BaseType RealType;
	typedef typename OpenCLRealType::VectorType VectorType;

    OpenCLHawkes(int embeddingDimension, int locationCount, long flags, int deviceNumber)
        : AbstractHawkes(embeddingDimension, locationCount, flags),
          sigmaXprec(0.0), storedSigmaXprec(0.0),
          tauXprec(0.0), storedTauXprec(0.0),
          tauTprec(0.0), storedTauTprec(0.0),
          omega(0.0), storedOmega(0.0),
          theta(0.0), storedTheta(0.0),
          mu0(0.0), storedMu0(0.0),

          sumOfLikContribs(0.0), storedSumOfLikContribs(0.0),
          times(locationCount),
          locations(locationCount * OpenCLRealType::dim),
		  locationsPtr(&locations),

          likContribs(locationCount),
          storedLikContribs(locationCount),

          isStoredLikContribsEmpty(false)

    {
#ifdef RBUILD
      Rcpp::Rcout << "ctor OpenCLHawkes" << std::endl;

      Rcpp::Rcout << "All devices:" << std::endl;

      const auto devices = boost::compute::system::devices();

      for(const auto &device : devices){
        Rcpp::Rcout << "\t" << device.name() << std::endl;
      }


      if (deviceNumber < 0 || deviceNumber >= devices.size()) {
        device = boost::compute::system::default_device();
      } else {
        device = devices[deviceNumber];
      }

      //device = devices[devices.size() - 1]; // hackishly chooses correct device TODO do this correctly

      if (device.type()!=CL_DEVICE_TYPE_GPU){
          Rcpp::stop("Error: selected device not GPU.");
      } else {
          Rcpp::Rcout << "Using: " << device.name() << std::endl;
      }

      ctx = boost::compute::context(device, 0);
      queue = boost::compute::command_queue{ctx, device
        , boost::compute::command_queue::enable_profiling
      };

      dObservations = mm::GPUMemoryManager<RealType>(observations.size(), ctx);

      Rcpp::Rcout << "\twith vector-dim = " << OpenCLRealType::dim << std::endl;

#else //RBUILD
      std::cerr << "ctor OpenCLHawkes" << std::endl;

      std::cerr << "All devices:" << std::endl;

      const auto devices = boost::compute::system::devices();

      for(const auto &device : devices){
        std::cerr << "\t" << device.name() << std::endl;
      }


      if (deviceNumber < 0 || deviceNumber >= devices.size()) {
        device = boost::compute::system::default_device();
      } else {
        device = devices[deviceNumber];
      }

      if (device.type()!=CL_DEVICE_TYPE_GPU){
          std::cerr << "Error: selected device not GPU." << std::endl;
          exit(-1);
      } else {
          std::cerr << "Using: " << device.name() << std::endl;
      }

      ctx = boost::compute::context(device, 0);
      queue = boost::compute::command_queue{ctx, device
        , boost::compute::command_queue::enable_profiling
      };

      dLocations = mm::GPUMemoryManager<RealType>(locations.size(), ctx);
      dTimes = mm::GPUMemoryManager<RealType>(times.size(), ctx);

        std::cerr << "\twith vector-dim = " << OpenCLRealType::dim << std::endl;
#endif //RBUILD

#ifdef USE_VECTORS
//		dLocations = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
		//dGradient   = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
#else
//		dLocations0 = mm::GPUMemoryManager<RealType>(locations0.size(), ctx);
//		dLocations1 = mm::GPUMemoryManager<RealType>(locations1.size(), ctx);
#endif // USE_VECTORS

		dLocationsPtr = &dLocations;
//		dStoredLocationsPtr = &dLocations1;

		dLikContribs = mm::GPUMemoryManager<RealType>(likContribs.size(), ctx);
		dStoredLikContribs = mm::GPUMemoryManager<RealType>(storedLikContribs.size(), ctx);

		createOpenCLKernels();
    }

    int getInternalDimension() override { return OpenCLRealType::dim; }

//	void getLogLikelihoodGradient(double* result, size_t length) override {
//
//		kernelGradientVector.set_arg(0, *dLocationsPtr);
//		kernelGradientVector.set_arg(3, static_cast<RealType>(precision));
//
//		queue.enqueue_1d_range_kernel(kernelGradientVector, 0,
//                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
//		queue.finish();
//
//        if (length == locationCount * OpenCLRealType::dim) {
//
//            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
//                                       result, buffer, queue);
//            queue.finish();
//
//        } else {
//
//            if (doubleBuffer.size() != locationCount * OpenCLRealType::dim) {
//                doubleBuffer.resize(locationCount * OpenCLRealType::dim);
//            }
//
//            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
//                                       doubleBuffer.data(), buffer, queue);
//            queue.finish();
//
//            mm::paddedBufferedCopy(begin(doubleBuffer), OpenCLRealType::dim, embeddingDimension,
//                                   result, embeddingDimension,
//                                   locationCount, buffer);
//        }
// 	}

    void computeLikelihood() {
		if (!incrementsKnown) {
		    computeSumLikContribs();
			incrementsKnown = true;
		} else {
#ifdef RBUILD
		  Rcpp::Rcout << "SHOULD NOT BE HERE" << std::endl;
#else
			std::cerr << "SHOULD NOT BE HERE" << std::endl;
#endif
//			updateSumOfLikContribs();
		}
    }

    double getSumOfLikContribs() override {
    	if (!sumOfLikContribsKnown) {
			computeLikContribs();
			sumOfLikContribsKnown = true;
		}

    	return sumOfLikContribs;
 	}

    void storeState() override {
    	storedSumOfLikContribs = sumOfLikContribs;
    	isStoredSquaredResidualsEmpty = true;

    	updatedLocation = -1;

    	storedPrecision = precision;
    	storedOneOverSd = oneOverSd;

    }

    void acceptState() override {
        if (!isStoredSquaredResidualsEmpty) {
    		for (int j = 0; j < locationCount; ++j) {
                squaredResiduals[j * locationCount + updatedLocation] = squaredResiduals[
                        updatedLocation * locationCount + j];
            }
    	}
    }

    void restoreState() override {
    	sumOfLikContribs = storedSumOfLikContribs;
    	sumOfLikContribsKnown = true;

		if (!isStoredSquaredResidualsEmpty) {
    		std::copy(
    			begin(storedSquaredResiduals),
    			end(storedSquaredResiduals),
    			begin(squaredResiduals) + updatedLocation * locationCount
    		);

    		// COMPUTE
    		boost::compute::copy(
    			dStoredSquaredResiduals.begin(),
    			dStoredSquaredResiduals.end(),
    			dSquaredResiduals.begin() + updatedLocation * locationCount, queue
    		);

    		incrementsKnown = true;
    	} else {
    		incrementsKnown = false; // Force recompute;
    	}

    	precision = storedPrecision;
    	oneOverSd = storedOneOverSd;

    	auto tmp1 = storedLocationsPtr;
    	storedLocationsPtr = locationsPtr;
    	locationsPtr = tmp1;

    	// COMPUTE
    	auto tmp2 = dStoredLocationsPtr;
    	dStoredLocationsPtr = dLocationsPtr;
    	dLocationsPtr = tmp2;

    }

    void setPairwiseData(double* data, size_t length) override {
		assert(length == observations.size());
		mm::bufferedCopy(data, data + length, begin(observations), buffer);

		// COMPUTE
		mm::bufferedCopyToDevice(data, data + length, dObservations.begin(),
			buffer, queue);
    }

    void setParameters(double* data, size_t length) override {
		assert(length == 1); // Call only with precision
		precision = data[0]; // TODO Remove
		oneOverSd = std::sqrt(data[0]);
    }

    void makeDirty() override {
    	sumOfLikContribsKnown = false;
    	incrementsKnown = false;
    }

	int count = 0;

	template <bool withTruncation>
	void computeSumOfLikContribs() {

		RealType lSumOfLikContribs = 0.0;

#ifdef USE_VECTORS
		kernelSumOfLikContribsVector.set_arg(0, *dLocationsPtr);

		if (isLeftTruncated) {
			kernelSumOfLikContribsVector.set_arg(3, static_cast<RealType>(precision));
			kernelSumOfLikContribsVector.set_arg(4, static_cast<RealType>(oneOverSd));
		}

		const size_t local_work_size[2] = {TILE_DIM, TILE_DIM};
		size_t work_groups = locationCount / TILE_DIM;
		if (locationCount % TILE_DIM != 0) {
			++work_groups;
		}
		const size_t global_work_size[2] = {work_groups * TILE_DIM, work_groups * TILE_DIM};

		queue.enqueue_nd_range_kernel(kernelSumOfLikContribsVector, 2, 0, global_work_size, local_work_size);

#else
		kernelSumOfLikContribs.set_arg(0, *dLocationsPtr);
		queue.enqueue_1d_range_kernel(kernelSumOfLikContribs, 0, locationCount * locationCount, 0);
#endif // USE_VECTORS

		queue.finish();

		RealType sum = RealType(0.0);
		boost::compute::reduce(dSquaredResiduals.begin(), dSquaredResiduals.end(), &sum, queue);

		queue.finish();

	    lSumOfLikContribs = sum;

	    lSumOfLikContribs /= 2.0;
    	sumOfLikContribs = lSumOfLikContribs;

	    incrementsKnown = true;
	    sumOfLikContribsKnown = true;

	    count++;
	}

#ifdef SSE
    template <typename HostVectorType, typename Iterator>
    double calculateDistance(Iterator iX, Iterator iY) const {

        using AlignedValueType = typename HostVectorType::allocator_type::aligned_value_type;

        auto sum = static_cast<AlignedValueType>(0);
        AlignedValueType* x = &*iX;
        AlignedValueType* y = &*iY;

        for (int i = 0; i < OpenCLRealType::dim; ++i, ++x, ++y) {
            const auto difference = *x - *y; // TODO Why does this seg-fault?
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }


#else // SSE
    template <typename HostVectorType, typename Iterator>
    double calculateDistance(Iterator x, Iterator y, int length) const {

        assert (false);

        auto sum = static_cast<RealType>(0);

        for (int i = 0; i < 2; ++i, ++x, ++y) {
            const auto difference = *x - *y;
            sum += difference * difference;
        }
        return std::sqrt(sum);
    }
#endif // SSE

	template <typename Integer, typename Function, typename Real>
	inline Real accumulate(Integer begin, Integer end, Real sum, Function function) {
		for (; begin != end; ++begin) {
			sum += function(begin);
		}
		return sum;
	}

	void createOpenCLSummationKernel() {

        std::stringstream code;
        std::stringstream options;

        options << "-DTILE_DIM=" << TILE_DIM;

        if (sizeof(RealType) == 8) { // 64-bit fp
            code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
            options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim << " -DCAST=long"
                    << " -DZERO=0.0 -DHALF=0.5";

        } else { // 32-bit fp
            options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DHALF=0.5f";
        }

        code <<
             " __kernel void computeSum(__global const REAL_VECTOR *summand,           \n" <<
             "                                 __global REAL *partialSum,              \n" <<
             "						           const uint locationCount) {             \n";

        code <<
             "   const uint lid = get_local_id(0);                                   \n" <<
             "   uint j = get_local_id(0);                                           \n" <<
             "                                                                       \n" <<
             "   __local REAL_VECTOR scratch[TPB];                                   \n" <<
             "                                                                       \n" <<
             "   REAL sum = ZERO;                                                    \n" <<
             "                                                                       \n" <<
             "   while (j < locationCount) {                                         \n";


        code <<
             "     sum += summand[j];                                                \n" <<
             "     j += TPB;                                                         \n" <<
             "     scratch[lid] = sum;                                               \n";
#ifdef USE_VECTOR
        code << reduce::ReduceBody1<RealType,false>::body();
#else
        code << (isNvidia ? reduce::ReduceBody2<RealType,true>::body() : reduce::ReduceBody2<RealType,false>::body());
#endif
        code <<
             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
             "   if (lid == 0) {                                                     \n";

        code <<
             "     partialSum    =  scratch[0];                                      \n" <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
		    Rcpp::Rcout << "Summation kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Summation kernel\n" << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelLikSum = boost::compute::kernel(program, "computeSum");

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

#ifdef DOUBLE_CHECK
        #ifdef RBUILD
        Rcpp::Rcout << kernelSumOfLikContribsVector.get_program().source() << std::endl;
#else
        std::cerr << kernelSumOfLikContribsVector.get_program().source() << std::endl;
#endif
//        exit(-1);
#endif // DOUBLE_CHECK

        size_t index = 0;
        kernelLikSum.set_arg(index++, dLikContribs);
        kernelLikSum.set_arg(index++, dLikelihood);
        kernelLikSum.set_arg(index++, boost::compute::uint_(locationCount));
	}

	void createOpenCLLikContribsKernel() {

        const char pdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static double pdf(double);

                static double pdf(double value) {
                    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp( - pow(value,2.0) * 0.5);
                }
        );

        const char pdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static float pdf(float);

                static float pdf(float value) {

                    const float rSqrt2f = 0.70710678118655f;
                    const float rSqrtPif = 0.56418958354775f;
                    return rSqrt2f * rSqrtPif * exp( - pow(value,2.0f) * 0.5f);
                }
        );

		const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			static double cdf(double);

			static double cdf(double value) {
	    		return 0.5 * erfc(-value * M_SQRT1_2);
	    	}
		);

		const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
			static float cdf(float);

			static float cdf(float value) {

				const float rSqrt2f = 0.70710678118655f;
	    		return 0.5f * erfc(-value * rSqrt2f);
	    	}
		);

		std::stringstream code;
		std::stringstream options;

		options << "-DTILE_DIM=" << TILE_DIM;

		if (sizeof(RealType) == 8) { // 64-bit fp
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
			options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim << " -DCAST=long"
                    << " -DZERO=0.0 -DHALF=0.5";
			code << cdfString1Double;

		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DHALF=0.5f";
			code << cdfString1Float;
		}

		code <<
			" __kernel void computeLikContribs(__global const REAL_VECTOR *locations, \n" <<
			"  						          __global const REAL *times,             \n" <<
			"						          __global REAL *likContribs,             \n" <<
            "                                 const REAL sigmaXprec,                  \n" <<
            "                                 const REAL tauXprec,                    \n" <<
			"                                 const REAL tauTprec,                    \n" <<
			"                                 const REAL omega,                       \n" <<
			"                                 const REAL theta,                       \n" <<
			"                                 const REAL mu0,                         \n" <<
			"                                 const uint dimX,                        \n" <<
			"						          const uint locationCount) {             \n";

		code <<
		    "   const uint i = get_group_id(0);                                     \n" <<
		    "                                                                       \n" <<
		    "   const uint lid = get_local_id(0);                                   \n" <<
		    "   uint j = get_local_id(0);                                           \n" <<
		    "                                                                       \n" <<
		    "   __local REAL_VECTOR scratch[TPB];                                   \n" <<
		    "                                                                       \n" <<
		    "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
		    "   REAL        sum = ZERO;                                             \n" <<
		    "                                                                       \n" <<
		    "   while (j < i) {                                                     \n" << // originally j < locationCount
		    "                                                                       \n" <<
		    "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
		    "     const REAL_VECTOR locDiff = vectorI - vectorJ;                    \n" <<
		    "     const REAL        timeDiff = times[i] - times[j];                 \n";

        if (OpenCLRealType::dim == 8) {
            code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                    const REAL distance = sqrt(dot(locDiff.lo, locDiff.lo) + dot(locDiff.hi, locDiff.hi));
            );
        } else {
            code << BOOST_COMPUTE_STRINGIZE_SOURCE(
					const REAL distance = length(locDiff);
            );
        }

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                    // TODO unsure whether I need to pown(tauXprec,dimX) pown(sigmaXprec,dimX) outside of pdf
                    REAL innerContrib = mu0 * tauXprec * tauTprec * pdf(distance * tauXprec) * pdf(timeDiff*tauTprec) +
                            select(theta, ZERO, times[j]<times[i]) * pdf(distance * sigmaXprec) * exp(-omega*timeDiff);
        );

        code <<
             "     sum += innerContrib;                                              \n" <<
             "     j += TPB;                                                         \n" <<
             "     scratch[lid] = sum;                                               \n";
#ifdef USE_VECTOR
        code << reduce::ReduceBody1<RealType,false>::body();
#else
        code << (isNvidia ? reduce::ReduceBody2<RealType,true>::body() : reduce::ReduceBody2<RealType,false>::body());
#endif
        code <<
             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
             "   if (lid == 0) {                                                     \n";

        code <<
             "     likContribs[i] =  log(scratch[0]) + theta / omega *               \n" <<
             "       ( exp(-omega*(times[locationCounts]-times[i]))-1 ) -            \n" <<
             "       mu0 * ( cdf((times[locationCounts]-times[i])/tauT)-             \n" <<
             "               cdf(-times[i]/tauT) )   ;                               \n" <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
		kernelLikContribsVector = boost::compute::kernel(program, "computeLikContribs");

		size_t index = 0;
        kernelLikContribsVector.set_arg(index++, dLocations0);
		kernelLikContribsVector.set_arg(index++, dTimes);
		kernelLikContribsVector.set_arg(index++, dLikContribs);
        kernelLikContribsVector.set_arg(index++, sigmaXprec);
        kernelLikContribsVector.set_arg(index++, tauXprec);
        kernelLikContribsVector.set_arg(index++, tauTprec);
        kernelLikContribsVector.set_arg(index++, omega);
        kernelLikContribsVector.set_arg(index++, theta);
        kernelLikContribsVector.set_arg(index++, mu0);
        kernelLikContribsVector.set_arg(index++, boost::compute::uint_(embeddingDimension));
        kernelLikContribsVector.set_arg(index++, boost::compute::uint_(locationCount));

	}

//	void createOpenCLGradientKernel() {
//
//        const char cdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
//                static double cdf(double);
//
//                static double cdf(double value) {
//                    return 0.5 * erfc(-value * M_SQRT1_2);
//                }
//        );
//
//        const char pdfString1Double[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
//                static double pdf(double);
//
//                static double pdf(double value) {
//                    return 0.5 * M_SQRT1_2 * M_2_SQRTPI * exp( - pow(value,2.0) * 0.5);
//                }
//        );
//
//        const char cdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
//                static float cdf(float);
//
//                static float cdf(float value) {
//
//                    const float rSqrt2f = 0.70710678118655f;
//                    return 0.5f * erfc(-value * rSqrt2f);
//                }
//        );
//
//        const char pdfString1Float[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
//                static float pdf(float);
//
//                static float pdf(float value) {
//
//                    const float rSqrt2f = 0.70710678118655f;
//                    const float rSqrtPif = 0.56418958354775f;
//                    return rSqrt2f * rSqrtPif * exp( - pow(value,2.0f) * 0.5f);
//                }
//        );
//
//		std::stringstream code;
//		std::stringstream options;
//
//		options << "-DTILE_DIM_I=" << TILE_DIM_I << " -DTPB=" << TPB << " -DDELTA=" << DELTA;
//
//		if (sizeof(RealType) == 8) { // 64-bit fp
//			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
//			options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim  << " -DCAST=long"
//                    << " -DZERO=0.0 -DONE=1.0 -DHALF=0.5";
//
//			if (isLeftTruncated) {
//				code << cdfString1Double;
//                code << pdfString1Double;
//			}
//
//		} else { // 32-bit fp
//			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
//                    << " -DZERO=0.0f -DONE=1.0f -DHALF=0.5f";
//
//			if (isLeftTruncated) {
//				code << cdfString1Float;
//				code << pdfString1Float;
//			}
//		}
//
//#ifndef USE_VECTOR
//		bool isNvidia = false; // TODO Check device name
//#endif
//
//		code <<
//			 " __kernel void computeGradient(__global const REAL_VECTOR *locations,  \n" <<
//			 "                               __global const REAL *observations,      \n" <<
//			 "                               __global REAL_VECTOR *output,           \n" <<
//		     "                               const REAL precision,                   \n" <<
//			 "                               const uint locationCount) {             \n" <<
//			 "                                                                       \n" <<
////			 "   const uint i = get_local_id(1) + get_group_id(0) * TILE_DIM_I;      \n" <<
//			 "   const uint i = get_group_id(0);                                     \n" <<
////			 "   const int inBounds = (i < locationCount);                           \n" <<
//			 "                                                                       \n" <<
//			 "   const uint lid = get_local_id(0);                                   \n" <<
//			 "   uint j = get_local_id(0);                                           \n" <<
//			 "                                                                       \n" <<
//			 "   __local REAL_VECTOR scratch[TPB];                                   \n" <<
//			 "                                                                       \n" <<
//			 "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
//			 "   REAL_VECTOR sum = ZERO;                                             \n" <<
//			 "                                                                       \n" <<
//			 "   while (j < locationCount) {                                         \n" <<
//			 "                                                                       \n" <<
//			 "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
//             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";
//
//        if (OpenCLRealType::dim == 8) {
//            code << "     const REAL distance = sqrt(                                \n" <<
//                    "              dot(difference.lo, difference.lo) +               \n" <<
//                    "              dot(difference.hi, difference.hi)                 \n" <<
//                    "      );                                                        \n";
//
//        } else {
//            code << "     const REAL distance = length(difference);                  \n";
//        }
//
//        // TODO Handle missing values by  `!isnan(observation) * `
//
//        code <<
//             "     const REAL observation = observations[i * locationCount + j];     \n" <<
//             "     REAL residual = select(observation - distance, ZERO,              \n" <<
//             "                                  (CAST)isnan(observation));           \n";
//
//        if (isLeftTruncated) {
//            code << "     const REAL trncDrv = select(-ONE / sqrt(precision) *        \n" << // TODO speed up this part
//                    "                              pdf(distance * sqrt(precision)) / \n" <<
//                    "                              cdf(distance * sqrt(precision)),  \n" <<
//                    "                                 ZERO,                          \n" <<
//                    "                                 (CAST)isnan(observation));     \n" <<
//                    "     residual += trncDrv;                                       \n";
//        }
//
//        code <<
//             "     REAL contrib = residual * precision / distance;                   \n" <<
//			 "                                                                       \n" <<
//             "     if (i != j) { sum += (vectorI - vectorJ) * contrib * DELTA;  }    \n" <<
//			 "                                                                       \n" <<
//			 "     j += TPB;                                                         \n" <<
//			 "   }                                                                   \n" <<
//			 "                                                                       \n" <<
//			 "   scratch[lid] = sum;                                                 \n";
//#ifdef USE_VECTOR
//			 code << reduce::ReduceBody1<RealType,false>::body(); // TODO Try NVIDIA version at some point
//#else
//		code << (isNvidia ? reduce::ReduceBody2<RealType,true>::body() : reduce::ReduceBody2<RealType,false>::body());
//#endif
//		code <<
//			 "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
//			 "   if (lid == 0) {                                                     \n";
//
//        code <<
//             "     REAL_VECTOR mask = (REAL_VECTOR) (";
//
//        for (int i = 0; i < embeddingDimension; ++i) {
//            code << " ONE";
//            if (i < (OpenCLRealType::dim - 1)) {
//                code << ",";
//            }
//        }
//        for (int i = embeddingDimension; i < OpenCLRealType::dim; ++i) {
//            code << " ZERO";
//            if (i < (OpenCLRealType::dim - 1)) {
//                code << ",";
//            }
//        }
//        code << " ); \n";
//
//        code <<
//			 "     output[i] = mask * scratch[0];                                    \n" <<
//			 "   }                                                                   \n" <<
//			 " }                                                                     \n ";
//
//
//
//		program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
//		kernelGradientVector = boost::compute::kernel(program, "computeGradient");
//
//		kernelGradientVector.set_arg(0, dLocations0); // Must update
//		kernelGradientVector.set_arg(1, dObservations);
//		kernelGradientVector.set_arg(2, dGradient);
//		kernelGradientVector.set_arg(3, static_cast<RealType>(precision)); // Must update
//		kernelGradientVector.set_arg(4, boost::compute::uint_(locationCount));
//	}

	void createOpenCLKernels() {

        createOpenCLSummationKernel();
        createOpenCLLikContribsKernel();
//		createOpenCLGradientKernel();

	}

private:
	double precision;
	double storedPrecision;

	double oneOverSd;
	double storedOneOverSd;

    double sumOfLikContribs;
    double storedSumOfLikContribs;

    boost::compute::device device;
    boost::compute::context ctx;
    boost::compute::command_queue queue;

    mm::MemoryManager<RealType> observations;

    mm::MemoryManager<RealType> locations0;
    mm::MemoryManager<RealType> locations1;

    mm::MemoryManager<RealType>* locationsPtr;
    mm::MemoryManager<RealType>* storedLocationsPtr;

    mm::MemoryManager<RealType> squaredResiduals;
    mm::MemoryManager<RealType> storedSquaredResiduals;

    mm::MemoryManager<RealType> truncations;
    mm::MemoryManager<RealType> storedTruncations;

	mm::MemoryManager<RealType> gradient;

    mm::GPUMemoryManager<RealType> dObservations;

#ifdef USE_VECTORS
    mm::GPUMemoryManager<VectorType> dLocations0;
    mm::GPUMemoryManager<VectorType> dLocations1;

    mm::GPUMemoryManager<VectorType>* dLocationsPtr;
    mm::GPUMemoryManager<VectorType>* dStoredLocationsPtr;

	mm::GPUMemoryManager<VectorType> dGradient;
#else
    mm::GPUMemoryManager<RealType> dLocations0;
    mm::GPUMemoryManager<RealType> dLocations1;

    mm::GPUMemoryManager<RealType>* dLocationsPtr;
    mm::GPUMemoryManager<RealType>* dStoredLocationsPtr;
#endif // USE_VECTORS


    mm::GPUMemoryManager<RealType> dSquaredResiduals;
    mm::GPUMemoryManager<RealType> dStoredSquaredResiduals;

    mm::GPUMemoryManager<RealType> dTruncations;
    mm::GPUMemoryManager<RealType> dStoredTruncations;

    bool isStoredSquaredResidualsEmpty;
    bool isStoredTruncationsEmpty;

    mm::MemoryManager<RealType> buffer;
    mm::MemoryManager<double> doubleBuffer;

    boost::compute::program program;
    boost::compute::kernel kernelLikSum;  // TODO guessing this goes here

#ifdef USE_VECTORS
	boost::compute::kernel kernelLikContribsVector;
//	boost::compute::kernel kernelGradientVector;
#else
    boost::compute::kernel kernelLikContribs;
#endif // USE_VECTORS
};

} // namespace hph

#endif // _OPENCL_MULTIDIMENSIONAL_SCALING_HPP
