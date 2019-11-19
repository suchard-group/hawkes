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

//#define DEBUG_KERNELS

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

          locDists(locationCount * locationCount),
          timDiffs(locationCount * locationCount),

          times(locationCount),

          sumOfLikContribs(0.0), storedSumOfLikContribs(0.0),

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

      dLocDists = mm::GPUMemoryManager<RealType>(locDists.size(), ctx);
      dTimDiffs = mm::GPUMemoryManager<RealType>(timDiffs.size(), ctx);

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

      dLocDists = mm::GPUMemoryManager<RealType>(locDists.size(), ctx);
      dTimDiffs = mm::GPUMemoryManager<RealType>(timDiffs.size(), ctx);

      dTimes = mm::GPUMemoryManager<RealType>(times.size(), ctx);

        std::cerr << "\twith vector-dim = " << OpenCLRealType::dim << std::endl;
#endif //RBUILD

#ifdef USE_VECTORS

        //dGradient   = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
#else

#endif // USE_VECTORS

		dLikContribs = mm::GPUMemoryManager<RealType>(likContribs.size(), ctx);
		dStoredLikContribs = mm::GPUMemoryManager<RealType>(storedLikContribs.size(), ctx);

		createOpenCLKernels();
    }

    int getInternalDimension() override { return OpenCLRealType::dim; }

    void getLogLikelihoodGradient(double* result, size_t length) override {
        std::cerr << "nothing to see here, move along now" << std::endl;
    }
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

    double getSumOfLikContribs() override {
        computeSumOfLikContribs();
    	return sumOfLikContribs;
 	}

    void storeState() override {
    	storedSumOfLikContribs = sumOfLikContribs;
        storedSigmaXprec = sigmaXprec;
        storedTauXprec = tauXprec;
        storedTauTprec = tauTprec;
        storedOmega = omega;
        storedTheta = theta;
        storedMu0 = mu0;

        storedLikContribs = likContribs;
    }

    void acceptState() override {

    }

    void restoreState() override {
    	sumOfLikContribs = storedSumOfLikContribs;

//		if (!isStoredSquaredResidualsEmpty) {
    		std::copy(
    			begin(storedLikContribs),
    			end(storedLikContribs),
    			begin(likContribs) + updatedLocation * locationCount
    		);

    		// COMPUTE
    		boost::compute::copy(
    			dStoredLikContribs.begin(),
    			dStoredLikContribs.end(),
    			dLikContribs.begin() + updatedLocation * locationCount, queue
    		);

//    		incrementsKnown = true;
//    	} else {
//    		incrementsKnown = false; // Force recompute;
//    	}

        sigmaXprec = storedSigmaXprec;
        tauXprec = storedTauXprec;
        tauTprec = storedTauTprec;
        omega = storedOmega;
        theta = storedTheta;
        mu0 = storedMu0;

    }

    void setLocDistsData(double* data, size_t length) override {
		assert(length == locDists.size());
		mm::bufferedCopy(data, data + length, begin(locDists), buffer);

		// COMPUTE
		mm::bufferedCopyToDevice(data, data + length, dLocDists.begin(),
			buffer, queue);
    }

    void setTimDiffsData(double* data, size_t length) override {
        assert(length == timDiffs.size());
        mm::bufferedCopy(data, data + length, begin(timDiffs), buffer);

        // COMPUTE
        mm::bufferedCopyToDevice(data, data + length, dTimDiffs.begin(),
                                 buffer, queue);
    }

    void setTimesData(double* data, size_t length) override {
        assert(length == times.size());
        mm::bufferedCopy(data, data + length, begin(times), buffer);

        // COMPUTE
        mm::bufferedCopyToDevice(data, data + length, dTimes.begin(),
                                 buffer, queue);
    }

    void setParameters(double* data, size_t length) override {
		assert(length == 6);
		sigmaXprec = data[0];
		tauXprec = data[1];
		tauTprec = data[2];
		omega = data[3];
		theta = data[4];
		mu0 = data[5];
    }


//    void makeDirty() override { // not sure this is needed
////    	sumOfLikContribsKnown = false;
////    	incrementsKnown = false;
//    }

	int count = 0;

	void computeSumOfLikContribs() {

		RealType lSumOfLikContribs = 0.0;

#ifdef USE_VECTORS
        kernelLikContribsVector.set_arg(0, dLocDists);
        kernelLikContribsVector.set_arg(1, dTimDiffs);
        kernelLikContribsVector.set_arg(2, dTimes);
        kernelLikContribsVector.set_arg(3, dLikContribs);
        kernelLikContribsVector.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelLikContribsVector.set_arg(5, static_cast<RealType>(tauXprec));
        kernelLikContribsVector.set_arg(6, static_cast<RealType>(tauTprec));
        kernelLikContribsVector.set_arg(7, static_cast<RealType>(omega));
        kernelLikContribsVector.set_arg(8, static_cast<RealType>(theta));
        kernelLikContribsVector.set_arg(9, static_cast<RealType>(mu0));
        kernelLikContribsVector.set_arg(10, boost::compute::int_(embeddingDimension));
        kernelLikContribsVector.set_arg(11, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelLikContribsVector, 0,
                static_cast<unsigned int>(locationCount) * TPB, TPB);


#else
        kernelLikContribs.set_arg(0, dLocDists);
        kernelLikContribs.set_arg(1, dTimDiffs);
        kernelLikContribs.set_arg(2, dTimes);
        kernelLikContribs.set_arg(3, dLikContribs);
        kernelLikContribs.set_arg(4, sigmaXprec);
        kernelLikContribs.set_arg(5, tauXprec);
        kernelLikContribs.set_arg(6, tauTprec);
        kernelLikContribs.set_arg(7, omega);
        kernelLikContribs.set_arg(8, theta);
        kernelLikContribs.set_arg(9, mu0);
        kernelLikContribs.set_arg(10, boost::compute::uint_(embeddingDimension));
        kernelLikContribs.set_arg(11, boost::compute::uint_(locationCount));
        queue.enqueue_1d_range_kernel(kernelLikContribs, 0, locationCount * locationCount, 0);
#endif // USE_VECTORS

		queue.finish();

		for(int l = 0; l < locationCount; l++) {
            std::cout << dLikContribs[l] << std::endl;
        };

        RealType sum = RealType(0.0);
        boost::compute::reduce(dLikContribs.begin(), dLikContribs.end(), &sum, queue);
        queue.finish();

        std::cout << "sum = " << sum << std::endl;

        sumOfLikContribs = sum + locationCount*(embeddingDimension-1)*log(M_1_SQRT_2PI);

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

//	void createOpenCLSummationKernel() {
//
//        std::stringstream code;
//        std::stringstream options;
//
//        options << "-DTILE_DIM=" << TILE_DIM << " -DTPB=" << TPB;
//
//        if (sizeof(RealType) == 8) { // 64-bit fp
//            code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
//            options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim << " -DCAST=long"
//                    << " -DZERO=0.0 -DHALF=0.5";
//
//        } else { // 32-bit fp
//            options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
//                    << " -DZERO=0.0f -DHALF=0.5f";
//        }
//
//        code <<
//             " __kernel void computeSum(__global const REAL *summand,           \n" <<
//             "                                 __global REAL *partialSum,              \n" <<
//             "						           const uint locationCount) {             \n";
//
//        code <<
//             "   const uint lid = get_local_id(0);                                   \n" <<
//             "   uint j = get_local_id(0);                                           \n" <<
//             "                                                                       \n" <<
//             "   __local REAL scratch[TPB];                                   \n" <<
//             "                                                                       \n" <<
//             "   REAL sum = ZERO;                                                    \n" <<
//             "                                                                       \n" <<
//             "   while (j < locationCount) {                                         \n";
//
//
//        code <<
//             "     sum += summand[j];                                                \n" <<
//             "     j += TPB;                                                         \n" <<
//             "  }                                                                    \n" <<
//             "     scratch[lid] = sum;                                               \n";
//#ifdef USE_VECTOR
//        code << reduce::ReduceBody1<RealType,false>::body();
//#else
//        code << (isNvidia ? reduce::ReduceBody2<RealType,true>::body() : reduce::ReduceBody2<RealType,false>::body());
//#endif
//        code <<
//             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
//             "   if (lid == 0) {                                                     \n";
//
//        code <<
//             "     partialSum[0]    =  scratch[0];                                      \n" <<
//             "   }                                                                   \n" <<
//             " }                                                                     \n ";
//
//#ifdef DEBUG_KERNELS
//        #ifdef RBUILD
//		    Rcpp::Rcout << "Summation kernel\n" << code.str() << std::endl;
//#else
//        std::cerr << "Summation kernel\n" << options.str() << code.str() << std::endl;
//#endif
//#endif
//
//        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
//        kernelLikSum = boost::compute::kernel(program, "computeSum");
//
//#ifdef DEBUG_KERNELS
//        #ifdef RBUILD
//        Rcpp:Rcout << "Successful build." << std::endl;
//#else
//        std::cerr << "Successful build." << std::endl;
//#endif
//#endif
//
//#ifdef DOUBLE_CHECK
//        #ifdef RBUILD
//        Rcpp::Rcout << kernelSumOfLikContribsVector.get_program().source() << std::endl;
//#else
//        std::cerr << kernelSumOfLikContribsVector.get_program().source() << std::endl;
//#endif
////        exit(-1);
//#endif // DOUBLE_CHECK
//
//        kernelLikSum.set_arg(0, dLikContribs);
//        kernelLikSum.set_arg(1, sumOfLikContribs);
//        kernelLikSum.set_arg(2, boost::compute::uint_(locationCount));
//	}

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

		const char safeExpStringFloat[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
	        static float safe_exp(float);

	        static float safe_exp(float value) {
	            if (value < -103.0f) {
	                return 0.0f;
	            } else if (value > 88.0f) {
	                return MAXFLOAT;
	            } else {
	                return exp(value);
	            }
	        }
	    );

        const char safeExpStringDouble[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
                static double safe_exp(double);

                static double safe_exp(double value) {
                    return exp(value);
                }
        );

		std::stringstream code;
		std::stringstream options;

		options << "-DTILE_DIM=" << TILE_DIM << " -DTPB=" << TPB;

		if (sizeof(RealType) == 8) { // 64-bit fp
			code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
			options << " -DREAL=double -DREAL_VECTOR=double" << OpenCLRealType::dim << " -DCAST=long"
                    << " -DZERO=0.0 -DHALF=0.5 -DONE=1.0";
			code << cdfString1Double;
			code << pdfString1Double;
			code << safeExpStringDouble;

		} else { // 32-bit fp
			options << " -DREAL=float -DREAL_VECTOR=float" << OpenCLRealType::dim << " -DCAST=int"
                    << " -DZERO=0.0f -DHALF=0.5f -DONE=1.0f";
			code << cdfString1Float;
			code << pdfString1Float;
			code << safeExpStringFloat;
		}

		code <<
			" __kernel void computeLikContribs(__global const REAL *locDists,         \n" <<
			"  						          __global const REAL *timDiffs,          \n" <<
			"                                 __global const REAL *times,             \n" <<
			"						          __global REAL *likContribs,             \n" <<
            "                                 const REAL sigmaXprec,                  \n" <<
            "                                 const REAL tauXprec,                    \n" <<
			"                                 const REAL tauTprec,                    \n" <<
			"                                 const REAL omega,                       \n" <<
			"                                 const REAL theta,                       \n" <<
			"                                 const REAL mu0,                         \n" <<
			"                                 const int dimX,                         \n" <<
			"						          const uint locationCount) {             \n";

		code <<
		    "   const uint i = get_group_id(0);                                     \n" <<
		    "                                                                       \n" <<
		    "   const uint lid = get_local_id(0);                                   \n" <<
		    "   uint j = get_local_id(0);                                           \n" <<
		    "                                                                       \n" <<
		    "   __local REAL scratch[TPB];                                          \n" <<
		    "                                                                       \n" <<
		    "   REAL        sum = ZERO;                                             \n" <<
		    "                                                                       \n" <<
		    "   while (j < locationCount) {                                         \n" << // originally j < locationCount
		    "                                                                       \n" <<
		    "     const REAL timDiff = timDiffs[i * locationCount + j];            \n" <<
            "     const REAL distance = locDists[i * locationCount + j];            \n";

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                    const REAL innerContrib = mu0 * pow(tauXprec,dimX) *
                            tauTprec * pdf(distance * tauXprec) * pdf(timDiff*tauTprec) +
                            select(ZERO, theta, timDiff>ZERO)  *
                            pow(sigmaXprec,dimX) * pdf(distance * sigmaXprec) * safe_exp(-omega * timDiff);
        );

        code <<
             "     sum += innerContrib;                                              \n" <<
             "     j += TPB;                                                         \n" <<
             "     }                                                                 \n" <<
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
             "     likContribs[i] = log(scratch[0]) + theta / omega *               \n" <<
             "       ( exp(-omega*(times[locationCount-1]-times[i]))-1 ) -            \n" <<
             "       mu0 * ( cdf((times[locationCount-1]-times[i])*tauTprec)-             \n" <<
             "               cdf(-times[i]*tauTprec) )   ;                               \n" <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
#ifdef RBUILD
        Rcpp::Rcout << "Likelihood contributions kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Likelihood contributions kernel\n" << options.str() << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
		kernelLikContribsVector = boost::compute::kernel(program, "computeLikContribs");

#ifdef DEBUG_KERNELS
#ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

//		size_t index = -1;
//        kernelLikContribsVector.set_arg(index++, dLocDists);
//		kernelLikContribsVector.set_arg(index++, dTimDiffs);
//        kernelLikContribsVector.set_arg(index++, dTimes);
//        kernelLikContribsVector.set_arg(index++, dLikContribs);
//        kernelLikContribsVector.set_arg(index++, sigmaXprec);
//        kernelLikContribsVector.set_arg(index++, tauXprec);
//        kernelLikContribsVector.set_arg(index++, tauTprec);
//        kernelLikContribsVector.set_arg(index++, omega);
//        kernelLikContribsVector.set_arg(index++, theta);
//        kernelLikContribsVector.set_arg(index++, mu0);
//        kernelLikContribsVector.set_arg(index++, boost::compute::uint_(locationCount));

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

//        createOpenCLSummationKernel();
        createOpenCLLikContribsKernel();
//		createOpenCLGradientKernel();

	}

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

    boost::compute::device device;
    boost::compute::context ctx;
    boost::compute::command_queue queue;

    mm::MemoryManager<RealType> locDists;
    mm::MemoryManager<RealType> timDiffs;

    mm::MemoryManager<RealType> times;


    mm::MemoryManager<RealType> likContribs;
    mm::MemoryManager<RealType> storedLikContribs;

//	mm::MemoryManager<RealType> gradient;
    mm::GPUMemoryManager<RealType> dLocDists;
    mm::GPUMemoryManager<RealType> dTimDiffs;

    mm::GPUMemoryManager<RealType> dTimes;

#ifdef USE_VECTORS
//	mm::GPUMemoryManager<VectorType> dGradient;
#endif // USE_VECTORS


    mm::GPUMemoryManager<RealType> dLikContribs;
    mm::GPUMemoryManager<RealType> dStoredLikContribs;

    bool isStoredLikContribsEmpty;

    mm::MemoryManager<RealType> buffer;
    mm::MemoryManager<double> doubleBuffer;

    boost::compute::program program;
//    boost::compute::kernel kernelLikSum;

#ifdef USE_VECTORS
	boost::compute::kernel kernelLikContribsVector;
//	boost::compute::kernel kernelGradientVector;
#else
    boost::compute::kernel kernelLikContribs;
#endif // USE_VECTORS
};

} // namespace hph

#endif // _OPENCL_HAWKES_HPP
