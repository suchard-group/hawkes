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


//#define MICRO_BENCHMARK

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
          randomRates(locationCount), storedRandomRates(locationCount),

          likContribs(locationCount),
          storedLikContribs(locationCount),

          probsSelfExcite(locationCount),

          innerGradsContribs(locationCount),

          locations0(locationCount * OpenCLRealType::dim),
          locations1(locationCount * OpenCLRealType::dim),
          locationsPtr(&locations0),
          storedLocationsPtr(&locations1),

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

      dTimes = mm::GPUMemoryManager<RealType>(times.size(), ctx);
      dRandomRates = mm::GPUMemoryManager<RealType>(randomRates.size(), ctx);
      dStoredRandomRates= mm::GPUMemoryManager<RealType>(storedRandomRates.size(), ctx);
      dRandomRatesGradient = mm::GPUMemoryManager<RealType>(locationCount, ctx);

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

      dTimes = mm::GPUMemoryManager<RealType>(times.size(), ctx);
      dRandomRates = mm::GPUMemoryManager<RealType>(randomRates.size(), ctx);
      dStoredRandomRates = mm::GPUMemoryManager<RealType>(storedRandomRates.size(), ctx);
      dRandomRatesGradient = mm::GPUMemoryManager<RealType>(locationCount, ctx);
      dRandomRatesHessian = mm::GPUMemoryManager<RealType>(locationCount, ctx);


        std::cerr << "\twith vector-dim = " << OpenCLRealType::dim << std::endl;
#endif //RBUILD

#ifdef USE_VECTORS
        dLocations0 = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
        dLocations1 = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
        dGradient   = mm::GPUMemoryManager<VectorType>(locationCount, ctx);
#else
        dLocations0 = mm::GPUMemoryManager<RealType>(locations0.size(), ctx);
		dLocations1 = mm::GPUMemoryManager<RealType>(locations1.size(), ctx);
#endif // USE_VECTORS

        dLocationsPtr = &dLocations0;
        dStoredLocationsPtr = &dLocations1;

        dInnerGradsContribs = mm::GPUMemoryManager<RealType>(locationCount, ctx);

		dLikContribs = mm::GPUMemoryManager<RealType>(likContribs.size(), ctx);
		dStoredLikContribs = mm::GPUMemoryManager<RealType>(storedLikContribs.size(), ctx);

        dProbsSelfExcite = mm::GPUMemoryManager<RealType>(probsSelfExcite.size(), ctx);


#ifdef MICRO_BENCHMARK
	    timer.fill(0.0);
#endif

		createOpenCLKernels();
    }

#ifdef MICRO_BENCHMARK
    virtual ~OpenCLHawkes() {

//      std::cerr << "micro-benchmarks" << std::endl;
//      for (int i = 0; i < timer.size(); ++i) {
//          std::cerr << "\t" << timer[i] << std::endl;
//      }
    }
#endif

    void updateLocations(int locationIndex, double* location, size_t length) override {

        size_t offset{0};
        size_t deviceOffset{0};

        if (locationIndex == -1) {
            // Update all locations
            assert(length == OpenCLRealType::dim * locationCount ||
                   length == embeddingDimension * locationCount);

//            incrementsKnown = false;
//            isStoredSquaredResidualsEmpty = true;
//            isStoredTruncationsEmpty = true;

        } else {
            // Update a single location
            assert(length == OpenCLRealType::dim ||
                   length == embeddingDimension);

            if (updatedLocation != - 1) {
                // more than one location updated -- do a full recomputation
//                incrementsKnown = false;
//                isStoredSquaredResidualsEmpty = true;
//                isStoredTruncationsEmpty = true;
            }

            updatedLocation = locationIndex;

            offset = locationIndex * OpenCLRealType::dim;
#ifdef USE_VECTORS
            deviceOffset = static_cast<size_t>(locationIndex);
#else
            deviceOffset = locationIndex * embeddingDimension;
#endif
        }

        // If requires padding
        if (embeddingDimension != OpenCLRealType::dim) {
            if (locationIndex == -1) {

                mm::paddedBufferedCopy(location, embeddingDimension, embeddingDimension,
                                       begin(*locationsPtr) + offset, OpenCLRealType::dim,
                                       locationCount, buffer);

                length = OpenCLRealType::dim * locationCount; // New padded length

            } else {

                mm::bufferedCopy(location, location + length,
                                 begin(*locationsPtr) + offset,
                                 buffer);

                length = OpenCLRealType::dim;
            }
        } else {
            // Without padding
            mm::bufferedCopy(location, location + length,
                             begin(*locationsPtr) + offset,
                             buffer
            );
        }

        // COMPUTE
        mm::copyToDevice<OpenCLRealType>(begin(*locationsPtr) + offset,
                                         begin(*locationsPtr) + offset + length,
                                         dLocationsPtr->begin() + deviceOffset,
                                         queue
        );

//        sumOfIncrementsKnown = false;
    }

    int getInternalDimension() override { return OpenCLRealType::dim; }

//    void getLogLikelihoodGradient(double* result, size_t length) override {
//
//#ifdef MICRO_BENCHMARK
//        auto startTime = std::chrono::steady_clock::now();
//#endif
//
//        kernelGradientVector.set_arg(0, dLocations0);
//        kernelGradientVector.set_arg(1, dTimes);
//        kernelGradientVector.set_arg(2, dGradContribs);
//        kernelGradientVector.set_arg(3, static_cast<RealType>(sigmaXprec));
//        kernelGradientVector.set_arg(4, static_cast<RealType>(tauXprec));
//        kernelGradientVector.set_arg(5, static_cast<RealType>(tauTprec));
//        kernelGradientVector.set_arg(6, static_cast<RealType>(omega));
//        kernelGradientVector.set_arg(7, static_cast<RealType>(theta));
//        kernelGradientVector.set_arg(8, static_cast<RealType>(mu0));
//        kernelGradientVector.set_arg(9, boost::compute::int_(embeddingDimension));
//        kernelGradientVector.set_arg(10, boost::compute::uint_(locationCount));
//
//        queue.enqueue_1d_range_kernel(kernelGradientVector, 0,
//                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
//        queue.finish();
//
//#ifdef MICRO_BENCHMARK
//        timer[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
//#endif
//
////#ifdef MICRO_BENCHMARK
////        startTime = std::chrono::steady_clock::now();
////#endif
////
////        // TODO Start of extremely expensive part
////        std::vector<RealType> sum(6);
////        boost::compute::reduce(dSigmaXGradContribs.begin(), dSigmaXGradContribs.end(), &sum[0], queue);
////        boost::compute::reduce(dTauXGradContribs.begin(), dTauXGradContribs.end(), &sum[1], queue);
////        boost::compute::reduce(dTauTGradContribs.begin(), dTauTGradContribs.end(), &sum[2], queue);
////        boost::compute::reduce(dOmegaGradContribs.begin(), dOmegaGradContribs.end(), &sum[3], queue);
////        boost::compute::reduce(dThetaGradContribs.begin(), dThetaGradContribs.end(), &sum[4], queue);
////        boost::compute::reduce(dMu0GradContribs.begin(), dMu0GradContribs.end(), &sum[5], queue);
////        // TODO End of extremely expensive part
////
////#ifdef MICRO_BENCHMARK
////        timer[3] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
////#endif
//
////        sum[0] *= theta * pow(sigmaXprec,embeddingDimension+1);
////        sum[1] *= mu0 * pow(tauXprec,embeddingDimension+1) * tauTprec;
////        sum[2] *= mu0 * tauTprec * tauTprec;
////        sum[3] *= theta;
//
////        gradient = sum;
//
////        mm::bufferedCopy(std::begin(sum), std::end(sum), result, buffer);
//
//#ifdef MICRO_BENCHMARK
//        startTime = std::chrono::steady_clock::now();
//#endif
//
//        kernelLikSum.set_arg(0,dGradContribs);
//        kernelLikSum.set_arg(1,dGradient);
//        kernelLikSum.set_arg(2,boost::compute::uint_(locationCount));
//
//        queue.enqueue_1d_range_kernel(kernelLikSum, 0,
//                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
//        queue.finish();
//
//#ifdef MICRO_BENCHMARK
//        timer[3] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
//#endif
//
//        std::vector<double> middleMan(8);
//
//        mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
//                                                   middleMan.data(), buffer, queue);
//        queue.finish();
//
//        middleMan[0] *= theta * pow(sigmaXprec,embeddingDimension+1);
//        middleMan[1] *= mu0 * pow(tauXprec,embeddingDimension+1) * tauTprec;
//        middleMan[2] *= mu0 * tauTprec * tauTprec;
//        middleMan[3] *= theta;
//
//        memcpy(result,middleMan.data(), 48);
//
//    }

    void getLogLikelihoodGradient(double* result, size_t length) override {

        // TODO Buffer gradients

        kernelInnerGradsLoop.set_arg(0, dLocations0);
        kernelInnerGradsLoop.set_arg(1, dTimes);
        kernelInnerGradsLoop.set_arg(2, dInnerGradsContribs);
        kernelInnerGradsLoop.set_arg(3, dRandomRates);
        kernelInnerGradsLoop.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelInnerGradsLoop.set_arg(5, static_cast<RealType>(tauXprec));
        kernelInnerGradsLoop.set_arg(6, static_cast<RealType>(tauTprec));
        kernelInnerGradsLoop.set_arg(7, static_cast<RealType>(omega));
        kernelInnerGradsLoop.set_arg(8, static_cast<RealType>(theta));
        kernelInnerGradsLoop.set_arg(9, static_cast<RealType>(mu0));
        kernelInnerGradsLoop.set_arg(10, boost::compute::int_(embeddingDimension));
        kernelInnerGradsLoop.set_arg(11, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelInnerGradsLoop, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();

        kernelGradientVector.set_arg(0, dLocations0);
        kernelGradientVector.set_arg(1, dTimes);
        kernelGradientVector.set_arg(2, dInnerGradsContribs);
        kernelGradientVector.set_arg(3, dGradient);
        kernelGradientVector.set_arg(4, dRandomRates);
        kernelGradientVector.set_arg(5, static_cast<RealType>(sigmaXprec));
        kernelGradientVector.set_arg(6, static_cast<RealType>(tauXprec));
        kernelGradientVector.set_arg(7, static_cast<RealType>(tauTprec));
        kernelGradientVector.set_arg(8, static_cast<RealType>(omega));
        kernelGradientVector.set_arg(9, static_cast<RealType>(theta));
        kernelGradientVector.set_arg(10, static_cast<RealType>(mu0));
        kernelGradientVector.set_arg(11, boost::compute::int_(embeddingDimension));
        kernelGradientVector.set_arg(12, boost::compute::uint_(locationCount));


        queue.enqueue_1d_range_kernel(kernelGradientVector, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();

        if (length == locationCount * OpenCLRealType::dim) {

            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
                                                       result, buffer, queue);
            queue.finish();

        } else {

            if (doubleBuffer.size() != locationCount * OpenCLRealType::dim) {
                doubleBuffer.resize(locationCount * OpenCLRealType::dim);
            }

            mm::bufferedCopyFromDevice<OpenCLRealType>(dGradient.begin(), dGradient.end(),
                                                       doubleBuffer.data(), buffer, queue);
            queue.finish();

            mm::paddedBufferedCopy(begin(doubleBuffer), OpenCLRealType::dim, embeddingDimension,
                                   result, embeddingDimension,
                                   locationCount, buffer);
        }

    }

    void getRandomRatesLogLikelihoodGradient(double* result, size_t length) override {

        assert(length == locationCount);

        kernelInnerGradsLoop.set_arg(0, dLocations0);
        kernelInnerGradsLoop.set_arg(1, dTimes);
        kernelInnerGradsLoop.set_arg(2, dInnerGradsContribs);
        kernelInnerGradsLoop.set_arg(3, dRandomRates);
        kernelInnerGradsLoop.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelInnerGradsLoop.set_arg(5, static_cast<RealType>(tauXprec));
        kernelInnerGradsLoop.set_arg(6, static_cast<RealType>(tauTprec));
        kernelInnerGradsLoop.set_arg(7, static_cast<RealType>(omega));
        kernelInnerGradsLoop.set_arg(8, static_cast<RealType>(theta));
        kernelInnerGradsLoop.set_arg(9, static_cast<RealType>(mu0));
        kernelInnerGradsLoop.set_arg(10, boost::compute::int_(embeddingDimension));
        kernelInnerGradsLoop.set_arg(11, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelInnerGradsLoop, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();


        kernelRandomRatesGradient.set_arg(0, dLocations0);
        kernelRandomRatesGradient.set_arg(1, dTimes);
        kernelRandomRatesGradient.set_arg(2, dRandomRatesGradient);
        kernelRandomRatesGradient.set_arg(3, dInnerGradsContribs);
        kernelRandomRatesGradient.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelRandomRatesGradient.set_arg(5, static_cast<RealType>(omega));
        kernelRandomRatesGradient.set_arg(6, static_cast<RealType>(theta));
        kernelRandomRatesGradient.set_arg(7, boost::compute::int_(embeddingDimension));
        kernelRandomRatesGradient.set_arg(8, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelRandomRatesGradient, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();

        mm::bufferedCopyFromDevice<OpenCLRealType>(dRandomRatesGradient.begin(), dRandomRatesGradient.end(),
                                                   result, buffer, queue);
        queue.finish();
    }

    void getRandomRatesLogLikelihoodHessian(double* result, size_t length) override {

        assert(length == locationCount);

        kernelInnerGradsLoop.set_arg(0, dLocations0);
        kernelInnerGradsLoop.set_arg(1, dTimes);
        kernelInnerGradsLoop.set_arg(2, dInnerGradsContribs);
        kernelInnerGradsLoop.set_arg(3, dRandomRates);
        kernelInnerGradsLoop.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelInnerGradsLoop.set_arg(5, static_cast<RealType>(tauXprec));
        kernelInnerGradsLoop.set_arg(6, static_cast<RealType>(tauTprec));
        kernelInnerGradsLoop.set_arg(7, static_cast<RealType>(omega));
        kernelInnerGradsLoop.set_arg(8, static_cast<RealType>(theta));
        kernelInnerGradsLoop.set_arg(9, static_cast<RealType>(mu0));
        kernelInnerGradsLoop.set_arg(10, boost::compute::int_(embeddingDimension));
        kernelInnerGradsLoop.set_arg(11, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelInnerGradsLoop, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();


        kernelRandomRatesHessian.set_arg(0, dLocations0);
        kernelRandomRatesHessian.set_arg(1, dTimes);
        kernelRandomRatesHessian.set_arg(2, dRandomRatesHessian);
        kernelRandomRatesHessian.set_arg(3, dInnerGradsContribs);
        kernelRandomRatesHessian.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelRandomRatesHessian.set_arg(5, static_cast<RealType>(omega));
        kernelRandomRatesHessian.set_arg(6, static_cast<RealType>(theta));
        kernelRandomRatesHessian.set_arg(7, boost::compute::int_(embeddingDimension));
        kernelRandomRatesHessian.set_arg(8, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelRandomRatesHessian, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();

        mm::bufferedCopyFromDevice<OpenCLRealType>(dRandomRatesHessian.begin(), dRandomRatesHessian.end(),
                                                   result, buffer, queue);
        queue.finish();
    }

	void getProbsSelfExcite(double* result, size_t length) override {

        assert(length == locationCount);

        kernelProbsSelfExcite.set_arg(0, dLocations0);
        kernelProbsSelfExcite.set_arg(1, dTimes);
        kernelProbsSelfExcite.set_arg(2, dProbsSelfExcite);
        kernelProbsSelfExcite.set_arg(3, dRandomRates);
        kernelProbsSelfExcite.set_arg(4, static_cast<RealType>(sigmaXprec));
        kernelProbsSelfExcite.set_arg(5, static_cast<RealType>(tauXprec));
        kernelProbsSelfExcite.set_arg(6, static_cast<RealType>(tauTprec));
        kernelProbsSelfExcite.set_arg(7, static_cast<RealType>(omega));
        kernelProbsSelfExcite.set_arg(8, static_cast<RealType>(theta));
        kernelProbsSelfExcite.set_arg(9, static_cast<RealType>(mu0));
        kernelProbsSelfExcite.set_arg(10, boost::compute::int_(embeddingDimension));
        kernelProbsSelfExcite.set_arg(11, boost::compute::uint_(locationCount));

        queue.enqueue_1d_range_kernel(kernelProbsSelfExcite, 0,
                                      static_cast<unsigned int>(locationCount) * TPB, TPB);
        queue.finish();

        mm::bufferedCopyFromDevice<OpenCLRealType>(dProbsSelfExcite.begin(), dProbsSelfExcite.end(),
                                       result, buffer, queue);
        queue.finish();
 	}

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
        storedRandomRates = randomRates;

        std::copy(begin(*locationsPtr), end(*locationsPtr),
                  begin(*storedLocationsPtr));

        // COMPUTE
        boost::compute::copy(dLocationsPtr->begin(), dLocationsPtr->end(),
                             dStoredLocationsPtr->begin(), queue);

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

        randomRates = storedRandomRates;
        boost::compute::copy(
                dStoredRandomRates.begin(),
                dStoredRandomRates.end(),
                dRandomRates.begin(), queue
        );

        auto tmp1 = storedLocationsPtr;
        storedLocationsPtr = locationsPtr;
        locationsPtr = tmp1;

        // COMPUTE
        auto tmp2 = dStoredLocationsPtr;
        dStoredLocationsPtr = dLocationsPtr;
        dLocationsPtr = tmp2;

    }

    void setTimesData(double* data, size_t length) override {
        assert(length == times.size());
        mm::bufferedCopy(data, data + length, begin(times), buffer);

        // COMPUTE
        mm::bufferedCopyToDevice(data, data + length, dTimes.begin(),
                                 buffer, queue);
    }

    void setRandomRates(double* data, size_t length) override {
        assert(length == locationCount);
        mm::bufferedCopy(data, data + length, begin(randomRates), buffer);

        // COMPUTE
        mm::bufferedCopyToDevice(data, data + length, dRandomRates.begin(),
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

		//RealType lSumOfLikContribs = 0.0;

#ifdef MICRO_BENCHMARK
        auto startTime = std::chrono::steady_clock::now();
#endif

#ifdef USE_VECTORS
        kernelLikContribsVector.set_arg(0, dLocations0);
        kernelLikContribsVector.set_arg(1, dTimes);
        kernelLikContribsVector.set_arg(2, dLikContribs);
        kernelLikContribsVector.set_arg(3, dRandomRates);
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

#ifdef MICRO_BENCHMARK
        timer[0] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
#endif

//		for(int l = 0; l < locationCount; l++) {
//            std::cout << dLikContribs[l] << std::endl;
//        };

#ifdef MICRO_BENCHMARK
        startTime = std::chrono::steady_clock::now();
#endif

        RealType sum = RealType(0.0);
        boost::compute::reduce(dLikContribs.begin(), dLikContribs.end(), &sum, queue);

#ifdef MICRO_BENCHMARK
        timer[1] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
#endif

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

//    void createOpenCLSummationKernel() {
//
//        std::stringstream code;
//        std::stringstream options;
//
//        options << "-DTILE_DIM=" << TILE_DIM << " -DTPB=" << TPB;
//
//        if (sizeof(RealType) == 8) { // 64-bit fp
//            code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
//            options << " -DREAL=double -DREAL_VECTOR=double8" << " -DCAST=long"
//                    << " -DZERO=0.0 -DHALF=0.5";
//
//        } else { // 32-bit fp
//            options << " -DREAL=float -DREAL_VECTOR=float8" << " -DCAST=int"
//                    << " -DZERO=0.0f -DHALF=0.5f";
//        }
//
//        code <<
//             " __kernel void computeSum(__global const REAL_VECTOR *summand,           \n" <<
//             "                          __global REAL_VECTOR *outputSum,               \n" <<
//             "						    const uint locationCount) {                    \n";
//
//        code <<
//             "   const uint lid = get_local_id(0);                                   \n" <<
//             "   uint j = get_local_id(0);                                           \n" <<
//             "                                                                       \n" <<
//             "   __local REAL_VECTOR scratch[TPB];                                   \n" <<
//             "                                                                       \n" <<
//             "   REAL_VECTOR sum = ZERO;                                             \n" <<
//             "                                                                       \n" <<
//             "   while (j < locationCount) {                                         \n";
//
//        code <<
//             "     sum += summand[j];                                                \n" <<
//             "     j += TPB;                                                        \n" <<
//             "      }                                                               \n" <<
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
//             "     outputSum[0] = scratch[0];                                      \n" <<
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
//    }

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
			code << " #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
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
			" __kernel void computeLikContribs(__global const REAL_VECTOR *locations, \n" <<
			"                                 __global const REAL *times,             \n" <<
			"						          __global REAL *likContribs,             \n" <<
			"						          __global const REAL *randomRates,             \n" <<
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
		    "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
		    "   const REAL timeI = times[i];                                        \n" <<
		    "                                                                       \n" <<
		    "   REAL        sum = ZERO;                                             \n" <<
		    "   REAL mu0TauXprecDTauTprec = mu0 * pow(tauXprec,dimX) * tauTprec;    \n" <<
		    "   REAL thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;  \n" <<
		    "                                                                       \n" <<
		    "   while (j < locationCount) {                                         \n" << // originally j < locationCount
		    "                                                                       \n" <<
		    "     const REAL timDiff = timeI - times[j];                            \n" << // timDiffs[i * locationCount + j];
		    "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
		    "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL distance = sqrt(                                \n" <<
                 "              dot(difference.lo, difference.lo) +               \n" <<
                 "              dot(difference.hi, difference.hi)                 \n" <<
                 "      );                                                        \n";

        } else {
            code << "     const REAL distance = length(difference);                  \n";
        }

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL innerContrib = mu0TauXprecDTauTprec *
                                           pdf(distance * tauXprec) *
                                           select(ZERO, pdf(timDiff*tauTprec),
                                                   (CAST)isnotequal(timDiff,ZERO))  +
                                           thetaSigmaXprecDOmega *
                                           select(ZERO, randomRates[j] * exp(-omega * timDiff),
                                                   (CAST)isgreater(timDiff,ZERO)) * pdf(distance * sigmaXprec);
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
             "     likContribs[i] = log(scratch[0]) + theta *               \n" <<
             "       randomRates[i] * ( exp(-omega*(times[locationCount-1]-times[i]))-1 ) -            \n" <<
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

	}


    void createOpenCLProbsSelfExciteKernel() {

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
            code << " #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
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
        //TODO: possible speedup from smaller vector size to match embedding dimension (right now always 8)
        code <<
             " __kernel void computeProbsSelfExcite(__global const REAL_VECTOR *locations, \n" <<
             "                                 __global const REAL *times,             \n" <<
             "						          __global REAL *probsSelfExcite,             \n" <<
             "                                __global const REAL *randomRates,                \n" <<
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
             "   __local REAL scratch1[TPB];                                          \n" <<
             "   __local REAL scratch2[TPB];                                          \n" <<
             "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
             "   const REAL timeI = times[i];                                        \n" <<
             "                                                                       \n" <<
             "   REAL        sum1 = ZERO;                                             \n" <<
             "   REAL        sum2 = ZERO;                                             \n" <<
             "   REAL mu0TauXprecDTauTprec = mu0 * pow(tauXprec,dimX) * tauTprec;    \n" <<
             "   REAL thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;  \n" <<
             "                                                                       \n" <<
             "   while (j < locationCount) {                                         \n" << // originally j < locationCount
             "                                                                       \n" <<
             "     const REAL timDiff = timeI - times[j];                            \n" << // timDiffs[i * locationCount + j];
             "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL distance = sqrt(                                \n" <<
                 "              dot(difference.lo, difference.lo) +               \n" <<
                 "              dot(difference.hi, difference.hi)                 \n" <<
                 "      );                                                        \n";

        } else {
            code << "     const REAL distance = length(difference);                  \n";
        }

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL background = mu0TauXprecDTauTprec *
                                          pdf(distance * tauXprec) *
                                          select(ZERO, pdf(timDiff*tauTprec), (CAST)isnotequal(timDiff,ZERO));
        );

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL selfexcite = thetaSigmaXprecDOmega *
                                          select(ZERO, randomRates[j] * exp(-omega * timDiff), (CAST)isgreater(timDiff,ZERO)) *
                                          pdf(distance * sigmaXprec);
        );

        code <<
             "     sum1 += background;                                              \n" <<
             "     sum2 += selfexcite;                                              \n" <<
             "     j += TPB;                                                         \n" <<
             "     }                                                                 \n" <<
             "     scratch1[lid] = sum1;                                             \n" <<
             "     scratch2[lid] = sum2;                                               \n";

        code <<
             "     for(int k = 1; k < TPB; k <<= 1) {                                 \n" <<
             "       barrier(CLK_LOCAL_MEM_FENCE);                                    \n" <<
             "       uint mask = (k << 1) - 1;                                        \n" <<
             "       if ((lid & mask) == 0) {                                         \n" <<
             "           scratch1[lid] += scratch1[lid + k];                  \n" <<
             "           scratch2[lid] += scratch2[lid + k];                  \n" <<
             "       }                                                                \n" <<
             "   }                                                                    \n";

        code <<
             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
             "   if (lid == 0) {                                                     \n";

        code <<
             "     probsSelfExcite[i] = scratch2[0] / (scratch1[0] + scratch2[0]);   \n" <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp::Rcout << "ProbsSelfExcite contributions kernel\n" << code.str() << std::endl;
#else
        std::cerr << "ProbsSelfExcite contributions kernel\n" << options.str() << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelProbsSelfExcite = boost::compute::kernel(program, "computeProbsSelfExcite");

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

    }


    void createOpenCLInnerGradsLoopKernel() {

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
            code << " #pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
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
             " __kernel void innerGradsLoop(__global const REAL_VECTOR *locations, \n" <<
             "                                 __global const REAL *times,             \n" <<
             "						          __global REAL *output,       \n" <<
             "                                __global const REAL *randomRates,                \n" <<
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
             "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
             "   const REAL timeI = times[i];                                        \n" <<
             "                                                                       \n" <<
             "   REAL        sum = ZERO;                                             \n" <<
             "   REAL mu0TauXprecDTauTprec = mu0 * pow(tauXprec,dimX) * tauTprec;    \n" <<
             "   REAL thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;  \n" <<
             "                                                                       \n" <<
             "   while (j < locationCount) {                                         \n" << // originally j < locationCount
             "                                                                       \n" <<
             "     const REAL timDiff = timeI - times[j];                            \n" << // timDiffs[i * locationCount + j];
             "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL distance = sqrt(                                \n" <<
                 "              dot(difference.lo, difference.lo) +               \n" <<
                 "              dot(difference.hi, difference.hi)                 \n" <<
                 "      );                                                        \n";

        } else {
            code << "     const REAL distance = length(difference);                  \n";
        }

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL innerContrib = mu0TauXprecDTauTprec *
                                          pdf(distance * tauXprec) *
                                          select(ZERO, pdf(timDiff*tauTprec), (CAST)isnotequal(timDiff,ZERO))  +
                                          thetaSigmaXprecDOmega *
                                          select(ZERO, randomRates[j] * exp(-omega * timDiff), (CAST)isgreater(timDiff,ZERO)) * pdf(distance * sigmaXprec);
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
             "     output[i] = scratch[0];                                      \n" <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp::Rcout << "Inner gradient kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Inner gradient kernel\n" << options.str() << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelInnerGradsLoop = boost::compute::kernel(program, "innerGradsLoop");

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

    }


    void createOpenCLGradientKernel() {

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
             " __kernel void computeGradient(__global const REAL_VECTOR *locations,         \n" <<
             "                                 __global const REAL *times,             \n" <<
             "                                 __global const REAL *innerGradsContribs, \n" <<
             "						          __global REAL_VECTOR *output,           \n" <<
             "                                __global const REAL *randomRates,                \n" <<
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
             "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
             "   const REAL timeI = times[i];                                        \n" <<
             "                                                                       \n" <<
             "   __local REAL_VECTOR nNprimeScratch[TPB];                                 \n" <<
             "   __local REAL_VECTOR nprimeNScratch[TPB];                                 \n" <<
             "                                                                       \n" <<
             "   REAL_VECTOR        nNprimeSum = ZERO;                                      \n" <<
             "   REAL_VECTOR        nprimeNSum = ZERO;                                      \n" <<
             "                                                                           \n" <<
             "   const REAL sigmaXprec2 = sigmaXprec * sigmaXprec;                       \n" <<
             "   const REAL sigmaXprecD = pow(sigmaXprec, dimX);                             \n" <<
             "   const REAL tauXprec2 = tauXprec * tauXprec;                             \n" <<
             "   const REAL tauXprecD = pow(tauXprec, dimX);                             \n" <<
             "   const REAL tauTprec2 = tauTprec * tauTprec;                             \n" <<
             "   const REAL mu0TauXprecDTauTprec = mu0 * tauXprecD * tauTprec;           \n" <<
             "   const REAL thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;  \n" <<
             "                                                                       \n" <<
             "   while (j < locationCount) {                                         \n" << // originally j < locationCount
             "                                                                       \n" <<
             "     const REAL timDiff = timeI - times[j];                            \n" << // timDiffs[i * locationCount + j];
             "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL locDist = sqrt(                                \n" <<
                 "              dot(difference.lo, difference.lo) +               \n" <<
                 "              dot(difference.hi, difference.hi)                 \n" <<
                 "      );                                                        \n";

        } else {
            code << "     const REAL locDist = length(difference);                  \n";
        }
        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL pdfLocDistSigmaXPrec = pdf(locDist * sigmaXprec);
                const REAL pdfLocDistTauXPrec = pdf(locDist * tauXprec);
                const REAL pdfTimDiffTauTPrec = select(ZERO, pdf(timDiff*tauTprec), (CAST)isnotequal(timDiff,ZERO));//pdf(timDiff * tauTprec);
                const REAL expOmegaTimDiffNNprime = select(ZERO, randomRates[j] * exp(-omega * timDiff), (CAST)isgreater(timDiff,ZERO));
                const REAL expOmegaTimDiffNprimeN = select(ZERO, randomRates[i] * exp(omega * timDiff), (CAST)isless(timDiff,ZERO));

                const REAL baseRate = pdfLocDistTauXPrec * pdfTimDiffTauTPrec;
                const REAL seRateNNprime = pdfLocDistSigmaXPrec * expOmegaTimDiffNNprime;
                const REAL seRateNprimeN = pdfLocDistSigmaXPrec * expOmegaTimDiffNprimeN;

                const REAL_VECTOR nNprimeGrad = - (tauXprec2 * mu0TauXprecDTauTprec * baseRate +
                        sigmaXprec2 * thetaSigmaXprecDOmega * seRateNNprime) * difference;
                const REAL_VECTOR nprimeNGrad = - (tauXprec2 * mu0TauXprecDTauTprec * baseRate +
                        sigmaXprec2 * thetaSigmaXprecDOmega * seRateNprimeN) * difference / innerGradsContribs[j];

                nNprimeSum += nNprimeGrad;
                nprimeNSum += nprimeNGrad;


        j += TPB;
	        }

	        nNprimeScratch[lid] = nNprimeSum;
	        nprimeNScratch[lid] = nprimeNSum;

        );

        code <<
             "     for(int k = 1; k < TPB; k <<= 1) {                                 \n" <<
             "       barrier(CLK_LOCAL_MEM_FENCE);                                    \n" <<
             "       uint mask = (k << 1) - 1;                                        \n" <<
             "       if ((lid & mask) == 0) {                                         \n" <<
             "           nNprimeScratch[lid]   += nNprimeScratch[lid + k];            \n" <<
             "           nprimeNScratch[lid]   += nprimeNScratch[lid + k];            \n" <<
             "       }                                                                \n" <<
             "   }                                                                    \n";

        code <<
             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
             "   if (lid == 0) {                                                     \n";

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                output[i] = nNprimeScratch[0] / innerGradsContribs[i] + nprimeNScratch[0];
                );

        code <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp::Rcout << "Outer gradient kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Outer gradient kernel\n" << options.str() << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelGradientVector = boost::compute::kernel(program, "computeGradient");

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

    }

    void createOpenCLRandomRatesGradientKernel() {

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
             " __kernel void computeRandomRatesGradient(__global const REAL_VECTOR *locations,         \n" <<
             "                                 __global const REAL *times,             \n" <<
             "						          __global REAL *output,           \n" <<
             "                                 __global const REAL *innerGradsContribs, \n" <<
             "                                 const REAL sigmaXprec,                  \n" <<
             "                                 const REAL omega,                       \n" <<
             "                                 const REAL theta,                       \n" <<
             "                                 const int dimX,                         \n" <<
             "						          const uint locationCount) {             \n";

        code <<
             "   const uint i = get_group_id(0);                                     \n" <<
             "                                                                       \n" <<
             "   const uint lid = get_local_id(0);                                   \n" <<
             "   uint j = get_local_id(0);                                           \n" <<
             "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
             "   const REAL timeI = times[i];                                        \n" <<
             "                                                                       \n" <<
             "   __local REAL scratch[TPB];                                 \n" <<
             "                                                                       \n" <<
             "   REAL             sum = ZERO;                                      \n" <<
             "                                                                           \n" <<
             "   const REAL thetaSigmaXprecDOmega = theta * pow(sigmaXprec,dimX) * omega;  \n" <<
             "                                                                       \n" <<
             "   while (j < locationCount) {                                         \n" << // originally j < locationCount
             "                                                                       \n" <<
             "     const REAL timDiff = timeI - times[j];                            \n" << // timDiffs[i * locationCount + j];
             "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL locDist = sqrt(                                \n" <<
                 "              dot(difference.lo, difference.lo) +               \n" <<
                 "              dot(difference.hi, difference.hi)                 \n" <<
                 "      );                                                        \n";

        } else {
            code << "     const REAL locDist = length(difference);                  \n";
        }
        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL pdfLocDistSigmaXPrec = pdf(locDist * sigmaXprec);
                const REAL expOmegaTimDiffNprimeN = select(ZERO,  exp(omega * timDiff), (CAST)isless(timDiff,ZERO));

                const REAL seRateNprimeN = thetaSigmaXprecDOmega * pdfLocDistSigmaXPrec * expOmegaTimDiffNprimeN / innerGradsContribs[j];

                sum += seRateNprimeN;


                j += TPB;
        }

                scratch[lid] = sum;

        );

        code <<
             "     for(int k = 1; k < TPB; k <<= 1) {                                 \n" <<
             "       barrier(CLK_LOCAL_MEM_FENCE);                                    \n" <<
             "       uint mask = (k << 1) - 1;                                        \n" <<
             "       if ((lid & mask) == 0) {                                         \n" <<
             "           scratch[lid]   += scratch[lid + k];                          \n" <<
             "       }                                                                \n" <<
             "   }                                                                    \n";

        code <<
             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
             "   if (lid == 0) {                                                     \n";

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                output[i] = scratch[0] + theta * ( exp(-omega*(times[locationCount-1]-times[i]))-1 ); //nNprimeScratch[0] / innerGradsContribs[i] + nprimeNScratch[0];
        );

        code <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp::Rcout << "Random rates gradient kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Random rates gradient kernel\n" << options.str() << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelRandomRatesGradient = boost::compute::kernel(program, "computeRandomRatesGradient");

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

    }

    void createOpenCLRandomRatesHessianKernel() {

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
             " __kernel void computeRandomRatesHessian(__global const REAL_VECTOR *locations,         \n" <<
             "                                 __global const REAL *times,             \n" <<
             "						          __global REAL *output,           \n" <<
             "                                 __global const REAL *innerGradsContribs, \n" <<
             "                                 const REAL sigmaXprec,                  \n" <<
             "                                 const REAL omega,                       \n" <<
             "                                 const REAL theta,                       \n" <<
             "                                 const int dimX,                         \n" <<
             "						          const uint locationCount) {             \n";

        code <<
             "   const uint i = get_group_id(0);                                     \n" <<
             "                                                                       \n" <<
             "   const uint lid = get_local_id(0);                                   \n" <<
             "   uint j = get_local_id(0);                                           \n" <<
             "   const REAL_VECTOR vectorI = locations[i];                           \n" <<
             "   const REAL timeI = times[i];                                        \n" <<
             "                                                                       \n" <<
             "   __local REAL scratch[TPB];                                 \n" <<
             "                                                                       \n" <<
             "   REAL             sum = ZERO;                                      \n" <<
             "                                                                           \n" <<
             "   const REAL thetaSigmaXprecDOmega = pow(theta * pow(sigmaXprec,dimX) * omega, 2);  \n" <<
             "                                                                       \n" <<
             "   while (j < locationCount) {                                         \n" << // originally j < locationCount
             "                                                                       \n" <<
             "     const REAL timDiff = timeI - times[j];                            \n" << // timDiffs[i * locationCount + j];
             "     const REAL_VECTOR vectorJ = locations[j];                         \n" <<
             "     const REAL_VECTOR difference = vectorI - vectorJ;                 \n";

        if (OpenCLRealType::dim == 8) {
            code << "     const REAL locDist = sqrt(                                \n" <<
                 "              dot(difference.lo, difference.lo) +               \n" <<
                 "              dot(difference.hi, difference.hi)                 \n" <<
                 "      );                                                        \n";

        } else {
            code << "     const REAL locDist = length(difference);                  \n";
        }
        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                const REAL pdfLocDistSigmaXPrec = pow(pdf(locDist * sigmaXprec),2);
                const REAL expOmegaTimDiffNprimeN = select(ZERO,  exp(2 * omega * timDiff), (CAST)isless(timDiff,ZERO));

                const REAL seRateNprimeN = thetaSigmaXprecDOmega * pdfLocDistSigmaXPrec * expOmegaTimDiffNprimeN / pow(innerGradsContribs[j],2);

                sum += seRateNprimeN;

                j += TPB;
        }

                scratch[lid] = sum;

        );

        code <<
             "     for(int k = 1; k < TPB; k <<= 1) {                                 \n" <<
             "       barrier(CLK_LOCAL_MEM_FENCE);                                    \n" <<
             "       uint mask = (k << 1) - 1;                                        \n" <<
             "       if ((lid & mask) == 0) {                                         \n" <<
             "           scratch[lid]   += scratch[lid + k];                          \n" <<
             "       }                                                                \n" <<
             "   }                                                                    \n";

        code <<
             "   barrier(CLK_LOCAL_MEM_FENCE);                                       \n" <<
             "   if (lid == 0) {                                                     \n";

        code << BOOST_COMPUTE_STRINGIZE_SOURCE(
                output[i] = - scratch[0];
        );

        code <<
             "   }                                                                   \n" <<
             " }                                                                     \n ";

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp::Rcout << "Random rates gradient kernel\n" << code.str() << std::endl;
#else
        std::cerr << "Random rates gradient kernel\n" << options.str() << code.str() << std::endl;
#endif
#endif

        program = boost::compute::program::build_with_source(code.str(), ctx, options.str());
        kernelRandomRatesHessian = boost::compute::kernel(program, "computeRandomRatesHessian");

#ifdef DEBUG_KERNELS
        #ifdef RBUILD
        Rcpp:Rcout << "Successful build." << std::endl;
#else
        std::cerr << "Successful build." << std::endl;
#endif
#endif

    }


	void createOpenCLKernels() {

        createOpenCLLikContribsKernel();
		createOpenCLGradientKernel();
		createOpenCLRandomRatesGradientKernel();
        createOpenCLRandomRatesHessianKernel();
        createOpenCLProbsSelfExciteKernel();
        createOpenCLInnerGradsLoopKernel();

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

    mm::MemoryManager<RealType> times;
    mm::MemoryManager<RealType> randomRates;
    mm::MemoryManager<RealType> storedRandomRates;

    mm::MemoryManager<RealType> likContribs;
    mm::MemoryManager<RealType> storedLikContribs;

    mm::MemoryManager<RealType> probsSelfExcite;

    mm::MemoryManager<RealType> innerGradsContribs;

    mm::GPUMemoryManager<RealType> dTimes;
    mm::GPUMemoryManager<RealType> dRandomRates;
    mm::GPUMemoryManager<RealType> dStoredRandomRates;

    mm::GPUMemoryManager<RealType> dInnerGradsContribs;
    mm::GPUMemoryManager<VectorType> dGradient;

    mm::GPUMemoryManager<RealType> dRandomRatesGradient;
    mm::GPUMemoryManager<RealType> dRandomRatesHessian;

    mm::MemoryManager<RealType> locations0;
    mm::MemoryManager<RealType> locations1;

    mm::MemoryManager<RealType>* locationsPtr;
    mm::MemoryManager<RealType>* storedLocationsPtr;

#ifdef USE_VECTORS
    mm::GPUMemoryManager<VectorType> dLocations0;
    mm::GPUMemoryManager<VectorType> dLocations1;

    mm::GPUMemoryManager<VectorType>* dLocationsPtr;
    mm::GPUMemoryManager<VectorType>* dStoredLocationsPtr;
    #else
    mm::GPUMemoryManager<RealType> dLocations0;
    mm::GPUMemoryManager<RealType> dLocations1;

    mm::GPUMemoryManager<RealType>* dLocationsPtr;
    mm::GPUMemoryManager<RealType>* dStoredLocationsPtr;
#endif // USE_VECTORS


    mm::GPUMemoryManager<RealType> dLikContribs;
    mm::GPUMemoryManager<RealType> dStoredLikContribs;

    mm::GPUMemoryManager<RealType> dProbsSelfExcite;

    bool isStoredLikContribsEmpty;

    mm::MemoryManager<RealType> buffer;
    mm::MemoryManager<double> doubleBuffer;

    boost::compute::program program;

#ifdef USE_VECTORS
	boost::compute::kernel kernelLikContribsVector;
	boost::compute::kernel kernelGradientVector;
    boost::compute::kernel kernelProbsSelfExcite;
    boost::compute::kernel kernelInnerGradsLoop;
    boost::compute::kernel kernelRandomRatesGradient;
    boost::compute::kernel kernelRandomRatesHessian;

#else
    boost::compute::kernel kernelLikContribs;
#endif // USE_VECTORS

#ifdef MICRO_BENCHMARK
    std::array<double,4> timer;
#endif
};

} // namespace hph

#endif // _OPENCL_HAWKES_HPP
