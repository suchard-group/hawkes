#include "AbstractHawkes.hpp"

// forward reference
namespace hph {
#ifdef HAVE_OPENCL
    SharedPtr constructOpenCLHawkesDouble(int, int, long, int);
    SharedPtr constructOpenCLHawkesFloat(int, int, long, int);
#endif
    SharedPtr constructNewHawkesDoubleNoParallelNoSimd(int, int, long, int);
    SharedPtr constructNewHawkesDoubleTbbNoSimd(int, int, long, int);

    SharedPtr constructNewHawkesFloatNoParallelNoSimd(int, int, long, int);
    SharedPtr constructNewHawkesFloatTbbNoSimd(int, int, long, int);

#ifdef USE_SSE
    SharedPtr constructNewHawkesDoubleTbbSse(int, int, long, int);
    SharedPtr constructNewHawkesDoubleNoParallelSse(int, int, long, int);
    SharedPtr constructNewHawkesFloatNoParallelSse(int, int, long, int);
    SharedPtr constructNewHawkesFloatTbbSse(int, int, long, int);
#endif

#ifdef USE_AVX
    SharedPtr constructNewHawkesDoubleTbbAvx(int, int, long, int);
    SharedPtr constructNewHawkesDoubleNoParallelAvx(int, int, long, int);
#endif

#ifdef USE_AVX512
    SharedPtr constructNewHawkesDoubleTbbAvx512(int, int, long, int);
    SharedPtr constructNewHawkesDoubleNoParallelAvx512(int, int, long, int);
#endif

SharedPtr factory(int dim1, int dim2, long flags, int device, int threads) {
	bool useFloat = flags & hph::Flags::FLOAT;
	bool useOpenCL = flags & hph::Flags::OPENCL;
	bool useTbb = flags & hph::Flags::TBB;
    bool useAvx512 = flags & hph::Flags::AVX512;
	bool useAvx = flags & hph::Flags::AVX;
	bool useSse = flags & hph::Flags::SSE;

	if (useFloat) {
		if (useOpenCL) {
#ifdef HAVE_OPENCL
			return constructOpenCLHawkesFloat(dim1, dim2, flags, device);
#else
		  return constructNewHawkesFloatNoParallelNoSimd(dim1, dim2, flags, threads);
#endif
		} else {
#ifdef USE_SSE
		    if (useSse) {
                if (useTbb) {
                    return constructNewHawkesFloatTbbSse(dim1, dim2, flags, threads);
                } else {
                    return constructNewHawkesFloatNoParallelSse(dim1, dim2, flags, threads);
                }
		    } else {
#endif
                if (useTbb) {
                    return constructNewHawkesFloatTbbNoSimd(dim1, dim2, flags, threads);
                } else {
                    return constructNewHawkesFloatNoParallelNoSimd(dim1, dim2, flags, threads);
                }
#ifdef USE_SSE
            }
#endif
		}
	} else {
		if (useOpenCL) {
#ifdef HAVE_OPENCL
			return constructOpenCLHawkesDouble(dim1, dim2, flags, device);
#else
		  return constructNewHawkesDoubleNoParallelNoSimd(dim1, dim2, flags, threads);
#endif
		} else {

#ifdef USE_AVX512
            if (useAvx512) {
                if (useTbb) {
                    return constructNewHawkesDoubleTbbAvx512(dim1, dim2, flags, threads);
                } else {
                    return constructNewHawkesDoubleNoParallelAvx512(dim1, dim2, flags, threads);
                }
            } else
#else
              useAvx512 = false; // stops unused variable warning when AVX512 is unavailable
#endif // USE_AVX512

#ifdef USE_AVX
		    if (useAvx) {
                if (useTbb) {
                    return constructNewHawkesDoubleTbbAvx(dim1, dim2, flags, threads);
                } else {
                    return constructNewHawkesDoubleNoParallelAvx(dim1, dim2, flags, threads);
                }
            } else
#else
              useAvx = false;
#endif // USE_AVX

#ifdef USE_SSE
		    if (useSse) {
                if (useTbb) {
                    return constructNewHawkesDoubleTbbSse(dim1, dim2, flags, threads);
                } else {
                    return constructNewHawkesDoubleNoParallelSse(dim1, dim2, flags, threads);
                }
		    } else
#endif // USE_SSE
            {

                if (useTbb) {
                    return constructNewHawkesDoubleTbbNoSimd(dim1, dim2, flags, threads);
                } else {
                    return constructNewHawkesDoubleNoParallelNoSimd(dim1, dim2, flags, threads);
                }
            }
		}
	}
}

} // namespace hph
