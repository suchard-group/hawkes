#ifndef _ABSTRACT_HAWKES_HPP
#define _ABSTRACT_HAWKES_HPP

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <complex>
#include <future>

#ifdef USE_SIMD
#include <emmintrin.h>
#include <smmintrin.h>
#endif

#define USE_TBB

#ifdef USE_TBB
    #include "tbb/parallel_reduce.h"
    #include "tbb/blocked_range.h"
    #include "tbb/parallel_for.h"
    #include "tbb/task_scheduler_init.h"
#endif

#include "MemoryManagement.hpp"
#include "ThreadPool.h"
#include "CDF.h"
#include "flags.h"

namespace hph {

class AbstractHawkes {
public:
    AbstractHawkes(int embeddingDimension, int locationCount, long flags)
        : embeddingDimension(embeddingDimension), locationCount(locationCount),
          observationCount(locationCount * (locationCount - 1) / 2),
          flags(flags) { }

    virtual ~AbstractHawkes() = default;

    // Interface
    virtual double getSumOfLikContribs() = 0;
//    virtual void getLogLikelihoodGradient(double*, size_t) = 0;
    virtual void storeState() = 0;
    virtual void restoreState() = 0;
    virtual void acceptState() = 0;
    virtual void setLocDistsData(double*, size_t)  = 0;
    virtual void setTimDiffsData(double*, size_t)  = 0;
    virtual void setTimesData(double*, size_t)  = 0;
    virtual void setParameters(double*, size_t) = 0;
//    virtual void makeDirty() = 0;
    virtual int getInternalDimension() = 0;

protected:
    int embeddingDimension;
    int locationCount;
    int observationCount;
    long flags;

    int updatedLocation = -1;
    bool incrementsKnown = false;
    bool sumOfIncrementsKnown = false;
};

typedef std::shared_ptr<hph::AbstractHawkes> SharedPtr;

SharedPtr factory(int dim1, int dim2, long flags, int device, int threads);

struct CpuAccumulate { };

#ifdef USE_TBB
struct TbbAccumulate{ };
#endif


} // namespace hph

#endif // _ABSTRACT_HAWKES_HPP
