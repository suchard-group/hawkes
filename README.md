
hpHawkes: high performance Hawkes process library
===

hpHawkes facilitates fast Bayesian inference for the Hawkes process through GPU, multi-core CPU, and SIMD vectorization powered implementations of an adaptive Metropolis-Hastings algorithm. 

GPU capabilities for either build require installation of OpenCL computing framework. See section **Configurations** below.

# R package

[![Build Status](https://travis-ci.com/suchard-group/hawkes.svg?branch=master)](https://travis-ci.com/suchard-group/hawkes)

[![Build status](https://ci.appveyor.com/api/projects/status/github/suchard-group/hawkes?branch=master&svg=true)](https://ci.appveyor.com/project/andrewjholbrook/hawkes)


[![DOI](https://zenodo.org/badge/200700494.svg)](https://zenodo.org/badge/latestdoi/200700494)



### Testing

After cloning `hpHawkes` to your prefered directory, build the `R` package via the command line with
```
R CMD build hawkes
```
Open `R` and use the `timeTest` function to time the different implementations of the spatio-temporal Hawkes process log likelihood calculations for `maxIts` iterations.  First check the serial implementation for 1000 locations.

```
timeTest(locationCount=1000)
```
We are interested in the number under `elapsed`.  The implementation using AVX SIMD should be roughly twice as fast.

```
timeTest(locationCount=1000, simd=2) 
```

Even faster should be a combination of AVX and a 4 core approach.

```
timeTest(locationCount=1000, simd=2, threads=4) 
```

The GPU implementation should be fastest of all.

```
timeTest(locationCount=1000, gpu=1) 
```

Not all GPUs have double precision capabilities. You might need to set `single=1` to use your GPU. If you have an eGPU connected, try fixing `gpu=2`. 

Speed computing the log likelihood should translate directly to faster MCMC times. Compare these MCMC implementations:

```
# 2010 Washington D.C. gunfire data `dcData` is already loaded
is.unsorted(dcData$Time) # check that Time is ordered correctly

one_thread_no_simd <- sampler(n_iter=10, locations=cbind(dcData$X,dcData$Y), times=dcData$Time)

two_threads_avx <- sampler(n_iter=10, locations=cbind(dcData$X,dcData$Y), times=dcData$Time, threads=2, simd=2)

gpu <- sampler(n_iter=10, locations=cbind(dcData$X,dcData$Y), times=dcData$Time, gpu=2)

one_thread_no_simd$Time
two_threads_avx$Time
gpu$Time
```
Again, you might need to set `single=1` to get the GPU implementation working.  Hopefully, the elapsed times are fastest for the GPU implementation and slowest for the single threaded, no SIMD implementation.

Starting values for parameters are fed to `sampler` with 6-element long `params` vector. Currently, elements 2 and 3 of `params` are not sampled over: these are the spatial and temporal precisions (1/lengthscale) for the Gaussian kernel background intensity.  They remain fixed and must be specified with care.  Elements 1, 4, 5 and 6 are the self-excitatory spatial precision (1/lengthscale), self-excitatory temporal precision, the self-excitatory weight and the background weight, respectively.

Get MCMC sample using GPU with initial values of 1 and spatial and temporal background lengthscales set to 1.6 kilometers and 14 days:
```
gpu <- sampler(n_iter=2000,
burnIn=1000,
params=c(1,1/1.6,1/(14*24),1,1,1), locations=cbind(dcData$X,dcData$Y),
times=dcData$Time,
gpu=2)
```
Create trace plot for, say, the self-excitatory weight:
```
plot(gpu$samples[5,])
```

# Standalone library

### Compilation

The standalone build requires CMake Version ≥ 2.8. Use the terminal to navigate to directory `hawkes/build`.

```
cd build
cmake ..
make
```

### Testing

Once the library is built, test the various implementation settings. The `benchmark` program computes the MDS log likelihood and its gradient for a given number of iterations and returns the time taken for each. First check the serial implementation for 1000 locations.

```
./benchmark --locations 1000 
```

The following implementation using AVX SIMD should be roughly twice as fast.

```
./benchmark --locations 1000 --avx
```

Even faster should be a combination of AVX and a 4 core approach.

```
./benchmark --locations 1000 --avx --tbb 4
```

The GPU implementation should be fastest of all. Make sure that your GPU can handle double precision floating points.  If not, make sure to toggle `--float`.  

```
./benchmark --locations 1000 --gpu 2
```

Test the different methods by increasing `iterations` and `locations`.



# Configurations

### OpenCL

Both builds of hpHawkes rely on the OpenCL framework for their GPU capabilities. Builds using OpenCL generally require access to the OpenCL headers <https://github.com/KhronosGroup/OpenCL-Headers> and the shared library `OpenCL.so` (or dynamically linked library `OpenCL.dll` for Windows).  Since we have included the headers in the package, one only needs acquire the shared library. Vendor specific drivers include the OpenCL shared library and are available here:

NVIDIA <https://www.nvidia.com/Download/index.aspx?lang=en-us>

AMD <https://www.amd.com/en/support>

Intel <https://downloadcenter.intel.com/product/80939/Graphics-Drivers> .


Another approach is to download vendor specific SDKs, which also include the shared libraries. <https://github.com/cdeterman/gpuR/wiki/Installing-OpenCL> has more details on this approach.

#### OpenCL on Windows
Building the hpHawkes R package on Windows with OpenCL requires copying (once installed) `OpenCl.dll` to the hpHawkes library.  For a 64 bit machine use

```
cd hawkes
scp /C/Windows/System32/OpenCL.dll inst/lib/x64
```
and for a 32 bit machine use the following.
```
cd hawkes
scp /C/Windows/SysWOW64/OpenCL.dll inst/lib/i386
```
Finally, uncomment the indicated lines in `src/Makevars.win`.


### C++14 on Windows

Compiling with C++14 and the default Rtools Mingw64 compiler causes an error. Circumvent the error by running the following R code prior to build (cf. [RStan for Windows](https://github.com/stan-dev/rstan/wiki/Installing-RStan-from-source-on-Windows#configuration)).

```
dotR <- file.path(Sys.getenv("HOME"), ".R")
if (!file.exists(dotR))
  dir.create(dotR)
M <- file.path(dotR, "Makevars.win")
if (!file.exists(M))
  file.create(M) 
cat("\nCXX14FLAGS=-O3",
  "CXX14 = $(BINPREF)g++ -m$(WIN) -std=c++1y",
  "CXX11FLAGS=-O3", file = M, sep = "\n", append = TRUE)
```

### eGPU

External GPUs can be difficult to set up. For OSX, we have had success with `macOS-eGPU.sh`

<https://github.com/learex/macOS-eGPU> .

For Windows 10 on Mac (using bootcamp), we have had success with the `automate-eGPU EFI`

<https://egpu.io/forums/mac-setup/automate-egpu-efi-egpu-boot-manager-for-macos-and-windows/> .

The online community <https://egpu.io/> is helpful for other builds.
