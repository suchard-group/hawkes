
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>

#include "AbstractHawkes.hpp"


//int cnt = 0;

template <typename T, typename PRNG, typename D>
void generateLocation(T& locations, D& d, PRNG& prng) {
    //int i = cnt++;
	for (auto& location : locations) {
		location = d(prng);
//        location = i++;
	}
}

//double getGradient

int main(int argc, char* argv[]) {

	// Set-up CLI
	namespace po = boost::program_options;
	po::options_description desc("Allowed options");
	desc.add_options()
            ("help", "produce help message")
            ("gpu", po::value<int>()->default_value(0), "number of GPU on which to run")
            ("tbb", po::value<int>()->default_value(0), "use TBB with specified number of threads")
            ("float", "run in single-precision")
            ("iterations", po::value<int>()->default_value(1), "number of iterations")
            ("locations", po::value<int>()->default_value(3), "number of locations")
            ("dimension", po::value<int>()->default_value(2), "number of dimensions")
			("internal", "use internal dimension")
            ("sse", "use hand-rolled SSE")
            ("avx", "use hand-rolled AVX")
            ("avx512", "use hand-rolled AVX-512")
	;
	po::variables_map vm;

	try {
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
	} catch (std::exception& e) {
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("help")) {
		std::cout << desc << std::endl;
		return 1;
	}

	long seed = 666L;

	auto prng = std::mt19937(seed);
	auto prng2 = std::mt19937(seed);

	std::cout << "Loading data" << std::endl;

	int embeddingDimension = vm["dimension"].as<int>();
	int locationCount = vm["locations"].as<int>();

	long flags = 0L;

	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, locationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);
	auto normalData = std::normal_distribution<double>(0.0, 1.0);
	auto toss = std::bernoulli_distribution(0.25);
	
	std::shared_ptr<tbb::task_scheduler_init> task{nullptr};

    int deviceNumber = -1;
    int threads = 0;
	if (vm["gpu"].as<int>() > 0) {
		std::cout << "Running on GPU" << std::endl;
		flags |= hph::Flags::OPENCL;
        deviceNumber = vm["gpu"].as<int>() - 1;
	} else {
		std::cout << "Running on CPU" << std::endl;
		
		threads = vm["tbb"].as<int>();
		if (threads != 0) {
			std::cout << "Using TBB with " << threads << " out of " 
			          << tbb::task_scheduler_init::default_num_threads()
			          << " threads" << std::endl;
			flags |= hph::Flags::TBB;
			task = std::make_shared<tbb::task_scheduler_init>(threads);
		}
	}
	
	if (vm.count("float")) {
		std::cout << "Running in single-precision" << std::endl;
		flags |= hph::Flags::FLOAT;
	} else {
		std::cout << "Running in double-precision" << std::endl;
	}

    int simdCount = 0;
    auto simd = "no simd";
    if (vm.count("sse")){
        ++simdCount;
        simd = "sse";
    }
    if (vm.count("avx")){
        ++simdCount;
        simd = "avx";
    }
    if (vm.count("avx512")){
        ++simdCount;
        simd = "avx512";
    }

    if (simdCount > 0) {
#if not defined(USE_SSE) && not defined(USE_AVX) && not defined(USE_AVX512)
        std::cerr << "SIMD is not implemented" << std::endl;
        exit(-1);
#else
        if (simdCount > 1) {
            std::cerr << "Can not request more than one SIMD simultaneously" << std::endl;
            exit(-1);
        }
        if (vm.count("avx512")) {
#ifndef USE_AVX512
            std::cerr << "AVX-512 is not implemented" << std::endl;
            exit(-1);
#else
            flags |= hph::Flags::AVX512;
#endif // USE_AVX512

		} else if (vm.count("avx")) {
#ifndef USE_AVX
			std::cerr << "AVX is not implemented" << std::endl;
			exit(-1);
#else
            flags |= hph::Flags::AVX;
#endif // USE_AVX
        } else {
            flags |= hph::Flags::SSE;
        }
#endif // not defined(USE_SSE) && not defined(USE_AVX) && not defined(USE_AVX512)
	}

	bool internalDimension = vm.count("internal");

	hph::SharedPtr instance = hph::factory(embeddingDimension, locationCount, flags, deviceNumber, threads);

	auto elementCount = locationCount * locationCount;
	std::vector<double> data(elementCount);
	for (int i = 0; i < locationCount; ++i) {
	    data[i * locationCount + i] = 0.0;
	    for (int j = i + 1; j < locationCount; ++j) {

	        const double draw = normalData(prng);
	        double distance = draw * draw;

	        data[i * locationCount + j] = distance;
	        data[j * locationCount + i] = distance;
	    }
	}

	instance->setPairwiseData(&data[0], elementCount);

	int dataDimension = internalDimension ? instance->getInternalDimension() : embeddingDimension;

	std::vector<double> location(dataDimension);
	std::vector<double> allLocations;
	allLocations.resize(dataDimension * locationCount);

	for (int i = 0; i < locationCount; ++i) {
		generateLocation(location, normal, prng);
		instance->updateLocations(i, &location[0], dataDimension);
	}

//    int gradientIndex = 1;

	double precision = 1.0;
	instance->setParameters(&precision, 1);

	instance->makeDirty();
	auto logLik = instance->getSumOfIncrements();

//    std::vector<double> gradient(locationCount * dataDimension);

//    instance->getLogLikelihoodGradient(gradient.data(), locationCount * dataDimension);
//    double sumGradient = gradient[gradientIndex];

	std::cout << "Starting HPH benchmark" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	int iterations = vm["iterations"].as<int>();
	
	double timer = 0;
    double timer2 = 0;

	for (auto itr = 0; itr < iterations; ++itr) {

		instance->storeState();

		generateLocation(allLocations, normal, prng);
		instance->updateLocations(-1, &allLocations[0], dataDimension * locationCount);
		
		auto startTime1 = std::chrono::steady_clock::now();

		double inc = instance->getSumOfIncrements();
		logLik += inc;
		
		auto duration1 = std::chrono::steady_clock::now() - startTime1;
		timer += std::chrono::duration<double, std::milli>(duration1).count();

		bool restore = binomial(prng);
		if (restore) {
			instance->restoreState();
		} else {
		    instance->acceptState();
		}

//        auto startTime2 = std::chrono::steady_clock::now();

//        instance->getLogLikelihoodGradient(gradient.data(), locationCount * dataDimension);

//        auto duration2 = std::chrono::steady_clock::now() - startTime2;
//        timer2 += std::chrono::duration<double, std::milli>(duration2).count();

//        sumGradient += gradient[gradientIndex];

	}
	logLik /= iterations + 1;
//    sumGradient /= iterations + 1;

	auto endTime = std::chrono::steady_clock::now();
	auto duration = endTime - startTime;

	std::cout << "End HPH benchmark" << std::endl;
	std::cout << "AvgLogLik = " << logLik << std::endl;
//    std::cout << "AvgSumGradient = " << sumGradient << std::endl;
	std::cout << timer  << " ms" << std::endl;
    std::cout << timer2 << " ms" << std::endl;

	std::cout << std::chrono::duration<double, std::milli> (duration).count() << " ms "
			  << std::endl;

	std::ofstream outfile;
	outfile.open("report.txt",std::ios_base::app);
    outfile << deviceNumber << " " << threads << " " << simd << " " << locationCount << " " << embeddingDimension << " " << iterations << " " << timer << " " << timer2 << "\n" ;
	outfile.close();

}
