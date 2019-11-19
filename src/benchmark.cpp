
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
    int iterations = vm["iterations"].as<int>();

    long flags = 0L;

	auto normal = std::normal_distribution<double>(0.0, 1.0);
	auto uniform = std::uniform_int_distribution<int>(0, locationCount - 1);
	auto binomial = std::bernoulli_distribution(0.75);
	auto normalData = std::normal_distribution<double>(0.0, 1.0);
	auto toss = std::bernoulli_distribution(0.25);
	auto expo = std::exponential_distribution<double>(1);
	
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

    auto elementCount = locationCount * locationCount; // size of pairwise data

    std::vector<double> times(locationCount);
	times[0] = expo(prng);
	for (int i = 1; i < locationCount; ++i) {
	    times[i] = times[i-1] + expo(prng);
	}
    instance->setTimesData(&times[0], locationCount);

    std::vector<double> data(elementCount); // pairwise distance data
    for (int i = 0; i < locationCount; ++i) {
        data[i * locationCount + i] = 0.0;
        for (int j = i + 1; j < locationCount; ++j) {

            const double draw = normalData(prng);
            double distance = draw * draw;

//            if(i==0 && j==1){ distance = 1;}//draw * draw;
//            if(i==0 && j==2){ distance = 1.5;}//draw * draw;
//            if(i==1 && j==2){ distance = 1.5;}//draw * draw;


            data[i * locationCount + j] = distance;
            data[j * locationCount + i] = distance;
        }
    }
	instance->setLocDistsData(&data[0], elementCount);

    for (int i = 0; i < locationCount; ++i) { // pairwise times data
        data[i * locationCount + i] = 0.0;
        for (int j = i + 1; j < locationCount; ++j) {
            data[i * locationCount + j] = times[i]-times[j]; //100% correct loadings
            data[j * locationCount + i] = times[j]-times[i];
        }
    }
    instance->setTimDiffsData(&data[0], elementCount);

	std::vector<double> parameters(6);
    for (int i = 0; i < 6; ++i) {
        parameters[i] = expo(prng2);
    }
	instance->setParameters(&parameters[0], 6);

	auto logLik = instance->getSumOfLikContribs();

    std::vector<double> gradient(6);
	instance->getLogLikelihoodGradient(gradient.data(),6);
    auto sumGradient = gradient;

	std::cout << "Starting HPH benchmark" << std::endl;
	auto startTime = std::chrono::steady_clock::now();

	double timer = 0;
    double timer2 = 0;

	for (auto itr = 0; itr < iterations; ++itr) {


        for (int i = 0; i < 6; ++i) {
            parameters[i] = expo(prng2);
        }
        instance->setParameters(&parameters[0], 6);

		auto startTime1 = std::chrono::steady_clock::now();

		auto inc = instance->getSumOfLikContribs();

        logLik += inc;
		
		auto duration1 = std::chrono::steady_clock::now() - startTime1;
		timer += std::chrono::duration<double, std::milli>(duration1).count();


        auto startTime2 = std::chrono::steady_clock::now();

        instance->getLogLikelihoodGradient(gradient.data(),6);

        auto duration2 = std::chrono::steady_clock::now() - startTime2;
        timer2 += std::chrono::duration<double, std::milli>(duration2).count();

        std::transform(sumGradient.begin(),sumGradient.end(),
                gradient.begin(),sumGradient.begin(),std::plus<double>());

	}
	logLik /= iterations + 1;
 //   sumGradient /= iterations + 1;

	auto endTime = std::chrono::steady_clock::now();
	auto duration = endTime - startTime;

	std::cout << "End HPH benchmark" << std::endl;
	std::cout << "AvgLogLik = " << logLik << std::endl;
    std::cout << "AvgGradient = " << "(" << sumGradient[0] << ", " << sumGradient[1] << ", " <<  sumGradient[2] << ", " <<
    sumGradient[3] << ", " <<  sumGradient[4] << ", " <<  sumGradient[5] << ")" <<  std::endl;
	std::cout << timer  << " ms" << std::endl;
    std::cout << timer2 << " ms" << std::endl;

	std::cout << std::chrono::duration<double, std::milli> (duration).count() << " ms "
			  << std::endl;

	std::ofstream outfile;
	outfile.open("report.txt",std::ios_base::app);
    outfile << deviceNumber << " " << threads << " " << simd << " " << locationCount << " " << embeddingDimension << " " << iterations << " " << timer << " " << timer2 << "\n" ;
	outfile.close();

}
