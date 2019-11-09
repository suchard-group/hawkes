

#' Serially computed Hawkes process log likelihood and gradient
#'
#' Called by \code{hpHawkes::test()} to compare output with parallel implementations.
#' Slow. Not recommended for use.
#'
#' @param locations Matrix of spatial locations (nxp).
#' @param times    Vector of times.
#' @param parameters Hawkes process parameters (length=6).
#' @param gradient Return gradient (or log likelihood)? Defaults to FALSE.
#' @return Hawkes process log likelihood or its gradient.
#'
#' @export
computeLoglikelihood <- function(locations, times, parameters, gradient = FALSE) {

  if (gradient) {
    gradLogLikelihood <- rep(0,6)
    gradLogLikelihood[1] <- num_grad("h",params=parameters,obs_x=locations,obs_t=times )#theta_grad(params=parameters,obs_x=locations,obs_t=times)
    gradLogLikelihood[2] <- num_grad("tau_x",params=parameters,obs_x=locations,obs_t=times )#theta_grad(params=parameters,obs_x=locations,obs_t=times)
    gradLogLikelihood[3] <- num_grad("tau_t",params=parameters,obs_x=locations,obs_t=times )#theta_grad(params=parameters,obs_x=locations,obs_t=times)
    gradLogLikelihood[4] <- num_grad("omega",params=parameters,obs_x=locations,obs_t=times )#theta_grad(params=parameters,obs_x=locations,obs_t=times)
    gradLogLikelihood[5] <- num_grad("theta",params=parameters,obs_x=locations,obs_t=times )#theta_grad(params=parameters,obs_x=locations,obs_t=times)
    gradLogLikelihood[6] <- num_grad("mu_0",params=parameters,obs_x=locations,obs_t=times )#theta_grad(params=parameters,obs_x=locations,obs_t=times)
    return(gradLogLikelihood)
  } else {
    logLikelihood <- log_lik(params=parameters,obs_x=locations,obs_t=times)
    return(logLikelihood)
  }
}


#' Compare serially and parallel-ly computed log likelihoods and gradients
#'
#' Compare outputs for serially and parallel-ly computed Hawkes process log likelihoods and gradients across
#' a range of implementations. Randomly generates
#' distance matrix and latent locations.
#'
#' @param locationCount Size of distance matrix or number of latent locations.
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @return Returns MDS log likelihoods (should be equal) and distance between gradients (should be 0).
#'
#' @export
test <- function(locationCount=10, threads=0, simd=0, gpu=0, single=0) {

  embeddingDimension <- 2

  locations <- matrix(rnorm(n = locationCount * embeddingDimension),
                 ncol = embeddingDimension, nrow = locationCount)
  times <- seq(from=1, to=locationCount, by=1)

  locDists <- as.matrix(dist(locations))
  timDiffs <- matrix(0,locationCount,locationCount)
  for(i in 1:locationCount){
    for(j in 1:locationCount){
      timDiffs[i,j] <- times[j]-times[i]
    }
  }
  params <- rexp(6)

  engine <- hpHawkes::createEngine(embeddingDimension, locationCount, threads, simd, gpu,single)
  engine <- hpHawkes::setLocDistsData(engine, locDists)
  engine <- hpHawkes::setTimDiffsData(engine, timDiffs)
  engine <- hpHawkes::setTimDiffsData(engine, timDiffs)
  engine <- hpHawkes::setTimesData(engine, times)
  engine <- hpHawkes::setParameters(engine, params)
  params2 <- list()
  params2$h     <- 1/params[1]
  params2$tau_x <- 1/params[2]
  params2$tau_t <- 1/params[3]
  params2$omega <- params[4]
  params2$theta <- params[5]
  params2$mu_0  <- params[6]

  cat("logliks\n")
  print(hpHawkes::getLogLikelihood(engine))
  print(computeLoglikelihood(locations=locations,
                             times=times,
                             parameters=params2))

 cat("grads\n")
print(hpHawkes::getGradient(engine))
print(computeLoglikelihood(locations=locations,
                           times=times,
                           parameters=params2,gradient = TRUE))
}


#' Time log likelihood and gradient calculations
#'
#' Time log likelihood and gradient calculations for serial and parallel implementations.
#'
#' @param locationCount Size of distance matrix or number of latent locations.
#' @param maxIts Number of times to compute the log likelihood and gradient.
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @return User, system and elapsed time. Elapsed time is most important.
#'
#' @export
timeTest <- function(locationCount=5000, maxIts=1, threads=0, simd=0,gpu=0,single=0) {
  # function returns length of time to compute log likelihood and gradient
  # threads is number of CPU cores used
  # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
  embeddingDimension <- 2
  truncation <- TRUE

  data <- matrix(rnorm(n = locationCount * locationCount, sd = 2),
                 ncol = locationCount, nrow = locationCount)
  data <- data * data    # Make positive
  data <- data + t(data) # Make symmetric
  diag(data) <- 0        # Make pairwise distance

  locations <- matrix(rnorm(n = embeddingDimension * locationCount, sd = 1),
                      ncol = embeddingDimension, nrow = locationCount)
  engine <- hpHawkes::createEngine(embeddingDimension, locationCount, truncation, threads, simd, gpu, single)
  engine <- hpHawkes::setPairwiseData(engine, data)
  engine <- hpHawkes::updateLocations(engine, locations)
  engine <- hpHawkes::setPrecision(engine, 2.0)

  ptm <- proc.time()
  for(i in 1:maxIts){
    hpHawkes::getLogLikelihood(engine)
    hpHawkes::getGradient(engine)
  }
  proc.time() - ptm
}


#' Initialize HPH engine object
#'
#' Takes data, parameters and implementation details and returns HPH engine object. Used within
#' \code{hpHawkes::hmcsampler()}.
#'
#' @param locDists N by N locations distances matrix.
#' @param N Number of locations and size of distance matrix.
#' @param P Dimension of locations.
#' @param times    Vector of times.
#' @param parameters Hawkes process parameters (length=6).
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @return MDS engine object.
#'
#' @export
engineInitial <- function(locDists,N,P,times,parameters=c(1,6),
                          threads,simd,gpu,single) {

  # Build reusable object
  engine <- hpHawkes::createEngine(embeddingDimension = P,
                                   locationCount = N,
                                   tbb = threads, simd=simd,
                                   gpu=gpu, single=single)

  # Set locDists data
  engine <- hpHawkes::setLocDistsData(engine, as.matrix(locDists))

  timDiffs <- matrix(0,N,N)
  for(i in 1:N){
    for(j in 1:N){
      timDiffs[i,j] <- times[j]-times[i]
    }
  }

  # Set Time Diffs Data
  engine <- hpHawkes::setTimDiffsData(engine, timDiffs)

  # Set Times Data
  engine <- hpHawkes::setTimesData(engine, times)

  # Call every time precision changes
  engine <- hpHawkes::setParameters(engine, parameters = parameters)

  return(engine)
}

#' Potential function and gradient for HMC
#'
#' Takes HPH engine object and model parameters. Returns
#' potential (proportional to log posterior) function or its gradient. Using
#' lognormal(0,1) prior.
#'
#' @param engine HPH engine object.
#' @param parameters (Exponentiated) spatio-temporal Hawkes process parameters (d=6).
#' @param gradient Return gradient (instead of potential)? Defaults \code{FALSE}.
#' @return Potential or its gradient.
#'
#' @export
Potential <- function(engine,parameters,gradient=FALSE) {
    # HMC potential (log posterior) and gradient
    if (gradient) {
    logPriorGrad <- -log(parameters)
    logLikelihoodGrad <- hpHawkes::getGradient(engine) * parameters

    return(-logLikelihoodGrad-logPriorGrad)
  }
  else {
    logPrior <- sum(dlnorm(parameters,log=TRUE))
    logLikelihood <- hpHawkes::getLogLikelihood(engine)

    return(-logLikelihood-logPrior)
  }
}

#' HMC for Bayesian inference of Hawkes model parameters
#'
#' Takes a number of settings and returns posterior samples of parameters
#' for spatio-temporal Hawkes process model.
#'
#' @param n_iter Number of MCMC iterations.
#' @param burnIn Number of initial samples to throw away.
#' @param locDists Locations distance matrix.
#' @param times Observation times.
#' @param params Length 6, default 1.
#' @param latentDimension Dimension of latent space. Integer ranging from 2 to 8.
#' @param trajectory HMC trajectory length.
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @return List containing posterior samples, negative log likelihood values (\code{target}) and time to compute (\code{Time}).
#'
#' @importFrom RcppXsimd supportsSSE supportsAVX supportsAVX512
#'
#' @export
hmcsampler <- function(n_iter,
                       burnIn=0,
                       locDists=NULL,
                       times=NULL,
                       params=rep(1,6),
                       latentDimension=2,
                       trajectory = 0.05,              # length of HMC proposal trajectory
                       threads=1,                     # number of CPU cores
                       simd=0,                        # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
                       gpu=0,
                       single=0) {

  # Check availability of SIMD  TODO Move into hidden function
  if (simd > 0) {
    if (simd == 1 && !RcppXsimd::supportsSSE()) {
      stop("CPU does not support SSE")
    } else if (simd == 2 && !RcppXsimd::supportsAVX()) {
      stop("CPU does not support AVX")
    } else if (simd == 3 && !RcppXsimd::supportsAVX512()) {
      stop("CPU does not support AVX512")
    }
  }

  set.seed(666)

  # Set up the parameters
  NumOfIterations = n_iter
  NumOfLeapfrog = 20
  StepSize = trajectory/NumOfLeapfrog

  # Allocate output space
  ParametersSaved = matrix(0,6,NumOfIterations)
  Target = vector()
  savedLikEvals <- rep(0,n_iter)
  P <- latentDimension

  if(is.null(locDists)){
      stop("No locDists found.")
  }
  if(is.null(times)){
    stop("No times found.")
  }
  N <- dim(locDists)[1]


  # Build reusable object to compute Loglikelihood (gradient)
  engine <- engineInitial(locDists,N,P,times,params,threads,simd,gpu,single)

  Accepted = 0;
  Proposed = 0;
  likEvals = 0;

  # Initialize the location
  CurrentParams = params;

  CurrentU = Potential(engine,params)

  cat(paste0('Initial log-likelihood: ', hpHawkes::getLogLikelihood(engine), '\n'))

  # track number of likelihood evaluations
  likEvals = likEvals + 1;

  # Perform Hamiltonian Monte Carlo
  for (Iteration in 1:NumOfIterations) {

    ProposedParams = CurrentParams
    engine <- hpHawkes::setParameters(engine, ProposedParams)

    # Sample the marginal momentum
    CurrentMomentum = rnorm(6)
    ProposedMomentum = CurrentMomentum

    Proposed = Proposed + 1

    # Simulate the Hamiltonian Dynamics
    for (StepNum in 1:NumOfLeapfrog) {
      ProposedMomentum = ProposedMomentum - StepSize/2 * Potential(engine,ProposedParams, gradient=T)
      likEvals = likEvals + 1;

      ProposedParams = exp(log(ProposedParams) + StepSize * ProposedMomentum)
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedMomentum = ProposedMomentum - StepSize/2 * Potential(engine,ProposedParams, gradient=T)
      likEvals = likEvals + 1;
    }

    ProposedMomentum = - ProposedMomentum

    # Compute the Potential
    ProposedU = Potential(engine,ProposedParams)
    likEvals = likEvals + 1;

    # Compute the Hamiltonian
    CurrentH = CurrentU + sum(CurrentMomentum^2)/2
    ProposedH = ProposedU + sum(ProposedMomentum^2)/2

    # Accept according to ratio
    Ratio = - ProposedH + CurrentH
    if (Ratio > min(0,log(runif(1)))) {
      CurrentParams = ProposedParams
      CurrentU = ProposedU
      Accepted = Accepted + 1
    }

    # Save if sample is required
    if (Iteration > burnIn) {
      ParametersSaved[,Iteration] = CurrentParams
      Target[Iteration-burnIn] = CurrentU
      savedLikEvals[Iteration-burnIn] = likEvals
    }

    # Show acceptance rate every 20 iterations
    if (Iteration %% 20 == 0) {
      cat(Iteration, "iterations completed. HMC acceptance rate: ",Accepted/Proposed,"\n")

      Proposed = 0
      Accepted = 0
    }

    # Start timer after burn-in
    if (Iteration == burnIn) { # If burnIn > 0
      cat("burn-in complete, now drawing samples ...\n")
      timer = proc.time()
    }
    if (burnIn==0 & Iteration==1) { # If burnIn = 0
      cat("burn-in complete, now drawing samples ...\n")
      timer = proc.time()
    }
  }   # end iterations for loop

  time = proc.time() - timer

  # only HMC, return ...
  return(list(samples = ParametersSaved, target = Target, Time = time,
              likEvals = savedLikEvals))

   # end iterations for loop
} # end HMC function
