

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

#' Get event specific probabilities self-excitatory
#'
#' Takes locations, times and parameters and returns the probability
#' that each event is a parent (backround) or a child (self-excitatory).
#'
#' @param locations Matrix of spatial locations (nxp).
#' @param times    Vector of times.
#' @param params Hawkes process parameters (length=6).
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @param naive Just use naive R implementation (very very slow).
#' @return n vector of probabilities.
#'
#' @export
probability_se <- function(locations, times, params,
                           threads=0, simd=0, gpu=0, single=0,
                           naive=FALSE) {

  if (naive) {

    params2 <- list()
    params2$h     <- 1/params[1]
    params2$tau_x <- 1/params[2]
    params2$tau_t <- 1/params[3]
    params2$omega <- params[4]
    params2$theta <- params[5]
    params2$mu_0  <- params[6]
    n <- dim(locations)[1]
    output <- rep(0,n)
    for (i in 1:n) {
      output[i] <- prob_se(x=locations[i,],
                           t=times[i],
                           params=params2,
                           obs_x=locations, obs_t=times)
    }
    return(output)

  } else {
    embeddingDimension <- 2
    locationCount <- dim(locations)[1]
    engine <- hpHawkes::createEngine(embeddingDimension, locationCount, threads, simd, gpu,single)
    engine <- hpHawkes::updateLocations(engine, locations)
    engine <- hpHawkes::setTimesData(engine, times)
    engine <- hpHawkes::setParameters(engine, params)
    return( hpHawkes::getProbsSelfExcite(engine) )

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

  set.seed(666)
  embeddingDimension <- 2

  locations <- matrix(rnorm(n = locationCount * embeddingDimension),
                 ncol = embeddingDimension, nrow = locationCount)
  times <- seq(from=1, to=locationCount, by=1)

  params <- rexp(6)

  engine <- hpHawkes::createEngine(embeddingDimension, locationCount, threads, simd, gpu,single)
  engine <- hpHawkes::updateLocations(engine, locations)
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


#' Time log likelihood calculations
#'
#' Time log likelihood calculations for serial and parallel implementations.
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
  # function returns length of time to compute log likelihood
  # threads is number of CPU cores used
  # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
  embeddingDimension <- 2

  locations <- matrix(rnorm(n = locationCount * embeddingDimension),
                      ncol = embeddingDimension, nrow = locationCount)
  times <- seq(from=1, to=locationCount, by=1)

  params <- rexp(6)

  engine <- hpHawkes::createEngine(embeddingDimension, locationCount, threads, simd, gpu,single)
  engine <- hpHawkes::updateLocations(engine, locations)
  engine <- hpHawkes::setTimesData(engine, times)
  engine <- hpHawkes::setParameters(engine, params)
  params2 <- list()
  params2$h     <- 1/params[1]
  params2$tau_x <- 1/params[2]
  params2$tau_t <- 1/params[3]
  params2$omega <- params[4]
  params2$theta <- params[5]
  params2$mu_0  <- params[6]

  ptm <- proc.time()
  for(i in 1:maxIts){
    hpHawkes::getLogLikelihood(engine)
    #hpHawkes::getGradient(engine)
  }
  proc.time() - ptm
}


#' Initialize HPH engine object
#'
#' Takes data, parameters and implementation details and returns HPH engine object. Used within
#' \code{hpHawkes::hmcsampler()}.
#'
#' @param locations N by P locations matrix.
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
engineInitial <- function(locations,N,P,times,parameters=c(1,6),
                          threads,simd,gpu,single) {

  # Build reusable object
  engine <- hpHawkes::createEngine(embeddingDimension = P,
                                   locationCount = N,
                                   tbb = threads, simd=simd,
                                   gpu=gpu, single=single)

  # Set locations data
  engine <- hpHawkes::updateLocations(engine, locations)

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
#' truncated normal priors.
#'
#' @param engine HPH engine object.
#' @param parameters (Exponentiated) spatio-temporal Hawkes process parameters (d=6).
#' @param lbs Lower bounds on spatial and temporal lengthscales.
#' @param ubs Upper bounds on spatial and temporal lengthscales.
#' @return Potential or its gradient.
#'
#' @export
Potential <- function(engine,parameters,lbs,ubs) {
  logPrior <- log(truncnorm::dtruncnorm(x=parameters[1],sd=10,a=1/ubs[1],b=1/lbs[1])) +
              log(truncnorm::dtruncnorm(x=parameters[2],a=1/ubs[2],b=1/lbs[2])) +
              log(truncnorm::dtruncnorm(x=parameters[3],a=1/ubs[3],b=1/lbs[3])) +
              log(truncnorm::dtruncnorm(x=parameters[4],sd=10,a=1/ubs[4],b=1/lbs[4])) +
              log(truncnorm::dtruncnorm(x=parameters[5],a=0)) +
              log(truncnorm::dtruncnorm(x=parameters[6],a=0))
  logLikelihood <- hpHawkes::getLogLikelihood(engine)

  return(-logLikelihood-logPrior)
}

#' M-H for Bayesian inference of Hawkes model parameters
#'
#' Takes a number of settings and returns posterior samples of parameters
#' for spatio-temporal Hawkes process model.
#'
#' @param n_iter Number of MCMC iterations.
#' @param burnIn Number of initial samples to throw away.
#' @param locations N x P locations matrix.
#' @param times Observation times.
#' @param radius Standard deviations of proposal distributions.
#' @param params Initial values of all parameters (lengthscales self-excitatory spatial, background spatial, background temporal and self-excitatory temporal and self-excitatory and background rates, in order), default 1.
#' @param upperbounds Vector of upper bounds on lengthscales self-excitatory spatial, background spatial, background temporal and self-excitatory temporal, in order.
#' @param lowerbounds Vector of lower bounds on lengthscales self-excitatory spatial, background spatial, background temporal and self-excitatory temporal, in order.
#' @param latentDimension Dimension of latent space. Integer ranging from 2 to 8.
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @return List containing posterior samples, negative log likelihood values (\code{target}) and time to compute (\code{Time}).
#'
#' @importFrom RcppXsimd supportsSSE supportsAVX supportsAVX512
#'
#' @export
sampler <- function(n_iter,
                       burnIn=0,
                       locations=NULL,
                       times=NULL,
                       radius = 2,                 # radius for uniform proposals
                       params=c(1,1,1,1,1,1),
                       upperbounds=c(Inf,Inf,Inf,Inf),
                       lowerbounds=c(0,0,0,0),
                       latentDimension=2,
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

  # enforce constraints on starting values
  if(params[1]>1/lowerbounds[1]) params[1] <- (1/lowerbounds[1]+1/upperbounds[1])/2
  if(params[2]>1/lowerbounds[2]) params[2] <- (1/lowerbounds[2]+1/upperbounds[2])/2
  if(params[3]>1/lowerbounds[3]) params[3] <- (1/lowerbounds[3]+1/upperbounds[3])/2
  if(params[4]>1/lowerbounds[4]) params[4] <- (1/lowerbounds[4]+1/upperbounds[4])/2

  if(params[1]<1/upperbounds[1]) params[1] <- (1/lowerbounds[1]+1/upperbounds[1])/2
  if(params[2]<1/upperbounds[2]) params[2] <- (1/lowerbounds[2]+1/upperbounds[2])/2
  if(params[3]<1/upperbounds[3]) params[3] <- (1/lowerbounds[3]+1/upperbounds[3])/2
  if(params[4]<1/upperbounds[4]) params[4] <- (1/lowerbounds[4]+1/upperbounds[4])/2

  # Set up the parameters
  NumOfIterations = n_iter

  # Allocate output space
  ParametersSaved = matrix(0,6,NumOfIterations-burnIn)
  Target = vector()
  savedLikEvals <- rep(0,n_iter)
  P <- latentDimension

  if(is.null(locations)){
    stop("No locations found.")
  }
  if(is.null(times)){
    stop("No times found.")
  }
  N <- dim(locations)[1]

  # Build reusable object to compute Loglikelihood (gradient)
  engine <- engineInitial(locations,N,P,times,params,threads,simd,gpu,single)

  engine <- hpHawkes::setParameters(engine,params)

  Accepted = 0;
  Acceptances = rep(0,6) # total acceptances within adaptation run (<= SampBound)
  SampBound = rep(5,6)   # current total samples before adapting radius
  SampCount = rep(0,6)   # number of samples collected (adapt when = SampBound)
  Radii  = rep(radius,6)
  Proposed = 0;

  # Initialize the location
  CurrentParams = as.vector(params);

  CurrentU = Potential(engine,params,lowerbounds,upperbounds)

  cat(paste0('Initial log-likelihood: ', hpHawkes::getLogLikelihood(engine), '\n'))

  # Perform Adaptive M-H
  for (Iteration in 1:NumOfIterations) {

    ProposedParams = CurrentParams
    engine <- hpHawkes::setParameters(engine, ProposedParams)

    Proposed = Proposed + 1

    #propose new parameters with UNIVARIATE M-H proposal
    index <- sample(1:6,size = 1) # random scan of 1:6 parameters

    Former <- ProposedParams[index]

    if (index<5) {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=1/upperbounds[index],b=1/lowerbounds[index],
                                                     mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams,lowerbounds,upperbounds)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=1/upperbounds[index],b=1/lowerbounds[index],
                                   mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=1/upperbounds[index],b=1/lowerbounds[index],
                                   mean=ProposedParams[index], sd=Radii[index]) )

    } else {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=0,mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams,lowerbounds,upperbounds)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=0, mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=0, mean=ProposedParams[index], sd=Radii[index]) )

    }

    Ratio = - ProposedH + CurrentH
    if (Ratio > min(0,log(runif(1)))) {
      CurrentParams = ProposedParams
      CurrentU = ProposedU
      Accepted = Accepted + 1
      Acceptances[index] = Acceptances[index] + 1
    }
    SampCount[index] <- SampCount[index] + 1

    if (SampCount[index] == SampBound[index]) {

      AcceptRatio <- Acceptances[index] / SampBound[index]
      AdaptRatio  <- AcceptRatio / 0.44
      if (AdaptRatio>2) AdaptRatio <- 2
      if (AdaptRatio<0.5) AdaptRatio <- 0.5
      Radii[index] <- Radii[index] * AdaptRatio

      SampCount[index] <- 0
      SampBound[index] <- ceiling(SampBound[index]^1.1)
      Acceptances[index] <- 0
    }

    # Save if sample is required
    if (Iteration > burnIn) {
      ParametersSaved[,Iteration-burnIn] = CurrentParams
      Target[Iteration-burnIn] = CurrentU
    }

    # Show acceptance rate every 100 iterations
    if (Iteration %% 100 == 0) {
      cat(Iteration, "iterations completed. Acceptance rate: ",Accepted/Proposed,"\n")
      cat("Radii ",Radii,"\n")

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
  return(list(samples = ParametersSaved, target = Target, Time = time))

} # end M-H function

