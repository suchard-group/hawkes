

#' Serially computed Hawkes process log likelihood and gradient
#'
#' Called by \code{hpHawkes::test()} to compare output with parallel implementations.
#' Slow. Not recommended for use.
#'
#' @param locations Matrix of spatial locations (nxp).
#' @param times    Vector of times (length=n).
#' @param randomRates Vector of random rates (length=n).
#' @param parameters Hawkes process parameters (length=6).
#' @param gradient Return gradient (or log likelihood)? Defaults to FALSE.
#' @return Hawkes process log likelihood or its gradient.
#'
#' @export
computeLoglikelihood <- function(locations, times, randomRates, parameters, gradient = FALSE) {

  wrap_func <- function (x){ # takes vector x
    return(log_lik(params=parameters,
                   obs_x=matrix(x,nrow=dim(locations)[1],ncol=dim(locations)[2]),
                   obs_t=times, randomRates=randomRates))
  }

  if (gradient) {
    gradLogLikelihood <- numDeriv::grad(wrap_func,x=as.vector(locations))
    gradLogLikelihood <- matrix(gradLogLikelihood,nrow=dim(locations)[1],ncol=dim(locations)[2])
    return(gradLogLikelihood)
  } else {
    logLikelihood <- log_lik(params=parameters,obs_x=locations,
                             obs_t=times, randomRates=randomRates)
    return(logLikelihood)
  }
}

#' Get event specific probabilities self-excitatory
#'
#' Takes locations, times and parameters and returns the probability
#' that each event is a child (self-excitatory) as opposed to a parent (background.
#'
#' @param locations Matrix of spatial locations (nxp).
#' @param times    Vector of times.
#' @param params Hawkes process parameters (length=6).
#' @param threads Number of CPU cores to be used.
#' @param simd For CPU implementation: no SIMD (\code{0}), SSE (\code{1}) or AVX (\code{2}).
#' @param gpu Which GPU to use? If only 1 available, use \code{gpu=1}. Defaults to \code{0}, no GPU.
#' @param single Set \code{single=1} if your GPU does not accommodate doubles.
#' @param naive Just use naive R implementation (very very slow).
#' @param dimension Dimension of space
#' @return n vector of probabilities.
#'
#' @export
probability_se <- function(locations, times, params,
                           threads=0, simd=0, gpu=0, single=0,
                           naive=FALSE, dimension) {

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
    embeddingDimension <- dimension
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
  randomRates <- rexp(locationCount)

  params <- rexp(6)

  engine <- hpHawkes::createEngine(embeddingDimension, locationCount, threads, simd, gpu,single)
  engine <- hpHawkes::updateLocations(engine, locations)
  engine <- hpHawkes::setRandomRates(engine, randomRates)
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
                             parameters=params2,
                             randomRates = randomRates))

 cat("grads\n")
 hphGrad <- hpHawkes::getGradient(engine)
 print(hphGrad)
 naiveGrad <- computeLoglikelihood(locations=locations,
                                   times=times,
                                   parameters=params2,gradient = TRUE,
                                   randomRates = randomRates)
 print(naiveGrad)
 cat("max absolute difference between gradients:\n",max(abs(hphGrad-naiveGrad)))
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
#' @param r Should we use naive R implementation? (Default FALSE)
#' @return User, system and elapsed time. Elapsed time is most important.
#'
#' @export
timeTest <- function(locationCount=5000, maxIts=1, threads=0, simd=0,gpu=0,single=0, r=FALSE) {
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

  if(r) {
    ptm <- proc.time()
    for(i in 1:maxIts){
      computeLoglikelihood(locations = locations,times=times,parameters=params2)
      #hpHawkes::getGradient(engine)
    }
    cat(proc.time() - ptm)
    return(proc.time() - ptm)
  } else {
    ptm <- proc.time()
    for(i in 1:maxIts){
      hpHawkes::getLogLikelihood(engine)
      #hpHawkes::getGradient(engine)
    }
    cat(proc.time() - ptm)
    return(proc.time() - ptm)
  }
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
#' @return Potential or its gradient.
#'
#' @export
Potential <- function(engine,parameters) {
  logPrior <- log(truncnorm::dtruncnorm(x=parameters[1],sd=10,a=0)) +
    log(truncnorm::dtruncnorm(x=parameters[2],a=0)) +
    log(truncnorm::dtruncnorm(x=parameters[4],sd=10,a=0)) +
    log(truncnorm::dtruncnorm(x=parameters[3],a=0)) +
  # logPrior <- log(truncnorm::dtruncnorm(x=parameters[1],sd=10,a=parameters[2])) +
  #             log(truncnorm::dtruncnorm(x=parameters[2],a=0,b=parameters[1])) +
  #             log(truncnorm::dtruncnorm(x=parameters[4],sd=10,a=parameters[3])) +
  #             log(truncnorm::dtruncnorm(x=parameters[3],a=0,b=parameters[4])) +
              log(truncnorm::dtruncnorm(x=parameters[5],sd=1,a=0)) +
              log(truncnorm::dtruncnorm(x=parameters[6],sd=1,a=0))
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
#' @param thinPeriod Collect once every thinPeriod samples (default=1).
#' @param sampleLocations Do we want to sample locations? (Default FALSE).
#' @param windowWidth Width of uncertainty window for locations.
#' @param radius Standard deviations of proposal distributions.
#' @param params Length 6, default 1.
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
                       thinPeriod=1,
                       sampleLocations=FALSE,
                       windowWidth=NULL,
                       radius = 2,                 # radius for uniform proposals
                       params=c(1,1,1,1,1,1),
                       latentDimension=2,
                       threads=1,                     # number of CPU cores
                       simd=0,                        # simd = 0, 1, 2 for no simd, SSE, and AVX, respectively
                       gpu=0,
                       single=0) {

  if (sampleLocations==TRUE) {
    if (is.null(windowWidth)) stop("Select window width.")
    B <- 100 # number of locations to update at a time
    locUp <- locations + windowWidth/2
    locLow <- locations - windowWidth/2
  }

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

  if(params[2]>=params[1]) params[2] <- params[1]/2
  if(params[3]>=params[4]) params[3] <- params[4]/2

  # Set up the parameters
  NumOfIterations = n_iter

  # Allocate output space
  ParametersSaved = vector()
  if (sampleLocations) LocationsSaved = list()
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
  Proposed = 0;
  if(sampleLocations){
    Acceptances = rep(0,7) # total acceptances within adaptation run (<= SampBound)
    SampBound = rep(5,7)   # current total samples before adapting radius
    SampCount = rep(0,7)   # number of samples collected (adapt when = SampBound)
    Radii  = rep(radius,7)
  }else{
    Acceptances = rep(0,6) # total acceptances within adaptation run (<= SampBound)
    SampBound = rep(5,6)   # current total samples before adapting radius
    SampCount = rep(0,6)   # number of samples collected (adapt when = SampBound)
    Radii  = rep(radius,6)
  }

  # Initialize locations and parameters
  if (sampleLocations) CurrentLocations = locations
  CurrentParams = as.vector(params);

  CurrentU = Potential(engine,params)

  cat(paste0('Initial log-likelihood: ', hpHawkes::getLogLikelihood(engine), '\n'))

  # Perform Adaptive M-H
  for (Iteration in 1:NumOfIterations) {

    ProposedParams = CurrentParams
    if (sampleLocations) ProposedLocations = CurrentLocations
    engine <- hpHawkes::setParameters(engine, ProposedParams)
    if (sampleLocations) engine <- hpHawkes::updateLocations(engine,ProposedLocations)

    Proposed = Proposed + 1

    #propose new parameters with UNIVARIATE M-H proposal
    if(sampleLocations){
      index <- sample(1:7,size = 1,prob = c(rep(1/12,6),0.5)) # random scan of 1:6 parameters
    } else {
      index <- sample(1:6,size = 1)
    }

    if (index < 7) {
      Former <- ProposedParams[index]
    }

    if (index==1) {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=ProposedParams[2],
                                                     mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=ProposedParams[2],
                                   mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=ProposedParams[2],
                                   mean=ProposedParams[index], sd=Radii[index]) )

    } else if (index==2) {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=0,b=ProposedParams[1],
                                                     mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=0,b=ProposedParams[1],
                                   mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=0,b=ProposedParams[1],
                                   mean=ProposedParams[index], sd=Radii[index]) )

    } else if (index==3) {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=0,b=ProposedParams[4],
                                                     mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=0,b=ProposedParams[4],
                                   mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=0,b=ProposedParams[4],
                                   mean=ProposedParams[index], sd=Radii[index]) )

    } else if (index==4) {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=ProposedParams[3],
                                                     mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=ProposedParams[3],
                                   mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=ProposedParams[3],
                                   mean=ProposedParams[index], sd=Radii[index]) )

    } else if (index == 5 | index == 6) {

      ProposedParams[index] <- truncnorm::rtruncnorm(1,a=0,mean=ProposedParams[index],sd=Radii[index])

      # get proposed log post
      engine <- hpHawkes::setParameters(engine, ProposedParams)
      ProposedU = Potential(engine,ProposedParams)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        log( truncnorm::dtruncnorm(x=ProposedParams[index], a=0, mean=Former, sd=Radii[index]) )
      ProposedH = ProposedU -
        log( truncnorm::dtruncnorm(x=Former, a=0, mean=ProposedParams[index], sd=Radii[index]) )

    } else {
      indices <- sample(1:N,size=B)
      Former <- as.vector(ProposedLocations[indices,])
      ProposedLocations[indices,] <- truncnorm::rtruncnorm(B*latentDimension,
                                                 a=as.vector(locLow[indices,]),
                                                 b=as.vector(locUp[indices,]),
                                                 mean=Former,
                                                 sd=Radii[index])

      engine <- hpHawkes::updateLocations(engine, ProposedLocations)
      ProposedU = Potential(engine,ProposedParams)

      # Compute the terms of accept/reject step
      CurrentH = CurrentU -
        sum( log( truncnorm::dtruncnorm(x=ProposedLocations[indices,],
                                   a=as.vector(locLow[indices,]),
                                   b=as.vector(locUp[indices,]),
                                   mean=Former,
                                   sd=Radii[index]) ) )
      ProposedH = ProposedU -
        sum( log( truncnorm::dtruncnorm(x=Former,
                                   a=as.vector(locLow[indices,]),
                                   b=as.vector(locUp[indices,]),
                                   mean=ProposedLocations[indices,],
                                   sd=Radii[index]) ) )
    }

    Ratio = - ProposedH + CurrentH
    if (Ratio > min(0,log(runif(1)))) {
      CurrentParams = ProposedParams
      if (sampleLocations) CurrentLocations = ProposedLocations
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
    if (Iteration > burnIn & (Iteration-burnIn) %% thinPeriod == 0) {
      ParametersSaved = cbind(ParametersSaved, CurrentParams)
      if(sampleLocations){
        LocationsSaved[[length(LocationsSaved)+1]] = CurrentLocations
      }
      Target[Iteration-burnIn] = CurrentU

      cat(CurrentParams, "\n",file = "output/params.txt",append = TRUE)
      if (sampleLocations) cat(as.vector(t(CurrentLocations)), "\n",file = "output/locs.txt",append = TRUE)
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
  if (sampleLocations) {
    return(list(samples = ParametersSaved, locations=LocationsSaved, target = Target, Time = time))
  } else {
    return(list(samples = ParametersSaved, target = Target, Time = time))
  }

  # end iterations for loop
} # end M-H function

