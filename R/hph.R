
#' Helper Hawkes process log likelihood function
#'
#' Takes HPH engine object and returns log likelihood.
#'
#' @param engine An HPH engine object.
#' @return Hawkes process log likelihood
#'
#' @export
getLogLikelihood <- function(engine) {

  if (!engine$locationsInitialized) {
    stop("locations not set")
  }

  if (!engine$timesInitialized) {
    stop("times not set")
  }

  if (!engine$randomRatesInitialized) {
    stop("random rates not set")
  }

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  sumOfLikContribs <- .getSumOfLikContribs(engine$engine)
  #observationCount <- (engine$locationCount * (engine$locationCount - 1)) / 2;

  logLikelihood <- sumOfLikContribs #0.5 * (log(engine$precision) - log(2 * pi)) * observationCount -
    #sumOfIncrements

  return(logLikelihood)
}

#' Helper Hawkes process probability self-excitatory function
#'
#' Takes HPH engine object and returns log likelihood.
#'
#' @param engine An HPH engine object.
#' @return vector of probabilities each event is self-excitatory in origin
#'
#' @export
getProbsSelfExcite <- function(engine) {

  if (!engine$locationsInitialized) {
    stop("locations not set")
  }

  if (!engine$timesInitialized) {
    stop("times not set")
  }

  if (!engine$randomRatesInitialized) {
    stop("random rates not set")
  }

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  probsSE <- .getProbsSelfExcite(engine$engine,engine$locationCount)
  return(probsSE)
}

#' Helper Hawkes process random rates gradient
#'
#' Takes HPH engine object and returns gradient of log likelihood wrt random rates.
#'
#' @param engine An HPH engine object.
#' @return vector of log likelihood gradient wrt random rates.
#'
#' @export
getRandomRatesGradient <- function(engine) {

  if (!engine$locationsInitialized) {
    stop("locations not set")
  }

  if (!engine$timesInitialized) {
    stop("times not set")
  }

  if (!engine$randomRatesInitialized) {
    stop("random rates not set")
  }

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  randomRatesGradient <- .getRandomRatesLogLikelihoodGradient(engine$engine,engine$locationCount)
  return(randomRatesGradient)
}

#' Helper Hawkes process random rates Hessian
#'
#' Takes HPH engine object and returns Hessian of log likelihood wrt random rates.
#'
#' @param engine An HPH engine object.
#' @return vector of log likelihood hessian wrt random rates.
#'
#' @export
getRandomRatesHessian <- function(engine) {

  if (!engine$locationsInitialized) {
    stop("locations not set")
  }

  if (!engine$timesInitialized) {
    stop("times not set")
  }

  if (!engine$randomRatesInitialized) {
    stop("random rates not set")
  }

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  randomRatesHessian <- .getRandomRatesLogLikelihoodHessian(engine$engine,engine$locationCount)
  return(randomRatesHessian)
}

#' Helper HPH log likelihood gradient function
#'
#' Takes HPH engine object and returns log likelihood gradient.
#'
#' @param engine An HPH engine object.
#' @return HPH log likelihood gradient.
#'
#' @export
getGradient <- function(engine) {

  if (!engine$locationsInitialized) {
    stop("locations not set")
  }

  if (!engine$timesInitialized) {
    stop("times not set")
  }

  if (!engine$randomRatesInitialized) {
    stop("random rates not set")
  }

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  matrix(.getLogLikelihoodGradient(engine$engine, engine$locationCount * engine$embeddingDimension),
         nrow = engine$locationCount, byrow = TRUE)
}

#' Deliver parameters to HPH engine object
#'
#' Helper function delivers Hawkes process likelihood parameters to HPH engine object.
#'
#' @param engine HPH engine object.
#' @param  parameters of Hawkes process likelihood
#' @return HPH engine object.
#'
#' @export
setParameters <- function(engine, parameters) {
  .setParameters(engine$engine, parameters)
  engine$parameters <- parameters
  return(engine)
}

#' Deliver times vector to HPH engine object
#'
#' Helper function delivers times vector to HPH engine object.
#'
#' @param engine HPH engine object.
#' @param data Times vector.
#' @return HPH engine object.
#'
#' @export
setTimesData <- function(engine, data) {
  data <- as.vector(data)
  if (length(data) != engine$locationCount) {
    stop("Invalid data size")
  }
  .setTimesData(engine$engine, data)
  engine$timesInitialized <- TRUE
  return(engine)
}

#' Deliver random rates vector to HPH engine object
#'
#' Helper function delivers random rates vector to HPH engine object.
#'
#' @param engine HPH engine object.
#' @param data Random rates vector.
#' @return HPH engine object.
#'
#' @export
setRandomRates <- function(engine, data) {
  data <- as.vector(data)
  if (length(data) != engine$locationCount) {
    stop("Invalid data size")
  }
  if (any(data<=0)) {
    stop("Random rates must be positive.")
  }
  .setRandomRates(engine$engine, data)
  engine$randomRatesInitialized <- TRUE
  return(engine)
}

#' Deliver latent locations matrix to MDS engine object
#'
#' Helper function delivers latent locations matrix to MDS engine object.
#'
#' @param engine MDS engine object.
#' @param locations N by P matrix of N P-dimensional latent locations.
#' @return MDS engine object.
#'
#' @export
updateLocations <- function(engine, locations) {
  locations <- as.vector(t(locations)) # C++ code assumes row-major
  if (length(locations) != engine$locationCount * engine$embeddingDimension) {
    stop("Invalid data size")
  }
  .updateLocations(engine$engine, locations)
  engine$locationsInitialized <- TRUE
  return(engine)
}
