
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

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  sumOfLikContribs <- .getSumOfLikContribs(engine$engine)
  #observationCount <- (engine$locationCount * (engine$locationCount - 1)) / 2;

  logLikelihood <- sumOfLikContribs #0.5 * (log(engine$precision) - log(2 * pi)) * observationCount -
    #sumOfIncrements

  return(logLikelihood)
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

  if (is.null(engine$parameters)) {
    stop("parameters not set")
  }

  .getLogLikelihoodGradient(engine$engine, 6)
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
