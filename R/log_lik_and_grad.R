
bg_rate <- function (x,t,params,obs_x,obs_t) {
  # takes the following inputs and returns the background rate

  # x is position vector
  # t is time
  # params is list of model parameters
  # obs_x is nxd matrix of observed positions
  # obs_t is n vector of observed times
  mu_0 <- params$mu_0

  tau_x    <- params$tau_x
  dim_x    <- length(x)
  kerns_x  <- dmvnorm(x=obs_x, mean=x, sigma=diag(dim_x)*(tau_x^2))

  tau_t    <- params$tau_t
  kerns_t  <- dnorm(x=obs_t, mean=t, sd=tau_t)
  kerns_t[obs_t==t] <- 0

  mu_xt <- sum( kerns_x*kerns_t )

  rate <- mu_0*mu_xt
  return(rate)
}

se_rate <- function (x,t,params,obs_x,obs_t,randomRates) {
  # takes the following inputs and returns the self-excitation rate

  # x is position vector
  # t is time
  # params is list of model parameters
  # obs_x is nxd matrix of observed positions
  # obs_t is n vector of observed times

  less_than_t <- obs_t < t

  theta <- params$theta

  h <- params$h
  dim_x   <- length(x)
  kerns_x <- dmvnorm(x=obs_x, mean=x, sigma=diag(dim_x)*(h^2))

  omega   <- params$omega
  kerns_t <- exp( -omega*(t-obs_t) ) * omega * randomRates

  lam_xt <- sum( less_than_t*kerns_x*kerns_t )

  rate <- theta*lam_xt
  return(rate)
}

prob_se <- function(x,t,params,obs_x,obs_t,randomRates) {
  SeRate <- se_rate(x,t,params,obs_x,obs_t,randomRates)
  out <- SeRate / (bg_rate(x,t,params,obs_x,obs_t) + SeRate)
  return(out)
}

rate <- function (x,t,params,obs_x,obs_t,randomRates) {
  return( bg_rate(x,t,params,obs_x,obs_t) +
            se_rate(x,t,params,obs_x,obs_t,randomRates) )
}

integral <- function (params,obs_x,obs_t, randomRates) {
  # returns integral component of log likelihood

  # background integral
  n             <- length(obs_t)
  tau_t         <- params$tau_t
  mu_0          <- params$mu_0
  contributions <- pnorm( (obs_t[n]-obs_t)/tau_t ) - pnorm(-obs_t/tau_t)
  bg_int        <- mu_0*sum(contributions)

  # self-excitatory integral
  theta  <- params$theta
  omega  <- params$omega
  se_int <- - theta * sum( randomRates * (exp(-omega*(obs_t[n]-obs_t))-1) )

  return( bg_int+se_int )
}

log_lik <- function (params,obs_x,obs_t,randomRates) {
  n       <- length(obs_t)

  non_int <- 0
  for (i in 1:n) {
    non_int <- non_int + log( rate(x=obs_x[i,],
                                   t=obs_t[i],
                                   params=params,
                                   obs_x=obs_x,
                                   obs_t=obs_t,
                                   randomRates=randomRates) )
  }

  integ <- integral(params, obs_x, obs_t, randomRates)
  return( non_int-integ )
}
