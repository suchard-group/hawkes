
se_rate <- function (x,t,params,obs_x,obs_t) {
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
  kerns_t <- exp( -omega*(t-obs_t) ) * omega

  lam_xt <- sum( less_than_t*kerns_x*kerns_t )

  rate <- theta*lam_xt
  return(rate)
}

prob_se <- function(x,t,params,obs_x,obs_t,backgroundRate) {
  SeRate <- se_rate(x,t,params,obs_x,obs_t)
  out <- SeRate / (params$mu_0*backgroundRate + SeRate)
  return(out)
}

integral <- function (params,obs_x,obs_t) {
  # returns integral component of log likelihood

  # self-excitatory integral
  n      <- length(obs_t)
  theta  <- params$theta
  omega  <- params$omega
  se_int <- - theta * sum( exp(-omega*(obs_t[n]-obs_t))-1 )

  return( params$mu_0+se_int )
}

log_lik <- function (params,obs_x,obs_t,backgroundRates) {
  n       <- length(obs_t)

  non_int <- 0
  for (i in 1:n) {
    non_int <- non_int + log( se_rate(x=obs_x[i,],
                                   t=obs_t[i],
                                   params=params,
                                   obs_x=obs_x,
                                   obs_t=obs_t) +
                              params$mu_0*backgroundRates[i] )
  }

  integ <- integral(params, obs_x, obs_t)
  return( non_int-integ )
}

