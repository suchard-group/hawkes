
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

prob_se <- function(x,t,params,obs_x,obs_t) {
  SeRate <- se_rate(x,t,params,obs_x,obs_t)
  out <- SeRate / (bg_rate(x,t,params,obs_x,obs_t) + SeRate)
  return(out)
}

rate <- function (x,t,params,obs_x,obs_t) {
  return( bg_rate(x,t,params,obs_x,obs_t)+se_rate(x,t,params,obs_x,obs_t) )
}

integral <- function (params,obs_x,obs_t) {
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
  se_int <- - theta * sum( exp(-omega*(obs_t[n]-obs_t))-1 )

  return( bg_int+se_int )
}

log_lik <- function (params,obs_x,obs_t) {
  n       <- length(obs_t)

  non_int <- 0
  for (i in 1:n) {
    non_int <- non_int + log( rate(x=obs_x[i,],
                                   t=obs_t[i],
                                   params=params,
                                   obs_x=obs_x,
                                   obs_t=obs_t) )
  }

  integ <- integral(params, obs_x, obs_t)
  return( non_int-integ )
}

# num_grad <- function (params, obs_x, obs_t, eps = 0.0001) {
#   # params[[param_name]] <- params[[param_name]] + eps
#   # ll_above             <- log_lik(params,obs_x,obs_t)
#   # params[[param_name]] <- params[[param_name]] - 2*eps
#   # ll_below             <- log_lik(params,obs_x,obs_t)
#
#
#
#   grad <- (ll_above-ll_below)/(2*eps)
#   return(grad)
# }

# theta_grad <- function (params,obs_x,obs_t) {
#   omega <- params$omega
#   h     <- params$h
#   n     <- length(obs_t)
#   dim_x <- dim(obs_x)[2]
#
#   grad  <- 0
#   for (i in 1:n) {
#     less_than_t <- obs_t < obs_t[i]
#     kerns_x     <- dmvnorm(x=obs_x, mean=obs_x[i,], sigma=diag(dim_x)*(h^2))
#     kerns_t     <- exp( -omega*(obs_t[i]-obs_t) )
#     lam_xt      <- sum( less_than_t*kerns_x*kerns_t )
#
#     grad <- grad + (exp(-omega*(obs_t[n]-obs_t[i]))-1)/omega +
#       1/rate(x=obs_x[i,],t=obs_t[i],params,obs_x,obs_t) *
#       lam_xt
#   }
#   return(grad)
# }
#
# omega_grad <- function (params,obs_x,obs_t) {
#   theta <- params$theta
#   omega <- params$omega
#   h     <- params$h
#   n     <- length(obs_t)
#   dim_x <- dim(obs_x)[2]
#
#   grad  <- 0
#   for (i in 1:n) {
#     less_than_t <- obs_t < obs_t[i]
#     kerns_x     <- dmvnorm(x=obs_x, mean=obs_x[i,], sigma=diag(dim_x)*(h^2))
#     kerns_t     <- (obs_t[i]-obs_t)*exp( -omega*(obs_t[i]-obs_t) )
#     lam_xt      <- sum( less_than_t*kerns_x*kerns_t )
#
#     grad <- grad + ( 1-(1+omega*(obs_t[n]-obs_t[i]))*
#                        exp(-omega*(obs_t[n]-obs_t[i])) )/(omega^2) -
#       1/rate(x=obs_x[i,],t=obs_t[i],params,obs_x,obs_t) *
#       lam_xt
#   }
#
#   grad <- theta*grad
#   return(grad)
# }
#
# h_grad <- function (params,obs_x,obs_t) {
#   theta <- params$theta
#   omega <- params$omega
#   h     <- params$h
#   n     <- length(obs_t)
#   dim_x <- dim(obs_x)[2]
#
#   grad  <- 0
#   for (i in 1:n) {
#     less_than_t <- obs_t < obs_t[i]
#
#     rep_row <- matrix(obs_x[i,], n, dim_x, byrow=TRUE)
#     squares <- rowSums((rep_row-obs_x)^2)
#
#     kerns_x     <- dmvnorm(x=obs_x, mean=obs_x[i,], sigma=diag(dim_x)*(h^2)) *
#                     (squares/(h^3) - dim_x/h)
#     kerns_t     <- exp( -omega*(obs_t[i]-obs_t) )
#     lam_xt      <- sum( less_than_t*kerns_x*kerns_t )
#
#     grad <- grad + 1/rate(x=obs_x[i,],t=obs_t[i],params,obs_x,obs_t) *
#       lam_xt
#   }
#
#   grad <- theta*grad
#   return(grad)
# }
#
# mu_0_grad <- function (params,obs_x,obs_t) {
#   tau_t <- params$tau_t
#   tau_x <- params$tau_x
#   n     <- length(obs_t)
#   dim_x <- dim(obs_x)[2]
#
#   grad <- 0
#   for (i in 1:n) {
#     kerns_x <- dmvnorm(x=obs_x, mean=obs_x[i,], sigma=diag(dim_x)*(tau_x^2))
#     kerns_t <- dnorm(x=obs_t, mean=obs_t[i], sd=tau_t)
#     mu_xt  <- sum( kerns_x*kerns_t )/rate(x=obs_x[i,],t=obs_t[i],params,obs_x,obs_t)
#
#     grad <- grad + mu_xt
#   }
#
#   contributions <- pnorm( (obs_t[n]-obs_t)/tau_t ) - pnorm(-obs_t/tau_t)
#   grad <- grad - sum(contributions)
#   return(grad)
# }
#
# tau_x_grad <- function (params,obs_x,obs_t) {
#   mu_0 <- params$mu_0
#   tau_t <- params$tau_t
#   tau_x <- params$tau_x
#
#   n     <- length(obs_t)
#   dim_x <- dim(obs_x)[2]
#
#   grad  <- 0
#   for (i in 1:n) {
#     rep_row <- matrix(obs_x[i,], n, dim_x, byrow=TRUE)
#     squares <- rowSums((rep_row-obs_x)^2)
#
#     kerns_x     <- dmvnorm(x=obs_x, mean=obs_x[i,], sigma=diag(dim_x)*(tau_x^2)) *
#       (squares/(tau_x^3) - dim_x/tau_x)
#     kerns_t     <- dnorm(x=obs_t, mean=obs_t[i], sd=tau_t)
#     lam_xt      <- sum( kerns_x*kerns_t )
#
#     grad <- grad + mu_0/rate(x=obs_x[i,],t=obs_t[i],params,obs_x,obs_t) *
#       lam_xt
#   }
#
#   return(grad)
# }
#
# tau_t_grad <- function (params,obs_x,obs_t) {
#   mu_0 <- params$mu_0
#   tau_t <- params$tau_t
#   tau_x <- params$tau_x
#
#   n     <- length(obs_t)
#   dim_x <- dim(obs_x)[2]
#
#   grad  <- 0
#   for (i in 1:n) {
#     squares <- (rep(obs_t[i],n) - obs_t)^2
#
#     kerns_x     <- dmvnorm(x=obs_x, mean=obs_x[i,], sigma=diag(dim_x)*(tau_x^2))
#     kerns_t     <- dnorm(x=obs_t, mean=obs_t[i], sd=tau_t) *
#       (squares/(tau_t^3) - 1/tau_t)
#     lam_xt      <- sum( kerns_x*kerns_t ) *
#       mu_0/rate(x=obs_x[i,],t=obs_t[i],params,obs_x,obs_t)
#
#     grad <- grad + lam_xt
#   }
#
#   other_bit <- dnorm(x=obs_t, mean=obs_t[n], sd=tau_t)*(obs_t[n]-obs_t) +
#     dnorm(x=obs_t, mean=0, sd=tau_t)*obs_t
#   grad <- grad + mu_0/tau_t * sum(other_bit)
#
#   return(grad)
# }
