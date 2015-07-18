model Ricker {

  param r;
  param phi;
  param theta;
  param sigma;
  noise Z;
  state N;
  obs y;

  sub parameter {
    r ~ uniform(lower=25, upper=65);
    phi ~ uniform(lower=5, upper=15);
    sigma ~ uniform(lower=0.15, upper=0.45);
    theta ~ uniform(lower=0.8, upper=1.2)
  }

  sub initial {
    N ~ gamma(shape=3, scale=1);
  }

  sub transition {
    Z ~ log_gaussian(mean=0, std=sigma);
    N <- r*N*exp(-pow(N, theta))*Z;
  }

  sub observation {
    y ~ pdf(pdf=-phi*N+y*log(phi*N)-lgamma(y+1), log=1);
  }

  sub proposal_parameter {
    r ~ truncated_normal(r, 5, 25, 65);
    phi ~ truncated_normal(phi, 0.5, 5, 15);
    theta ~ truncated_normal(theta, 0.05, 0.8, 1.2);
    sigma ~ truncated_normal(sigma, 0.1, 0.15, 0.45);
  }
}