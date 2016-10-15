functions {
  vector grsm_probs(real theta, real mu, real alpha, real beta, vector kappa) {
    vector[rows(kappa) + 1] unsummed;
    vector[rows(kappa) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), alpha*(theta + mu - beta - kappa));
    probs = softmax(cumulative_sum(unsummed));
    return probs;
  }
}
data {
  int<lower=1> I;                // # items
  int<lower=1> J;                // # persons
  int<lower=1> N;                // # responses
  int<lower=1,upper=I> ii[N];    // i for n
  int<lower=1,upper=J> jj[N];    // j for n
  int<lower=0> y[N];             // response for n; y in {1 ... m_i}
  int<lower=1> K;                // # person covariates
  matrix[J,K] W;                 // person covariate matrix
}
transformed data {
  int r[N];                      // modified response; r in {1 ... m_i + 1}
  int m;                         // # steps
  m = max(y) - 1;
}
parameters {
  vector<lower=0>[I] alpha;
  vector[I-1] beta_free;
  vector[m-1] kappa_free[I];
  vector[J] theta;
  vector[K] lambda;
}
transformed parameters {
  vector[I] beta;
  vector[m] kappa[I];
  beta = append_row(beta_free, rep_vector(-1*sum(beta_free), 1));
  for (i in 1:I)
    kappa[i] = append_row(kappa_free[i], rep_vector(-1*sum(kappa_free[i]), 1));
}
model {
  vector[J] mu;
  mu = W*lambda;
  theta ~ normal(0, 1);
  alpha ~ lognormal(1, 1);
  beta_free ~ normal(0, 5);
  for (i in 1:I)
    kappa_free[i] ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ categorical(grsm_probs(theta[jj[n]], mu[jj[n]], alpha[ii[n]],
                                  beta[ii[n]], kappa[ii[n]]));
}