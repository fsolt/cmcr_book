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
  int<lower=1> Q;                // # items
  int<lower=1> J;                // # persons
  int<lower=1> N;                // # responses
  int<lower=1,upper=Q> qq[N];    // q for n
  int<lower=1,upper=J> jj[N];    // j for n
  int<lower=0> y[N];             // response for n; y in {1 ... m_i}
  int<lower=1> K;                // # person covariates
  matrix[J,K] W;                 // person covariate matrix
  int<lower=1> S[Q];            // number of steps (max score - 1) for indicator q
  int<lower=1> MS;              // maximum steps across all indicators
}
transformed data {
  int<lower=0> pos[Q, MS-1]; // location within kappa_free for each q and step
  int<lower=0> kpos[Q, MS];  // location within kappa for each q and step
  int<lower=0> NUS; // number of unconstrained steps
  int<lower=1> NCS; // number of constrained steps
  int i;

  i = 0;
  for (q in 1:Q) {
    for (s in 1:(S[q]-1)) {
      i = i + 1;
      pos[q, s] = i; 
    }
    for (s in S[q]:MS-1) {
      pos[q, s] = 0;
    }
  }
  NUS = i;
  
  i = 0;
  for (q in 1:Q) {
    for (s in 1:S[q]) {
      i = i + 1;
      kpos[q, s] = i; 
    }
    for (s in (S[q]+1):MS) {
      kpos[q, s] = 0;
    }
  }
  NCS = i;
}
parameters {
  vector<lower=0>[Q] alpha;     // item discrimination
  vector[Q-1] beta_free;        // item difficulty
  vector[NUS] kappa_free;       // constant cutpoint difficulty
  vector[J] theta;              // person "ability"
  vector[K] lambda;             // coefficients for person covariates
}
transformed parameters {
  vector[Q] beta;
  vector[NCS] kappa;
  real total;
  
  beta = append_row(beta_free, rep_vector(-1*sum(beta_free), 1));

  for (q in 1:Q) { // indicators
    total = 0;
    for (s in 1:S[q]) { // steps in indicator q
      if (s < S[q]) {
        kappa[kpos[q, s]] = kappa_free[pos[q, s]];
        total = total + kappa_free[pos[q, s]];
      } else {
        kappa[kpos[q, s]] = -1 * total;
      }
    }
  }
  total = 0;
}
model {
  vector[J] mu;
  mu = W*lambda;
  theta ~ normal(0, 1);
  alpha ~ lognormal(1, 1);
  beta_free ~ normal(0, 5);
  kappa_free ~ normal(0, 5);
  for (n in 1:N)
    y[n] ~ categorical(grsm_probs(theta[jj[n]],
                                  mu[jj[n]],
                                  alpha[qq[n]],
                                  beta[qq[n]],
                                  segment(kappa, kpos[qq[n], 1], S[qq[n]])));
}
