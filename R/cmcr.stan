functions {
  vector grsm_probs(real alpha, real gamma, real beta, vector kappa) {
    vector[rows(kappa) + 1] unsummed;
    vector[rows(kappa) + 1] probs;
    unsummed = append_row(rep_vector(0, 1), gamma*(alpha - beta - kappa));
    probs = softmax(cumulative_sum(unsummed));
    return probs;
  }
}

data {
  int<lower=1> K;     		    // number of countries
  int<lower=1> T; 				// number of years
  int<lower=1> Q; 				// number of indicators
  int<lower=1> N; 				// number of KTQ observations
  int<lower=1, upper=K> kk[N]; 	// country for observation n
  int<lower=1, upper=T> tt[N]; 	// year for observation n
  int<lower=1, upper=Q> qq[N];  // indicator for observation n
  int<lower=1> y[N];            // score for observation n (1 to max score)
  int<lower=1> S[Q];            // number of steps (max score - 1) for indicator q
  int<lower=1> MS;              // maximum steps across all indicators
}
transformed data {
  int G[N-1]; // number of missing years until next observed country-year (G for "gap")
  int<lower=0> pos[Q, MS-1]; // location within kappa_free for each q and step
  int<lower=0> kpos[Q, MS];  // location within kappa for each q and step
  int<lower=0> NUS; // number of unconstrained steps
  int<lower=1> NCS; // number of constrained steps
  int i;
  
  for (n in 1:N-1) {
      G[n] = tt[n+1] - tt[n] - 1;
  }

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
  real<lower=0, upper=1> alpha[K, T]; // latent variable
  vector<lower=0>[Q] gamma; // discrimination of indicator q
  vector[Q-1] beta_free; // (unconstrained) difficulty of indicator q
  vector[NUS] kappa_free; // (unconstrained) step difficulties, all indicators
  vector<lower=0>[K] sigma_alpha; // country variance parameter
}
transformed parameters {
  vector[Q] beta; // difficulty of indicator q
  vector[NCS] kappa; // step difficulties, all indicators
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
}
model {
  gamma ~ lognormal(1, 1);
  beta_free ~ normal(0, 5);
  kappa_free ~ normal(0, 5);
  
  for (n in 1:N) {
    y[n] ~ categorical(grsm_probs(alpha[kk[n], tt[n]],
                                  gamma[qq[n]],
                                  beta[qq[n]], 
                                  segment(kappa, kpos[qq[n], 1], S[qq[n]])));
    // prior for alpha for the next observed year by country as well as for all intervening missing y
    if (n < N) {
      if (tt[n] < T) {
        for (g in 0:G[n]) {
            alpha[kk[n], tt[n]+g+1] ~ normal(alpha[kk[n], tt[n]+g], sigma_alpha[kk[n]]);
        }
      }
    }
  }
}
