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
  int<lower=0> us[Q];       // unconstrained steps (max score - 2) for indicator q
  int<lower=0> S;           // sum of unconstrained steps (us)
  int<lower=1> cs[Q];       // constrained steps (max score - 1) for indicator q
}
transformed data {
  int G[N-1]; // number of missing years until next observed country-year (G for "gap")
  int<lower=1> pos[Q]; // location within kappa_free for each q ("position")
  int<lower=1> kpos[Q]; // location within kappa for each q ("kappa position")
  
  for (n in 1:N-1) {
      G[n] = tt[n+1] - tt[n] - 1;
  }

  pos[1] = 1;
  for (q in 2:Q-1) {
      pos[q] = pos[q-1] + cs[q-1];
  }
  
  kpos[1] = 1;
  for (q in 2:Q-1) {
      kpos[q] = kpos[q-1] + us[q-1];
  }
  
}
parameters {
  real<lower=0, upper=1> alpha[K, T]; // latent variable
  vector<lower=0>[Q] gamma; // discrimination of indicator q
  vector[Q-1] beta_free; // (unconstrained) difficulty of indicator q
  vector[S] kappa_free; // (unconstrained) step difficulties, all indicators
  vector<lower=0>[K] sigma_alpha; // country variance parameter
}
transformed parameters {
  vector[Q] beta; // difficulty of indicator q
  vector[S+Q] kappa; // step difficulties, all indicators

  beta = append_row(beta_free, rep_vector(-1*sum(beta_free), 1));

  for (q in 1:Q) {
    segment(kappa, kpos[q], cs[q]) = append_row(segment(kappa_free, pos[q], us[q]),
      rep_vector(-1*sum(segment(kappa_free, pos[q], us[q])), 1));
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
                                  segment(kappa, kpos[qq[n]], cs[qq[n]])));
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
