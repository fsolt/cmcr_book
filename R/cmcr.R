load("data/ls2015_dat.Rdata")
dat2 <- dat %>%
    
    # Generate numeric codes for countries, years, questions, and question-cuts
    mutate(variable_cp = paste(variable, cutpoint, sep="_gt"),
           ccode = as.numeric(factor(country, levels = unique(country))),
           tcode = as.integer(year - min(year) + 1),
           qcode = as.numeric(factor(variable, levels = unique(variable)))) %>%
    arrange(ccode, tcode, qcode)

### Delete these when turning into a function
seed <- 324
iter <- 200
chains <- 3
cores <- chains
x <- dat
###



prot_data <- list(  K=max(x$ccode),
                    T=max(x$tcode),
                    Q=max(x$qcode),
                    N=length(x$y_r),
                    kk=x$ccode,
                    tt=x$tcode,
                    qq=x$qcode,
                    y_r=x$y_r,
                    n_r=x$n)

prot_code <- '
data {
int<lower=1> K;     		// number of countries
int<lower=1> T; 				// number of years
int<lower=1> Q; 				// number of questions
int<lower=1> N; 				// number of KTQ observations
int<lower=1, upper=K> kk[N]; 	// country for observation n
int<lower=1, upper=T> tt[N]; 	// year for observation n
int<lower=1, upper=Q> qq[N];  // question for observation n
int<lower=0> y_r[N];    // number of respondents giving selected answer for observation n
int<lower=0> n_r[N];    // total number of respondents for observation n
}
transformed data {
int G[N-1];				// number of missing years until next observed country-year (G for "gap")
for (n in 1:N-1) {
G[n] = tt[n+1] - tt[n] - 1;
}
}
parameters {
real<lower=0, upper=1> alpha[K, T]; // protest participation
real<lower=-1, upper=1> beta[Q]; // position ("difficulty") of question q minus mean (see Stan Development Team 2015, 61; Gelman and Hill 2007, 314-320; McGann 2014, 118-120 (using lambda))
real<lower=0> sigma_beta;   // scale of difficulties (see Stan Development Team 2015, 61)
real mu_beta; // mean question difficulty
real<lower=0, upper=1> gamma[Q]; // discrimination of each question (see Stan Development Team 2015, 61 (using 1/gamma); Gelman and Hill 2007, 314-320 (using 1/gamma); McGann 2014, 118-120 (using alpha))
real<lower=0> sigma_gamma;  // scale of indicator discriminations (see Stan Development Team 2015, 61)
real<lower=0, upper=1> p[N]; // final probability of random individual respondent giving selected answer for observation n (see McGann 2014, 120)
real<lower=0, upper=1> sigma_alpha[K]; 	// country mean opinion variance parameter (see Linzer and Stanton 2012, 12)
real<lower=0, upper=.1> sigma_alpha_var[K]; 	// country sd opinion variance parameter
real<lower=0, upper=30> b[Q];  // "the degree of stochastic variation between question administrations" (McGann 2014, 122)
}
transformed parameters {
real<lower=0, upper=1> m[N]; // expected probability of random individual giving selected answer

for (n in 1:N) {
m[n] = inv_logit((alpha[kk[n], tt[n]] - (beta[qq[n]] + mu_beta)) / gamma[qq[n]]);
}

}
model {
beta ~ normal(0, sigma_beta);
gamma ~ lognormal(0, sigma_gamma);
mu_beta ~ cauchy(0, 1);
sigma_beta ~ cauchy(0, .5);
sigma_gamma ~ cauchy(0, .5);

// actual number of respondents giving selected answer
y_r ~ binomial(n_r, p);

for (n in 1:N) {
// individual probability of selected answer
p[n] ~ beta(b[qq[n]]*m[n]/(1 - m[n]), b[qq[n]]);

// prior for alpha and var_alpha for the next observed year by country as well as for all intervening missing years
if (n < N) {
if (tt[n] < T) {
for (g in 0:G[n]) {
alpha[kk[n], tt[n]+g+1] ~ normal(alpha[kk[n], tt[n]+g], sigma_alpha[kk[n]]);
}
}
}
}
}
'

start <- proc.time()
out1 <- stan(model_code = prot_code,
             data = prot_data,
             seed = seed,
             iter = iter,
             cores = cores,
             chains = chains,
             control = list(max_treedepth = 20,
                            adapt_delta = .8))
runtime <- proc.time() - start
runtime

lapply(get_sampler_params(out1, inc_warmup = FALSE),
       summary, digits = 2)

#Chime
beep()
