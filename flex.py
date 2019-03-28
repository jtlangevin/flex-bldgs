#!/usr/bin/env python3
import pymc3 as pm
import theano as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd
tt.config.compute_test_value = "ignore"


class ModelIO(object):
    """Dummy observations to use in testing the Bayesian network models.

    Attributes:
        temp_out (numpy ndarray): Test outdoor temperature data.
        intercept (theano tensor): Vector of ones for intercept terms.
        X_temp (numpy ndarray): Temperature model input observations.
        Y_temp (theano tensor): Temperature model output observations.
        X_hum (numpy ndarray): Humidity model input observations.
        Y_hum (theano tensor): Humidity model output observations.
        X_co2 (numpy ndarray): CO2 model input observations.
        Y_co2 (theano tensor): CO2 model output observations.
        X_lt (numpy ndarray): Illuminance model input observations.
        Y_lt (theano tensor): Illuminance model output observations.
        plug_delta (theano tensor): Plug load power fraction observations.
        X_dmd_nc (numpy ndarray): Energy model inputs, no change point.
        X_dmd_c (numpy ndarray): Energy model input, change point.
        Y_dmd (theano tensor): Energy model output observations.
        Y_choice (numpy ndarray): Choice model output observations.
    """

    def __init__(self, th_rng, n_samples, n_choices):
        """Initialize class attributes."""
        # Whole building occupancy fraction
        occupancy = np.random.uniform(0, 1, n_samples)
        # Outdoor air temperature
        self.temp_out = np.random.uniform(5, 105, n_samples)
        # Outdoor relative humidity
        rh_out = np.random.uniform(0, 1, n_samples)
        # Cooling delta T
        tmp_delta = np.random.choice(
            [0, 2, 3, 6], n_samples, p=[0.25, 0.25, 0.25, 0.25])
        # Pre-cooling delta T
        pre_tmp_delta = np.random.choice(
            [0, 2, 3, 6], n_samples, p=[0.25, 0.25, 0.25, 0.25])
        # OA ventilation fraction reduction
        oaf_delta = np.random.uniform(0, 1, n_samples)
        # Set a vector of ones for intercept estimation
        intercept = np.ones(n_samples)
        # Convert to theano tensor
        self.intercept = tt.tensor._shared(intercept)

        # Temperature model X variables
        self.X_temp = np.column_stack([
            occupancy.T, self.temp_out.T, tmp_delta.T, pre_tmp_delta.T,
            oaf_delta.T, intercept.T])
        # Temperature model betas
        betas_temp = np.linspace(0.1, 0.75, self.X_temp.shape[1])
        # Temperature model Y variable
        self.Y_temp = pm.math.dot(self.X_temp, betas_temp) + \
            np.random.randn(n_samples)

        # Humidity model X variables
        self.X_hum = np.column_stack([
            occupancy.T, rh_out.T, tmp_delta.T, pre_tmp_delta.T,
            oaf_delta.T, intercept.T])
        # Humidity model betas
        betas_hum = np.linspace(0.1, 0.5, self.X_hum.shape[1])
        # Humidity model Y variable
        self.Y_hum = pm.math.dot(self.X_hum, betas_hum) + \
            np.random.randn(n_samples)

        # CO2 model X variables
        self.X_co2 = np.column_stack([occupancy.T, oaf_delta.T, intercept.T])
        # CO2 model betas
        betas_co2 = np.linspace(0.2, 0.75, self.X_co2.shape[1])
        # CO2 model Y variable
        self.Y_co2 = pm.math.dot(self.X_co2, betas_co2) + \
            np.random.randn(n_samples)

        # Exterior cloud cover fraction
        shade_out = np.random.uniform(0, 1, n_samples)
        # Lighting fraction reduction
        lt_delta = np.random.uniform(0, 1, n_samples)

        # Lighting model X variables
        self.X_lt = np.column_stack([
            occupancy.T, shade_out.T, lt_delta.T, intercept.T])
        # Lighting model betas
        betas_lt = np.linspace(0.3, 0.9, self.X_lt.shape[1])
        # Lighting model Y variable
        self.Y_lt = pm.math.dot(self.X_lt, betas_lt) + \
            np.random.randn(n_samples)

        # Plug load power fraction reduction
        plug_delta = np.random.uniform(0, 1, n_samples)
        # Convert to theano tensor for use in pm.Model()
        self.plug_delta = tt.tensor._shared(plug_delta)

        # Facility energy use model X variables (no change point behavior)
        self.X_dmd_nc = np.column_stack([
            occupancy.T, oaf_delta.T, lt_delta.T,
            plug_delta.T, tmp_delta.T, pre_tmp_delta.T])
        # Facility energy use model X variables (change point behavior)
        self.X_dmd_c = np.column_stack([self.temp_out.T, intercept.T])
        # Facility energy use model betas (no change point behavior)
        betas_dmd_nc = np.linspace(0.4, 0.6, self.X_dmd_nc.shape[1])
        # Facility energy use model betas (change point behavior)
        # Set switch points
        dmd_sp1, dmd_sp2 = [50, 70]
        # Set betas for each region between the switch points
        b_1, b_2, b_3 = [[-0.5, 3], [0.1, 1], [0.5, -3]]
        # Facility energy use model Y variable
        # Y values before switch point one, between switch point one and two
        y_dmd_c = pm.math.switch(
            dmd_sp1 >= self.temp_out,
            pm.math.dot(self.X_dmd_c, b_1), pm.math.dot(self.X_dmd_c, b_2))
        # Y values after switch point two
        self.Y_dmd = pm.math.switch(
            dmd_sp2 >= self.temp_out, y_dmd_c,
            pm.math.dot(self.X_dmd_c, b_3)) + \
            pm.math.dot(self.X_dmd_nc, betas_dmd_nc) + \
            np.random.randn(n_samples)

        # Control choice model X variables
        x_choice = tt.tensor.stack([
            self.Y_temp, self.Y_hum, self.Y_co2, self.Y_lt,
            self.plug_delta, self.Y_dmd, self.intercept]).T
        # Control choice model betas (one set for each control alternative,
        # here testing with three alternatives)
        betas_choice_1 = tt.tensor._shared(
            np.linspace(0.1, 0.25, x_choice.shape.eval()[1]))
        betas_choice_2 = tt.tensor._shared(
            np.linspace(0.2, 0.5, x_choice.shape.eval()[1]))
        betas_choice_3 = tt.tensor._shared(
            np.linspace(0.3, 0.4, x_choice.shape.eval()[1]))
        betas_choice = tt.tensor.stack([
            betas_choice_1, betas_choice_2, betas_choice_3]).T
        # Softmax transformation of linear estimator into multinomial choice
        # probabilities
        logit = pm.math.dot(x_choice, betas_choice) + \
            th_rng.normal((n_samples, n_choices))
        choice_probs = tt.tensor.nnet.softmax(logit)
        # Control choice model Y variable - theano yields a vector where the
        # elements are themselves vectors of length three, with ones in
        # the chosen index elements (e.g., [[0, 1, 0], [0, 0, 1], ...])
        y_choice = th_rng.multinomial(n=1, pvals=choice_probs).eval()
        # Reformat choice tensor such that it is a vector with the choice
        # index (0, 1, 2) in each of the elements (e.g., [1, 2, ...])
        self.Y_choice = np.zeros(len(y_choice))
        for ind, r in enumerate(y_choice):
            self.Y_choice[ind] = np.nonzero(r)[0][0]


def main(base_dir):
    """Implement Bayesian network and plot resultant parameter estimates."""
    # Set numpy and theano RNG seeds for consistency
    np.random.seed(123)
    th_rng = MRG_RandomStreams()
    th_rng.seed(123)

    # Set number of observations to use for parameter updating (for now,
    # assume model parameters are updated daily given 15 minute interval data
    # from all 24 hours)
    n_samples = 24 * 4
    # Set number of control choices
    n_choices = 3

    # Initialize test data
    io = ModelIO(th_rng, n_samples, n_choices)

    # Estimate Bayesian network model
    with pm.Model() as flex_bdn:

        # *** Temperature sub-model ***

        # Set parameter priors (betas, error)
        ta_params = pm.Normal('ta_params', 0, 20, shape=(io.X_temp.shape[1]))
        ta_sd = pm.Uniform('ta_sd', 0, 20)
        # Likelihood of temperature estimator
        ta_est = pm.math.dot(io.X_temp, ta_params)
        # Likelihood of temperature
        ta = pm.Normal('ta', mu=ta_est, sd=ta_sd, observed=io.Y_temp)

        # *** RH sub-model ***

        # Set parameter priors (betas, error)
        rh_params = pm.Normal('rh_params', 0, 20, shape=(io.X_hum.shape[1]))
        rh_sd = pm.Uniform('rh_sd', 0, 20)
        # Likelihood of humidity estimator
        rh_est = pm.math.dot(io.X_hum, rh_params)
        # Likelihood of humidity
        rh = pm.Normal('rh', mu=rh_est, sd=rh_sd, observed=io.Y_hum)

        # *** CO2 sub-model ***

        # Set parameter priors (betas, error)
        co2_params = pm.Normal('co2_params', 0, 20, shape=(io.X_co2.shape[1]))
        co2_sd = pm.Uniform('co2_sd', 0, 20)
        # Likelihood of CO2 estimator
        co2_est = pm.math.dot(io.X_co2, co2_params)
        # Likelihood of CO2
        co2 = pm.Normal('co2', mu=co2_est, sd=co2_sd, observed=io.Y_co2)

        # *** Lighting sub-model ***

        # Set parameter priors (betas, error)
        lt_params = pm.Normal('lt_params', 0, 20, shape=(io.X_lt.shape[1]))
        lt_sd = pm.Uniform('lt_sd', 0, 20)
        # Likelihood of lighting estimator
        lt_est = pm.math.dot(io.X_lt, lt_params)
        # Likelihood of lighting
        lt = pm.Normal('lt', mu=lt_est, sd=lt_sd, observed=io.Y_lt)

        # *** Demand sub-model ***

        # Set parameter priors (switch points, betas, error)
        # Switch points
        dmd_sp1 = pm.DiscreteUniform(
            'dmd_sp1', io.temp_out.min(), io.temp_out.max())
        dmd_sp2 = pm.DiscreteUniform('dmd_sp2', dmd_sp1, io.temp_out.max())
        # Betas
        dmd_params_c = pm.Normal('dmd_params_c', 0, 20, shape=(
            3, io.X_dmd_c.shape[1]))
        dmd_params_nc = pm.Normal('dmd_params_nc', 0, 20, shape=(
            io.X_dmd_nc.shape[1]))
        # Error
        dmd_sd = pm.Uniform('dmd_sd', 0, 20)
        # Likelihood of demand estimator
        dmd_est_c = pm.math.switch(
            dmd_sp1 >= io.temp_out,
            pm.math.dot(io.X_dmd_c, dmd_params_c[0]),
            pm.math.dot(io.X_dmd_c, dmd_params_c[1]))
        dmd_est = pm.math.switch(
            dmd_sp2 >= io.temp_out, dmd_est_c,
            pm.math.dot(io.X_dmd_c, dmd_params_c[2])) + pm.math.dot(
                io.X_dmd_nc, dmd_params_nc)
        # Likelihood of demand
        dmd = pm.Normal('dmd', mu=dmd_est, sd=dmd_sd, observed=io.Y_dmd)

        # *** Choice sub-model ***

        # X variables are the outputs of the above sub-models
        x_choice_bn = tt.tensor.stack([
            ta, rh, co2, lt, io.plug_delta, dmd,
            io.intercept]).T
        # Set parameter priors (betas)
        choice_params = pm.Normal(
            'choice_params', mu=0, sd=10, shape=(
                x_choice_bn.shape.eval()[1], n_choices))
        # Softmax transformation of linear estimator into multinomial choice
        # probabilities
        logit_bn = pm.math.dot(x_choice_bn, choice_params)
        choice_probs_bn = tt.tensor.nnet.softmax(logit_bn)
        # Set choice probabilities as deterministic PyMC3 variable type
        p = pm.Deterministic('p', choice_probs_bn)
        # Likelihood of choice
        choice = pm.Categorical('choice', p=p, observed=io.Y_choice)

        # Draw posterior samples
        trace = pm.sample()

    # # Sample from the posterior predictive distribution
    # ppc = pm.sample_posterior_predictive(trace, samples=1, model=flex_bdn)
    # print(ppc)
    # Plot parameter traces for diagnostic purposes
    pm.traceplot(trace)
    plt.show()


if __name__ == '__main__':
    base_dir = getcwd()
    main(base_dir)
