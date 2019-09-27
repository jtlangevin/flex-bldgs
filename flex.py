#!/usr/bin/env python3
import pymc3 as pm
import theano as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
from os import getcwd, path
from argparse import ArgumentParser
import pickle
from scipy import stats
tt.config.compute_test_value = "ignore"


class UsefulInputFiles(object):
    """Class of input file paths to be used by this routine.

    Attributes:
        stored_tmp: Stored temperature sub-model file.
        stored_rh: Stored relative humidity sub-model file.
        stored_co2: Stored CO2 sub-model file.
        stored_lt: Stored lighting sub-model file.
        stored_dmd: Stored demand sub-model file.
        fit_data: Training data to use in inferring model parameter values.
        predict_data: Data to use in testing model predictions.
    """

    def __init__(self):

        # Set file path to stored model information
        self.stored_tmp = ("model_stored", "tmp.pkl")
        self.stored_rh = ("model_stored", "rh.pkl")
        self.stored_co2 = ("model_stored", "co2.pkl")
        self.stored_lt = ("model_stored", "lt.pkl")
        self.stored_dmd = ("model_stored", "dmd.pkl")
        self.fit_data = ("data", "synth_new.csv")
        self.predict_data = ("data", "test.csv")


class ModelIOFit(object):
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
        Y_dmd (theano tensor): Energy model output observations.
    """

    def __init__(self, handyfiles):
        """Initialize class attributes."""

        # Import training data
        print("Retrieving training data...", end="", flush=True)
        data = np.genfromtxt(
            path.join(base_dir, *handyfiles.fit_data),
            skip_header=True,
            names=('id', 'vintage', 'climate', 'hour', 'day_typ',
                   'dmd_delt_sf', 't_out', 'rh_out', 'cloud_out',
                   't_in_delt', 'rh_in_delt', 'rh_in_delt_pct', 'lt_nat',
                   'lt_in_delt', 'lt_in_delt_pct', 'co2_in_delt',
                   'co2_in_delt_pct', 'tsp_delt', 'lt_pwr_delt_pct',
                   'ven_delt_pct', 'mels_delt_pct', 'hrs_since_dr_st',
                   'hrs_since_dr_end', 'hrs_since_pc_st', 'hrs_since_pc_end',
                   'occ_frac', 'tsp_delt_lag', 'lt_pwr_delt_pct_lag',
                   'mels_delt_pct_lag', 'ven_delt_pct_lag'),
            dtype=(['<i4'] * 2 + ['<U25'] + ['<i4'] * 2 + ['<f8'] * 25),
            delimiter=',')
        print("Data import complete")

        # Draw a training subset from the full dataset
        data = data[np.random.randint(0, len(data), size=500000)]

        # Initialize variables
        print("Initializing model variables...", end="", flush=True)
        # Whole building occupancy fraction
        occupancy = data['occ_frac']
        # Outdoor air temperature
        self.temp_out = data['t_out']
        # Outdoor relative humidity
        rh_out = data['rh_out']
        # Cooling delta T
        tmp_delta = data['tsp_delt']
        # Cooling delta T lag
        tmp_delta_lag = data['tsp_delt_lag']
        # Hours since DR event started (adjustment to normal op. condition)
        dr_start = data['hrs_since_dr_st']
        # Hours since DR event ended (adjustment to normal op. condition)
        dr_end = data['hrs_since_dr_end']
        # Hours since pre-cooling started (if applicable)
        pcool_start = data['hrs_since_pc_st']
        # Hours since pre-cooling ended (if applicable)
        pcool_end = data['hrs_since_pc_end']
        # OA ventilation fraction reduction
        oaf_delta = data['ven_delt_pct']
        # OA ventilation fraction reduction lag
        oaf_delta_lag = data['ven_delt_pct_lag']
        # Lighting fraction reduction
        lt_delta = data['lt_pwr_delt_pct']
        # Lighting fraction reduction lag
        lt_delta_lag = data['lt_pwr_delt_pct_lag']
        # Natural illuminance
        lt_out = data['lt_nat']
        # Plug load power fraction reduction
        plug_delta = data['mels_delt_pct']
        # Plug load power fraction reduction
        plug_delta_lag = data['mels_delt_pct_lag']
        # Convert to theano tensor for use in pm.Model()
        self.plug_delta = tt.tensor._shared(plug_delta)
        # Set a vector of ones for intercept estimation
        intercept = np.ones(len(oaf_delta))
        # Convert to theano tensor
        self.intercept = tt.tensor._shared(intercept)
        # Initialize interactive terms
        # temp_dr_interact = self.temp_out * tmp_delta * dr_start
        # vent_dr_interact = self.temp_out * oaf_delta * dr_start

        # Temperature model X variables
        self.X_temp = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out, tmp_delta.T,
            lt_delta.T, plug_delta.T,
            tmp_delta_lag.T, lt_delta_lag.T,
            plug_delta_lag.T, pcool_start.T, pcool_end.T, dr_start.T,
            dr_end.T, intercept.T])
        # Temperature model Y variable
        self.Y_temp = tt.tensor._shared(data['t_in_delt'])
        # self.Y_temp = pm.math.dot(self.X_temp, betas_temp) + \
        #     np.random.randn(n_samples)

        # Humidity model X variables
        self.X_hum = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out, tmp_delta.T, lt_delta.T,
            plug_delta.T, tmp_delta_lag.T, lt_delta_lag.T,
            plug_delta_lag.T, pcool_start.T, pcool_end.T, dr_start.T,
            dr_end.T, intercept.T])
        # Humidity model Y variable
        self.Y_hum = tt.tensor._shared(data['rh_in_delt'])

        # CO2 model X variables
        self.X_co2 = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out,
            dr_start.T, dr_end.T, intercept.T])
        # CO2 model Y variable
        self.Y_co2 = tt.tensor._shared(data['co2_in_delt_pct'])

        # Lighting model X variables
        self.X_lt = np.column_stack([
            lt_out.T, lt_delta.T, intercept.T])
        # Lighting model Y variable
        self.Y_lt = tt.tensor._shared(data['lt_in_delt_pct'])

        # Facility energy use model X variables (no change point behavior)
        self.X_dmd_nc = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out.T, tmp_delta.T, lt_delta.T,
            plug_delta.T, tmp_delta_lag.T, lt_delta_lag.T,
            plug_delta_lag.T, pcool_start.T, pcool_end.T, dr_start.T,
            dr_end.T, intercept.T])

        # Demand Y variable
        self.Y_dmd = data['dmd_delt_sf']

        print("Model initialization complete.")


class ModelIOPredict(object):
    """Scenario data to use in testing Bayesian model predictions.

    Attributes:
        names (numpy ndarray): Names of the candidate control choices.
        temp_out (numpy ndarray): Test outdoor temperature data.
        intercept (theano tensor): Vector of ones for intercept terms.
        X_temp (numpy ndarray): Temperature model input observations.
        X_hum (numpy ndarray): Humidity model input observations.
        X_co2 (numpy ndarray): CO2 model input observations.
        X_lt (numpy ndarray): Illuminance model input observations.
        plug_delta (theano tensor): Plug load power fraction observations.
        X_dmd_nc (numpy ndarray): Energy model inputs, no change point.
        price_delta (theano tensor): Electricity price/kWh saved over baseline.
    """

    def __init__(self, handyfiles, scn):
        """Initialize class attributes."""

        # Import testing data
        print("Retrieving testing data...", end="", flush=True)
        data = np.genfromtxt(
            path.join(base_dir, *handyfiles.predict_data),
            skip_header=True, delimiter=',',
            names=('Name', 'Scn', 't_out', 'rh_out', 'lt_nat', 'occ_frac',
                   'delt_price_kwh', 'hrs_since_dr_st', 'hrs_since_dr_end',
                   'hrs_since_pc_st', 'hrs_since_pc_end', 'tsp_delt',
                   'lt_pwr_delt_pct', 'ven_delt_pct', 'mels_delt_pct',
                   'tsp_delt_lag', 'lt_pwr_delt_pct_lag',
                   'ven_delt_pct_lag', 'mels_delt_pct_lag'),
            dtype=(['<U25'] + ['<f8'] * 18))
        print("Data import complete")
        data = data[np.in1d(data["Scn"], (scn + 1))]
        # Set measure names
        self.names = data['Name']
        # Whole building occupancy fraction
        occupancy = data['occ_frac']
        # Outdoor air temperature
        self.temp_out = data['t_out']
        # Outdoor relative humidity
        rh_out = data['rh_out']
        # Cooling delta T
        tmp_delta = data['tsp_delt']
        # Cooling delta T lag
        tmp_delta_lag = data['tsp_delt_lag']
        # Hours since DR event started (adjustment to normal op. condition)
        dr_start = data['hrs_since_dr_st']
        # Hours since DR event ended (adjustment to normal op. condition)
        dr_end = data['hrs_since_dr_end']
        # Hours since pre-cooling started (if applicable)
        pcool_start = data['hrs_since_pc_st']
        # Hours since pre-cooling ended (if applicable)
        pcool_end = data['hrs_since_pc_end']
        # OA ventilation fraction reduction
        oaf_delta = data['ven_delt_pct']
        # OA ventilation fraction reduction lag
        oaf_delta_lag = data['ven_delt_pct_lag']
        # oaf_delta = np.random.uniform(0, 1, n_samples)
        # Lighting fraction reduction
        lt_delta = data['lt_pwr_delt_pct']
        # Lighting fraction reduction lag
        lt_delta_lag = data['lt_pwr_delt_pct_lag']
        # Natural illuminance
        lt_out = data['lt_nat']
        # Plug load power fraction reduction
        plug_delta = data['mels_delt_pct']
        # Plug load power fraction reduction
        plug_delta_lag = data['mels_delt_pct_lag']
        # Convert to theano tensor for use in pm.Model()
        self.plug_delta = tt.tensor._shared(plug_delta)
        # Set a vector of ones for intercept estimation
        intercept = np.ones(len(oaf_delta))
        # Convert to theano tensor
        self.intercept = tt.tensor._shared(intercept)
        # Set a price vector
        price_delt = data['delt_price_kwh']
        # Convert to theano tensor
        self.price_delt = tt.tensor._shared(price_delt)

        # Temperature model X variables
        self.X_temp = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out, tmp_delta.T, lt_delta.T,
            plug_delta.T, tmp_delta_lag.T, lt_delta_lag.T,
            plug_delta_lag.T, pcool_start.T, pcool_end.T, dr_start.T,
            dr_end.T, intercept.T])

        # Humidity model X variables
        self.X_hum = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out, tmp_delta.T, lt_delta.T,
            plug_delta.T, tmp_delta_lag.T, lt_delta_lag.T,
            plug_delta_lag.T, pcool_start.T, pcool_end.T, dr_start.T,
            dr_end.T, intercept.T])

        # CO2 model X variables
        self.X_co2 = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out,
            dr_start.T, dr_end.T, intercept.T])

        # Lighting model X variables
        self.X_lt = np.column_stack([
            lt_out.T, lt_delta.T, intercept.T])

        # Facility energy use model X variables (no change point behavior)
        self.X_dmd_nc = np.column_stack([
            occupancy.T, self.temp_out.T, rh_out.T, tmp_delta.T, lt_delta.T,
            plug_delta.T, tmp_delta_lag.T, lt_delta_lag.T,
            plug_delta_lag.T, pcool_start.T, pcool_end.T, dr_start.T,
            dr_end.T, intercept.T])


def main(base_dir):
    """Implement Bayesian network and plot resultant parameter estimates."""

    # Set input file names
    handyfiles = UsefulInputFiles()
    # Set numpy and theano RNG seeds for consistency
    np.random.seed(123)
    th_rng = MRG_RandomStreams()
    th_rng.seed(123)

    # Proceed with model inference only if user flags doing so
    if opts.mod_est is True:

        # Initialize X variables for training data
        io = ModelIOFit(handyfiles)

        # *** Temperature change ***

        # Estimate temperature sub-model
        with pm.Model() as tmp_model_fit:
            print("Setting temperature sub-model priors and likelihood...",
                  end="", flush=True)
            # Unpack shared theano variable
            # io = io_shared.get_value()
            # Set parameter priors (betas, error)
            ta_params = pm.Normal(
                'ta_params', 0, 10, shape=(io.X_temp.shape[1]))
            ta_sd = pm.HalfNormal('ta_sd', 20)
            # Likelihood of temperature estimator
            ta_est = pm.math.dot(io.X_temp, ta_params)
            # Likelihood of temperature
            ta = pm.Normal('ta', mu=ta_est, sd=ta_sd, observed=io.Y_temp)
            print("Complete.")
            # Draw posterior samples
            trace = pm.sample(chains=2, cores=1, init="advi",
                              target_accept=0.9)

        # Store temperature model, trace, and predictor variables
        with open(path.join(
                base_dir, *handyfiles.stored_tmp), 'wb') as tmp_s:
            pickle.dump({'trace': trace, 'model': tmp_model_fit}, tmp_s)

        # *** Relative humidity change ***

        # Estimate RH sub-model
        with pm.Model() as rh_model:
            print("Setting RH sub-model priors and likelihood...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            rh_params = pm.Normal(
                'rh_params', 0, 10, shape=(io.X_hum.shape[1]))
            rh_sd = pm.HalfNormal('rh_sd', 20)
            # Likelihood of humidity estimator
            rh_est = pm.math.dot(io.X_hum, rh_params)
            # Likelihood of humidity
            rh = pm.Normal('rh', mu=rh_est, sd=rh_sd, observed=io.Y_hum)
            print("Complete.")
            # Draw posterior samples
            trace = pm.sample(chains=2, cores=1, init="advi",
                              target_accept=0.9)
        # Store RH model, trace, and predictor variables
        with open(path.join(
                base_dir, *handyfiles.stored_rh), 'wb') as rh_s:
            pickle.dump({'trace': trace, 'model': rh_model}, rh_s)

        # *** CO2 concentration change ***

        # Estimate CO2 sub-model
        with pm.Model() as co_model:
            print("Setting CO2 sub-model priors and likelihood...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            co2_params = pm.Normal(
                'co2_params', 0, 10, shape=(io.X_co2.shape[1]))
            co2_sd = pm.HalfNormal('co2_sd', 20)
            # Likelihood of CO2 estimator
            co2_est = pm.math.dot(io.X_co2, co2_params)
            # Likelihood of CO2
            co2 = pm.Normal('co2', mu=co2_est, sd=co2_sd, observed=io.Y_co2)
            print("Complete.")
            # Draw posterior samples
            trace = pm.sample(chains=2, cores=1, init="advi",
                              target_accept=0.9)
        # Store CO2 model, trace, and predictor variables
        with open(path.join(
                base_dir, *handyfiles.stored_co2), 'wb') as co_s:
            pickle.dump({'trace': trace, 'model': co_model}, co_s)

        # *** Lighting change ***

        # Estimate lighting sub-model
        with pm.Model() as lt_model:
            print("Setting lighting sub-model priors and likelihood...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            lt_params = pm.Normal('lt_params', 0, 10, shape=(io.X_lt.shape[1]))
            lt_sd = pm.HalfNormal('lt_sd', 20)
            # Likelihood of lighting estimator
            lt_est = pm.math.dot(io.X_lt, lt_params)
            # Likelihood of lighting
            lt = pm.Normal('lt', mu=lt_est, sd=lt_sd, observed=io.Y_lt)
            print("Complete.")
            # Draw posterior samples
            trace = pm.sample(chains=2, cores=1, init="advi",
                              target_accept=0.9)
        # Store lighting model, trace, and predictor variables
        with open(path.join(
                base_dir, *handyfiles.stored_lt), 'wb') as lt_s:
            pickle.dump({'trace': trace, 'model': lt_model}, lt_s)

        # *** Change in hourly electricity demand ***

        # Estimate demand sub-model
        with pm.Model() as dmd_model:
            print("Setting demand sub-model priors and likelihood...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            dmd_params_nc = pm.Normal('dmd_params_nc', 0, 10, shape=(
                io.X_dmd_nc.shape[1]))
            dmd_sd = pm.HalfNormal('dmd_sd', 20)
            # Likelihood of demand estimator
            dmd_est = pm.math.dot(io.X_dmd_nc, dmd_params_nc)
            # Likelihood of demand
            dmd = pm.Normal('dmd', mu=dmd_est, sd=dmd_sd, observed=io.Y_dmd)
            print("Complete.")
            # Draw posterior samples
            trace = pm.sample(
                chains=2, cores=1, init="advi", target_accept=0.9)
            # pm.traceplot(trace)
            # plt.show()
            # ppc = pm.sample_posterior_predictive(trace, samples=11)
            # print(ppc["dmd"], len(ppc["dmd"][0]))

        # Store demand model, trace, and predictor variables
        with open(path.join(
                base_dir, *handyfiles.stored_dmd), 'wb') as dmd_s:
            pickle.dump({'trace': trace, 'model': dmd_model}, dmd_s)

        # # *** Choice sub-model SUPPRESSED FOR NOW ***

        # with pm.Model() as choice_model:
        #     print("Setting choice sub-model priors and likelihood...",
        #           end="", flush=True)
        #     # X variables are the outputs of the above sub-models
        #     x_choice_bn = tt.tensor.stack([
        #         cost_delt, ta, rh, co2, lt, io.plug_delta,
        #         io.intercept]).T
        #     # Set parameter priors (betas)
        #     choice_params = pm.Normal(
        #         'choice_params', mu=0, sd=10, shape=(
        #             x_choice_bn.shape.eval()[1], n_choices))
        #     # Softmax transformation of linear estimator into multinomial
        #     # choice probabilities
        #     logit_bn = pm.math.dot(x_choice_bn, choice_params)
        #     choice_probs_bn = tt.tensor.nnet.softmax(logit_bn)
        #     # Set choice probabilities as deterministic PyMC3 variable type
        #     p = pm.Deterministic('p', choice_probs_bn)
        #     # Likelihood of choice
        #     choice = pm.Categorical('choice', p=p, observed=io.Y_choice)
        #     print("Complete.")
        #     # Draw posterior samples
        #     trace = pm.sample(chains=2, cores=1, init="advi")

        # # Store choice model, trace, and predictor variables
        # with open(path.join(
        #         base_dir, *handyfiles.stored_choice), 'wb') as choice_s:
        #     pickle.dump({
        #         'model': choice_model, 'trace': trace,
        #         'X_shared': x_choice_bn}, choice_s)

    # Set number of control choices
    n_choices = 16
    # Set number of samples to draw. for predictions
    n_samples = 100
    # Set number of scenarios to test
    n_scenarios = 10
    # Set square footage to assume
    sf = int(input("Enter the total square footage of your building: "))
    # Sample noise to use in the choice model
    rand_elem = th_rng.normal(n_choices, avg=0, std=1)
    # Set the constant set of betas across alternatives to use in the choice
    # model
    betas_choice = tt.tensor._shared(
        np.array([0.05, -20, 0, -50, -50, -100, 10]))

    # Loop through the set of scenarios considered for FY19 EOY deliverable
    for scn in range(n_scenarios):
        print(("Running input scenario " + str(scn+1)) + "...")
        # Reset x predictor variables according to the scenario
        iop = ModelIOPredict(handyfiles, scn=scn)
        # Reload temperature trace
        print("Loading temperature sub-model...", end="", flush=True)
        with open(path.join(
                base_dir, *handyfiles.stored_tmp), 'rb') as store_tmp:
            trace_tmp = pickle.load(store_tmp)['trace']
        # Re-estimate temperature sub-model with out-of sample data
        with pm.Model() as tmp_model_pred:
            print("Re-setting temperature sub-model inputs...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            ta_params = pm.Normal(
                'ta_params', 0, 10, shape=(iop.X_temp.shape[1]))
            ta_sd = pm.HalfNormal('ta_sd', 20)
            # Likelihood of temperature estimator
            ta_est = pm.math.dot(iop.X_temp, ta_params)
            # Likelihood of temperature
            ta = pm.Normal(
                'ta', mu=ta_est, sd=ta_sd,
                observed=np.zeros(iop.X_temp.shape[0]))
            print("Complete.")
            # Sample predictions for temperature
            ppc_ta = pm.sample_posterior_predictive(
                trace_tmp, samples=n_samples)

        # Reload relative humidity trace with out-of sample data
        print("Loading relative humidity sub-model...", end="", flush=True)
        with open(path.join(
                base_dir, *handyfiles.stored_rh), 'rb') as store_rh:
            trace_rh = pickle.load(store_rh)['trace']
        # Re-estimate humidity sub-model
        with pm.Model() as rh_model_pred:
            print("Re-setting RH sub-model inputs...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            rh_params = pm.Normal(
                'rh_params', 0, 10, shape=(iop.X_hum.shape[1]))
            rh_sd = pm.HalfNormal('rh_sd', 20)
            # Likelihood of temperature estimator
            rh_est = pm.math.dot(iop.X_hum, rh_params)
            # Likelihood of temperature
            rh = pm.Normal(
                'rh', mu=rh_est, sd=rh_sd,
                observed=np.zeros(iop.X_hum.shape[0]))
            print("Complete.")
            # Sample predictions for humidity
            ppc_rh = pm.sample_posterior_predictive(
                trace_rh, samples=n_samples)

        # Reload CO2 trace
        print("Loading CO2 sub-model...", end="", flush=True)
        with open(path.join(
                base_dir, *handyfiles.stored_co2), 'rb') as store_co2:
            trace_co2 = pickle.load(store_co2)['trace']
        # Re-estimate CO2 sub-model
        with pm.Model() as co2_model_pred:
            print("Re-setting CO2 sub-model inputs...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            co2_params = pm.Normal(
                'co2_params', 0, 10, shape=(iop.X_co2.shape[1]))
            co2_sd = pm.HalfNormal('co2_sd', 20)
            # Likelihood of temperature estimator
            co2_est = pm.math.dot(iop.X_co2, co2_params)
            # Likelihood of temperature
            co2 = pm.Normal('co2', mu=co2_est, sd=co2_sd,
                            observed=np.zeros(iop.X_co2.shape[0]))
            print("Complete.")
            # Sample predictions for co2
            ppc_co2 = pm.sample_posterior_predictive(
                trace_co2, samples=n_samples)

        # Reload lighting trace
        print("Loading lighting sub-model...", end="", flush=True)
        with open(path.join(
                base_dir, *handyfiles.stored_lt), 'rb') as store_lt:
            trace_lt = pickle.load(store_lt)['trace']
        # Re-estimate lighting sub-model
        with pm.Model() as lt_model_pred:
            print("Re-setting lighting sub-model inputs...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            lt_params = pm.Normal(
                'lt_params', 0, 10, shape=(iop.X_lt.shape[1]))
            lt_sd = pm.HalfNormal('lt_sd', 20)
            # Likelihood of temperature estimator
            lt_est = pm.math.dot(iop.X_lt, lt_params)
            # Likelihood of temperature
            lt = pm.Normal('lt', mu=lt_est, sd=lt_sd,
                           observed=np.zeros(iop.X_lt.shape[0]))
            print("Complete.")
            # Sample predictions for lighting
            ppc_lt = pm.sample_posterior_predictive(
                trace_lt, samples=n_samples)

        # Reload demand trace
        print("Loading demand sub-model...", end="", flush=True)
        with open(path.join(
                base_dir, *handyfiles.stored_dmd), 'rb') as store_dmd:
            trace_dmd = pickle.load(store_dmd)['trace']
        # Re-estimate demand sub-model
        with pm.Model() as dmd_model_pred:
            print("Re-setting demand sub-model inputs...",
                  end="", flush=True)
            # Set parameter priors (betas, error)
            dmd_params_nc = pm.Normal(
                'dmd_params_nc', 0, 10, shape=(iop.X_dmd_nc.shape[1]))
            dmd_sd = pm.HalfNormal('dmd_sd', 20)
            # Likelihood of temperature estimator
            dmd_est = pm.math.dot(iop.X_dmd_nc, dmd_params_nc)
            # Likelihood of temperature
            dmd = pm.Normal('dmd', mu=dmd_est, sd=dmd_sd,
                            observed=np.zeros(iop.X_dmd_nc.shape[0]))
            print("Complete.")
            # Sample predictions for demand
            ppc_dmd = pm.sample_posterior_predictive(
                trace_dmd, samples=n_samples)

        # Initialize choice logits
        choice_logits = [np.zeros(n_choices) for x in range(n_samples)]
        print("Setting choice probabilities based on sampled input values...",
              end="", flush=True)
        # Loop through each sample of input variables one-by-one
        for s in range(n_samples):
            # Calculate the total cost delta under the currently sampled
            # change in demand, paired with square footage and price per
            # kWh avoided
            cost_delt = ppc_dmd["dmd"][s] * sf * iop.price_delt.eval()
            # Assemble the input variables to the choice models using
            # values for those variables in the current sample
            x_choice = tt.tensor.stack([
                cost_delt, abs(ppc_ta["ta"][s]), ppc_rh["rh"][s],
                ppc_co2["co2"][s], ppc_lt["lt"][s], iop.plug_delta,
                iop.intercept])
            # Loop through all possible choices and calculate logits for
            # each choice given the betas and variable inputs
            for ind in range(n_choices):
                choice_logits[s][ind] = pm.math.dot(
                    x_choice.eval().T[ind], betas_choice.eval()).eval() + \
                    rand_elem.eval()[ind]
        # Use softmax transformation to yield choice probabilities across all
        # samples
        choice_probs = tt.tensor.nnet.softmax(choice_logits)
        print("Complete.")
        # Control choice model Y variable - theano yields a vector where the
        # elements are themselves vectors of length three, with ones in
        # the chosen index elements (e.g., [[0, 1, 0], [0, 0, 1], ...])
        y_choice = th_rng.multinomial(n=1, pvals=choice_probs).eval()
        # Reformat choice tensor such that it is a vector with the choice
        # index (0, 1, 2) in each of the elements (e.g., [1, 2, ...])
        Y_choice = np.zeros(len(y_choice))
        for ind, r in enumerate(y_choice):
            Y_choice[ind] = np.nonzero(r)[0][0]
        # Pull the mode (most frequently sampled choice) from the predicted
        # choice sample
        mode = stats.mode(Y_choice)
        # Print the most frequently sampled choice along with its frequency
        # in the sampled
        print(("Scenario " + str(scn+1) + " choice recommendation: " +
               iop.names[int(mode.mode[0])] + ", " +
              (str((mode.count[0] / n_samples)*100) + "%")))


if __name__ == '__main__':
    # Handle optional user-specified execution arguments
    parser = ArgumentParser()
    # Optional flag to re-estimate the stored model with new data
    # NOTE: currently parameter priors are regenerated to be vague, but
    # eventually the priors will be drawn from the previous posterior estimates
    parser.add_argument("--mod_est", action="store_true",
                        help="Restrict model estimation step")
    # Object to store all user-specified execution arguments
    opts = parser.parse_args()
    base_dir = getcwd()
    main(base_dir)
