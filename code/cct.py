import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


# Part 1:
#   Load the "plant knowledgeLinks to an external site." dataset. 
#   Represent it appropriately (e.g., as a NumPy array or Pandas DataFrame, excluding the Informant ID column).

def load_data():
    """custom function that loads the datsa and returns it"""
    # chatGBT was used to help write the following code to read and return the data within the csv file
    DATA_URL = "https://raw.githubusercontent.com/joachimvandekerckhove/cogs107s25/refs/heads/main/1-mpt/data/plant_knowledge.csv"
    df = pd.read_csv(DATA_URL, sep=',', skipinitialspace=True, header=0, engine='python')
    df = df.drop(columns=['Informant_ID'], errors = "ignore")
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.to_numpy(dtype=int)

# Part 2:
#   Implement the Model in PyMC

#   For each informant's competence Di, choose a suitable prior distribution.
#   Make sure to justify your choice in the report!!

#   Parameters:
#     N: number of informants.
#     M: number of items (questions).
#     Xij: response of informant i to item j (0 or 1).
#     Zj: latent "consensus" or "correct" answer for item j (0 or 1).
#     Di: latent "competence" of informant i (probability of knowing the correct answer), where 0.5 ≤ Di ≤ 1.

def fit_model(data):
    with pm.Model() as model:
        rows,cols = data.shape
        # the above ".shape" method and the "shape" parameter below were found using chatGBT
        
        # Define Priors:
        D = pm.Uniform("D", 0.5, 1, shape = rows)
        # chose a uniform prior distribution because we don't know anything about their knowledge and how competent each informant will be. all we know is that their probability ahs to be between 0.5(completely guessing) or 1(always knowing the correct answer)
        Z = pm.Bernoulli("Z", p=0.5, shape = cols)

        # Define the probability pij using the given formula:
        # (first reshape or broadcast D and Z appropriately to calculate p for all i and j.)
        D_reshaped = D[:, None] 
        Prob = Z * D_reshaped + (1-Z) * (1-D_reshaped)

        # link observed data X to the calculated probability p to define the likelihood (use pm.Dernoulli)
        pm.Bernoulli("likelihood", p=Prob, observed=data)
        
        # perform inference:
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, idata_kwargs={"log_likelihood": True})

    return trace

# Part 3:
#   Analyze Results
#   the following three functions were written and debugged using chatGBT

def analyze_results(trace):
    print("\nModel summary for competence (D):")
    print(az.summary(trace, var_names=["D"]))

    print("\nModel summary for consensus answers (Z):")
    print(az.summary(trace, var_names=["Z"]))

    # Plot posterior distributions for competence
    az.plot_posterior(trace, var_names=["D"])
    plt.tight_layout()
    plt.show()

    # Plot posterior distributions for consensus answers
    az.plot_posterior(trace, var_names=["Z"])
    plt.tight_layout()
    plt.show()

def compute_majority_vote(data):
    """
    Compute majority vote answer for each question (column).
    """
    return (np.mean(data, axis=0) > 0.5).astype(int)

def compare_with_majority(data, trace):
    """
    Compare model-estimated consensus answers with majority vote answers.
    """
    posterior_Z = trace.posterior["Z"].mean(dim=("chain", "draw")).values
    consensus_from_model = (posterior_Z > 0.5).astype(int)
    majority_vote = compute_majority_vote(data)

    print("\nConsensus vs Majority Vote (question-wise):")
    for i, (c_model, c_majority) in enumerate(zip(consensus_from_model, majority_vote)):
        print(f"Question {i+1:2d}: Model = {c_model}, Majority = {c_majority}")


###############################################################################
### --- Main execution block ---
###############################################################################

if __name__ == "__main__":
    data = load_data()
    print("> Fitting model...")
    results = fit_model(data)

    analyze_results(results)
    compare_with_majority(data, results)