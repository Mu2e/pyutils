import matplotlib.pyplot as plt
import numpy as np
import zfit

from zfit.loss import UnbinnedNLL
from zfit.minimize import Minuit
from zfit import ztypes
from hepstats.hypotests import Discovery, UpperLimit
from hepstats.hypotests.calculators import AsymptoticCalculator, FrequentistCalculator
from hepstats.hypotests.parameters import POI, POIarray

def plotlimit(ul, alpha=0.05, CLs=True, ax=None):
    """
    plot pvalue scan for different values of a parameter of interest (observed, expected and +/- sigma bands)

    Args:
        ul: UpperLimit instance
        alpha (float, default=0.05): significance level
        CLs (bool, optional): if `True` uses pvalues as $$p_{cls}=p_{null}/p_{alt}=p_{clsb}/p_{clb}$$
            else as $$p_{clsb} = p_{null}$
        ax (matplotlib axis, optionnal)
        
    FIXME: this is taken from the hepstats/utils.py script

    """
    if ax is None:
        ax = plt.gca()

    poivalues = ul.poinull.values
    pvalues = ul.pvalues(CLs=CLs)

    if CLs:
        cls_clr = "r"
        clsb_clr = "b"
    else:
        cls_clr = "b"
        clsb_clr = "r"

    color_1sigma = "mediumseagreen"
    color_2sigma = "gold"

    ax.plot(
        poivalues,
        pvalues["cls"],
        label="Observed CL$_{s}$",
        marker=".",
        color="k",
        markerfacecolor=cls_clr,
        markeredgecolor=cls_clr,
        linewidth=2.0,
        ms=11,
    )

    ax.plot(
        poivalues,
        pvalues["clsb"],
        label="Observed CL$_{s+b}$",
        marker=".",
        color="k",
        markerfacecolor=clsb_clr,
        markeredgecolor=clsb_clr,
        linewidth=2.0,
        ms=11,
        linestyle=":",
    )

    ax.plot(
        poivalues,
        pvalues["clb"],
        label="Observed CL$_{b}$",
        marker=".",
        color="k",
        markerfacecolor="k",
        markeredgecolor="k",
        linewidth=2.0,
        ms=11,
    )

    ax.plot(
        poivalues,
        pvalues["expected"],
        label="Expected CL$_{s}-$Median",
        color="k",
        linestyle="--",
        linewidth=1.5,
        ms=10,
    )

    ax.plot(
        [poivalues[0], poivalues[-1]],
        [alpha, alpha],
        color="r",
        linestyle="-",
        linewidth=1.5,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected"],
        pvalues["expected_p1"],
        facecolor=color_1sigma,
        label="Expected CL$_{s} \\pm 1 \\sigma$",
        alpha=0.8,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected"],
        pvalues["expected_m1"],
        facecolor=color_1sigma,
        alpha=0.8,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected_p1"],
        pvalues["expected_p2"],
        facecolor=color_2sigma,
        label="Expected CL$_{s} \\pm 2 \\sigma$",
        alpha=0.8,
    )

    ax.fill_between(
        poivalues,
        pvalues["expected_m1"],
        pvalues["expected_m2"],
        facecolor=color_2sigma,
        alpha=0.8,
    )

    ax.set_ylim(-0.01, 1.1)
    ax.set_ylabel("p-value")
    ax.set_xlabel("parameter of interest")
    ax.legend(loc="best", fontsize=14)

    return ax
def analyze_results(input_nsig, input_nbkg):
  """
  Analyzes results assuming simple Poisson stats

    Args:
        input_nsig (float): The observed number of signal events.
        input_nbkg (float): The observed number of background events.
  
  """

  Nsig = zfit.Parameter("Nsig", 0, 0, 20)
  Nbkg = zfit.Parameter("Nbkg", 5, 0, 10)
  Nobs = zfit.ComposedParameter("Nobs", lambda a, b: a + b, params=[Nsig, Nbkg])

  obs = zfit.Space("N", limits=(0, 30))
  model = zfit.pdf.Poisson(obs=obs, lamb=Nobs)

  n = input_nsig + input_nbkg
  nbkg = input_nbkg # will go infinite for 0 backgrounds

  data = zfit.data.Data.from_numpy(obs=obs, array=np.array([n]))
  Nbkg.set_value(nbkg)
  Nbkg.floating = False

  # likelihood function function
  nll = UnbinnedNLL(model=model, data=data)

  # Instantiate a minuit minimizer
  minimizer = Minuit(verbosity=0)

  # minimisation of the loss function
  minimum = minimizer.minimize(loss=nll)

  # instantation of the calculator
  calculator = AsymptoticCalculator(nll, minimizer)
  calculator.bestfit = minimum  # optionnal

  discovery_test = Discovery(calculator, POI(Nsig, 0))
  pnull, significance = discovery_test.result()
  print("result has ", significance, "sigma significance") #Will be inf if no backgrounds

  # parameter of interest to scan
  poi_scan = POIarray(Nsig, np.linspace(0.0, 20, 5))
  # parameter of interest set at the background only hypothesi
  poi_bkg_only = POI(Nsig, 0)

  # TODO optional limit od discovery
  # instantation of the discovery test
  ul = UpperLimit(calculator, poi_scan, poi_bkg_only)
  limits = ul.upperlimit(alpha=0.05, CLs=False)

  f = plt.figure(figsize=(9, 8))
  plotlimit(ul, alpha=0.05, CLs=False)
  plt.xlabel("Nsig");
  
  plt.show()
  print('observed', limits['observed'], 'expected (null)', limits['expected'])


def anlayze_result_with_err(input_nsig, input_nbkg, mom_uncertainty=0.1 ):

    """
    Analyzes results including a example systematic uncertainty.

    Args:
        input_nsig (float): The observed number of signal events.
        input_nbkg (float): The observed number of background events.
        mom_uncertainty (float): The fractional systematic uncertainty on the background.
    """

    # Define parameters using zfit
    Nsig_nominal = zfit.Parameter("Nsig_nominal", 0, 0, 20)
    Nbkg_nominal = zfit.Parameter("Nbkg_nominal", 5, 0, 10)
    
    # Define a nuisance parameter for the systematic uncertainty
    mom_nuisance = zfit.Parameter("mom_nuisance", 0.0, -5, 5)

    # Add a Gaussian constraint on the nuisance parameter
    mom_constraint = zfit.constraint.GaussianConstraint(
        params=mom_nuisance,
        observation=0.0, #ztypes.py_float(0.0),
        sigma=1.0#ztypes.py_float(1.0)
    )

    # Define the *actual* expected number of background events as a function of the nuisance parameter
    # e.g., a 10% uncertainty is applied to the background count.
    Nbkg_unc = zfit.ComposedParameter(
        f"Nbkg_unc({Nbkg_nominal},{mom_nuisance})",
        lambda nbkg_nom, mom_nuis: nbkg_nom * (1 + mom_uncertainty * mom_nuis),
        params={Nbkg_nominal, mom_nuisance}
    )

    # Combined expected number of events
    Nobs = zfit.ComposedParameter("Nobs", lambda a, b: a + b, params=[Nsig_nominal, Nbkg_unc])

    # Model and data setup
    obs = zfit.Space("N", limits=(0, 30))
    model = zfit.pdf.Poisson(obs=obs, lamb=Nobs)

    n = input_nsig + input_nbkg
    
    # We still need to set the value of the background for the background-only hypothesis.
    # The systematic uncertainty will be constrained during the fit.
    Nbkg_nominal.set_value(input_nbkg)
    Nbkg_nominal.floating = False

    data = zfit.data.Data.from_numpy(obs=obs, array=np.array([n]))

    # likelihood function
    nll = UnbinnedNLL(model=model, data=data, constraints=mom_constraint)

    # Instantiate a minuit minimizer
    minimizer = Minuit(verbosity=0)

    # minimisation of the loss function
    minimum = minimizer.minimize(loss=nll)
    
    # Instantiate the calculator, passing the constraints
    calculator = AsymptoticCalculator(nll, minimizer)
    calculator.bestfit = minimum

    # --- Discovery Test with new hepstats API ---
    discovery_test = Discovery(calculator, POI(Nsig_nominal, 0))
    pnull, significance = discovery_test.result()
    print("result has ", significance, "sigma significance") #Will be inf if no backgrounds

    # parameter of interest to scan
    poi_scan = POIarray(Nsig_nominal, np.linspace(0.0, 20, 5))
    poi_bkg_only = POI(Nsig_nominal, 0)

    # instantation of the discovery test
    ul = UpperLimit(calculator, poi_scan, poi_bkg_only)
    limits = ul.upperlimit(alpha=0.05, CLs=False)

    f = plt.figure(figsize=(9, 8))
    plotlimit(ul, alpha=0.05, CLs=False)
    plt.xlabel("Nsig")

    plt.show()
    print('observed', limits['observed'], 'expected (null)', limits['expected'])





