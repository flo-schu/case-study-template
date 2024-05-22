from pymob.utils.store_file import prepare_casestudy
from pymob.simulation import SimulationBase

import mod
import prob
import data
import plot

class Simulation(SimulationBase):
    __pymob_version__ = "current_pymob_version"
    mod = mod
    prob = prob
    dat = data
    mplot = plot

    def initialize(self, input):
        pass

    def parameterize(self):
        pass

    def set_coordinates(self):
        pass


if __name__ == "__main__":
    config = prepare_casestudy((
        "CASE_STUDY", 
        "guts_reduced"), 
        "settings.cfg"
    )
    
    sim = Simulation(config=config)

    # run a single simulation
    evaluator = sim.dispatch(theta=sim.model_parameter_dict)
    evaluator()
    evaluator.results

    # run inference
    sim.set_inferer(backend="numpyro")
    sim.prior_predictive_checks()
    sim.inferer.run()
    sim.inferer.store_results()
