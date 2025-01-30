# This Python Package contains code used to modify the base Lunar Lander Environment 
# and assess the performance of the selected algorithms over the introduced changes

# Defining which submodules to import when using from <package> import *
__all__ = ["MyLunarLander", "LunarLanderManager",
           "plotModelsTrainingPerformance", "plotModelSettingsPerformance",
           "plotModelsOverallPerformances", "plotOverallEvaluationResults",
           "plotOverallEvaluationResultsViolinplots", "plotCriticalDifferenceDiagram"]

from .Environment import (MyLunarLander)
from .LunarLanderManager import (LunarLanderManager)
from .DataVisualization import (plotModelsTrainingPerformance, plotModelSettingsPerformance,
                                plotModelsOverallPerformances, plotOverallEvaluationResults,
                                plotOverallEvaluationResultsViolinplots, plotCriticalDifferenceDiagram)