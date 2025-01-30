# This Python Package contains utility code used in various miscellaneous tasks throughout the project

# Defining which submodules to import when using from <package> import *
__all__ = ["CONFIG", "PATHS_CONFIG",
           "PPO_SETTINGS_1", "PPO_SETTINGS_2",
           "DQN_SETTING_1", "DQN_SETTING_2",
           "saveObject", "loadObject"]

from .Configuration import (CONFIG, PATHS_CONFIG,
                            PPO_SETTINGS_1, PPO_SETTINGS_2,
                            DQN_SETTING_1, DQN_SETTING_2)

from .pickleFileManagement import (saveObject, loadObject)