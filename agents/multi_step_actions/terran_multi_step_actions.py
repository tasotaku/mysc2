from pysc2.lib import actions, features, units

def do_nothing(self, obs):
    return actions.RAW_FUNCTIONS.no_op()

