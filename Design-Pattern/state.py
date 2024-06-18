class State:
    def perform():
        pass

class Machine:
    def __init__(self, init_state):
        self.state = init_state

    def change_state(self, new_state):
        self.state = new_state

    def strategy(self):
        """
        This function is used to implement the implicit 
        transition strategy within state sets
        """
        pass

    def perform(self):
        """
        Perform the current state
        """
        self.state.perform()