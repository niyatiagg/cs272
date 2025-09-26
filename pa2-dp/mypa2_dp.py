from mymdp import MDP
import math
import copy

class ValueAgent:
    """Value-based Agent template (Used as a parent class for VIAgent and PIAgent)
    An agent should maintain:
    - q table (dict[state,dict[action,q-value]])
    - v table (dict[state,v-value])
    - policy table (dict[state,dict[action,probability]])
    - mdp (An MDP instance)
    - v_update_history (list of the v tables): [Grading purpose only] Every time when you update the v table, you need to append the v table to this list. (include the initial value)
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization

        Args:
            mdp (MDP): An MDP instance
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.            
        """        
        self.q = dict()
        self.v = dict()
        self.pi = dict()
        self.mdp = mdp
        self.thresh = conv_thresh
        self.v_update_history = list()

        for state in self.mdp.states():
            self.v[state] = 0.0

        self.v_update_history.append(self.v.copy())

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """

        for state in self.mdp.states():
            num_actions = len(self.mdp.actions(state))
            self.pi[state] = dict.fromkeys(self.mdp.actions(state), 1/num_actions)

                    
    def computeq_fromv(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """

        for state in v.keys():
            total = 0
            if state not in self.q:
                self.q[state] = {}
            for action in self.mdp.actions(state):
                for next_state, trans_prob in self.mdp.T(state, action):
                    total += (trans_prob*(self.mdp.R(state, action, next_state) + (self.mdp.gamma * v.get(next_state))))
                self.q[state][action] = total

        return self.q


    def greedy_policy_improvement(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        self.q = self.computeq_fromv(v)
        max_q = float('-inf')

        max_q_action = None
        for state in v.keys():
            for action in self.mdp.actions(state):
                if self.q[state][action] > max_q:
                    max_q_action = action
                    max_q = self.q[state][action]

            for action in self.mdp.actions(state):
                self.pi[state][action] = 0.0

            self.pi[state][max_q_action] = 1.0

        return self.pi

    def check_term(self, v: dict[str,float], next_v: dict[str,float]) -> bool:
        """Return True if the state value has NOT converged.
        Convergence here is defined as follows: 
        For ANY state s, the update delta, abs(v'(s) - v(s)), is within the threshold (self.thresh).

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}
            next_v (dict[str,float]): a state value table (after update)

        Returns:
            bool: True if continue; False if converged
        """
        delta = 0.0
        for state in v.keys():
            delta = max(abs(next_v[state] - v[state]), delta)

        if delta < self.thresh:
            return False

        return True


class PIAgent(ValueAgent):
    """Policy Iteration Agent class
    """    
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def __iter_policy_eval(self, pi: dict[str,dict[str,float]]) -> dict[str,float]:
        """Iterative policy evaluation algorithm. Given a policy pi, evaluate the value of states (v).

        This function should be called in policy_iteration().

        Args:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}

        Returns:
            dict[str,float]: state-value table {state:v-value}
        """
        while True:
            new_v = {}
            for state in self.mdp.states():
                new_v[state] = 0.0
                total = 0
                for action in self.mdp.actions(state):
                    for next_state, trans_prob in self.mdp.T(state, action):
                        total += pi[state][action] * (trans_prob * (
                                    self.mdp.R(state, action, next_state) + (self.mdp.gamma * self.v.get(next_state))))

                new_v[state] = total
            self.v_update_history.append(new_v.copy())

            if not self.check_term(self.v, new_v):
                break

            self.v = new_v

        return self.v


    def policy_iteration(self) -> dict[str,dict[str,float]]:
        """Policy iteration algorithm. Iterating iter_policy_eval and greedy_policy_improvement, update the policy pi until convergence of the state-value function.

        You must use:
         - __iter_policy_eval
         - greedy_policy_improvement        

        This function is called to run PI. 
        e.g.
        mdp = MDP("./mdp1.json")
        dpa = PIAgent(mdp)
        dpa.policy_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        while True:
            is_policy_stable = False

            self.__iter_policy_eval(self.pi)
            old_policy = self.pi
            self.greedy_policy_improvement(self.v)

            if old_policy == self.pi:
                is_policy_stable = True

            if is_policy_stable:
                break

        return self.pi

class VIAgent(ValueAgent):
    """Value Iteration Agent class
    """
    def __init__(self, mdp: MDP, conv_thresh: float=0.000001) -> None:
        """Initialization (Use the functions from the parent class)
        - set up values for member variables
        - init the policy to the random policy     

        Args:
            mdp (MDP): An MDP
            conv_thresh (float, optional): a threshold for convergence approximation. Defaults to 0.000001.
        """        
        super().__init__(mdp, conv_thresh)
        super().init_random_policy() # initialize its policy function with the random policy

    def value_iteration(self) -> dict[str,dict[str,float]]:
        """Value iteration algorithm. Compute the optimal v values using the value iteration. After that, generate the corresponding optimal policy pi.

        You must use:
         - greedy_policy_improvement           

        This function is called to run VI. 
        e.g.
        mdp = MDP("./mdp1.json")
        via = VIAgent(mdp)
        via.value_iteration()

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        while True:
            self.computeq_fromv(self.v)
            new_v = {}
            for state in self.mdp.states():
                new_v[state] = 0.0
                total = 0
                max_q = float('-inf')
                max_q_action = None
                for action in self.mdp.actions(state):
                    if self.q[state][action] > max_q:
                        max_q_action = action
                        max_q = self.q[state][action]


                for next_state, trans_prob in self.mdp.T(state, max_q_action):
                    total += self.pi[state][max_q_action] * (trans_prob * (
                                self.mdp.R(state, max_q_action, next_state) + (self.mdp.gamma * self.v.get(next_state))))

                new_v[state] = total
            self.v_update_history.append(new_v.copy())

            if not self.check_term(self.v, new_v):
                break

            self.v = new_v

        self.greedy_policy_improvement(self.v)
        return self.pi