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

        # Initializing all state-values V to 0.0
        for state in self.mdp.states():
            self.v[state] = 0.0
        self.v_update_history.append(self.v.copy())

    def init_random_policy(self):
        """Initialize the policy function with equally distributed random probability.

        When n actions are available at state s, the probability of choosing an action should be 1/n.
        """

        # Creating a random policy 'pi' where all actions have equal probability
        for state in self.mdp.states():
            actions = self.mdp.actions(state)
            if not actions:
                self.pi[state] = {}
            else :
                num_actions = len(actions)
                self.pi[state] = dict.fromkeys(actions, 1/num_actions)

                    
    def computeq_fromv(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Given a state-value table, compute the action-state values.
        For deterministic actions, q(s,a) = E[r] + v(s'). Check the lecture slides.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            dict[str,dict[str,float]]: a q value table {state:{action:q-value}}
        """

        # Compute q-values from v-values
        for state in v.keys():
            if state not in self.q:
                self.q[state] = {}
            for action in self.mdp.actions(state):
                self.q[state][action] = 0.0
                for next_state, trans_prob in self.mdp.T(state, action):
                    self.q[state][action] += (trans_prob *
                                              (self.mdp.R(state, action, next_state) +
                                               (self.mdp.gamma * v.get(next_state))))

        return self.q


    def greedy_policy_improvement(self, v: dict[str,float]) -> dict[str,dict[str,float]]:
        """Greedy policy improvement algorithm. Given a state-value table, update the policy pi.

        Args:
            v (dict[str,float]): a state value table (before update) {state:v-value}

        Returns:
            pi (dict[str,dict[str,float]]): a policy table {state:{action:probability}}
        """

        # Compute all q-values for v-values
        self.q = self.computeq_fromv(v)
        for state in v.keys():
            max_q = float('-inf')
            max_q_action = None
            for action in self.mdp.actions(state):

                # Compute the best action i.e., action with maximum q-value
                if self.q[state][action] > max_q:
                    max_q_action = action
                    max_q = self.q[state][action]

            for action in self.mdp.actions(state):
                self.pi[state][action] = 0.0

            # Update policy greedily i.e., max-action probability = 1
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

        # Check for CONVERGENCE: i.e., if the difference between new and old v-values is within threshold(given)
        delta = 0.0
        for state in v.keys():
            delta = max(abs(next_v[state] - v[state]), delta)

        if delta < self.thresh:
            return False #Converged

        return True # not Converged, signal for continuance.


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

        # Evaluate the v-values for current policy-pi
        while True:
            new_v = {}
            for state in self.mdp.states():
                new_v[state] = 0.0
                for action in self.mdp.actions(state):
                    for next_state, trans_prob in self.mdp.T(state, action):
                        new_v[state] += pi[state][action] * (trans_prob * (
                                    self.mdp.R(state, action, next_state) + (self.mdp.gamma * self.v.get(next_state))))

            self.v_update_history.append(new_v.copy()) # Update all v-values history
            converged = not self.check_term(self.v, new_v) # Check for convergence
            self.v = new_v
            if converged:
                break

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
            self.__iter_policy_eval(self.pi)
            old_policy = copy.deepcopy(self.pi) # Save the old policy to later compare with new policy for 'policy stability'
            self.greedy_policy_improvement(self.v)

            if old_policy == self.pi:
                break # The policy is stable

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

        # Combining both steps of policy iteration in value iteration
        while True:

            # Compute v-values with max-action(according to greedy policy) until old and new v-values converge
            self.computeq_fromv(self.v)
            new_v = {}
            for state in self.mdp.states():
                new_v[state] = 0.0
                max_q = float('-inf')
                max_q_action = None

                # Compute maximum q-value
                for action in self.mdp.actions(state):
                    if self.q[state][action] > max_q:
                        max_q_action = action
                        max_q = self.q[state][action]

                new_v[state] = max_q # Here v-value is maximum q-value

            self.v_update_history.append(new_v.copy())

            converged = not self.check_term(self.v, new_v)
            self.v = new_v
            if converged:
                break

        # Update to deterministic policy
        self.greedy_policy_improvement(self.v)
        return self.pi