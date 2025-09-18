from mypa2_dp import MDP, VIAgent, PIAgent

def savelog(hisotry, method_name: str, step: int=100):
    with open(f'log_{method_name}.txt', 'w') as f:
        for i, h in enumerate(hisotry):
            if i % step == 0:
                f.write(f'{h}\n')

mdp = MDP("./mdp1.json")
via = VIAgent(mdp)
vi_p = via.value_iteration()
savelog(via.v_update_history, 'VI')
print(f'Value iteration V: {via.v_update_history[-1]}')
print(f'Value iteration pi: {vi_p}')

pia = PIAgent(mdp)
pi_p = pia.policy_iteration()
savelog(pia.v_update_history, 'PI')
print(f'Policy iteration V: {pia.v_update_history[-1]}')
print(f'Policy iteration pi: {pi_p}')