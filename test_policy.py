import numpy as np
from d3rlpy.algos import DiscreteBCQConfig, DiscreteCQLConfig
import pickle
import d3rlpy

# with open('scalers_action_mapping.pkl', 'wb') as f:
#     pickle.dump({'scalers': scalers, 'action_mapping': action_mapping}, f)

# load scalers and mapping
with open('scalers_action_mapping.pkl', 'rb') as f:
    data = pickle.load(f)
scalers = data['scalers']
action_mapping = data['action_mapping']


# load policy
loaded_bcq = d3rlpy.load_learnable('./bcq_ahu_model.d3')
loaded_cql = d3rlpy.load_learnable('./cql_ahu_model.d3')

# Test minimal inference with custom input
test_observation = {
    'temp_request': 22.0,
    'current_temperature_t': np.log(25.0),
    'current_co2_t': np.log(1500.0),
    'current_humidity_t': (60.0**3)/1000000,
    'AHU_MDT_t': 25.0**2,
    'AHU_RDT_t': np.log(25.0),
    'AHU_SDT_t': 25.0,
    'temperature_out_t': 34.0**1.5,
    'inertia_t': 0.0**(1/6),
    'temp_ratio_t': 0.9**3,
    'month': np.sin(2 * np.pi * 8 / 12),
    'hour': -np.cos(2 * np.pi * 13 / 24),
    'is_weekend': 0,
    'is_working_hour': 1
}

# scale test observation
test_obs_array = np.array([[
    scalers['tset_scaler'].transform([[test_observation['temp_request']]])[0][0],
    scalers['current_temp_scaler'].transform([[test_observation['current_temperature_t']]])[0][0],
    scalers['current_co2_scaler'].transform([[test_observation['current_co2_t']]])[0][0],
    test_observation['current_humidity_t'],
    scalers['mdt_scaler'].transform([[test_observation['AHU_MDT_t']]])[0][0],
    scalers['rdt_scaler'].transform([[test_observation['AHU_RDT_t']]])[0][0],
    scalers['sdt_scaler'].transform([[test_observation['AHU_SDT_t']]])[0][0],
    scalers['out_temp_scaler'].transform([[test_observation['temperature_out_t']]])[0][0],
    scalers['inertia_scaler'].transform([[test_observation['inertia_t']]])[0][0],
    scalers['ratio_scaler'].transform([[test_observation['temp_ratio_t']]])[0][0],
    test_observation['month'],
    test_observation['hour'],
    test_observation['is_weekend'],
    test_observation['is_working_hour']
]])

#sac_action = sac.predict(test_obs_array)
bcq_action = loaded_bcq.predict(test_obs_array)
cql_action = loaded_cql.predict(test_obs_array)

# print original action from mapping
inv_action_mapping = {v: k for k, v in action_mapping.items()}
#print("SAC Action:", inv_action_mapping[sac_action[0]])
print("T_SET, OBDM, FAN?, FCU ON?")
print("BCQ Action:", inv_action_mapping[bcq_action[0]])
print("CQL Action:", inv_action_mapping[cql_action[0]])