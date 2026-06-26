import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig, DiscreteSACConfig, DiscreteBCQConfig
from itertools import product


def compute_reward(row):
    comfort_penalty = 0
    energy_penalty = 0
    co2_penalty = 0

    # Comfort penalty based on temperature deviation from setpoint
    temp_deviation = abs(row['current_temperature'] - row['AHU_TMSET'])
    comfort_penalty += temp_deviation * 10  # weight for comfort

    # Energy penalty based on OBDM and FCU usage
    capacity = np.abs(row['AHU_SDT'] - row['AHU_MDT'])
    energy_penalty += capacity * 10  # weight for OBDM
    
    # co2 penalty
    co2_level = row['current_co2']
    if co2_level > 800:
        co2_penalty += (co2_level - 800) * 0.05

    total_penalty = comfort_penalty + energy_penalty + co2_penalty
    reward = -total_penalty  # negative penalty as reward

    return reward

def set_actions(df:pd.DataFrame):
    ahu = df.copy()

    ahu['AHU_TMSET'] = (ahu['AHU_TMSET'].round()).fillna(method='ffill')
    ahu['SET_ACT'] = ahu['AHU_TMSET'].copy() 

    # AHU_OBDM needs to be rounded to have either of 6 values, pairing with 0%, 20% ... 100%
    ahu['AHU_OBDM'] = (ahu['AHU_OBDM']/100*5).round()/5

    # If mode is 4 then set to 1, else 0
    ahu['MODE_ACT'] = ahu['AHU_MODE'].apply(lambda it: 1 if it==4 else 0)

    # for all FCU, set 1 or 0 by majority vite, single value in output
    fcu_cols = [f'FCU_{i}'for i in range(1,12)]
    ahu['FCU_ACT'] = ahu[fcu_cols].sum(axis=1)
    ahu['FCU_ACT'] = ahu['FCU_ACT'].apply(lambda x: 1 if x >= 6 else 0)

    # combine and change to single action space
    ahu['ACTION'] = list(zip(ahu['AHU_TMSET'], ahu['AHU_OBDM'], ahu['MODE_ACT'], ahu['FCU_ACT']))
    
    # all permuations of possible values for 4 lists
    tmset_values = sorted(ahu['AHU_TMSET'].unique())
    obdm_values = sorted(ahu['AHU_OBDM'].unique())
    mode_values = sorted(ahu['MODE_ACT'].unique())
    fcu_values = sorted(ahu['FCU_ACT'].unique())

    print(tmset_values, obdm_values, mode_values, fcu_values)

    possible_actions = list(product(tmset_values, obdm_values, mode_values, fcu_values))
    action_mapping = {action: idx for idx, action in enumerate(possible_actions)}
    
    ahu['ACTION_CODE'] = ahu['ACTION'].map(action_mapping)

    return ahu['ACTION_CODE'].to_numpy(), action_mapping


def set_observations(df:pd.DataFrame):
    ahu = df.copy()

    current_temp_scaler = StandardScaler()
    ahu['current_temperature_t'] = current_temp_scaler.fit_transform(np.log(ahu[['current_temperature']]))

    current_co2_scaler = StandardScaler()
    ahu['current_co2_t'] = current_co2_scaler.fit_transform(np.log(ahu[['current_co2']]))

    ahu['current_humidity_t'] = (ahu[['current_humidity']]**3)/1000000

    mdt_scaler = StandardScaler()
    ahu['AHU_MDT_t'] = mdt_scaler.fit_transform(ahu[['AHU_MDT']]**2)

    rdt_scaler = StandardScaler()
    ahu['AHU_RDT_t'] = rdt_scaler.fit_transform(np.log(ahu[['AHU_RDT']]))

    sdt_scaler = StandardScaler()
    ahu['AHU_SDT_t'] = sdt_scaler.fit_transform(ahu[['AHU_SDT']])

    out_temp_scaler = StandardScaler()
    ahu['temperature_out_t'] = out_temp_scaler.fit_transform((ahu[['temperature_out']]**1.5))

    inertia_scaler = StandardScaler()
    ahu['inertia_t'] = inertia_scaler.fit_transform((ahu[['inertia']]**(1/6)).fillna(0))

    ratio_scaler = StandardScaler()
    ahu['temp_ratio_t'] = ratio_scaler.fit_transform((ahu[['temp_ratio']]**3).fillna(0))

    tset_scaler = StandardScaler()
    ahu['temp_request'] = tset_scaler.fit_transform(ahu[['AHU_TMSET']].fillna(0))

    observation = ahu[['temp_request','current_temperature_t','current_co2_t','current_humidity_t','AHU_MDT_t','AHU_RDT_t','AHU_SDT_t',
                   'temperature_out_t','inertia_t','temp_ratio_t','month_sin','hour_sin','is_weekend','is_working_hour']]
    
    # group all scalers in dict
    scalers = {
        'current_temp_scaler': current_temp_scaler,
        'current_co2_scaler': current_co2_scaler,
        'mdt_scaler': mdt_scaler,
        'rdt_scaler': rdt_scaler,
        'sdt_scaler': sdt_scaler,
        'out_temp_scaler': out_temp_scaler,
        'inertia_scaler': inertia_scaler,
        'ratio_scaler': ratio_scaler,
        'tset_scaler': tset_scaler
    }

    return observation.fillna(method='bfill').to_numpy(), scalers


def set_terminals(df:pd.DataFrame):
    ahu = df.copy()
    ahu['date'] = ahu['timestamp'].dt.date
    ahu['terminal'] = ahu['date'] != ahu['date'].shift(-1)
    ahu['terminal'] = ahu['terminal'].astype(int)
    return ahu['terminal'].to_numpy()

def setup_ahu(id:int, df:pd.DataFrame):
    ahu = df[(df['ahu_id'] == id) & (df['AHU_MODE']!=2)].copy()
    ahu[[f'FCU_{i}'for i in range(1,12)]] = ahu[[f'FCU_{i}'for i in range(1,12)]].applymap(lambda it: 1 if it=='active' else 0)
    ahu['AHU_OBDM'].fillna(0,inplace=True)
    ahu['AHU_RUN'] = ahu['AHU_RUN'].apply(lambda it: 1 if it=='active' else 0)
    ahu['CHU_SS'] = ahu['CHU_SS'].apply(lambda it: 1 if it=='active' else 0)

    ahu.drop(columns=['floor_id','hvac_zone','floor', 'corner','ahu_id'],inplace=True)

    # ADD NEW FEATURES HERE
    ahum = ahu.groupby('timestamp').mean().reset_index()

    ahum['temp_ratio'] = ahum['AHU_SDT'] / ahum['AHU_RDT']
    ahum['delta_RDT'] = ahum['AHU_RDT'].diff().abs()
    ahum['delta_SDT'] = ahum['AHU_SDT'].diff().abs()
    ahum['inertia'] = (ahum['delta_SDT'] - ahum['delta_RDT']).abs()

    # get weekday, decide if it's weekend or not
    ahum['timestamp'] = pd.to_datetime(ahum['timestamp'])
    ahum['weekday'] = ahum['timestamp'].dt.weekday
    ahum['is_weekend'] = ahum['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    # get hour
    ahum['hour'] = ahum['timestamp'].dt.hour
    ahum['hour_sin'] = -np.cos(2 * np.pi * ahum['hour'] / 24)

    # decide if it's working hours or not
    ahum['is_working_hour'] = ahum['hour'].apply(lambda x: 1 if 8 <= x <= 18 else 0)

    ahum['month'] = ahum['timestamp'].dt.month
    ahum['month_sin'] = np.sin(2 * np.pi * ahum['month'] / 12)

    ahum['reward'] = ahum.apply(compute_reward, axis=1).fillna(method='bfill')
    
    return ahum


def setup_dataset(df:pd.DataFrame):
    ids = ['AHU1', 'AHU2', 'AHU3', 'AHU4', 'AHU5','AHU6','AHU7','AHU8']

    # setup ahu df and stack them
    ahu_list = [setup_ahu(id, df.copy()) for id in ids]
    
    ahum = pd.concat(ahu_list, ignore_index=True)

    reward_scaler = StandardScaler()
    reward_t = reward_scaler.fit_transform(ahum['reward'].to_numpy().reshape(-1, 1)).flatten()

    actions, action_mapping = set_actions(ahum)
    observations, scalers = set_observations(ahum)
    terminals = set_terminals(ahum)

    # check for nans
    assert not np.isnan(observations).any(), "Observations contain NaN values"
    assert not np.isnan(actions).any(), "Actions contain NaN values"
    assert not np.isnan(reward_t).any(), "Rewards contain NaN values"

    dataset = MDPDataset(observations, actions, reward_t, terminals=terminals)

    return dataset, action_mapping, scalers

if __name__ == "__main__":
    df = pd.read_csv('./data_complete.csv')
    dataset, action_mapping, scalers = setup_dataset(df)
    print("Dataset prepared with the following action mapping:")
    print(dataset.dataset_info, dataset.transition_count)

    # save scalers and mappings
    import pickle
    with open('scalers_action_mapping.pkl', 'wb') as f:
        pickle.dump({'scalers': scalers, 'action_mapping': action_mapping}, f)
    
    #sac = DiscreteSACConfig().create()
    bcq = DiscreteBCQConfig().create()
    cql = DiscreteCQLConfig().create()

    #sac.fit(dataset, n_steps=20000)
    bcq.fit(dataset, n_steps=20000)
    cql.fit(dataset, n_steps=20000)

    #sac.save_model('./sac_ahu_model')
    bcq.save('./bcq_ahu_model.d3')
    cql.save('./cql_ahu_model.d3')

    print("Models trained and saved.")

    # Test minimal inference with custom input
    test_observation = {
        'temp_request': 22.0,
        'current_temperature_t': np.log(25.0),
        'current_co2_t': np.log(1000.0),
        'current_humidity_t': (60.0**3)/1000000,
        'AHU_MDT_t': 24.0**2,
        'AHU_RDT_t': np.log(25.0),
        'AHU_SDT_t': 22.0,
        'temperature_out_t': 30.0**1.5,
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
    bcq_action = bcq.predict(test_obs_array)
    cql_action = cql.predict(test_obs_array)
    
    # print original action from mapping
    inv_action_mapping = {v: k for k, v in action_mapping.items()}
    #print("SAC Action:", inv_action_mapping[sac_action[0]])
    print("T_SET, OBDM, FAN?, FCU ON?")
    print("BCQ Action:", inv_action_mapping[bcq_action[0]])
    print("CQL Action:", inv_action_mapping[cql_action[0]])