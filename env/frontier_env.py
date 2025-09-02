import os
import numpy as np
import pyfmi
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import copy

FMU_PATH = os.path.dirname(os.path.abspath(__file__)) + "/LC_Frontier_5Cabinet_4_17_25.fmu"
EXOGENOUS_VAR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_04-07-24.csv')

TEMP_MIN_K = 293.15  # 20Â°C
TEMP_MAX_K = 313.15  # 40Â°C
KELVIN_TO_CELSIUS = 273.15

# Add these utility functions:
def kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - KELVIN_TO_CELSIUS

def celsius_to_kelvin(temp_c):
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + KELVIN_TO_CELSIUS

def scale_temperature_goal(norm_goal):
    """Scale normalized goal (-1 to 1) to temperature range (TEMP_MIN_K to TEMP_MAX_K)."""
    return ((norm_goal + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K

class exogenous_variable_generator:
    def __init__(self, path, Towb_offset_in_K=10, nCDUs=5, nBranches=3, parallel_nCabinets=5):
        self.exogenous_var = pd.read_csv(path)
        n = 5
        for col in self.exogenous_var.columns[1:1+n]:
            mean = self.exogenous_var[col].mean()
            std = self.exogenous_var[col].std()
            upper_limit = mean + 0.1 * std
            lower_limit = mean - 1.75 * std
            self.exogenous_var[col] = self.exogenous_var[col].clip(lower=lower_limit, upper=upper_limit)
        self.exogenous_var.iloc[:,-1] += 273.15 + Towb_offset_in_K
        
        self.exogenous_var = self.exogenous_var.to_numpy()
        
        Q_flow_totals = self.exogenous_var[:,1:1+nCDUs]/parallel_nCabinets
        Q_flow_totals /= nBranches
        Q_flow_totals = Q_flow_totals.repeat(nBranches, axis=1).round(2)
        columns_to_roll_dict = {1:1800, 2:3600, 4:1800, 5:3600, 7:1800, 8:3600, 10:1800, 11:3600, 13:1800, 14:3600}
        for col, roll in columns_to_roll_dict.items():
            Q_flow_totals[:,col] = np.roll(Q_flow_totals[:,col], roll, axis=0)
        self.exogenous_var_final = np.concatenate([Q_flow_totals, self.exogenous_var[:,-1].reshape(-1,1)], axis=1)
        
    def iterate_cyclically(self):
        while True:
            for row in self.exogenous_var_final:
                yield row

class SmallFrontierModel(gym.Env):
    CT_nominal_power_per_cell = 0.55 * 149140
    Towb_offset_in_K = 15.0
    variable_ranges = {
        'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_CT_RL_stpt': [298.15, 313.15],
        'cabinet_temperature_K': [273.15, 373.15],
        'valve_flow_rate': [0.0, 12.0],
        'simulator[1].datacenter[1].computeBlock[1].cdu[1].CabRet_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[2].cdu[1].CabRet_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[3].cdu[1].CabRet_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[4].cdu[1].CabRet_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[5].cdu[1].CabRet_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[1].cdu[1].CabSup_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[2].cdu[1].CabSup_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[3].cdu[1].CabSup_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[4].cdu[1].CabSup_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[5].cdu[1].CabSup_pT.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[1].cdu[1].valveCDU.m_flow': [0.0, 12.0],
        'simulator[1].datacenter[1].computeBlock[2].cdu[1].valveCDU.m_flow': [0.0, 12.0],
        'simulator[1].datacenter[1].computeBlock[3].cdu[1].valveCDU.m_flow': [0.0, 12.0],
        'simulator[1].datacenter[1].computeBlock[4].cdu[1].valveCDU.m_flow': [0.0, 12.0],
        'simulator[1].datacenter[1].computeBlock[5].cdu[1].valveCDU.m_flow': [0.0, 12.0],
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].CT.PFan': [0.0, CT_nominal_power_per_cell],
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[2].CT.PFan': [0.0, CT_nominal_power_per_cell],
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[3].CT.PFan': [0.0, CT_nominal_power_per_cell],
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[4].CT.PFan': [0.0, CT_nominal_power_per_cell],
        'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_1.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_1.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_1.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_1.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_1.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_2.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_2.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_2.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_2.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_2.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_3.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_3.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_3.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_3.port.T': [273.15, 373.15],
        'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_3.port.T': [273.15, 373.15],
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].controls.p_CTWR_Setpoint_Model.T_CT_setpoint': [280.15, 305.15],
        'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].waterSPTLvg': [273.15, 373.15],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Q_flow_total': [0.0, 1e6],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Q_flow_total': [0.0, 1e6],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Q_flow_total': [0.0, 1e6],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Q_flow_total': [0.0, 1e6],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Q_flow_total': [0.0, 1e6],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade1': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade1': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade1': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade1': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade1': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade2': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade2': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade2': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade2': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade2': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade3': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade3': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade3': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade3': [0.0, 0.34e6],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade3': [0.0, 0.34e6],
        'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb': [270.15, 373.15],
        'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_Tsec_supply_nom_RL': [20.0, 40.0],
        'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_Tsec_supply_nom_RL': [20.0, 40.0],
        'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_Tsec_supply_nom_RL': [20.0, 40.0],
        'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_Tsec_supply_nom_RL': [20.0, 40.0],
        'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_Tsec_supply_nom_RL': [20.0, 40.0],
        'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_dp_nom_RL': [25.0, 38.0],
        'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_dp_nom_RL': [25.0, 38.0],
        'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_dp_nom_RL': [25.0, 38.0],
        'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_dp_nom_RL': [25.0, 38.0],
        'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_dp_nom_RL': [25.0, 38.0],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[1]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[2]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[3]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[1]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[2]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[3]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[1]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[2]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[3]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[1]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[2]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[3]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[1]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[2]': [0.0, 1.0],
        'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[3]': [0.0, 1.0],
    }
    
    def __init__(self, start_time=0, stop_time=24*60*60, step_size=15.0, use_reward_shaping='reward_shaping_v1'):
        try:
            self.fmu = pyfmi.load_fmu(FMU_PATH, kind='CS', log_level=0)
            print(f"FMU file loaded correctly: {FMU_PATH}")
        except Exception as e:
            print(f"Error loading FMU file: {e}")
            
        self.start_time = start_time
        self.stop_time = stop_time
        self.fmu.setup_experiment(start_time=self.start_time, stop_time=self.stop_time)
        self.step_size = step_size
        self.fmu.initialize()
        self.current_time = 0
        
        self.observation_vars = {
            'cdu-cabinet-1': [
                'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_1.port.T',
                'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_2.port.T',
                'simulator[1].datacenter[1].computeBlock[1].cabinet[1].boundary_3.port.T',
                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade1',
                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade2',
                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade3'],
            'cdu-cabinet-2': [
                'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_1.port.T',
                'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_2.port.T',
                'simulator[1].datacenter[1].computeBlock[2].cabinet[1].boundary_3.port.T',
                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade1',
                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade2',
                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade3'],
            'cdu-cabinet-3': [
                'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_1.port.T',
                'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_2.port.T',
                'simulator[1].datacenter[1].computeBlock[3].cabinet[1].boundary_3.port.T',
                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade1',
                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade2',
                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade3'],
            'cdu-cabinet-4': [
                'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_1.port.T',
                'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_2.port.T',
                'simulator[1].datacenter[1].computeBlock[4].cabinet[1].boundary_3.port.T',
                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade1',
                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade2',
                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade3'],
            'cdu-cabinet-5': [
                'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_1.port.T',
                'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_2.port.T',
                'simulator[1].datacenter[1].computeBlock[5].cabinet[1].boundary_3.port.T',
                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade1',
                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade2',
                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade3'],
            'cooling-tower-1': [
                'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].CT.PFan',
                'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[2].CT.PFan',
                'simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].coolingTower[1].cell[1].waterSPTLvg',
                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb']
        }
        
        self.observation_space = spaces.Dict({
            'cdu-cabinet-1': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-2': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-3': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-4': spaces.Box(low=-1, high=1, shape=(6,)),
            'cdu-cabinet-5': spaces.Box(low=-1, high=1, shape=(6,)),
            'cooling-tower-1': spaces.Box(low=-1, high=1, shape=(4,)),
        })
        
        self.raw_observation_space_max = {
            'cdu-cabinet-1': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-1']]),
            'cdu-cabinet-2': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-2']]),
            'cdu-cabinet-3': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-3']]),
            'cdu-cabinet-4': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-4']]),
            'cdu-cabinet-5': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cdu-cabinet-5']]),
            'cooling-tower-1': np.array([self.variable_ranges[var_name][1] for var_name in self.observation_vars['cooling-tower-1']])
        }
        self.raw_observation_space_min = {
            'cdu-cabinet-1': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-1']]),
            'cdu-cabinet-2': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-2']]),
            'cdu-cabinet-3': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-3']]),
            'cdu-cabinet-4': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-4']]),
            'cdu-cabinet-5': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cdu-cabinet-5']]),
            'cooling-tower-1': np.array([self.variable_ranges[var_name][0] for var_name in self.observation_vars['cooling-tower-1']])
        }

        self.action_vars = {
            'cdu-cabinet-1': [
                'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_Tsec_supply_nom_RL',
                'simulator_1_datacenter_1_computeBlock_1_cdu_1_sources_dp_nom_RL',
                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[1]',
                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[2]',
                'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_Valve_Stpts[3]'
            ],
            'cdu-cabinet-2': [
                'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_Tsec_supply_nom_RL',
                'simulator_1_datacenter_1_computeBlock_2_cdu_1_sources_dp_nom_RL',
                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[1]',
                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[2]',
                'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_Valve_Stpts[3]'
            ],
            'cdu-cabinet-3': [
                'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_Tsec_supply_nom_RL',
                'simulator_1_datacenter_1_computeBlock_3_cdu_1_sources_dp_nom_RL',
                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[1]',
                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[2]',
                'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_Valve_Stpts[3]'
            ],
            'cdu-cabinet-4': [
                'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_Tsec_supply_nom_RL',
                'simulator_1_datacenter_1_computeBlock_4_cdu_1_sources_dp_nom_RL',
                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[1]',
                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[2]',
                'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_Valve_Stpts[3]'
            ],
            'cdu-cabinet-5': [
                'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_Tsec_supply_nom_RL',
                'simulator_1_datacenter_1_computeBlock_5_cdu_1_sources_dp_nom_RL',
                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[1]',
                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[2]',
                'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_Valve_Stpts[3]'
            ],
            'cooling-tower-1': [
                'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_CT_RL_stpt'
            ]
        }
        
        self.cooling_tower_action_decoding = {
            0: -0.20,
            1: -0.15,
            2: -0.10,
            3: -0.05,
            4: 0,
            5: 0.05,
            6: 0.10,
            7: 0.15,
            8: 0.20
        }
        
        self.action_space = spaces.Dict({
            'cdu-cabinet-1': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-2': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-3': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-4': spaces.Box(low=-1, high=1, shape=(5,)),
            'cdu-cabinet-5': spaces.Box(low=-1, high=1, shape=(5,)),
            'cooling-tower-1': spaces.Discrete(len(self.cooling_tower_action_decoding)),
        })
        
        self.raw_action_space_max = {
            'cdu-cabinet-1': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-1']]),
            'cdu-cabinet-2': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-2']]),
            'cdu-cabinet-3': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-3']]),
            'cdu-cabinet-4': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-4']]),
            'cdu-cabinet-5': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cdu-cabinet-5']]),
            'cooling-tower-1': np.array([self.variable_ranges[var_name][1] for var_name in self.action_vars['cooling-tower-1']])
        }
        self.raw_action_space_min = {
            'cdu-cabinet-1': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-1']]),
            'cdu-cabinet-2': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-2']]),
            'cdu-cabinet-3': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-3']]),
            'cdu-cabinet-4': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-4']]),
            'cdu-cabinet-5': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cdu-cabinet-5']]),
            'cooling-tower-1': np.array([self.variable_ranges[var_name][0] for var_name in self.action_vars['cooling-tower-1']])
        }
            
        self.nCDUs = 5
        self.parallel_nCabinets = 3
        self.nBranches = 3
        self.exogenous_var = [
            'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade1',
            'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade2',
            'simulator_1_datacenter_1_computeBlock_1_cabinet_1_sources_ComputePowerBlade3',
            'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade1',
            'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade2',
            'simulator_1_datacenter_1_computeBlock_2_cabinet_1_sources_ComputePowerBlade3',
            'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade1',
            'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade2',
            'simulator_1_datacenter_1_computeBlock_3_cabinet_1_sources_ComputePowerBlade3',
            'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade1',
            'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade2',
            'simulator_1_datacenter_1_computeBlock_4_cabinet_1_sources_ComputePowerBlade3',
            'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade1',
            'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade2',
            'simulator_1_datacenter_1_computeBlock_5_cabinet_1_sources_ComputePowerBlade3',
            'simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb']
        self.iter_exogenous_var = exogenous_variable_generator(EXOGENOUS_VAR_PATH, Towb_offset_in_K=self.Towb_offset_in_K).iterate_cyclically()
        
        self.reward = {}
        self.info = {}
        self.scaled_action = None
        self.previous_state = None
        self.use_reward_shaping = use_reward_shaping
        self.terminateds = {key: False for key in self.observation_vars}
        self.terminateds['cooling-tower-1'] = False
        self.truncateds = {key: False for key in self.observation_vars}
        self.truncateds['cooling-tower-1'] = False
        
    def seed(self, seed=None):
        pass
    
    def action_inverse_mapper(self, action):
        unscaled_action = {
            'cdu-cabinet-1': None,
            'cdu-cabinet-2': None,
            'cdu-cabinet-3': None,
            'cdu-cabinet-4': None,
            'cdu-cabinet-5': None,
            'cooling-tower-1': None
        }
        assert type(action) == dict, "Action must be a dict"
        for key, val in action.items():
            if key == 'cooling-tower-1':
                unscaled_action[key] = self.cooling_tower_action_decoding[val]
            else:
                assert type(val) == np.ndarray, "Actions for each category must be a numpy np.ndarray before being sent to the environment"
                unscaled_action[key] = self.raw_action_space_min[key] + 0.5 * (val + 1) * (self.raw_action_space_max[key] - self.raw_action_space_min[key])
        return unscaled_action
    
    def observation_mapper(self, raw_observation):
        scaled_observation = {
            'cdu-cabinet-1': None,
            'cdu-cabinet-2': None,
            'cdu-cabinet-3': None,
            'cdu-cabinet-4': None,
            'cdu-cabinet-5': None,
            'cooling-tower-1': None
        }
        assert type(raw_observation) == dict, "Observation must be a dict"
        for key, val in raw_observation.items():
            assert type(val) == np.ndarray, "Observations for each category must be a numpy np.ndarray before being sent from the environment"
            scaled_observation[key] = 2 * (val - self.raw_observation_space_min[key]) / (self.raw_observation_space_max[key] - self.raw_observation_space_min[key]) - 1
        return scaled_observation
    
    def get_exogenous_var(self):
        return next(self.iter_exogenous_var)
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.fmu.reset()
        self.fmu.setup_experiment(start_time=self.start_time, stop_time=self.stop_time)
        self.fmu.initialize()
        self.current_time = 0
        
        raw_observations = {
            'cdu-cabinet-1': None,
            'cdu-cabinet-2': None,
            'cdu-cabinet-3': None,
            'cdu-cabinet-4': None,
            'cdu-cabinet-5': None,
            'cooling-tower-1': None
        }
        for key, var_list in self.observation_vars.items():
            raw_observations[key] = np.array([i[0] for i in self.fmu.get(var_list)])
        
        self.previous_state = self.observation_mapper(raw_observations)
        self.info = raw_observations.copy()
        
        return self.previous_state, self.info

    def step_original(self, action, base_info=None):
        self.scaled_action = copy.deepcopy(action)
        action = self.action_inverse_mapper(action)
        
        for key, val in action.items():
            if key != 'cooling-tower-1':
                action[key][2:2+3] = np.exp(val[2:2+3]) / np.sum(np.exp(val[2:2+3]))
        
        for key, var_list in self.action_vars.items():
            if key != 'cooling-tower-1':
                self.fmu.set(var_list, [round(i, 2) for i in list(action[key])])
            else:
                latest_wetbulb = self.fmu.get('simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb')[0]
                rule_based_new_setpoint = latest_wetbulb + 10.0 * 5/9
                rl_new_setpoint = action[key] + rule_based_new_setpoint
                self.fmu.set(var_list, [round(rl_new_setpoint, 2)])
        
        self.fmu.do_step(current_t=self.current_time, step_size=self.step_size)
        
        exogenous_variables = self.get_exogenous_var()
        self.fmu.set(self.exogenous_var, list(exogenous_variables))
        
        raw_observations = {
            'cdu-cabinet-1': None,
            'cdu-cabinet-2': None,
            'cdu-cabinet-3': None,
            'cdu-cabinet-4': None,
            'cdu-cabinet-5': None,
            'cooling-tower-1': None
        }
        for key, var_list in self.observation_vars.items():
            raw_observations[key] = np.array([i[0] for i in self.fmu.get(var_list)])
        
        self.info = raw_observations.copy()
        self.info['actions'] = action
        self.info['actions_newsetpoint'] = rl_new_setpoint
        if base_info is not None:
            for key in self.info:
                if key in base_info:
                    self.info[key] = {'observations': self.info[key], **base_info[key]}
        
        scaled_observation = self.observation_mapper(raw_observations)
        
        self.reward = {}
        self.terminateds = {key: False for key in self.observation_vars}
        self.terminateds['cooling-tower-1'] = False
        self.truncateds = {key: False for key in self.observation_vars}
        self.truncateds['cooling-tower-1'] = False
        
        if self.use_reward_shaping == 'reward_shaping_v0':
            self.reward_shaping_v0(scaled_observation)
        elif self.use_reward_shaping == 'reward_shaping_v1':
            self.reward_shaping_v1()
        else:
            raise ValueError(f"Unknown reward shaping method: {self.use_reward_shaping}")
        
        # Compute cooling tower reward with temperature deviation penalty
        temp_deviation = 0.0
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            temp_scaled = scaled_observation[key][0:3]  # Boundary temperatures
            temp_K = ((temp_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            goal_scaled = self.info[key].get('goal', 0.0) if isinstance(self.info[key], dict) else 0.0
            goal_K = ((goal_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            temp_deviation += np.abs(temp_K - goal_K).mean() / 20.0  # Normalize by max deviation (20Â°C)
        temp_deviation /= 5  # Average over cabinets
        self.reward['cooling-tower-1'] = (2 - scaled_observation['cooling-tower-1'][0:2].sum()) / 2 - temp_deviation
        self.terminateds['cooling-tower-1'] = False
        self.truncateds['cooling-tower-1'] = False
        
        self.previous_state = scaled_observation
        
        return scaled_observation, self.reward, self.terminateds, self.truncateds, self.info

    def step(self, action, base_info=None):
        self.scaled_action = copy.deepcopy(action)
        action = self.action_inverse_mapper(action)
        
        for key, val in action.items():
            if key != 'cooling-tower-1':
                action[key][2:2+3] = np.exp(val[2:2+3]) / np.sum(np.exp(val[2:2+3]))
        
        for key, var_list in self.action_vars.items():
            if key != 'cooling-tower-1':
                self.fmu.set(var_list, [round(i, 2) for i in list(action[key])])
            else:
                latest_wetbulb = self.fmu.get('simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb')[0]
                rule_based_new_setpoint = latest_wetbulb + 10.0 * 5/9
                rl_new_setpoint = action[key] + rule_based_new_setpoint
                self.fmu.set(var_list, [round(rl_new_setpoint, 2)])
        
        self.fmu.do_step(current_t=self.current_time, step_size=self.step_size)
        self.current_time += self.step_size
        
        exogenous_variables = self.get_exogenous_var()
        self.fmu.set(self.exogenous_var, list(exogenous_variables))
        
        raw_observations = {
            'cdu-cabinet-1': None,
            'cdu-cabinet-2': None,
            'cdu-cabinet-3': None,
            'cdu-cabinet-4': None,
            'cdu-cabinet-5': None,
            'cooling-tower-1': None
        }
        for key, var_list in self.observation_vars.items():
            raw_observations[key] = np.array([i[0] for i in self.fmu.get(var_list)])
        
        self.info = raw_observations.copy()
        self.info['actions'] = action
        self.info['actions_newsetpoint'] = rl_new_setpoint
        if base_info is not None:
            for key in self.info:
                if key in base_info:
                    self.info[key] = {'observations': self.info[key], **base_info[key]}
        
        scaled_observation = self.observation_mapper(raw_observations)
        
        self.reward = {}
        self.terminateds = {key: False for key in self.observation_vars}
        self.terminateds['cooling-tower-1'] = False
        self.truncateds = {key: False for key in self.observation_vars}
        self.truncateds['cooling-tower-1'] = False
        
        if self.use_reward_shaping == 'reward_shaping_v0':
            self.reward_shaping_v0(scaled_observation)
        elif self.use_reward_shaping == 'reward_shaping_v1':
            self.reward_shaping_v1()
        else:
            raise ValueError(f"Unknown reward shaping method: {self.use_reward_shaping}")
        
        # Compute cooling tower reward with temperature deviation penalty
        temp_deviation = 0.0
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            temp_scaled = scaled_observation[key][0:3]
            temp_K = ((temp_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            goal_scaled = self.info[key].get('goal', 0.0) if isinstance(self.info[key], dict) else 0.0
            goal_K = ((goal_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            temp_deviation += np.abs(temp_K - goal_K).mean() / 20.0
        temp_deviation /= 5

        fan_power = scaled_observation['cooling-tower-1'][0:2].sum()
        efficiency_reward = 1.0 - (fan_power + 1) / 2  # Map to [0,1], lower power is better

        self.reward['cooling-tower-1'] = 0.6 * efficiency_reward - 0.4 * temp_deviation
        self.terminateds['cooling-tower-1'] = False
        self.truncateds['cooling-tower-1'] = False
        
        self.previous_state = scaled_observation
        
        return scaled_observation, self.reward, self.terminateds, self.truncateds, self.info
    
    def reward_shaping_v11(self):
        target_temp_min = 293.15
        target_temp_max = 313.15
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            # Original reward: Match valve setpoints to power levels
            reward = 6.0 - abs(self.scaled_action[key][2] - self.previous_state[key][3]) \
                         - abs(self.scaled_action[key][3] - self.previous_state[key][4]) \
                         - abs(self.scaled_action[key][4] - self.previous_state[key][5])
            
            # Add penalty for temperature deviation from goal
            temp_scaled = self.previous_state[key][0:3]  # Boundary temperatures
            # temp_K = ((temp_scaled + 1) / 2) * (373.15 - 273.15) + 273.15  # Map to [273.15, 373.15] K
            temp_K = ((temp_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            goal_scaled = self.info[key].get('goal', 0.0) if isinstance(self.info[key], dict) else 0.0
            # goal_K = ((goal_scaled + 1) / 2) * (373.15 - 273.15) + 273.15  # Map goal to [273.15, 373.15] K
            goal_K = ((goal_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            temp_deviation = np.abs(temp_K - goal_K).mean() / 20.0  # Normalize by max deviation (20Â°C)
            reward -= 2.0 * temp_deviation  # Weight the penalty to prioritize temperature control
            
            # Existing penalty for goals outside target range
            if goal_K < target_temp_min or goal_K > target_temp_max:
                reward -= 1.0
            
            self.reward[key] = reward
            self.terminateds[key] = False
            self.truncateds[key] = False
    
    
    def reward_shaping_v1(self):
        target_temp_min = 293.15
        target_temp_max = 313.15
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            # Original reward: Match valve setpoints to power levels
            alignment_reward = 6.0 - abs(self.scaled_action[key][2] - self.previous_state[key][3]) \
                                   - abs(self.scaled_action[key][3] - self.previous_state[key][4]) \
                                   - abs(self.scaled_action[key][4] - self.previous_state[key][5])
            
            # Efficiency term: Reward lower total valve openings
            valve_efficiency = 1.0 - np.sum(self.scaled_action[key][2:5]) / 3.0  # Normalized to [0,1]
            
            # Combine alignment and efficiency
            reward = 0.7 * alignment_reward + 0.3 * valve_efficiency
            
            # Reduced penalty for temperature deviation
            temp_scaled = self.previous_state[key][0:3]
            temp_K = ((temp_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            goal_scaled = self.info[key].get('goal', 0.0) if isinstance(self.info[key], dict) else 0.0
            goal_K = ((goal_scaled + 1) / 2) * (TEMP_MAX_K - TEMP_MIN_K) + TEMP_MIN_K
            temp_deviation = np.abs(temp_K - goal_K).mean() / 20.0
            reward -= 1.5 * temp_deviation  # Reduced from 2.0
            
            # Reduced penalty for goals outside target range
            if goal_K < target_temp_min or goal_K > target_temp_max:
                reward -= 0.5  # Reduced from 1.0
            
            self.reward[key] = reward
            self.terminateds[key] = False
            self.truncateds[key] = False
    
    
    
    def reward_shaping_v0(self, scaled_observation):
        for key in ['cdu-cabinet-1', 'cdu-cabinet-2', 'cdu-cabinet-3', 'cdu-cabinet-4', 'cdu-cabinet-5']:
            self.reward[key] = (3 - scaled_observation[key][0:3].sum()) / 3
            self.terminateds[key] = False
            self.truncateds[key] = False
            
            
    def get_all_observations(self):
        """Get all current observations from the FMU."""
        observations = {}
        
        # Get values for all observation variables
        for component, var_list in self.observation_vars.items():
            for var in var_list:
                observations[var] = self.fmu.get(var)[0]
        
        # Print for debugging
        print(f"Got {len(observations)} observations from FMU")
        
        return observations