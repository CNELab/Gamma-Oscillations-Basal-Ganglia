#This file simulates the BG network, disconnecting a single projection in each iteration.
#This allow the compuation of the gamma spectral power with and without each connection
from ANNarchy import *

import warnings
warnings.filterwarnings("ignore")
#all the projections in the model
possible_values_of_disc = [
    'original',
    'D1-D1',
    'D1-D2',
    'D2-D1',
    'D2-D2',
    'D2-GPe-TI',
    'FSN-D1',
    'FSN-FSN',
    'FSN-D2',
    'GPe-TI-GPe-TI',
    'GPe-TI-GPe-TA',
    'GPe-TI-FSN',
    'GPe-TI-STN',
    'GPe-TA-D1',
    'GPe-TA-D2',
    'GPe-TA-FSN',
    'GPe-TA-GPe-TA',
    'GPe-TA-GPe-TI',
    'STN-GPe-TA',
    'STN-GPe-TI'
]

#load the mean firing rate for each nuclues from a different file
import pickle
with open('dd_firing.p', 'rb') as f:
    result = pickle.load(f)
import numpy as np

# Extract results for the trials and compute the mean for each variable
results = np.array([result['1.0'][i] for i in range(len(result['1.0']))]).mean(axis=0)
#these are the 6 mean firing rates
d1, d2, fsn,  gpeta, gpeti, stn = results






    
    
result={}
#in each loop, disc is the name of the connection that will be disconnected
for disc in possible_values_of_disc:
  

  #required to run multiple simulations
  clear()
  #integration stepsize
  dt_value = 0.1

  setup(dt=dt_value, structural_plasticity= True, method='rk4')



  # Build the BG Network
  STN = Neuron(
      parameters="""
          C_m = 60 * 1e-3         : population
          G_l = 10.0 * 1e-3       : population
          E_l = -80.2             : population
          delta_T = 16.2          : population
          V_th = -64.0            : population
          E_ex = 0.0              : population
          E_in = -84.0            : population
          I_e = 5.0 * 1e-3        : population
          tau_ex = 4.0            : population
          tau_in = 8.0            : population
          tau_w = 333.0           : population
          a = 0.0 * 1e-3          : population
          b = 0.05 * 1e-3         : population
          V_peak = 15.0           : population
          V_reset = -70.0         : population
      """,
        equations="""
          prev_v = v
          prev_w = w
          I_ex=g_ex * (E_ex - v)
          I=g_ex * (E_ex - v) + g_in * (E_in - v )
          C_m * dv/dt = -G_l * (v - E_l) + G_l * delta_T * func_aux(v, V_th, delta_T, V_peak) - w + I+ I_e
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + a * (v - E_l)

      """,
      spike = "(v>V_peak) or (abs(v-prev_v)>150)",
      reset = """
                w = temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) + b
                v = V_reset
      """,
      functions = """
          func_aux(v, V_th, delta_T, V_peak) = if v>1100: exp((1100 - V_th)/delta_T) else: exp((v - V_th)/delta_T)
          temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) = if (abs(v-prev_v)>150) : prev_w + (dt/tau_w) * (a * (V_peak- E_l) - prev_w) else: w
      """,
      refractory = 0.0
  )
  GPe_TA = Neuron(
        parameters="""
          C_m = 60 * 1e-3         : population
          G_l = 1.0 * 1e-3        : population
          E_l = -55.1             : population
          delta_T = 2.55          : population
          V_th = -54.7            : population
          E_ex = 0.0              : population
          E_in = -65.0            : population
          I_e = 1.0 * 1e-3        : population
          tau_ex = 10.0           : population
          tau_in = 5.5            : population
          tau_w = 20.0            : population
          a = 2.5 * 1e-3          : population
          b = 105.0 * 1e-3        : population
          V_peak = 15.0           : population
          V_reset = -60.0         : population
      """,
        equations="""
          prev_v = v
          prev_w = w
          I_ex=g_ex * (E_ex - v)
          I=g_ex * (E_ex - v) + g_in * (E_in - v )
          C_m * dv/dt = -G_l * (v - E_l) + G_l * delta_T * func_aux(v, V_th, delta_T, V_peak) - w + I + I_e
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + a * (v - E_l)

      """,
      spike = "(v>V_peak) or (abs(v-prev_v)>150)",
      reset = """
                w = temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) + b
                v = V_reset
      """,
      functions = """
          func_aux(v, V_th, delta_T, V_peak) = if v>1100: exp((1100 - V_th)/delta_T) else: exp((v - V_th)/delta_T)
          temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) = if (abs(v-prev_v)>150) : prev_w + (dt/tau_w) * (a * (V_peak- E_l) - prev_w) else: w
      """,
      refractory = 0.0
  )
  GPe_TI = Neuron(
        parameters="""
          C_m = 40 * 1e-3         : population
          G_l = 1.0 * 1e-3        : population
          E_l = -55.1             : population
          delta_T = 1.7           : population
          V_th = -54.7            : population
          E_ex = 0.0              : population
          E_in = -65.0            : population
          I_e = 12.0 * 1e-3       : population
          tau_ex = 10.0           : population
          tau_in = 5.5            : population
          tau_rec = 7             : population
          tau_w = 20.0            : population
          a = 2.5 * 1e-3          : population
          b = 70.0 * 1e-3         : population
          V_peak = 15.0           : population
          V_reset = -60.0         : population
      """,
        equations="""
          prev_v = v
          prev_w = w
                
          C_m * dv/dt = -G_l * (v - E_l) + G_l * delta_T * func_aux(v, V_th, delta_T, V_peak) - w + g_ex * (E_ex - v) + g_in * (E_in - v )+ I_e
          
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + a * (v - E_l)

      """,
      spike = "(v>V_peak) or (abs(v-prev_v)>150)",
      reset = """
                w = temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) + b
                v = V_reset
      """,
      functions = """
          func_aux(v, V_th, delta_T, V_peak) = if v>1100: exp((1100 - V_th)/delta_T) else: exp((v - V_th)/delta_T)
          temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) = if (abs(v-prev_v)>150) : prev_w + (dt/tau_w) * (a * (V_peak- E_l) - prev_w) else: w
      """,
      refractory = 0.0
  )
  FSN = Neuron(
    parameters="""
          C_m = 80.0 * 1e-3       : population
          E_l = -80.0             : population
          V_th = -50.0            : population
          E_ex = 0.0              : population
          E_in = -74.0            : population
          I_e = 0.0 * 1e-3        : population
          tau_ex = 12.0           : population
          tau_in = 10.0           : population
          tau_w = 5.0             : population
          a = 0.025 * 1e-3        : population
          b = 0.0 * 1e-3          : population
          V_peak = 25.0           : population
          V_reset = -60.0         : population
          k = 1.0 * 1e-3          : population
          V_b = -55.0             : population
      """,
    equations="""
          I_ex=g_ex * (E_ex - v)
          I=g_ex * (E_ex - v) + g_in * (E_in - v )
          C_m * dv/dt = k * (v - E_l) * (v - V_th) - w +I+ I_e
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + var_temp(v, V_b, a)
      """,
      spike = "v > V_peak",
      reset = """v = V_reset
                w += b
      """,
      functions = """
          var_temp(v, s1, s2) = if s1 > v : s2 * pow((v - s1) , 3) else: 0
      """,
      refractory = 0.0
  )
  D2 = Neuron(
    parameters="""
          C_m = 15.2 * 1e-3       : population
          E_l = -80.0             : population
          V_th = -29.7            : population
          E_ex = 0.0              : population
          E_in = -74.0            : population
          I_e = 0.0 * 1e-3        : population
          tau_ex = 12.0           : population
          tau_in = 10.0           : population
          tau_w = 100.0           : population
          a = -20.0 * 1e-3        : population
          b = 91.0 * 1e-3         : population
          V_peak = 40.0           : population
          V_reset = -60.0         : population
          k = 1.0 * 1e-3          : population
      """,
    equations="""
          I_ex=g_ex * (E_ex - v)
          I=g_ex * (E_ex - v) + g_in * (E_in - v )
          C_m * dv/dt = k * (v - E_l) * (v - V_th) - w + I + I_e
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + a * (v - E_l)

      """,
      spike = "v > V_peak",
      reset = """v = V_reset
                w += b
      """,
      refractory = 0.0
  )
  D1 = Neuron(
    parameters="""
          C_m = 15.2 * 1e-3       : population
          E_l = -78.2             : population
          V_th = -29.7            : population
          E_ex = 0.0              : population
          E_in = -74.0            : population
          I_e = 0.0 * 1e-3        : population
          tau_ex = 12.0           : population
          tau_in = 10.0           : population
          tau_w = 100.0           : population
          a = -20.0 * 1e-3        : population
          b = 67.0 * 1e-3         : population
          V_peak = 40.0           : population
          V_reset = -60.0         : population
          k = 1.0 * 1e-3          : population
      """,
    equations="""
          I_ex=g_ex * (E_ex - v)
          I=g_ex * (E_ex - v) + g_in * (E_in - v )
          C_m * dv/dt = k * (v - E_l) * (v - V_th) - w + I + I_e
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + a * (v - E_l)

      """,
      spike = "v > V_peak",
      reset = """v = V_reset
                w += b
      """,
      refractory = 0.0
  )
  SNR = Neuron(
      parameters="""
          C_m = 80 * 1e-3         : population
          G_l = 3.0 * 1e-3        : population
          E_l = -55.8             : population
          delta_T = 1.8           : population
          V_th = -55.2            : population
          E_ex = 0.0              : population
          E_in = -72.0            : population
          I_e = 15.0 * 1e-3       : population
          tau_ex = 12.0           : population
          tau_in = 2.1            : population
          tau_w = 20.0            : population
          a = 3 * 1e-3            : population
          b = 200 * 1e-3          : population
          V_peak = 20.0           : population
          V_reset = -65.0         : population
      """,
        equations="""
          prev_v = v
          prev_w = w
          I_ex=g_ex * (E_ex - v)
          I=g_ex * (E_ex - v) + g_in * (E_in - v )
          C_m * dv/dt = -G_l * (v - E_l) + G_l * delta_T * func_aux(v, V_th, delta_T, V_peak) - w + I + I_e
          tau_ex * dg_ex/dt = - g_ex
          tau_in * dg_in/dt = - g_in
          tau_w * dw/dt = - w + a * (v - E_l)

      """,
      spike = "(v>V_peak) or (abs(v-prev_v)>150)",
      reset = """
                w = temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) + b
                v = V_reset
      """,
      functions = """
          func_aux(v, V_th, delta_T, V_peak) = if v>1100: exp((1100 - V_th)/delta_T) else: exp((v - V_th)/delta_T)
          temp_w(w, v, prev_v, prev_w, tau_w, a, V_peak, E_l, b, dt) = if (abs(v-prev_v)>150) : prev_w + (dt/tau_w) * (a * (V_peak- E_l) - prev_w) else: w
      """,
      refractory = 0.0
  )
  #defining the neural populations
  P_STN = Population(geometry=408, neuron=STN)
  P_GPe_TA = Population(geometry=264, neuron=GPe_TA)
  P_GPe_TI = Population(geometry=780, neuron=GPe_TI)
  P_FSN = Population(geometry=420, neuron=FSN)
  P_D2 = Population(geometry=6000, neuron=D2)
  P_D1 = Population(geometry=6000, neuron=D1)
  #external inputs to the populations
  P_ext_STN = PoissonPopulation(408, rates=0.5 * 1e3)
  P_ext_GPe_TA = PoissonPopulation(264, rates=0.17 * 1e3)
  P_ext_GPe_TI = PoissonPopulation(780, rates=600)
  P_ext_FSN = PoissonPopulation(420, rates=0.9444 * 1e3)
  P_ext_D2 = PoissonPopulation(6000, rates=(0.9729278004599998/0.9) * 1* 1e3*1.11)
  P_ext_D1 = PoissonPopulation(6000, rates=1.12 * 1e3*1.14)

  
  #for each connection, the weights are imposed to be null (i.e., the projection is not defined) and the projection is substited with an input for a poissonian source. 
  # The firings for the poissonian sources are the mean firing rates in parkinsonian condition.
  if disc=='D1-D1':
    print(disc)

    D1_aux = PoissonPopulation(6000, rates=d1)
    D1_auxtoD1 = Projection(pre=D1_aux, post=P_D1, target='in')
    D1_auxtoD1.connect_fixed_probability(weights=0.12 * 1e-3, probability=0.0607, delays=1.7)
  else:

    D1toD1 = Projection(pre=P_D1, post=P_D1, target='in')
    D1toD1.connect_fixed_probability(weights=0.12 * 1e-3, probability=0.0607, delays=1.7)

  if disc=='D1-D2':
    print(disc)

    D1_aux = PoissonPopulation(6000, rates=d1)
    D1_auxtoD2 = Projection(pre=D1_aux, post=P_D2, target='in')
    D1_auxtoD2.connect_fixed_probability(weights=0.30 * 1e-3, probability=0.0140, delays=1.7)
  else:
    D1toD2 = Projection(pre=P_D1, post=P_D2, target='in')
    D1toD2.connect_fixed_probability(weights=0.30 * 1e-3, probability=0.0140, delays=1.7)

  if disc=='D2-D1':
    print(disc)

    D2_aux = PoissonPopulation(6000, rates=d2)
    D2_auxtoD1 = Projection(pre=D2_aux, post=P_D1, target='in')
    D2_auxtoD1.connect_fixed_probability(weights=0.36 * 1e-3, probability=0.0653, delays=1.7)
  else:
    D2toD1 = Projection(pre=P_D2, post=P_D1, target='in')
    D2toD1.connect_fixed_probability(weights=0.36 * 1e-3, probability=0.0653, delays=1.7)

  if disc=='D2-D2':
    print(disc)

   
    D2_aux = PoissonPopulation(6000, rates=d2)
    D2_auxtoD2 = Projection(pre=D2_aux, post=P_D2, target='in')
    D2_auxtoD2.connect_fixed_probability(weights=0.25 * 1e-3, probability=0.0840, delays=1.7)
  else:
    D2toD2 = Projection(pre=P_D2, post=P_D2, target='in')
    D2toD2.connect_fixed_probability(weights=0.25 * 1e-3, probability=0.0840, delays=1.7)

  if disc=='D2-GPe-TI':
    print(disc)

    D2_aux = PoissonPopulation(6000, rates=d2)
    D2_auxtoGPe_TI = Projection(pre=D2_aux, post=P_GPe_TI, target='in')
    D2_auxtoGPe_TI.connect_fixed_probability(weights=1.28 * 1e-3, probability=0.0833, delays=7.0)
  else:
    D2toGPe_TI = Projection(pre=P_D2, post=P_GPe_TI, target='in')
    D2toGPe_TI.connect_fixed_probability(weights=1.28 * 1e-3, probability=0.0833, delays=7.0)

  if disc=='FSN-D1':
    print(disc)

 
    FSN_aux = PoissonPopulation(420, rates=fsn)
    FSN_auxtoD1 = Projection(pre=FSN_aux, post=P_D1, target='in')
    FSN_auxtoD1.connect_fixed_probability(weights=6.6 * 1e-3, probability=0.0381, delays=1.7)
  else:
    FSNtoD1 = Projection(pre=P_FSN, post=P_D1, target='in')
    FSNtoD1.connect_fixed_probability(weights=6.6 * 1e-3, probability=0.0381, delays=1.7)

  if disc=='FSN-FSN':
    print(disc)

    FSN_aux = PoissonPopulation(420, rates=fsn)
    FSN_auxtoFSN = Projection(pre=FSN_aux, post=P_FSN, target='in')
    FSN_auxtoFSN.connect_fixed_probability(weights=0.5 * 1e-3, probability=0.0238, delays=1.0)
  else:
    FSNtoFSN = Projection(pre=P_FSN, post=P_FSN, target='in')
    FSNtoFSN.connect_fixed_probability(weights=0.5 * 1e-3, probability=0.0238, delays=1.0)

  if disc=='FSN-D2':
    print(disc)
    
    FSN_aux = PoissonPopulation(420, rates=fsn)
    FSN_auxtoD2 = Projection(pre=FSN_aux, post=P_D2, target='in')
    FSN_auxtoD2.connect_fixed_probability(weights=4.8 * 1e-3, probability=0.0262, delays=1.7)
  else:
    FSNtoD2 = Projection(pre=P_FSN, post=P_D2, target='in')
    FSNtoD2.connect_fixed_probability(weights=4.8 * 1e-3, probability=0.0262, delays=1.7)

  if disc=='GPe-TI-GPe-TI':
    print(disc)


    GPe_TI_aux = PoissonPopulation(780, rates= gpeti)
    GPe_TI_auxtoGPe_TI = Projection(pre=GPe_TI_aux, post=P_GPe_TI, target='in')
    GPe_TI_auxtoGPe_TI.connect_fixed_probability(weights=1.1 * 1e-3, probability=0.0321, delays=1.8)
  else:
    GPe_TItoGPe_TI = Projection(pre=P_GPe_TI, post=P_GPe_TI, target='in')
    GPe_TItoGPe_TI.connect_fixed_probability(weights=1.1 * 1e-3, probability=0.0321, delays=1.8)

  if disc=='GPe-TI-GPe-TA':
    print(disc)
  
    GPe_TI_aux = PoissonPopulation(780, rates= gpeti)
    GPe_TI_auxtoGPe_TA = Projection(pre=GPe_TI_aux, post=P_GPe_TA, target='in')
    GPe_TI_auxtoGPe_TA.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0321, delays=1.0)
  else:
    GPe_TItoGPe_TA = Projection(pre=P_GPe_TI, post=P_GPe_TA, target='in')
    GPe_TItoGPe_TA.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0321, delays=1.0)

  if disc=='GPe-TI-FSN':
    print(disc)

    GPe_TI_aux = PoissonPopulation(780, rates= gpeti)
    GPe_TI_auxtoFSN = Projection(pre=GPe_TI_aux, post=P_FSN, target='in')
    GPe_TI_auxtoFSN.connect_fixed_probability(weights=1.6 * 1e-3, probability=0.0128, delays=7.0)
  else:
    GPe_TItoFSN = Projection(pre=P_GPe_TI, post=P_FSN, target='in')
    GPe_TItoFSN.connect_fixed_probability(weights=1.6 * 1e-3, probability=0.0128, delays=7.0)

  if disc=='GPe-TI-STN':
    print(disc)
  
    GPe_TI_aux = PoissonPopulation(780, rates= gpeti)
    GPe_TI_auxtoSTN= Projection(pre=GPe_TI_aux, post=P_STN, target='in')
    GPe_TI_auxtoSTN.connect_fixed_probability(weights=0.08 * 1e-3, probability=0.0385, delays=1.0)
  else:
    GPe_TItoSTN = Projection(pre=P_GPe_TI, post=P_STN, target='in')
    GPe_TItoSTN.connect_fixed_probability(weights=0.08 * 1e-3, probability=0.0385, delays=1.0)

  if disc=='GPe-TA-D1':
    print(disc)

    GPe_TA_aux = PoissonPopulation(264, rates= gpeta)
    GPe_TA_auxtoD1 = Projection(pre=GPe_TA_aux, post=P_D1, target='in')
    GPe_TA_auxtoD1.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0379, delays=7.0)
  else:
    GPe_TAtoD1 = Projection(pre=P_GPe_TA, post=P_D1, target='in')
    GPe_TAtoD1.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0379, delays=7.0)

  if disc=='GPe-TA-D2':
    print(disc)
  
    GPe_TA_aux = PoissonPopulation(264, rates= gpeta)
    GPe_TA_auxtoD2 = Projection(pre=GPe_TA_aux, post=P_D2, target='in')
    GPe_TA_auxtoD2.connect_fixed_probability(weights=0.61 * 1e-3, probability=0.0379, delays=7.0)
  else:
    GPe_TAtoD2 = Projection(pre=P_GPe_TA, post=P_D2, target='in')
    GPe_TAtoD2.connect_fixed_probability(weights=0.61 * 1e-3, probability=0.0379, delays=7.0)

  if disc=='GPe-TA-FSN':
    print(disc)


    GPe_TA_aux = PoissonPopulation(264, rates= gpeta)
    GPe_TA_auxtoFSN = Projection(pre=GPe_TA_aux, post=P_FSN, target='in')
    GPe_TA_auxtoFSN.connect_fixed_probability(weights=1.85 * 1e-3, probability=0.0379, delays=7.0)
  else:
    GPe_TAtoFSN = Projection(pre=P_GPe_TA, post=P_FSN, target='in')
    GPe_TAtoFSN.connect_fixed_probability(weights=1.85 * 1e-3, probability=0.0379, delays=7.0)

  if disc=='GPe-TA-GPe-TA':
    print(disc)

    GPe_TA_aux = PoissonPopulation(264, rates= gpeta)
    GPe_TA_auxtoGPe_TA = Projection(pre=GPe_TA_aux, post=P_GPe_TA, target='in')
    GPe_TA_auxtoGPe_TA.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0189, delays=1.0)
  else:
    GPe_TAtoGPe_TA = Projection(pre=P_GPe_TA, post=P_GPe_TA, target='in')
    GPe_TAtoGPe_TA.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0189, delays=1.0)

  if disc=='GPe-TA-GPe-TI':
    print(disc)

    GPe_TA_aux = PoissonPopulation(264, rates= gpeta)
    GPe_TA_auxtoGPe_TI = Projection(pre=GPe_TA_aux, post=P_GPe_TI, target='in')
    GPe_TA_auxtoGPe_TI.connect_fixed_probability(weights=1.2 * 1e-3, probability=0.0189, delays=1.0)
  else:
    GPe_TAtoGPe_TI = Projection(pre=P_GPe_TA, post=P_GPe_TI, target='in')
    GPe_TAtoGPe_TI.connect_fixed_probability(weights=1.2 * 1e-3, probability=0.0189, delays=1.0)

  if disc=='STN-GPe-TA':
    print(disc)


    STN_aux = PoissonPopulation(408, rates=stn)
    STN_auxtoGPe_TA = Projection(pre=STN_aux, post=P_GPe_TA, target='ex')
    STN_auxtoGPe_TA.connect_fixed_probability(weights=0.13 * 1e-3, probability=0.0735, delays=2.0)
  else:
    STNtoGPe_TA = Projection(pre=P_STN, post=P_GPe_TA, target='ex')
    STNtoGPe_TA.connect_fixed_probability(weights=0.13 * 1e-3, probability=0.0735, delays=2.0)

  if disc=='STN-GPe-TI':
    print(disc)

    STN_aux = PoissonPopulation(408, rates=stn)
    STN_auxtoGPe_TI = Projection(pre=STN_aux, post=P_GPe_TI, target='ex')
    STN_auxtoGPe_TI.connect_fixed_probability(weights=0.42 * 1e-3, probability=0.0735, delays=2.0)
  else:
    STNtoGPe_TI = Projection(pre=P_STN, post=P_GPe_TI, target='ex')
    STNtoGPe_TI.connect_fixed_probability(weights=0.42 * 1e-3, probability=0.0735, delays=2.0)

  if disc=='original':
    print(disc)

  #external excitatory inputs
  exttoSTN = Projection(P_ext_STN, P_STN, 'ex')
  exttoSTN.connect_one_to_one(weights=Uniform(0.2 * 1e-3, 0.3 * 1e-3), delays = 0)

  exttoGPe_TA = Projection(P_ext_GPe_TA, P_GPe_TA, 'ex')
  exttoGPe_TA.connect_one_to_one(weights=Uniform(0.10 * 1e-3, 0.2 * 1e-3), delays = 0)


  exttoGPe_TI = Projection(P_ext_GPe_TI, P_GPe_TI, 'ex')
  exttoGPe_TI.connect_one_to_one(weights=Uniform(0.2 * 1e-3, 0.3 * 1e-3), delays = 0)
 

  exttoFSN = Projection(P_ext_FSN, P_FSN, 'ex')
  exttoFSN.connect_one_to_one(weights=Uniform(0.45 * 1e-3, 0.55 * 1e-3), delays = 0)


  exttoD2 = Projection(P_ext_D2, P_D2, 'ex')
  exttoD2.connect_one_to_one(weights=Uniform(0.4 * 1e-3, 0.5 * 1e-3), delays = 0)
 
  exttoD1 = Projection(P_ext_D1, P_D1, 'ex')
  exttoD1.connect_one_to_one(weights=Uniform(0.4 * 1e-3, 0.5 * 1e-3), delays = 0)


  compile()


  # Define Monitors and Simulate the Network

  #only D2 and GPe-TI activities are recorded
  m_GPe_TI = Monitor(P_GPe_TI, ['spike'],start=False,period=1)
  m_D2 = Monitor(P_D2,  ['spike'],start=False,period=1)



  def simulation(idx,net):
    net.get(P_STN).v=np.random.normal(-80.2,1,size=408)
    net.get(P_GPe_TA).v = np.random.normal(-55.1, 1,size=264)
    net.get(P_GPe_TI).v = np.random.normal(-55.1, 1,size=780)
    net.get(P_FSN).v = np.random.normal(-80.0, 1,size=420)
    net.get(P_D2).v = np.random.normal(-80.0, 1,size=6000)
    net.get(P_D1).v = np.random.normal(-78.2, 1,size=6000)
   
    net.simulate(1000)
    
    net.get(m_GPe_TI).start()
    net.get(m_D2).start()

    print('skipped fist 1 s')

    t_f=5000
    print(idx)
    net.simulate(t_f)
    print('simulated')


    data_GPe_TI =net.get( m_GPe_TI)
    data_D2 = net.get(m_D2)

   


    bin_value=1
   
    data_GPe_TI = net.get( m_GPe_TI).get('spike')
    data_D2 = net.get(m_D2).get('spike')
    #computation of the activities of D2 and GPe-TI
    hist_GPe_TI = histogram(data_GPe_TI, bins=bin_value)
    hist_D2 = histogram(data_D2, bins=bin_value)

    net.get(m_GPe_TI).stop()
    net.get(m_D2).stop()


    return hist_GPe_TI,hist_D2

  #for each disconnection, 5 simulations are performed
  n_trials=5
  result[disc]=parallel_run(method=simulation, number=n_trials,max_processes=5)

#saving the data in a pickle file
import pickle


pickle.dump(result, open( "disc.p", "wb" ) )

