

from ANNarchy import *
#dopamine depletion parameter: 0.75 correspond to healthy condition, 1 to Parkinsonian
D_d=np.linspace(0.75,1.05,31)
result={}
for Dd in D_d:
  #this is required to perform multiple simulation with a for loop
  clear()
  #integration step
  dt_value = 0.1

  setup(dt=dt_value, structural_plasticity= True, method='rk4')
 
  # Definition of the neuronal model for each nucleus
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
          tau_in = 7.0            : population
          tau_w = 20.0            : population
          a = 2.5 * 1e-3          : population
          b = 70.0 * 1e-3         : population
          V_peak = 15.0           : population
          V_reset = -60.0         : population
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

  # Creation of the neural populations  
  P_STN = Population(geometry=408, neuron=STN)
  P_GPe_TA = Population(geometry=264, neuron=GPe_TA)
  P_GPe_TI = Population(geometry=780, neuron=GPe_TI)
  P_FSN = Population(geometry=420, neuron=FSN)
  P_D2 = Population(geometry=6000, neuron=D2)
  P_D1 = Population(geometry=6000, neuron=D1)
  # Creation of the external populations (excitatory poissonian inputs)
  P_ext_STN = PoissonPopulation(408, rates=0.5 * 1e3)
  P_ext_GPe_TA = PoissonPopulation(264, rates=0.17 * 1e3)
  P_ext_GPe_TI = PoissonPopulation(780, rates=600)
  P_ext_FSN = PoissonPopulation(420, rates=0.9444 * 1e3)

  # Dd (i.e., the dopamine depletion parameter) changes the rate of external inputs to D2
  P_ext_D2 = PoissonPopulation(6000, rates=(0.9729278004599998/0.9) * Dd * 1e3*1.11)

  P_ext_D1 = PoissonPopulation(6000, rates=1.12 * 1e3*1.14)

  # Definition of the synaptic projections
  D1toD1 = Projection(pre=P_D1, post=P_D1, target='in')
  D1toD1.connect_fixed_probability(weights=0.12 * 1e-3, probability=0.0607, delays=1.7)

  D1toD2 = Projection(pre=P_D1, post=P_D2, target='in')
  D1toD2.connect_fixed_probability(weights=0.30 * 1e-3, probability=0.0140, delays=1.7)

  D2toD1 = Projection(pre=P_D2, post=P_D1, target='in')
  D2toD1.connect_fixed_probability(weights=0.36 * 1e-3, probability=0.0653, delays=1.7)

  D2toD2 = Projection(pre=P_D2, post=P_D2, target='in')
  D2toD2.connect_fixed_probability(weights=0.25 * 1e-3, probability=0.0840, delays=1.7)

  D2toGPe_TI = Projection(pre=P_D2, post=P_GPe_TI, target='in')
  D2toGPe_TI.connect_fixed_probability(weights=1.28 * 1e-3, probability=0.0833, delays=7.0)

  FSNtoD1 = Projection(pre=P_FSN, post=P_D1, target='in')
  FSNtoD1.connect_fixed_probability(weights=6.6 * 1e-3, probability=0.0381, delays=1.7)

  FSNtoFSN = Projection(pre=P_FSN, post=P_FSN, target='in')
  FSNtoFSN.connect_fixed_probability(weights=0.5 * 1e-3, probability=0.0238, delays=1.0)

  FSNtoD2 = Projection(pre=P_FSN, post=P_D2, target='in')
  FSNtoD2.connect_fixed_probability(weights=4.8 * 1e-3, probability=0.0262, delays=1.7)
  
  GPe_TItoGPe_TI = Projection(pre=P_GPe_TI, post=P_GPe_TI, target='in')
  GPe_TItoGPe_TI.connect_fixed_probability(weights=1.1 * 1e-3, probability=0.0321, delays=1.8)

  GPe_TItoGPe_TA = Projection(pre=P_GPe_TI, post=P_GPe_TA, target='in')
  GPe_TItoGPe_TA.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0321, delays=1.0)

  GPe_TItoFSN = Projection(pre=P_GPe_TI, post=P_FSN, target='in')
  GPe_TItoFSN.connect_fixed_probability(weights=1.6 * 1e-3, probability=0.0128, delays=7.0)

  GPe_TItoSTN = Projection(pre=P_GPe_TI, post=P_STN, target='in')
  GPe_TItoSTN.connect_fixed_probability(weights=0.08 * 1e-3, probability=0.0385, delays=1.0)
  
  GPe_TAtoD1 = Projection(pre=P_GPe_TA, post=P_D1, target='in')
  GPe_TAtoD1.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0379, delays=7.0)
  
  GPe_TAtoD2 = Projection(pre=P_GPe_TA, post=P_D2, target='in')
  GPe_TAtoD2.connect_fixed_probability(weights=0.61 * 1e-3, probability=0.0379, delays=7.0)
 
  GPe_TAtoFSN = Projection(pre=P_GPe_TA, post=P_FSN, target='in')
  GPe_TAtoFSN.connect_fixed_probability(weights=1.85 * 1e-3, probability=0.0379, delays=7.0)
  
  GPe_TAtoGPe_TA = Projection(pre=P_GPe_TA, post=P_GPe_TA, target='in')
  GPe_TAtoGPe_TA.connect_fixed_probability(weights=0.35 * 1e-3, probability=0.0189, delays=1.0)

  GPe_TAtoGPe_TI = Projection(pre=P_GPe_TA, post=P_GPe_TI, target='in')
  GPe_TAtoGPe_TI.connect_fixed_probability(weights=1.2 * 1e-3, probability=0.0189, delays=1.0)
  
  STNtoGPe_TA = Projection(pre=P_STN, post=P_GPe_TA, target='ex')
  STNtoGPe_TA.connect_fixed_probability(weights=0.13 * 1e-3, probability=0.0735, delays=2.0)

  STNtoGPe_TI = Projection(pre=P_STN, post=P_GPe_TI, target='ex')
  STNtoGPe_TI.connect_fixed_probability(weights=0.42 * 1e-3, probability=0.0735, delays=2.0)
 
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
  
  # Define monitors to record the spiking activity of the neurons
  m_STN = Monitor(P_STN,['spike'],start=False,period=1)
  m_GPe_TA = Monitor(P_GPe_TA, ['spike'],start=False,period=1)
  m_GPe_TI = Monitor(P_GPe_TI, ['spike'],start=False,period=1)
  m_FSN = Monitor(P_FSN, ['spike'],start=False,period=1)
  m_D2 = Monitor(P_D2,  ['spike'],start=False,period=1)
  m_D1 = Monitor(P_D1, ['spike'],start=False,period=1)


  # Function defined to simulates 5 seconds of the network dynamics 
  def simulation(idx,net):
    #Initial conditions for the membrane potentials
    net.get(P_STN).v=np.random.normal(-80.2,1,size=408)
    net.get(P_GPe_TA).v = np.random.normal(-55.1, 1,size=264)
    net.get(P_GPe_TI).v = np.random.normal(-55.1, 1,size=780)
    net.get(P_FSN).v = np.random.normal(-80.0, 1,size=420)
    net.get(P_D2).v = np.random.normal(-80.0, 1,size=6000)
    net.get(P_D1).v = np.random.normal(-78.2, 1,size=6000)
    
    # First second is skipped
    net.simulate(1000)
    #start of the recording
    net.get(m_STN).start()
    net.get(m_GPe_TA).start()
    net.get(m_GPe_TI).start()
    net.get(m_FSN).start()
    net.get(m_D2).start()
    net.get(m_D1).start()

    # Actual simulation
    t_f=5000
    net.simulate(t_f)

    # Recovering the spike data
    data_STN = net.get(m_STN).get('spike')
    data_GPe_TA = net.get(m_GPe_TA).get('spike')
    data_GPe_TI = net.get( m_GPe_TI).get('spike')
    data_FSN =  net.get(m_FSN).get('spike')
    data_D2 = net.get(m_D2).get('spike')
    data_D1 = net.get(m_D1).get('spike')

    # Computation of the activities 
    bin_value=1
    hist_STN = histogram(data_STN, bins=bin_value)
    hist_GPe_TA = histogram(data_GPe_TA, bins=bin_value)
    hist_GPe_TI = histogram(data_GPe_TI, bins=bin_value)
    hist_FSN =histogram(data_FSN, bins=bin_value)
    hist_D2 = histogram(data_D2, bins=bin_value)
    hist_D1 = histogram(data_D1, bins=bin_value)

    # This is the right way to free the monitors at the end of each simulation. Required to avoid excessive RAM consumption.
    net.get(m_STN).stop()
    net.get(m_GPe_TA).stop()
    net.get(m_GPe_TI).stop()
    net.get(m_FSN).stop()
    net.get(m_D2).stop()
    net.get(m_D1).stop()
    # return the activities for each nucleus
    return  hist_D1, hist_D2, hist_FSN, hist_GPe_TA, hist_GPe_TI, hist_STN 

  n_trials=5
  # Simulations in parallel and storing of recorded activities. Results are returned in a dictionary with keys equal to the value of Dd.
  result[str(Dd)]=parallel_run(method=simulation, number=n_trials,max_processes=2)

#saving the data in a pickle file
import pickle
pickle.dump(result, open( "dd.p", "wb" ) )
