##############################################################################################################################
#################################################### 1. IMPORTS ##############################################################
##############################################################################################################################

from config import *                                #all constants, defined in the config.py file
from functions.useful_functions import *            #useful functions, defined in the functions/useful_functions.py file

##############################################################################################################################
############################################ 2  REDSHIFT DISTRIBUTIONS #######################################################
##############################################################################################################################

from functions.redshift_distributions import *      #redshift distributions class

redshift_distribution_E = Redshift_Distributions(NGal, binscheme = binscheme_E, Nbinz = Nbinz_E, zmax_dist = zmax_E)
redshift_distribution_P = Redshift_Distributions(NGal, binscheme = binscheme_P, Nbinz = Nbinz_P, zmax_dist = zmax_P)

redshift_distributions = {"E" : redshift_distribution_E,
                          "P" : redshift_distribution_P}

save_pickle(redshift_distributions, f'data/{suffix}/redshift_distributions', f"Saved redshift distributions")
add_dict(redshift_distributions)

print(f"Finished 2. Redshift Distributions.")

##############################################################################################################################
################################################# 3. CORRELATIONS ############################################################
##############################################################################################################################

if compute_correlations:

    Thetamin = arcmintorad(Thetamin_arcmin)  #minimum theta from which we calculate correlation functions (in radians)
    
    ####################################### 3.1 Euclid lenses and galaxies ##################################################
    #########################################################################################################################
    
    #reading in the forecasted sample of Euclid lenses
    Euclid_lenses = np.loadtxt('lenses_Euclid.txt')
    zd = Euclid_lenses[:, 0]
    zs = Euclid_lenses[:, 1]
    
    # convert into comoving distances (in Mpc)
    chid = background.comoving_radial_distance(zd)
    chis = background.comoving_radial_distance(zs)
    
    chimax_L = max(chis)

    chimax_E = background.comoving_radial_distance(zmax_E)
    chimax_P = background.comoving_radial_distance(zmax_P)

    chimax = max(chimax_L,chimax_E,chimax_P) 
    
    #place these variables in the global dictionary
    add_dict(chimax, chid, chis, zd, zs)

    ##############################################################################################################################
    ############################################ 3.2 AUTOCORRELATION FUNCTIONS ###################################################
    ##############################################################################################################################
    
    from functions.correlations.get_correlations import *
    
    ######################################################### 3.2.1 LL ###########################################################
    ##############################################################################################################################
    
    from functions.correlations.LL import *
    
    ################################################# 3.2.1.1 weight functions ######################################################
    
    # Interpolate to get a fast 1D weight function
    W_LOS_mean_vec = np.vectorize(W_LOS_mean)
    chi = np.linspace(0, chimax, 100)
    W = W_LOS_mean_vec(chi)
    W_LOS_mean_intp = CubicSpline(chi, W)
    
    # Interpolate to get a fast 1D weight function
    WW_LOS_mean_vec = np.vectorize(WW_LOS_mean)
    chi = np.linspace(0, chimax, 100)
    WW = WW_LOS_mean_vec(chi)
    WW_rms = np.sqrt(WW)
    WW_LOS_rms_intp = CubicSpline(chi, WW_rms)
    
    add_dict(W_LOS_mean_intp, WW_LOS_rms_intp)
    
    ######################################################## 3.2.1.2 cls ###########################################################
    
    ls, cl2, cl1 = get_cl_L(chimax, lmax, nl)
    cl2_LOS_intp = CubicSpline(ls, cl2)
    cl1_LOS_intp = CubicSpline(ls, cl1)
    
    ############################################# 3.2.1.3 correlation functions ####################################################
    
    Theta, LLp, LLx, LL_plus, LL_minus = get_correlations(
        cl2_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    
    LLp = CubicSpline(Theta, LLp)
    LLx = CubicSpline(Theta, LLx)
    
    LL_plus = CubicSpline(Theta, LL_plus)
    LL_minus = CubicSpline(Theta, LL_minus)
    
    Theta, L0p, L0x, L0_plus, L0_minus = get_correlations(
        cl1_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    
    L0_plus = CubicSpline(Theta, L0_plus)

    L0 = L0_plus(0)

    LL_plus_primitive = compute_antiderivative(LL_plus, Thetamax_LL_plus)
    LL_minus_primitive = compute_antiderivative(LL_minus, Thetamax_LL_minus)
    
    print('Finished 3.2.1 LL autocorrelation functions')
    
    add_dict(LLp, LLx, L0, LL_plus, LL_minus, LL_plus_primitive, LL_minus_primitive)
    
    ######################################################## 3.2.2 EE ###########################################################
    #############################################################################################################################
    
    from functions.correlations.EE import *
    
    ################################################# 3.2.2.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    W_os_mean_intp = []
    
    for b in range(Nbinz_E):
        W_os_mean_vec = np.vectorize(W_os_mean)
        chi = np.linspace(1e-3, chimax, 100)
        W = W_os_mean_vec(chi, b)
        W_os_mean_intp.append(CubicSpline(chi, W))
    
    # Interpolate to get fast 1D weight functions
    
    WW_os_rms_intp = []
    
    for b in range(Nbinz_E):
        WW_os_mean_vec = np.vectorize(WW_os_mean)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        WW = WW_os_mean_vec(chi, b)
        WW_rms = np.sqrt(WW)                             #potential for confusion - WW_rms is actually order W
        WW_os_rms_intp.append(CubicSpline(chi, WW_rms))
    
    add_dict(W_os_mean_intp, WW_os_rms_intp)
    
    ######################################################## 3.2.2.2 cls ###########################################################
    
    ls_list = []
    cl2_eps_intp = []
    cl1_eps_intp = []
    
    for b1 in range(Nbinz_E): #loop through b
    
        cl2_eps_intp.append([])
    
        for b2 in range(Nbinz_E): #loop through b'
            
            ls, cl2, cl1 = get_cl_E(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_eps_intp[b1].append(CubicSpline(ls, cl2))

            if b1 == b2:
                ls_list.append(ls)
                cl1_eps_intp.append(CubicSpline(ls, cl1))
    
    ############################################# 3.2.2.3 correlation functions ####################################################
    
    Theta_list = []
    EEp_list = []
    EEx_list = []
    E0_list = []
    EE_plus_list = []
    EE_minus_list = []
    
    for b1 in range(Nbinz_E):
            
        EEp_list.append([])
        EEx_list.append([])
            
        EE_plus_list.append([])
        EE_minus_list.append([])
            
        Theta, E0p, E0x, E0_plus, E0_minus = get_correlations(
            cl1_eps_intp[b1], Thetamin, Thetamax, nTheta)
            
        E0_list.append(E0_plus)
    
        for b2 in range(Nbinz_E):
            Theta, EEp, EEx, EE_plus, EE_minus = get_correlations(
                cl2_eps_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            EEp_list[b1].append(EEp)
            EEx_list[b1].append(EEx)
            
            EE_plus_list[b1].append(EE_plus)
            EE_minus_list[b1].append(EE_minus)
        
        Theta_list.append(Theta)
    
    EEp = []
    EEx = []
    
    EE_plus = []
    EE_minus = []
    
    E0 = []
    
    for b1 in range(Nbinz_E):
        
        EEp.append([])
        EEx.append([])
        
        EE_plus.append([])
        EE_minus.append([])
        
        E0.append(CubicSpline(Theta_list[b1], E0_list[b1])(0))
    
        for b2 in range(Nbinz_E):
            EEp[b1].append(CubicSpline(Theta_list[b1], EEp_list[b1][b2]))
            EEx[b1].append(CubicSpline(Theta_list[b1], EEx_list[b1][b2]))
            
            EE_plus[b1].append(CubicSpline(Theta_list[b1], EE_plus_list[b1][b2]))
            EE_minus[b1].append(CubicSpline(Theta_list[b1], EE_minus_list[b1][b2]))
            
    print('Finished 3.2.2 EE autocorrelation functions')
    
    add_dict(EEp, EEx, E0, EE_plus, EE_minus)

    ######################################################### 3.2.3 PP ###########################################################
    ##############################################################################################################################
    
    from functions.correlations.PP import *
    
    ################################################# 3.2.3.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    W_d_mean_intp = []
    
    for b in range(Nbinz_P):
        W_d_mean_vec = np.vectorize(W_d)
        chi = np.linspace(1e-5, chimax, 100)
        W = W_d_mean_vec(chi, b)
        W_d_mean_intp.append(CubicSpline(chi, W))
    
    # Interpolate to get fast 1D weight functions
    
    WW_d_rms_intp = []
    
    for b in range(Nbinz_P):
        WW_d_mean_vec = np.vectorize(WW_d)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        WW = WW_d_mean_vec(chi, b)
        WW_rms = np.sqrt(WW)
        WW_d_rms_intp.append(CubicSpline(chi, WW_rms))

    W_d_intp = W_d_mean_intp  #redundant, fix this
    add_dict(W_d_mean_intp, WW_d_rms_intp)
    add_dict(W_d_intp)
    
    ######################################################## 3.2.3.2 cls ###########################################################
    
    ls_list = []
    cl2_d_intp = []
    
    for b1 in range(Nbinz_P): #loop through b
    
        cl2_d_intp.append([])
        
        for b2 in range(Nbinz_P): #loop through b'
            
            ls, cl2 = get_cl_P(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_d_intp[b1].append(CubicSpline(ls, cl2))

            if b1 == b2:
                ls_list.append(ls)

    #Note - because of the weight function, all of these will be zero unless b1 == b2 
    
    ############################################# 3.2.3.3 correlation functions ####################################################
    
    Theta_list = []
    PP_list = []
    
    for b1 in range(Nbinz_P):
            
        PP_list.append([])
    
        for b2 in range(Nbinz_P):
            
            Theta, PP = get_DD_correlations(
                cl2_d_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            PP_list[b1].append(PP)
        
        Theta_list.append(Theta)
    
    PP = []
    
    for b1 in range(Nbinz_P):
        
        PP.append([])
    
        for b2 in range(Nbinz_P):
            PP[b1].append(CubicSpline(Theta_list[b1], PP_list[b1][b2]))
            
    print('Finished 3.2.3 PP autocorrelation functions')
    
    add_dict(PP)
    
    ##############################################################################################################################
    ############################################ 3.3 MIXED CORRELATION FUNCTIONS #################################################
    ##############################################################################################################################
    
    ######################################################## 3.3.1 LE ############################################################
    ##############################################################################################################################
    
    from functions.correlations.LE import *
    
    ls_list = []
    cl2LOSos_intp_list = []
    
    for b in range(Nbinz_E):

        ls, cl2LOSos = get_cls_mixed_LE(b, chimax, lmax, nl)

        ls_list.append(ls)

        cl2LOSos_intp_list.append(CubicSpline(ls, cl2LOSos))
    
    Theta_list = []
    
    LEp_list = []
    LEx_list = []

    LE_plus_list = []
    LE_minus_list = []
    
    for b1 in range(Nbinz_E):
        
        Theta, LEp, LEx, LE_plus, LE_minus = get_correlations(
            cl2LOSos_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        LEp_list.append(LEp)
        LEx_list.append(LEx)
        
        LE_plus_list.append(LE_plus)
        LE_minus_list.append(LE_minus)
    
        Theta_list.append(Theta)
    
    LEp = []
    LEx = []
    
    LE_plus = []
    LE_minus = []
    
    LE_plus_primitive = []
    LE_minus_primitive = []
    
    for b1 in range(Nbinz_E):
        
        LEp.append(CubicSpline(Theta_list[b1], LEp_list[b1]))
        LEx.append(CubicSpline(Theta_list[b1], LEx_list[b1]))
        
        LE_plus.append(CubicSpline(Theta_list[b1], LE_plus_list[b1]))
        LE_minus.append(CubicSpline(Theta_list[b1], LE_minus_list[b1]))
        
        LE_plus_primitive.append(compute_antiderivative(LE_plus[b1], Thetamax_LE_plus))
        LE_minus_primitive.append(compute_antiderivative(LE_minus[b1], Thetamax_LE_minus))
    
    print('Finished 3.3.1 LE correlation functions')
    
    add_dict(LEp, LEx, LE_plus, LE_minus, LE_plus_primitive, LE_minus_primitive)
    
    ######################################################### 3.3.2 LP ###########################################################
    ##############################################################################################################################
    
    from functions.correlations.LP import *
    
    ls_list = []
    cl2LOSd_intp_list = []
    
    for b in range(Nbinz_P):
        
        ls, cl2LOSd = get_cls_mixed_LP(b, chimax, lmax, nl)
            
        ls_list.append(ls)
        cl2LOSd_intp_list.append(CubicSpline(ls, cl2LOSd))
    
    Theta_list = []
    
    LP_list = []
    
    for b in range(Nbinz_P):
        
        Theta, LP = get_gD_correlations(
            cl2LOSd_intp_list[b], Thetamin, Thetamax, nTheta)
        
        LP_list.append(LP)
    
        Theta_list.append(Theta)
    
    LP = []
    
    LP_primitive = []
    
    for b in range(Nbinz_P):
        
        LP.append(CubicSpline(Theta_list[b], LP_list[b]))
        
        LP_primitive.append(compute_antiderivative(LP[b], Thetamax_LP))
    
    print('Finished 3.3.2 LP correlation functions')
    
    add_dict(LP, LP_primitive)
    
    ######################################################### 3.3.3 EP ##########################################################
    ##############################################################################################################################

    from functions.correlations.EP import *
    
    ls_list = []
    cl2dos_intp_list = []
    
    for b1 in range(Nbinz_E):
        
        cl2dos_intp_list.append([])
    
        for b2 in range(Nbinz_P):
        
            ls, cl2dos = get_cls_mixed_EP(b1, b2, chimax, lmax, nl)
            
            cl2dos_intp_list[b1].append(CubicSpline(ls, cl2dos))
            
        ls_list.append(ls)
    
    Theta_list = []
    
    EP_list = []
    
    for b1 in range(Nbinz_E):
        
        EP_list.append([])
        
        for b2 in range(Nbinz_P):
        
            Theta, EP = get_gD_correlations(
                cl2dos_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            EP_list[b1].append(EP)
    
        Theta_list.append(Theta)
    
    EP = []
    
    for b1 in range(Nbinz_E):
        
        EP.append([])
    
        for b2 in range(Nbinz_P):
            
            EP[b1].append(CubicSpline(Theta_list[b1], EP_list[b1][b2]))
            
    print('Finished 3.3.3 shape position correlation functions')
    
    add_dict(EP)
        
    save_pickle(global_dict, f"correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}", f"Saved all correlations")

load_correlations(filename=f"correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}")
correlations_folder = f'data/{suffix}/binned_correlations'
            
print('Finished 3. Correlation Functions')

##############################################################################################################################
############################################## 4 ANGULAR DISTRIBUTIONS #######################################################
##############################################################################################################################

from functions.angular_distributions import *       #angular distributions class

if supply_binscheme:

    ######## LL
    
    angular_distribution_LL_plus = Angular_Distributions(Nlens, binscheme=binscheme_LL_plus, Nbin_a=Nbina_LL_plus, Thetamax=Thetamax_LL_plus)
    angular_distribution_LL_minus = Angular_Distributions(Nlens, binscheme=binscheme_LL_minus, Nbin_a=Nbina_LL_minus, Thetamax=Thetamax_LL_minus)  

    ######## LE

    angular_distribution_LE_plus = []
    angular_distribution_LE_minus = []

    ad_plus = Angular_Distributions(NGal, binscheme=binscheme_LE_plus, Nbin_a=Nbina_LE_plus, Thetamax=Thetamax_LE_plus)
    ad_minus = Angular_Distributions(NGal, binscheme=binscheme_LE_minus, Nbin_a=Nbina_LE_minus, Thetamax=Thetamax_LE_minus)
    
    for zbin in range(Nbinz_E):
        
        angular_distribution_LE_plus.append(ad_plus)
        angular_distribution_LE_minus.append(ad_minus)

    ######## LP
    
    ad = Angular_Distributions(NGal, binscheme=binscheme_LP, Nbin_a=Nbina_LP, Thetamax=Thetamax_LP)
    
    for zbin in range(Nbinz_P):
        
        angular_distribution_LP_plus.append(ad)
    
else:
    get_item('LL_plus', 'LL_plus_primitive', 'LL_minus', 'LL_minus_primitive', 'LE_plus', 'LE_plus_primitive', 'LE_minus', 'LE_minus_primitive', 'LP', 'LP_primitive')

    ######## LL
    
    binscheme_LL_plus = optimise_bins(LL_plus, 'LL', LL_plus_primitive, b = None, SNR_goal = SNR_goal_LL_plus, Nbin_max = Nbin_max_LL_plus, SNR_min = SNR_min_LL_plus)
    binscheme_LL_minus = optimise_bins(LL_minus, 'LL', LL_minus_primitive, b = None, SNR_goal = SNR_goal_LL_minus, Nbin_max = Nbin_max_LL_minus, SNR_min = SNR_min_LL_minus)

    angular_distribution_LL_plus = Angular_Distributions(Nlens, binscheme=binscheme_LL_plus, Nbin_a = None, Thetamax=Thetamax_LL_plus)
    angular_distribution_LL_minus = Angular_Distributions(Nlens, binscheme=binscheme_LL_minus, Nbin_a = None, Thetamax=Thetamax_LL_minus)  

    ######## LE

    binscheme_LE_plus = []
    binscheme_LE_minus = []

    angular_distribution_LE_plus = []
    angular_distribution_LE_minus = []
    
    for zbin in range(Nbinz_E):
        
        binscheme_LE_plus.append(optimise_bins(LE_plus[zbin], 'LE', LE_plus_primitive[zbin], b = zbin, SNR_goal = SNR_goal_LE_plus, Nbin_max = Nbin_max_LE_plus, SNR_min = SNR_min_LE_plus))
        binscheme_LE_minus.append(optimise_bins(LE_minus[zbin], 'LE', LE_minus_primitive[zbin], b = zbin, SNR_goal = SNR_goal_LE_minus, Nbin_max = Nbin_max_LE_minus, SNR_min = SNR_min_LE_minus))
        
        angular_distribution_LE_plus.append(Angular_Distributions(NGal, binscheme=binscheme_LE_plus[zbin], Nbin_a = None, Thetamax=Thetamax_LE_plus))
        angular_distribution_LE_minus.append(Angular_Distributions(NGal, binscheme=binscheme_LE_minus[zbin], Nbin_a = None, Thetamax=Thetamax_LE_minus))

    ######## LP

    binscheme_LP = []

    angular_distribution_LP = []
    
    for zbin in range(Nbinz_P):
        
        binscheme_LP.append(optimise_bins(LP[zbin], 'LP', LP_primitive[zbin], b = zbin, SNR_goal = SNR_goal_LP, Nbin_max = Nbin_max_LP, SNR_min = SNR_min_LP))
        
        angular_distribution_LP.append(Angular_Distributions(NGal, binscheme=binscheme_LP[zbin], Nbin_a = None, Thetamax=Thetamax_LP))

    print(f'binscheme LL_plus: {binscheme_LL_plus}')
    print(f'binscheme LL_minus: {binscheme_LL_minus}')
    print(f'binscheme LE_plus: {binscheme_LE_plus}')
    print(f'binscheme LE_minus: {binscheme_LE_minus}')
    print(f'binscheme_LP: {binscheme_LP}')

angular_distributions = {"LL_plus" : angular_distribution_LL_plus,
                         "LL_minus" : angular_distribution_LL_minus,
                         "LE_plus" : angular_distribution_LE_plus,
                         "LE_minus" : angular_distribution_LE_minus,
                         "LP" : angular_distribution_LP}

save_pickle(angular_distributions, f'data/{suffix}/angular_distributions', f"Saved angular distributions")
add_dict(angular_distributions)

print('Finished 4. Angular Distributions')

##############################################################################################################################
############################################## 5 PREPARING FOR THE RUN #######################################################
##############################################################################################################################

########################################## 5.1 Saving binned correlations ####################################################
##############################################################################################################################

os.makedirs(correlations_folder, exist_ok=True)

LL_binned = generate_binned_correlation('LL', 0)

save_pickle(LL_binned, f"{correlations_folder}/LL", f"correlation=LL")

for b1 in range(Nbinz_E):

    LE_binned = generate_binned_correlation('LE', b1)
    save_pickle(LE_binned, f"{correlations_folder}/LE{b1}", f"b1={b1}, correlation=LE")

for b1 in range(Nbinz_P):

    LP_binned = generate_binned_correlation('LP', b1)
    save_pickle(LP_binned, f"{correlations_folder}/LP{b1}", f"b1={b1}, correlation=LP")

############################################### 5.2 Creating .txt file ########################$##############################
##############################################################################################################################


# Output file name
task_file = "tasks.txt"

lines = []

Elist = list(range(Nbinz_E))
Plist = list(range(Nbinz_P))

# Full (b1, b2) iteration
for cov_matrix in cov_matrices_full:
    for cov_type in cov_types:
        if cov_matrix == 'LELP':
            b1b2_pairs = [(i, j) for i in Elist for j in Plist]
        elif cov_matrix == 'LELE':
            b1b2_pairs = [(i, j) for i in Elist for j in Elist if i <= j]
        elif cov_matrix == 'LPLP':
            b1b2_pairs = [(i, j) for i in Plist for j in Plist if i <= j]

        for b1, b2 in b1b2_pairs:
            lines.append(f"{b1} {b2} {cov_matrix} {cov_type}")

for cov_matrix in cov_matrices_b1:
    # Choose b1_values depending on cov_matrix
    if cov_matrix == 'LLLE':
        b1_values = Elist
    elif cov_matrix == 'LLLP':
        b1_values = Plist
    else:
        raise ValueError(f"Unknown cov_matrix type: {cov_matrix}")

    for b1 in b1_values:
        for cov_type in cov_types:
            lines.append(f"{b1} {cov_matrix} {cov_type}")

# No b1, b2 iteration (only one call)
for cov_matrix in cov_matrices_no_b:
    for cov_type in cov_types:
        lines.append(f"{cov_matrix} {cov_type}")

# Write all lines without trailing newline
with open(task_file, "w") as f:
    f.write("\n".join(lines))

print(f"Task file '{task_file}' created successfully with all required jobs.")

############################################### 5.3 Creating .sh file ########################$##############################
##############################################################################################################################

ntasks = 2*(Nbinz_E * Nbinz_P + Nbinz_E + Nbinz_P + 1) + Nbinz_E * (Nbinz_E + 1) + Nbinz_P * (Nbinz_P + 1)
ncpus = min(ntasks, max_cpus)

job_name = f"all_covariance"
output_dir = f"output{suffix}"

runtime_log = f"runtime_{format_sci(nsamp)}.log"

if os.path.exists(runtime_log):
    os.remove(runtime_log)
    
lockfile = f"/tmp/runtime_{format_sci(nsamp)}.lock"

script_content = f"""#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/parallel-job-output_%A_%a.log
#SBATCH --error={output_dir}/parallel-job-error_%A_%a.log
#SBATCH --mail-user=daniel.johnson@umontpellier.fr
#SBATCH --mail-type=TIME_LIMIT_80
#SBATCH --time=15-00:00:00
#SBATCH --array=0-{int(ntasks-1)}%{ncpus}
#SBATCH --mem=10000
#SBATCH --partition=lupm
#SBATCH --exclude=tumce[2-4]
#SBATCH --dependency=singleton

source ~/lenstronomyenv/bin/activate

# Record start time
start_time=$(date +%s)

# Read task from task list
TASK_FILE="tasks.txt"
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" ${{TASK_FILE}})

# Run the corresponding task
python -u job.py ${{TASK}}

# Record end time
end_time=$(date +%s)
runtime_seconds=$((end_time - start_time))
runtime_minutes=$((runtime_seconds / 60))

# Log runtime with task parameters safely
{{
  flock -x 201
  echo "Job ${{SLURM_ARRAY_TASK_ID}} | Task: ${{TASK}} | Runtime: ${{runtime_minutes}} min (${{runtime_seconds}} sec)" >> {runtime_log}
}} 201>{lockfile}

cat {output_dir}/parallel-job-output_${{SLURM_ARRAY_JOB_ID}}_*.log > {output_dir}/combined_output.log
cat {output_dir}/parallel-job-error_${{SLURM_ARRAY_JOB_ID}}_*.log > {output_dir}/combined_errors.log
"""

with open("parallel_jobs.sh", "w") as f:
    f.write(script_content)

print(f"Slurm file created.")
