
############################ reading in files ############################

def load_file(filename):
    """Loads a pickled dictionary"""
    with open(filename, "rb") as f:
        data = pickle.load(f)  # Load the dictionary from the pickle file

    return(data)

############################ reading correlations ############################

def read_correlations(base_folder=folder_name):

    correlations_data = {}

    LL_file = f'data/{base_folder}/binned_correlations/LL'
    
    try:
        with open(LL_file, "rb") as f:
            correlations_data['LL'] = pickle.load(f)
    except Exception as ex:
        print(f"Error reading LL in {LL_file}: {ex}")

    for corr in ['LE', 'LP']: 

        correlations_data[corr] = []

        if corr == 'LE':
            loop = Nbinz_E
        elif corr == 'LP':
            loop = Nbinz_P
            
        for b1 in range(loop):
            file_name = f'data/{base_folder}/binned_correlations/{corr}{b1}'
            
            try:
                with open(file_name, "rb") as f:
                    correlations_data[corr].append(pickle.load(f))
            except Exception as ex:
                print(f"Error reading {corr} in {file_name}: {ex}")
                
    
    return correlations_data

############################ reading covariance ############################

def read_all_matrices(base_folder=folder_name):
    """
    Reads ccov, ncov, scov files from each folder in folder_structure,
    now expecting filenames like ccov_0, scov_1_2, etc.
    """
    covariance_data = {}

    for folder in folder_structure:
        folder_path = os.path.join("data", base_folder, "covariance", folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} not found, skipping")
            continue

        # Split into the two 2-letter codes
        comp1, comp2 = folder[:2], folder[2:]
        n1, n2 = Nbinz_map.get(comp1), Nbinz_map.get(comp2)
        if n1 is None or n2 is None:
            print(f"Unknown comp in '{folder}', skipping")
            continue

        # Determine which dimensions vary
        vary1 = (n1 > 0 and comp1 != 'LL')
        vary2 = (n2 > 0 and comp2 != 'LL')

        # Build index tuples and string formatters
        if vary1 and vary2:
            index_iter = itertools.product(range(n1), range(n2))
            fmt = lambda i, j: f"_{i}_{j}"
        elif vary1:
            index_iter = ((i,) for i in range(n1))
            fmt = lambda i: f"_{i}"
        elif vary2:
            index_iter = ((j,) for j in range(n2))
            fmt = lambda j: f"_{j}"
        else:
            index_iter = [()]
            fmt = lambda: ""

        covariance_data[folder] = {name: {} for name in ("ccov", "ncov", "scov")}

        for idx in index_iter:
            idx_str = fmt(*idx)

            for cov_name in ("ccov", "ncov", "scov"):
                fpath = os.path.join(folder_path, f"{cov_name}{idx_str}")
                try:
                    with open(fpath, "rb") as f:
                        mat = pickle.load(f)
                    if mat is not None:
                        covariance_data[folder][cov_name][idx] = mat
                except FileNotFoundError:
                    pass  # Silently skip missing files
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")

    return covariance_data

############################ constructing large matrices ############################

def construct_large_matrices(covariance_data: Dict[str, Dict[str, Dict[Tuple, np.ndarray]]]
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs three large covariance matrices (noise, cosmic, sparsity) by arranging
    blocks according to their redshift bin indices and component types.
    
    Key features:
    - Each redshift bin can have different sizes
    - Diagonal blocks determine the row/column dimensions for each bin
    - Off-diagonal blocks must match the dimensions set by diagonal blocks
    - LL has only 1 block (no redshift binning)
    - LE and LP can have arbitrary numbers of redshift bins with arbitrary sizes
    
    Args:
        covariance_data: {
            folder_name: {
                'ncov': { key_tuple: ndarray, ... },
                'ccov': { ... },
                'scov': { ... }
            }, ...
        }
    
    Returns:
        noise_matrix, cosmic_matrix, sparsity_matrix
    """
    
    def parse_folder_name(folder: str) -> Tuple[str, str]:
        """Extract component types from folder name (e.g., 'LELE' -> ('LE', 'LE'))"""
        return folder[:2], folder[2:]
    
    def analyze_block_structure(covariance_data: Dict) -> Dict[str, Dict]:
        """
        Analyze the block structure to determine:
        1. Which (component, bin) combinations exist
        2. The size of each (component, bin) combination
        
        The size of each bin is determined from diagonal blocks where possible.
        """
        # Get any covariance type to analyze structure
        sample_cov_type = None
        for folder_data in covariance_data.values():
            for cov_type, blocks in folder_data.items():
                if blocks:
                    sample_cov_type = cov_type
                    break
            if sample_cov_type:
                break
        
        if not sample_cov_type:
            return {}
        
        # Track sizes for each (component, bin) pair
        component_bin_sizes = {}  # {(component, bin): size}
        all_component_bins = set()  # All (component, bin) pairs that exist
        
        # First pass: identify all (component, bin) pairs and their sizes from diagonal blocks
        for folder, matrices in covariance_data.items():
            if sample_cov_type not in matrices:
                continue
            
            row_comp, col_comp = parse_folder_name(folder)
            
            for key, matrix in matrices[sample_cov_type].items():
                if matrix is None:
                    continue
                
                # Determine what (component, bin) pairs this block represents
                if len(key) == 0:  # LLLL case
                    all_component_bins.add(('LL', 0))
                    # For LLLL, both row and col are LL bin 0
                    component_bin_sizes[('LL', 0)] = matrix.shape[0]  # Assuming square
                
                elif len(key) == 1:  # LLLE, LLLP case
                    if row_comp == 'LL':
                        all_component_bins.add(('LL', 0))
                        all_component_bins.add((col_comp, key[0]))
                        component_bin_sizes[('LL', 0)] = matrix.shape[0]
                        component_bin_sizes[(col_comp, key[0])] = matrix.shape[1]
                    elif col_comp == 'LL':
                        all_component_bins.add((row_comp, key[0]))
                        all_component_bins.add(('LL', 0))
                        component_bin_sizes[(row_comp, key[0])] = matrix.shape[0]
                        component_bin_sizes[('LL', 0)] = matrix.shape[1]
                    else:
                        # Both components vary, diagonal block
                        all_component_bins.add((row_comp, key[0]))
                        all_component_bins.add((col_comp, key[0]))
                        if row_comp == col_comp:  # True diagonal
                            component_bin_sizes[(row_comp, key[0])] = matrix.shape[0]
                        else:
                            component_bin_sizes[(row_comp, key[0])] = matrix.shape[0]
                            component_bin_sizes[(col_comp, key[0])] = matrix.shape[1]
                
                elif len(key) == 2:  # LELE, LPLP, LELP case
                    all_component_bins.add((row_comp, key[0]))
                    all_component_bins.add((col_comp, key[1]))
                    
                    # For diagonal blocks, we can determine the size
                    if row_comp == col_comp and key[0] == key[1]:
                        component_bin_sizes[(row_comp, key[0])] = matrix.shape[0]
                    else:
                        # Off-diagonal: sizes should be consistent with diagonal blocks
                        # We'll validate this later
                        if (row_comp, key[0]) not in component_bin_sizes:
                            component_bin_sizes[(row_comp, key[0])] = matrix.shape[0]
                        if (col_comp, key[1]) not in component_bin_sizes:
                            component_bin_sizes[(col_comp, key[1])] = matrix.shape[1]
        
        # Organize by component
        component_info = defaultdict(lambda: {'bins': [], 'sizes': {}})
        
        for (comp, bin_idx), size in component_bin_sizes.items():
            component_info[comp]['bins'].append(bin_idx)
            component_info[comp]['sizes'][bin_idx] = size
        
        # Sort bins and remove duplicates
        for comp in component_info:
            component_info[comp]['bins'] = sorted(set(component_info[comp]['bins']))
        
        return dict(component_info)
    
    def create_single_matrix(cov_type: str) -> Optional[np.ndarray]:
        """Create a single large matrix for the given covariance type."""
        
        # Step 1: Analyze block structure
        component_info = analyze_block_structure(covariance_data)
        
        if not component_info:
            print(f"No component info found for {cov_type}")
            return None
        
        # print(f"Component analysis for {cov_type}:")
        for comp in ['LL', 'LP', 'LE']:
            if comp in component_info:
                info = component_info[comp]
                # print(f"  {comp}: bins {info['bins']}")
                for bin_idx in info['bins']:
                    size = info['sizes'].get(bin_idx, 'unknown')
                    # print(f"    bin {bin_idx}: size {size}")
        
        # Step 2: Calculate offsets for each (component, bin) pair
        offsets = {}  # {(component, bin): offset}
        total_size = 0
        
        for comp in ['LL', 'LP', 'LE']:
            if comp in component_info:
                info = component_info[comp]
                for bin_idx in info['bins']:
                    offsets[(comp, bin_idx)] = total_size
                    bin_size = info['sizes'][bin_idx]
                    total_size += bin_size
        
        # Step 3: Create the large matrix
        large_matrix = np.zeros((total_size, total_size))
        
        # Step 4: Place all blocks
        for folder, matrices in covariance_data.items():
            if cov_type not in matrices or not matrices[cov_type]:
                continue
            
            row_comp, col_comp = parse_folder_name(folder)
            
            for key, matrix in matrices[cov_type].items():
                if matrix is None:
                    continue
                
                # Determine the (component, bin) pairs for this block
                row_comp_bins, col_comp_bins = determine_component_bins(folder, key)
                
                # Place the block for each combination
                for (r_comp, r_bin) in row_comp_bins:
                    for (c_comp, c_bin) in col_comp_bins:
                        
                        # Get offsets
                        row_offset = offsets.get((r_comp, r_bin))
                        col_offset = offsets.get((c_comp, c_bin))
                        
                        if row_offset is None or col_offset is None:
                            print(f"WARNING: Missing offset for ({r_comp}, {r_bin}) or ({c_comp}, {c_bin})")
                            continue
                        
                        # Get expected sizes
                        expected_row_size = component_info[r_comp]['sizes'][r_bin]
                        expected_col_size = component_info[c_comp]['sizes'][c_bin]
                        
                        # Verify matrix dimensions
                        if matrix.shape != (expected_row_size, expected_col_size):
                            print(f"WARNING: Block {folder}[{key}] has shape {matrix.shape}, "
                                  f"expected ({expected_row_size}, {expected_col_size}) for "
                                  f"({r_comp},{r_bin}) × ({c_comp},{c_bin})")
                            continue
                        
                        # Place the block
                        row_start = row_offset
                        row_end = row_start + expected_row_size
                        col_start = col_offset
                        col_end = col_start + expected_col_size
                        
                        large_matrix[row_start:row_end, col_start:col_end] += matrix
                        
                        # Handle transpose for cross-component blocks
                        if should_place_transpose(folder, r_comp, c_comp, r_bin, c_bin):
                            # Transpose: swap row and column components/bins
                            t_row_offset = offsets[(c_comp, c_bin)]
                            t_col_offset = offsets[(r_comp, r_bin)]
                            
                            t_row_start = t_row_offset
                            t_row_end = t_row_start + expected_col_size
                            t_col_start = t_col_offset
                            t_col_end = t_col_start + expected_row_size
                            
                            large_matrix[t_row_start:t_row_end, t_col_start:t_col_end] += matrix.T
        
        return large_matrix
    
    def determine_component_bins(folder: str, key: Tuple) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Determine which (component, bin) pairs a block represents.
        
        Returns: (row_component_bins, col_component_bins)
        Each is a list of (component, bin) tuples.
        """
        row_comp, col_comp = parse_folder_name(folder)
        
        if len(key) == 0:  # LLLL case
            return [('LL', 0)], [('LL', 0)]
        
        elif len(key) == 1:  # LLLE, LLLP case
            if row_comp == 'LL' and col_comp != 'LL':
                return [('LL', 0)], [(col_comp, key[0])]
            elif row_comp != 'LL' and col_comp == 'LL':
                return [(row_comp, key[0])], [('LL', 0)]
            else:
                # Both components vary - diagonal block
                return [(row_comp, key[0])], [(col_comp, key[0])]
        
        elif len(key) == 2:  # LELE, LPLP, LELP case
            return [(row_comp, key[0])], [(col_comp, key[1])]
        
        else:
            raise ValueError(f"Unexpected key length: {key}")
    
    def should_place_transpose(folder: str, row_comp: str, col_comp: str, 
                             row_bin: int, col_bin: int) -> bool:
        """
        Determine if we should place the transpose of this block.
        Only for off-diagonal component blocks.
        """
        # Only transpose cross-component blocks
        if row_comp == col_comp:
            return False
        
        # Only transpose certain folder types that represent symmetric relationships
        if folder not in ['LLLE', 'LLLP', 'LELP']:
            return False
        
        return True
    
    # Create each matrix type
    # print("=== Creating noise matrix ===")
    noise_matrix = create_single_matrix('ncov')
    
    # print("\n=== Creating cosmic matrix ===")
    cosmic_matrix = create_single_matrix('ccov')
    
    # print("\n=== Creating sparsity matrix ===")
    sparsity_matrix = create_single_matrix('scov')
    
    return noise_matrix, cosmic_matrix, sparsity_matrix

############################ matrix plotting ############################

def format_sci(n):
    return f'{n:.0e}'.replace('e+00', '').replace('+0', '').replace('+', '').replace('-0', '-')
                    
def find_block_idx(i):
    """
    Finds the block number of a cell in the matrix and the number of angular bins.
    Block 0 = LL minus
    Block 1 = LL plus
    Block 2..2+Nbinz_P-1 = LP redshift bins
    Block 2+Nbinz_P.. = LE redshift bins (split into minus/plus)
    """
    # LL block (block 0 and 1)
    ll_minus = angular_bin_dictionary_pm['LL']['minus'][0]
    ll_plus  = angular_bin_dictionary_pm['LL']['plus'][0]
    
    if i < ll_minus:
        return 0, ll_minus
    elif i < ll_minus + ll_plus:
        return 1, ll_plus

    offset = ll_minus + ll_plus

    # LP blocks (block 2 .. 2+Nbinz_P-1)
    for b in range(Nbinz_P):
        bins = angular_bin_dictionary['LP'][b]
        if i < offset + bins:
            return 2 + b, bins
        offset += bins

    # LE blocks (block 2+Nbinz_P .. )
    for b in range(Nbinz_E):
        minus_bins = angular_bin_dictionary_pm['LE']['minus'][b]
        plus_bins  = angular_bin_dictionary_pm['LE']['plus'][b]
        
        if i < offset + minus_bins:
            return 2 + Nbinz_P + b * 2, minus_bins
        offset += minus_bins

        if i < offset + plus_bins:
            return 2 + Nbinz_P + b * 2 + 1, plus_bins
        offset += plus_bins

    raise ValueError(f"Index {i} is out of bounds in total angular bin configuration.")

def plot_colored_covariance(covariance_data, correlations_data, scale=True, normalize=True, log_scale=True, plot_noise = True, plot_cosmic = True, plot_sparsity = True, figsize=(12, 10), eps = 1e-6, gamma = 1, eta = 1):
    """
    Creates a colored visualization of the three covariance matrices with separate colorbars.
    
    Args:
        covariance_data: Dictionary containing ncov, ccov, and scov matrices
        scale: scale the matrix by the corresponding correlations
        normalize: Whether to normalize the color intensities
        log_scale: Whether to use log scale for color intensities
        figsize: Size of the figure (width, height)
        eps:higher numbers = higher contrast
        gamma: <1 brightens darks, >1 darkens
        eta: >1 oversaturates, <1 undersaturates
    """
    # Get the three matrices
    noise_mat, cosmic_mat, sparsity_mat = construct_large_matrices(covariance_data)
    
    #here, we scale our matrices by the corresponding product of correlations. Thus we have an indication of SNR
    if scale:
        noise_mat /= np.outer(correlations_list, correlations_list)
        cosmic_mat /= np.outer(correlations_list, correlations_list)
        sparsity_mat /= np.outer(correlations_list, correlations_list)

    #here, we keep only the absolute values (for plotting)
    noise_mat = np.abs(noise_mat)
    cosmic_mat = np.abs(cosmic_mat)
    sparsity_mat = np.abs(sparsity_mat)

    max_value = max(np.max(noise_mat), np.max(cosmic_mat), np.max(sparsity_mat))
    true_max = max_value
    half_value = np.sqrt(true_max)
    
    # Apply log scale if required (np.log1p(x) is log(1+x))
    if log_scale:
        noise_mat = np.log1p(noise_mat/eps)
        cosmic_mat = np.log1p(cosmic_mat/eps)
        sparsity_mat = np.log1p(sparsity_mat/eps)
        half_value = np.sqrt(true_max * eps)
        
    noise_mat **= gamma
    cosmic_mat **= gamma
    sparsity_mat **= gamma 
    
    #here, we normalise our matrices by the maximum value
    if normalize:        
        max_val = max(np.max(noise_mat), np.max(cosmic_mat), np.max(sparsity_mat))
        max_val = max_val / eta
        if max_val > 0:
            noise_mat /= max_val
            cosmic_mat /= max_val
            sparsity_mat /= max_val
    
    # Create RGB array
    rgb_matrix = np.zeros(noise_mat.shape + (3,))
    if plot_noise:
        rgb_matrix[..., 1] = noise_mat  # Green for noise
    if plot_cosmic:
        rgb_matrix[..., 0] = cosmic_mat  # Red for cosmic
    if plot_sparsity:
        rgb_matrix[..., 2] = sparsity_mat  # Blue for sparsity
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    N = rgb_matrix.shape[0]
    im = ax.imshow(rgb_matrix, aspect='equal', extent=[-0.5, N-0.5, N-0.5, -0.5])
    
    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # For example: vertical lines after columns 2, 5, and 7
    x_lines = [angular_bin_dictionary['LL'][0], angular_bin_dictionary['LL'][0]+ np.sum([angular_bin_dictionary['LP'][b] for b in range(Nbinz_P)])]
    
    # Shift by -0.5 to align with imshow cell edges
    x_minor = [x - 0.5 for x in x_lines]
    
    # Set minor ticks only at those specific positions
    ax.set_xticks(x_minor, minor=True)
    ax.set_yticks(x_minor, minor=True)

    # Draw gridlines only at those positions
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    # Optional: Hide default tick labels if undesired
    # ax.tick_params(which='major', labelbottom=False, labelleft=False)

    # Get matrix size to determine label positioning
    matrix_size = noise_mat.shape[0]      #this gives the number of cells on an axis 
    
    # Create custom tick positions and labels
    inner_labels = []
    # LL once
    inner_labels.extend(range(1, angular_bin_dictionary_pm['LL']['minus'][0] + 1))
    inner_labels.extend(range(1, angular_bin_dictionary_pm['LL']['plus'][0] + 1))
    # LP repeated Nbinz_P times
    for b in range(Nbinz_P):
        inner_labels.extend(range(1, angular_bin_dictionary['LP'][b] + 1))
    # LE repeated 2*Nbinz_E times
    for b in range(Nbinz_E):
        inner_labels.extend(range(1, angular_bin_dictionary_pm['LE']['minus'][b] + 1))
        inner_labels.extend(range(1, angular_bin_dictionary_pm['LE']['plus'][b] + 1))

    inner_ticks = np.arange(len(inner_labels))
    
    # Create the figure with reduced distance between plot and colorbars
    # Adjust right margin to be closer to the plot
    plt.subplots_adjust(right=0.85)
    
    # Create separate colorbars closer to the plot
    cbar_ax1 = fig.add_axes([0.87, 0.7, 0.02, 0.2])  # Red (cosmic)
    cbar_ax2 = fig.add_axes([0.87, 0.4, 0.02, 0.2])  # Green (noise)
    cbar_ax3 = fig.add_axes([0.87, 0.1, 0.02, 0.2])  # Blue (sparsity)
    
    # Define color maps
    red_cmap = plt.cm.Reds_r
    green_cmap = plt.cm.Greens_r
    blue_cmap = plt.cm.Blues_r
    
    # Create ScalarMappables
    norm = plt.Normalize(vmin=0, vmax=1)
    sm_red = plt.cm.ScalarMappable(cmap=red_cmap, norm=norm)
    sm_green = plt.cm.ScalarMappable(cmap=green_cmap, norm=norm)
    sm_blue = plt.cm.ScalarMappable(cmap=blue_cmap, norm=norm)
    
    # Add colorbars
    cbar1 = plt.colorbar(sm_red, cax=cbar_ax1)
    cbar2 = plt.colorbar(sm_green, cax=cbar_ax2)
    cbar3 = plt.colorbar(sm_blue, cax=cbar_ax3)
    
    cbar1.set_label("Cosmic", color='red')
    cbar2.set_label("Noise", color='green')
    cbar3.set_label("Sparsity", color='blue')
    
    cbar1.set_ticks([0, 0.5, 1])
    cbar2.set_ticks([0, 0.5, 1])
    cbar3.set_ticks([0, 0.5, 1])
    
    cbar1.set_ticklabels(["0", f'{format_sci(half_value)}', f'{format_sci(true_max)}'])
    cbar2.set_ticklabels(["0", f'{format_sci(half_value)}', f'{format_sci(true_max)}'])
    cbar3.set_ticklabels(["0", f'{format_sci(half_value)}', f'{format_sci(true_max)}'])
    
    # Add first level (innermost) labels
    ax.set_xticks(inner_ticks)
    ax.set_yticks(inner_ticks)
    ax.set_xticklabels(inner_labels, fontsize= 5 * 75/side_length)
    ax.set_yticklabels(inner_labels, fontsize= 5 * 75/side_length, rotation = 90)
    
    # Define label positions with evenly spaced tiers
    # We need to ensure enough space for the additional tiers of labels
    plt.subplots_adjust(left=0.12, top=0.85, bottom=0.05)
    
    # Define evenly spaced offsets for the different tiers of labels
    # offset_unit = matrix_size / 30  # Tune this divisor to space things properly
    offset_unit = 3 * side_length / 81  # Tune this to space things properly
    middle_tier_offset = -1.2 * offset_unit
    outer_tier_offset = -1.7 * offset_unit
    extra_tier_offset = -2.5 * offset_unit
    
    # CORRECTED LABELS ACCORDING TO SPECIFICATIONS
    
    # Middle tier: +/- labels
    # "-" for first block of Nbina, "+" for next, nothing for next Nbin_z blocks, alternating -/+ for last 2*Nbin_z blocks
    middle_labels = []
    for i in range(matrix_size):       #looping through the blocks
        block_idx, Na = find_block_idx(i)
        if block_idx == 0:
            middle_labels.append(r'$-$')
        elif block_idx == 1:
            middle_labels.append(r'$+$')
        elif block_idx < Nbinz_P+2:  # blocks 2 to (Nbinz_P + 1) (Nbinz_P blocks)
            middle_labels.append('')
        else:  # blocks (Nbinz_P + 2) to (Nbinz_E + Nbinz_P + 2) (Nbinz_E blocks)
            if (block_idx - (Nbinz_P+2)) % 2 == 0:
                middle_labels.append(r'$-$')
            else:
                middle_labels.append(r'$+$')
    
    # Function to add section label
    def add_section_label(pos, label, offset, fontsize, is_x_axis=True):
        if label and is_x_axis:
            ax.text(pos, offset, label, ha='center', va='center', fontsize=fontsize)
        if label and not is_x_axis:
            ax.text(offset, pos, label, ha='center', va='center', fontsize=fontsize, rotation=90)
    
    # Add middle tier labels (-/+) - with spacing adjustment to avoid overlap
    checklist = []
    for i in range(0, matrix_size):  
        block_idx, Na = find_block_idx(i)
        if block_idx in checklist:
            continue
        else:
            checklist.append(block_idx)
            if middle_labels[i]:
                # Only add label once per block
                add_section_label(i + Na / 2 -0.5, middle_labels[i], middle_tier_offset, 10, True)
                add_section_label(i + Na / 2 -0.5, middle_labels[i], middle_tier_offset, 10, False)
            
    # Add outer tier labels (numbers)
    # Nothing for first 2 blocks
    # 1 to Nbinz_P for next Nbinz_P blocks (blocks 2 to Nbinz_P + 2)
    for block_idx in range(2, Nbinz_P+2):
        center_pos = angular_bin_dictionary['LL'][0] + np.sum([angular_bin_dictionary['LP'][b] for b in range(block_idx-2)]) + angular_bin_dictionary['LP'][block_idx-2] / 2.5 
        add_section_label(center_pos, "B = " + str(block_idx-1), outer_tier_offset, 5 * np.sqrt(angular_bin_dictionary['LP'][block_idx-2] / 5), True)
        add_section_label(center_pos, "D = " + str(block_idx-1), outer_tier_offset, 5 * np.sqrt(angular_bin_dictionary['LP'][block_idx-2] / 5), False)
    
    # 0-Nbinz_E for last 2*Nbinz_E blocks (1 label per 2 blocks)
    for block_idx in range(0, Nbinz_E):
        # Position at the center of each pair of blocks
        center_pos = np.sum([angular_bin_dictionary['LE'][b] for b in range(block_idx)]) + np.sum([angular_bin_dictionary['LP'][b] for b in range(Nbinz_P)]) + angular_bin_dictionary['LL'][0] + angular_bin_dictionary['LE'][block_idx] / 2.5
        add_section_label(center_pos, "B = " + str(block_idx+1), outer_tier_offset, 5 * np.sqrt(angular_bin_dictionary['LE'][block_idx] / 5), True)
        add_section_label(center_pos, "D = " + str(block_idx+1), outer_tier_offset, 5 * np.sqrt(angular_bin_dictionary['LE'][block_idx]  / 5), False)
    
    # Add extra tier labels (LL, LP, LE) - only once per section
    # LL (first 2 blocks)
    add_section_label(angular_bin_dictionary['LL'][0] / 2.1, r'$\mathrm{LL}$', extra_tier_offset, 12, True)
    add_section_label(angular_bin_dictionary['LL'][0] / 2.1, r'$\mathrm{LL}$', extra_tier_offset, 12, False)
    
    # LP (next Nbinz_P blocks)
    add_section_label(angular_bin_dictionary['LL'][0] + np.sum([angular_bin_dictionary['LP'][b] for b in range(Nbinz_P)]) / 2.1, r'$\mathrm{LP}$', extra_tier_offset, 12, True)
    add_section_label(angular_bin_dictionary['LL'][0] + np.sum([angular_bin_dictionary['LP'][b] for b in range(Nbinz_P)]) / 2.1, r'$\mathrm{LP}$', extra_tier_offset, 12, False)
    
    # LE (last 2*Nbinz_E blocks)
    add_section_label(angular_bin_dictionary['LL'][0] + np.sum([angular_bin_dictionary['LP'][b] for b in range(Nbinz_P)]) + np.sum([angular_bin_dictionary['LE'][b] for b in range(Nbinz_E)]) / 2.1, r'$\mathrm{LE}$', extra_tier_offset, 12, True)
    add_section_label(angular_bin_dictionary['LL'][0] + np.sum([angular_bin_dictionary['LP'][b] for b in range(Nbinz_P)]) + np.sum([angular_bin_dictionary['LE'][b] for b in range(Nbinz_E)]) / 2.1, r'$\mathrm{LE}$', extra_tier_offset, 12, False)
    
    return fig

############################ extracting blocks ############################

def extract_block(matrix, corr1, corr2):
    """
    Extracts a specified block from the covariance matrix.
    
    Args:
        matrix: Full covariance matrix
        corr1: the first correlation function ("LL", "LE", "LP")
        corr2: the second correlation function ("LL", "LE", "LP")
    
    Returns:
        Extracted submatrix
    """
    
    return matrix[indices[corr1][0]:indices[corr1][1], indices[corr2][0]:indices[corr2][1]]


def extract_redshift_bin(submatrix, corr1, corr2, redshifts = (0,0)):
    
    redshift_index1 = redshifts[0]
    redshift_index2 = redshifts[1]

    start1 = int(np.sum([angular_bin_dictionary[corr1][b] for b in range(redshift_index1)]))
    end1 = int(start1 + angular_bin_dictionary[corr1][redshift_index1])

    start2 = int(np.sum([angular_bin_dictionary[corr2][b] for b in range(redshift_index2)]))
    end2 = int(start2 + angular_bin_dictionary[corr2][redshift_index2])
    
    result = submatrix[start1:end1, start2:end2]
    
    return result

############################ plotting covariance block ############################

def plot_covariance_block(covariance_data, correlations_data, corrs, 
                          scale=True, normalize=True, log_scale=True, figsize=(8, 6),
                          plot_noise=True, plot_cosmic=True, plot_sparsity=True, redshift_bins = None):
    """
    Plots a specific block of the covariance matrix, allowing selective inclusion of noise, cosmic, and sparsity components.
    """

    corr1 = corrs[0]
    corr2 = corrs[1]
    
    stop = False

    if redshift_bins is not None:
        b1_name = str(redshift_bins[0])
        b2_name = str(redshift_bins[1])
        for i, redshift_bin in enumerate(redshift_bins):
            if type(redshift_bin) != int or redshift_bin > Nbinz[corrs[i]]:
                print(f'Error: Invalid redshift binning - bins must be integers from 0 to {Nbinz[corrs[i]]-1}')
                stop = True
            elif corrs[i] == 'LL' and redshift_bin > 0:
                print(f'Error: No redshift binning for LOS shear, only valid index is 0')
                stop = True
    else:
        b1_name = 'b'
        b2_name = 'b'

    if not stop:
        noise_mat, cosmic_mat, sparsity_mat = construct_large_matrices(covariance_data)
        
        if scale:
            noise_mat /= np.outer(correlations_list, correlations_list)
            cosmic_mat /= np.outer(correlations_list, correlations_list)
            sparsity_mat /= np.outer(correlations_list, correlations_list)
        
        noise_mat = extract_block(noise_mat, corr1, corr2)
        cosmic_mat = extract_block(cosmic_mat, corr1, corr2)
        sparsity_mat = extract_block(sparsity_mat, corr1, corr2)
    
        if redshift_bins is not None:
            noise_mat = extract_redshift_bin(noise_mat, corr1, corr2, redshifts = redshift_bins)
            cosmic_mat = extract_redshift_bin(cosmic_mat, corr1, corr2, redshifts = redshift_bins)
            sparsity_mat = extract_redshift_bin(sparsity_mat, corr1, corr2, redshifts = redshift_bins)
            
    
        eps = 1e-6
        eps = max(eps, np.min(np.abs([noise_mat, cosmic_mat, sparsity_mat])))
        
        if log_scale:
            noise_mat = np.log1p((np.abs(noise_mat)) / eps)
            cosmic_mat = np.log1p((np.abs(cosmic_mat)) / eps)
            sparsity_mat = np.log1p((np.abs(sparsity_mat)) / eps)
    
        # Select matrices to plot based on user input
        components = []
        colors = []
        
        if plot_noise:
            components.append(noise_mat)
            colors.append(1)  # Green
        if plot_cosmic:
            components.append(cosmic_mat)
            colors.append(0)  # Red
        if plot_sparsity:
            components.append(sparsity_mat)
            colors.append(2)  # Blue
    
        # Normalize if enabled
        if normalize and components:
            max_val = max(np.max(mat) for mat in components)
            if max_val > 0:
                components = [mat / max_val for mat in components]
    
        # Initialize RGB matrix
        rgb_matrix = np.zeros(noise_mat.shape + (3,))
        
        # Assign selected components to RGB channels
        for mat, color in zip(components, colors):
            rgb_matrix[..., color] = mat  
    
        # Plot the covariance block
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(rgb_matrix, aspect='equal')
        ax.set_title(r'$[$'+corr1+ '(a; ' + b1_name + '), '+corr2+ '(a; ' + b2_name + r')$]$')
        plt.show()