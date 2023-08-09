import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def read_data(file_list, verbose=True, warning=True):
    """
    Reads LogPhase600 Excel files and returns two matrices: n_files x n_plates x time x wells of raw OD readings
    and corresponding metadata. Certain assumptions are made for the data -- the most important ones
    is that the number of wells per plate is 96 and that the sheets of the Excel file are named
    consistently.
    
    Parameter file_list: an array containing the paths to the Excel files.
    """

    all_raw = []
    all_meta = []
    all_plate_numbers = []
    ref_wells = None
    ref_meta = None
    ref_strains = None
    ref_time = None

    def original_datetime_converter(x):
        t = str(x)
        if len(t) != 8:
            split_date = t.split(" ")
            day_mul = int(split_date[0].split("-")[-1])
            new_t = split_date[-1].split(":")
            new_t[0] = str(int(new_t[0]) + 24*day_mul)
            t = ":".join(new_t)
            if verbose: print("Replaced imported date", str(x), "with", t)
        return t

    if verbose: 
        print("Starting data import...")
        print("File list:")
        for f in file_list:
            print(f)
        print()

    for f in file_list:
        if verbose: print("Processing:", f)
        exp_raw = []
        exp_meta = []
        for pn in range(1, 5):
            if verbose: print("Plate", pn)
            plate_meta = pd.read_excel(f, sheet_name="Plate " + str(pn) + " - Results", header=1)
            plate_meta.iloc[:, 4] = plate_meta.iloc[:, 4].fillna("")
            plate_raw = pd.read_excel(f, sheet_name="Plate " + str(pn) + " - Raw Data", header=1, converters={"Time": original_datetime_converter})
            # plate_raw = plate_raw[["Time"] + list(plate_meta["Well"].values)]

            tn = np.array([np.round(int(x.split(":")[0]) + (int(x.split(":")[1])/60), 2) for x in plate_raw["Time"]])

            if ref_time is None:
                if verbose: print("Using", f, "time values as reference")
                ref_time = tn

            if ref_meta is None:
                if verbose: print("Using", f, "metadata columns as reference")
                ref_meta = plate_meta.columns.values

            if ref_wells is None:
                if verbose: print("Using", f, "raw data well names as reference")
                ref_wells = plate_raw.columns.values

            # Assertions
            if not np.all(ref_meta == plate_meta.columns.values) and warning:
                print("--------------------------------------------")
                print("Warning: in file", f, "plate", str(pn), "metadata columns do not match to the reference.")
                mismatch_idx = np.where(ref_meta != plate_meta.columns.values)[0]
                for mm in mismatch_idx:
                    print("    Column", plate_meta.columns.values[mm], "!= reference", ref_meta[mm])
                print("--------------------------------------------")

            if not np.all(ref_wells == plate_raw.columns.values) and warning:
                print("--------------------------------------------")
                print("Warning: in file", f, "plate", str(pn), "raw data columns do not match to the reference.")
                mismatch_idx = np.where(ref_wells != plate_raw.columns.values)[0]
                for mm in mismatch_idx:
                    print("    Column", plate_raw.columns.values[mm], "!= reference", ref_wells[mm])
                print("--------------------------------------------")

            if not np.all(ref_time == tn) and warning:
                print("--------------------------------------------")
                print("Warning: in file", f, "plate", str(pn), "time vector does not match to the reference.")
                print("--------------------------------------------")
                assert 0

            assert np.all(np.array([len(str(x)) for x in plate_raw.Time]) == 8)
            assert np.all(plate_meta.iloc[:, 3] + " " + plate_meta.iloc[:, 4] == plate_meta.iloc[0, 3] + " " + plate_meta.iloc[0, 4])
            assert plate_meta.shape[0] == 96
            
            plate_meta = plate_meta.set_index("Well").loc[ref_wells[1:]].reset_index()

            # all_plate_numbers += [pn]*plate_meta.shape[0]
            exp_raw.append(plate_raw.to_numpy())
            exp_meta.append(plate_meta.to_numpy())

        all_raw.append(np.array(exp_raw))
        all_meta.append(np.array(exp_meta))
        if verbose: print()
            
    if verbose: print("Done")
    return np.array(all_raw), np.array(all_meta)

def plot_od_rho_heatmaps(all_raw, all_meta, file_save_path=None, cmap1="YlOrBr", cmap2="YlOrBr", figsize=(20, 8), show=True, normalize=False):
    
    all_plate_numbers = []
    for i in range(all_meta.shape[0]):
        for j in range(all_meta.shape[1]):
            all_plate_numbers += [j+1]*96
    
    conditions = [all_meta[i, j, 0, 3] + " " + all_meta[i, j, 0, 4] if all_meta[i, j, 0, 4] != "" else all_meta[i, j, 0, 3] for i in range(all_meta.shape[0]) for j in range(4)]
    temp = all_raw[:, :, :, 1:].reshape(-1, all_raw.shape[2], all_raw.shape[3]-1).transpose(0, 2, 1)
    m = temp.reshape(temp.shape[0]*temp.shape[1], -1).T.astype(float)
    if normalize:
        m = m/np.max(m)
    t = all_raw[0, 0, :, 0]
    rho = m.copy()
    for j in range(rho.shape[1]):
        rho[:, j] = np.array([0] + [(rho[i, j] - rho[i-1, j])/rho[i-1, j] if rho[i-1, j] != 0 else 0 for i in range(1, rho.shape[0])])

    labels = np.array([all_meta.reshape(-1, 8)[i, 3] + "\n" + all_meta.reshape(-1, 8)[i, 4] if all_meta.reshape(-1, 8)[i, 4] != "" else all_meta.reshape(-1, 8)[i, 3] for i in range(len(all_meta.reshape(-1, 8)))])

    if "Control" in conditions:
        c_idx = np.where(labels == "Control")[0]
        nc_idx = np.where(labels != "Control")[0]

        c_labels = labels[c_idx]
        nc_labels = labels[nc_idx]

        c_plates = np.array(all_plate_numbers)[c_idx]
        nc_plates = np.array(all_plate_numbers)[nc_idx]

        nc_sorted_labels, nc_sorted_idx = zip(*sorted(zip(list(nc_labels), nc_idx)))
        _, nc_sorted_plates = zip(*sorted(zip(list(nc_labels), list(nc_plates))))

        final_plates = list(c_plates) + list(nc_sorted_plates)
        final_labels = list(c_labels) + list(nc_sorted_labels)
        final_idx = list(c_idx) + list(nc_sorted_idx)
    else:
        final_labels, final_idx = zip(*sorted(zip(labels, range(len(labels)))))
        _, final_plates = zip(*sorted(zip(labels, all_plate_numbers)))

    last = final_labels[0]
    envs = [last]
    env_indices = [0]
    for i in range(len(final_labels)):
        x = final_labels[i]
        if x != last:
            last = x
            envs.append(final_labels[i])
            env_indices.append(i)
    env_indices.append(len(final_labels))

    last = final_plates[0]
    plate_envs = [last]
    plate_env_indices = [0]
    for i in range(len(final_plates)):
        x = final_plates[i]
        if x != last:
            last = x
            plate_envs.append(final_plates[i])
            plate_env_indices.append(i)
    plate_env_indices.append(len(final_plates))

    m = m[:, final_idx]
    rho = rho[:, final_idx]

    x_tick_env_pos = [(env_indices[i] + env_indices[i+1])/2 for i in range(len(env_indices) - 1)]
    x_tick_plate_pos = [(plate_env_indices[i] + plate_env_indices[i+1])/2 for i in range(len(plate_env_indices) - 1)]

    plt.subplots(2, 1, figsize=figsize)

    plt.subplot(211)
    plt.imshow(rho, aspect="auto", cmap=cmap1)
    plt.colorbar(label="Per-capita growth rate", pad=0.025)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.set_xticks([])

    plt.yticks(range(0, m.shape[0], 12), range(0, 25, 2))
    plt.vlines(plate_env_indices, -1, m.shape[0], color="black", alpha=0.5, lw=0.5)
    plt.vlines(env_indices, -1, m.shape[0], color="black", lw=0.5)
    plt.margins(0)
    plt.ylabel("Time (hours)")

    plt.subplot(212)
    plt.imshow(m, aspect="auto", cmap=cmap2)
    plt.colorbar(label="OD", pad=0.025)
    plt.xticks(x_tick_plate_pos, plate_envs)
    plt.yticks(range(0, m.shape[0], 12), range(0, 25, 2))
    plt.vlines(plate_env_indices, -1, m.shape[0], color="black", alpha=0.5, lw=0.5)
    plt.vlines(env_indices, -1, m.shape[0], color="black", lw=0.5)
    plt.margins(0)
    plt.ylabel("Time (hours)")

    for i in range(len(x_tick_env_pos)):
        plt.text(x_tick_env_pos[i], m.shape[0]+m.shape[0]*0.125, envs[i], rotation=15, va="top", ha="center")

    plt.text(m.shape[1]/2, m.shape[0]+m.shape[0]*0.5, "Experiment/plate", ha="center")
    plt.subplots_adjust(hspace=0.05)
    if file_save_path is not None: plt.savefig(file_save_path, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()


def plot_end_ods(all_raw, all_meta, figsize=(25, 4), file_save_path=None, show=True, normalize=False):
    
    def get_xtick_labels_and_positions(a):
        last = a[0]
        envs = [last]
        env_indices = [0]
        for i in range(len(a)):
            x = a[i]
            if x != last:
                last = x
                envs.append(a[i])
                env_indices.append(i)
        env_indices.append(len(a))
        return envs, env_indices
    
    all_plate_numbers = []
    for i in range(all_meta.shape[0]):
        for j in range(all_meta.shape[1]):
            all_plate_numbers += [j+1]*96

    conditions = [all_meta[i, j, 0, 3] + " " + all_meta[i, j, 0, 4] if all_meta[i, j, 0, 4] != "" else all_meta[i, j, 0, 3] for i in range(all_meta.shape[0]) for j in range(4)]
    temp = all_raw[:, :, :, 1:].reshape(-1, all_raw.shape[2], all_raw.shape[3]-1).transpose(0, 2, 1)
    m = temp.reshape(temp.shape[0]*temp.shape[1], -1).T.astype(float)
    if normalize:
        m = m/np.max(m)
    t = all_raw[0, 0, :, 0]

    strains = np.array(all_meta.reshape(-1, 8)[:, 1])
    labels = np.array([all_meta.reshape(-1, 8)[i, 3] + "\n" + all_meta.reshape(-1, 8)[i, 4] if all_meta.reshape(-1, 8)[i, 4] != "" else all_meta.reshape(-1, 8)[i, 3] for i in range(len(all_meta.reshape(-1, 8)))])
    plot_control_overlay = False

    if "Control" in conditions:
        c_idx = np.where(labels == "Control")[0]
        nc_idx = np.where(labels != "Control")[0]

        c_labels = labels[c_idx]
        nc_labels = labels[nc_idx]

        c_strains = strains[c_idx]
        nc_strains = strains[nc_idx]

        c_plates = np.array(all_plate_numbers)[c_idx]
        nc_plates = np.array(all_plate_numbers)[nc_idx]

        nc_sorted_labels, nc_sorted_idx = zip(*sorted(zip(list(nc_labels), nc_idx)))
        # _, nc_sorted_plates = zip(*sorted(zip(list(nc_labels), list(nc_plates))))
        # _, nc_sorted_strains = zip(*sorted(zip(list(nc_labels), list(nc_strains))))
        
        nc_sorted_plates = np.array(all_plate_numbers)[np.array(nc_sorted_idx)]
        nc_sorted_strains = strains[np.array(nc_sorted_idx)]

        final_plates = list(c_plates) + list(nc_sorted_plates)
        final_labels = list(c_labels) + list(nc_sorted_labels)
        final_idx = list(c_idx) + list(nc_sorted_idx)
        final_strains = list(c_strains) + list(nc_sorted_strains)
        plot_control_overlay = True
    else:
        final_labels, final_idx = zip(*sorted(zip(labels, range(len(labels)))))
        final_plates = np.array(all_plate_numbers)[list(final_idx)]
        final_strains = strains[list(final_idx)]

    envs, env_indices = get_xtick_labels_and_positions(final_labels)
    plate_envs, plate_env_indices = get_xtick_labels_and_positions(final_plates)

    m = m[:, final_idx]

    x_tick_env_pos = [(env_indices[i] + env_indices[i+1])/2 for i in range(len(env_indices) - 1)]
    x_tick_plate_pos = [(plate_env_indices[i] + plate_env_indices[i+1])/2 for i in range(len(plate_env_indices) - 1)]

    exps = np.array(final_labels)
    _, idx = np.unique(exps, return_index=True)
    max_od = np.max(m)

    step = 96*2
    x = range(0, m.shape[1], step)

    if plot_control_overlay:
        control_m = m[-1, x[0]:x[0]+step]
        control_strains = final_strains[x[0]:x[0]+step]
        custom_lines = [Line2D([0], [0], color="green", lw=4, label="Control")]

    plt.subplots(len(x), 1, figsize=figsize)
    sc = 1
    for i in x:
        title = str(exps[np.sort(idx)][sc-1])

        sorted_m, sorted_strains = zip(*sorted(zip(m[-1, i:i+step], final_strains[i:i+step]), reverse=True))
        sorted_strains_bottom = [sorted_strains[i] if i % 2 == 0 else "" for i in range(len(sorted_strains))]
        sorted_strains_top = [sorted_strains[i] if i % 2 != 0 else "" for i in range(len(sorted_strains))]

        plt.subplot(len(x), 1, sc)
        plt.plot(range(len(sorted_m)), sorted_m)
        plt.scatter(range(0, len(sorted_strains), 2), sorted_m[0::2], edgecolors="tab:blue", s=10)
        plt.scatter(range(1, len(sorted_strains), 2), sorted_m[1::2], facecolors="none", edgecolors="tab:orange", s=10)

        if plot_control_overlay:
            sorted_control_idx = [np.where(np.array(control_strains) == x)[0][0] for x in sorted_strains]
            plt.plot(range(len(sorted_control_idx)), np.array(control_m)[sorted_control_idx], color="green")
            plt.legend(handles=custom_lines, loc="upper right")

        plt.ylim(0, max_od)
        plt.xlabel("Strain", fontsize=15)
        plt.ylabel("End OD", fontsize=15)
        plt.xticks(range(len(sorted_strains_bottom)), sorted_strains_bottom, rotation=90)
        for j in range(len(sorted_strains_top)):
            plt.text(j, max_od+0.1, sorted_strains_top[j], color="tab:orange", rotation=90, ha="center")
        plt.vlines(range(0, len(sorted_strains), 2), 0, max_od, color="black", alpha=0.125)
        plt.vlines(range(1, len(sorted_strains), 2), -0.25, max_od, color="black", alpha=0.125/2)
        plt.title(title, fontsize=15, pad=40)
        ax = plt.gca()
        for j in range(0, len(sorted_strains), 2):
            ax.get_xticklabels()[j].set_color("tab:blue")

        sc += 1
        plt.margins(0)
    plt.subplots_adjust(hspace=1.5)
    if file_save_path is not None: plt.savefig(file_save_path, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()