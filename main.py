from pathlib import Path
from discover.discover import Discover as disc
from discover.timing import Timing as timing
from discover import util

def mine_timings_from_dcr(input_event_log_path, output_dcr_graph_path, output_timing_results_folder):
    log, graph = mine_dcr(input_event_log_path, output_dcr_graph_path)
    print(f'[i] Mining timings started')
    timing_input_dict = timing.create_timing_input_dict(output_dcr_graph_path)
    event_pair_timing_data = timing.get_timings(log, timing_input_dict)
    print(f'[i] Mining timings finished')
    Path(output_timing_results_folder).mkdir(parents=True, exist_ok=True)
    print(f'[i] Started creating boxplot data')
    boxplot_values = util.create_timing_box_plots(event_pair_timing_data, output_timing_results_folder)
    print(f'[i] Finished creating boxplot data')

    print(f'[i] Started writing median values as delays and deadlines to DCR graph')
    to_print_values = {}
    to_print_values_outliers = {}
    total_conditions = 0
    total_responses = 0
    total_not_enough_values = 0
    for (k, v) in boxplot_values.items():
        if v:
            if k[0] == 'CONDITION':  # min for condition
                to_print_values[k] = v[6]
                to_print_values_outliers[k] = v[0]
                total_conditions = total_conditions + 1
            elif k[0] == 'RESPONSE':  # max for response
                to_print_values[k] = v[7]
                to_print_values_outliers[k] = v[4]
                total_responses = total_responses + 1
        else:
            total_not_enough_values = total_not_enough_values + 1
    disc.writeGraph(f'{output_dcr_graph_path}.txt', to_print_values)
    disc.write_with_do_subprocesses(f'{output_dcr_graph_path}_do_subprocesses.txt', to_print_values)
    disc.writeGraph(f'{output_dcr_graph_path}_no_outliers.txt', to_print_values_outliers)

    mean_values = util.get_mean_values(event_pair_timing_data)
    print(f'[i] Finished writing values as delays and deadlines to DCR graph')

    print(f'[i] Started creating histograms')
    histogram_values = util.create_histograms(event_pair_timing_data, output_timing_results_folder)

    print(f'[i] Started fitting single distributions')
    single_distribution_fits = timing.simple_distribution_fit_all_timings(event_pair_timing_data, output_timing_results_folder)
    print(f'[i] Finished fitting single distributions')

def mine_dcr(input_event_log_path, output_dcr_graph_path):
    print(f'[i] Mining dcr graph started')
    log = util.load_log(input_event_log_path)
    graph = disc.mine(log=log,graph_path=output_dcr_graph_path)
    print(f'[i] Mining dcr graph finished')
    return log, graph

if __name__ == '__main__':
    input_event_log_path = 'data/Road_Traffic_Fine_Management_Process.xes'
    output_dcr_graph_path = 'models/road_traffic_dcr_model'
    mine_dcr(input_event_log_path,output_dcr_graph_path)