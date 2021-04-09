import argparse
import os
# import time
from datetime import datetime
import pytz
from pathlib import Path
import osmnx

if osmnx.__version__ == '0.11.4':
    from FireABM_opt import load_road_graph, get_node_edge_gdf, create_bboxes, load_shpfile, setup_sim, NetABM
else:
    from FireABM_opt_Keel import load_road_graph, get_node_edge_gdf, create_bboxes, load_shpfile, setup_sim, NetABM

########################################################################################################
    # This file runs the wildfire evacuation ABM from the command line. Most of the file checks input parameters for validity. Some parameters are set to default for ease of use.

########################################################################################################

def main():

    time_zone = pytz.timezone('America/Chicago')
    print('\n!! starting file parse at:', datetime.now(time_zone).strftime("%H:%M:%S"))

    ########################################################################################################
    # #### parse input arguments
    parser = argparse.ArgumentParser(description='Run FireABM simulation.')

    # example run strings:
    # python run_fireabm.py -nv 10 -sd 0 1 -epath '/home/jovyan/work/Fire_Optimization_ABM/src' -ofd ett_test -strat quickest dist+road_type_weight dist -rg Sta_Rosa_2000.pkl -exdsc 'Test parser' -strd 1.0 1.0 1.0

    # python run_fireabm.py -nv 10 -sd 0 1 -epath '/home/jovyan/work/Fire_Optimization_ABM/src' -ofd ett_test -strat quickest -rg Sta_Rosa_2000.pkl -exdsc 'Test parser' -strd 1.0 -rfn 'sumer_result' -vfn 'summer_output'

    # ## REQUIRED
    # should be the same per experiment
    parser.add_argument('-nv', '--number_vehicles', required=True, type=int, dest='num_veh',
                        help='the number of vehicles used in the simulation')
    # must be unique per strategy per run but same across strategies
    # ex: major roads total seeds =[1, 2, 3...], quickest = [1, 2, 3...]
    parser.add_argument('-sd', '--seed', required=True, type=int, dest='seed', nargs='+',
                        help='master seed used for an individual simulation run')
    # typically the same per experiment
    parser.add_argument('-epath', '--experiment_path', required=True, type=str, dest='experiment_path',
                        help='path to directory used to run simulation and store simulation results. Use foward slashes.')
    # typically the same per experiment
    parser.add_argument('-ofd', '--output_folder', required=True, type=str, dest='out_folder',
                        help='a directory used to store simulation results')

    # more useful when split strategies, helpful for checking

    parser.add_argument('-strat', '--strategy', required=True, type=str, dest='major_strat', nargs='+',
                        choices=['dist', 'dist+road_type_weight', 'quickest'], help='strategy used in simulation (distance, major roads, or quickest')

    # typically the same per experiment (road graph may change for testing)
    parser.add_argument('-rg', '--road_graph', required=True, type=str, dest='road_graph_pkl',
                        help='a Osmnx road graph stored as a pickel file')

    # typically the same per experiment
    parser.add_argument('-exdsc', '--experiment_discription', required=True, type=str, dest='exp_desc',
                        help='short discription of overall experiment')

    # ## DEFAULTS
    # typically the same per experiment
    parser.add_argument('-fsp', '--fire_shape_file', default='santa_rosa_fire.shp', type=str, dest='fire_shapefile',
                        help='a shapefile with fire perimeters (expected export from FlamMap), must have "Simtime" column. Store fire shapefile in "fire_input" folder.')

    # typically the same per fire shapefile, can be advanced to simulate later fire stages. value must be present in shapefile
    parser.add_argument('-sft', '--start_fire_time', default=60, type=int, dest='start_fire_time',
                        help='"Simtime" value for initial wildfire perimeter')
    # typically the same per each simulation run currently
    parser.add_argument('-strd', '--strategy_distribution', default=1.0, type=float, dest='strat_distribution', nargs='+',
                        help='ammount of vehicles using strategy')
    # typically the same per study area
    parser.add_argument('-bbb', '--bbox_buffer', default=[2.1, 4.5, 3, 3], type=list, dest='bbox_buffer',
                        help='buffer to adjust location of disaster area bounding box relative to study area extent')
    # use for bookkeeping
    parser.add_argument('-expno', '--experiment_no', default=1, type=int, dest='exp_no',
                        help='experiment number')
    # use for bookkeeping
    parser.add_argument('-nbno', '--notebook_no', default=1, type=int, dest='nb_no',
                        help='notebook or run section number')
    # use for file naming
    parser.add_argument('-rfn', '--result_file_name', default='result_file', type=str, dest='rslt_file_name',
                        help='experiment number')
    # use for file naming
    parser.add_argument('-vfn', '--video_file_name', default='output_file', type=str, dest='vid_file_name',
                        help='notebook or run section number')
    parser.add_argument('-pargs', '--print_args', default=False, type=bool, dest='print_args',
                        help='if true, print arg values')

    args = parser.parse_args()

    ########################################################################################################
    # check args
    print('\n!! checking input parameters')

    if args.print_args:
        print('!! passed args: ...')
        print('\tnum_veh:', args.num_veh)  # type(args.num_veh))
        print('\tseeds:', args.seed)  # type(args.seed))
        print('\texperiment_path:', args.experiment_path)  # type(args.experiment_path))
        print('\tout_folder:', args.out_folder)  # type(args.out_folder))
        print('\tmajor_strat:', args.major_strat)  # type(args.major_strat))
        print('\troad_graph_pkl:', args.road_graph_pkl)  # type(args.road_graph_pkl))
        print('\tfire_shapefile:', args.fire_shapefile)  # type(args.fire_shapefile))
        print('\texp_desc:', args.exp_desc)  # type(args.exp_desc))
        print('\tstart_fire_time:', args.start_fire_time)  # type(args.start_fire_time))
        print('\tstrat_distribution:', args.strat_distribution)  # type(args.strat_distribution))
        print('\tbbox_buffer:', args.bbox_buffer)  # type(args.bbox_buffer))
        print('\texp_no:', args.exp_no)  # type(args.exp_no))
        print('\tnb_no:', args.nb_no)  # type(args.nb_no))
        print('\trslt_file_name:', args.rslt_file_name)  # type(args.rslt_file_name))
        print('\tvid_file_name:', args.vid_file_name)  # type(args.vid_file_name))
        print('!! end passed args: ...')

    # check args
    pars_exp_path = Path(args.experiment_path)
    assert os.path.isdir(pars_exp_path / args.out_folder), "path to output folder not found"

    assert len(args.major_strat) < 4, "too many strategies"
    assert len(set(args.major_strat)) == len(args.major_strat), "duplicate strategies"
    assert len(args.strat_distribution) < 4, "too many strategy distrbutions"
    assert len(args.major_strat) == len(args.strat_distribution), "strategy lists (major strat and distribution) are unequal"

    assert os.path.isfile(pars_exp_path / args.road_graph_pkl), "path to road graph not found"
    assert os.path.isfile(pars_exp_path / 'fire_input' / args.fire_shapefile), "path to fire shapefile not found"
    fire_shp_ex = args.fire_shapefile.split('.')[0] + '.shx'
    assert os.path.isfile(pars_exp_path / 'fire_input' / fire_shp_ex), 'required file (.shx) is missing for fire shapefile'
    fire_shp_ex = args.fire_shapefile.split('.')[0] + '.dbf'
    assert os.path.isfile(pars_exp_path / 'fire_input' / fire_shp_ex), 'required file (.dbf) is missing for fire shapefile'

    print('!! input parameters OK\n')

    ########################################################################################################
    # #### setup run information

    st_trt_dict = {'dist': '100% shortest distance', 'dist+road_type_weight': '100% short major roads', 'quickest': '100% quickest'}
    treatments = [st_trt_dict[ms_val] for ms_val in args.major_strat]
    run_count = 0
    bad_seeds = []
    start_full_run_time = datetime.now(time_zone)
    print('!! starting full run at', start_full_run_time.strftime("%H:%M:%S"), '\n')

    ########################################################################################################
    # #### run simulation
    print('!! run simulation')

    for j in args.seed:
        for i in range(len(args.major_strat)):
            run_start_time = datetime.now(time_zone)
            treat_desc = treatments[i]
            strat_perc = args.strat_distribution[i]
            major_strat = args.major_strat[i]
            seed = j
            print('\nrun params:', treat_desc, 'i:', i, 'j:', j, 'SEED:', seed, 'strat_perc', strat_perc)

            try:
                road_graph = load_road_graph(args.road_graph_pkl)
                gdf_nodes, gdf_edges = get_node_edge_gdf(road_graph)
                (bbox, lbbox, poly, x, y) = create_bboxes(gdf_nodes, 0.1, buff_adj=args.bbox_buffer)
                sr_fire = load_shpfile(road_graph, ("fire_input", args.fire_shapefile))
                # init_fire = sr_fire[sr_fire['SimTime'] == args.start_fire_time]

                fig, ax = setup_sim(road_graph, seed)
                simulation = NetABM(road_graph, args.num_veh, bbox=lbbox, fire_perim=sr_fire, fire_ignit_time=args.start_fire_time,
                fire_des_ts_sec=100, reset_interval=True, placement_prob='Pct_HH_Cpd',
                init_strategies={major_strat: strat_perc})

                simulation.run(save_args=(fig, ax, args.rslt_file_name, args.vid_file_name, args.out_folder,
                             i, j, seed, treat_desc, args.exp_desc, args.exp_no, args.nb_no, args.road_graph_pkl), mutate_rate=0.005, update_interval=100)
                run_count += 1
                print("\nsuccess! no:", run_count, 'run_time:', datetime.now(time_zone) - run_start_time, 'timestamp:', datetime.now(time_zone).strftime("%H:%M:%S"))

            except:  # noqa: E722 # sometimes crashes
                print("Issue with this: ", treat_desc, i, j, seed, strat_perc)
                bad_seeds.append(seed)
    print('\nbad_seeds', bad_seeds)

    ########################################################################################################
    # #### end simulation run

    end_full_run_time = datetime.now(time_zone)
    print('\n!! ending full run at', end_full_run_time.strftime("%H:%M:%S") + ',', 'elapsed time:', (end_full_run_time - start_full_run_time))
    print('!! runs completed:', run_count, '/', len(args.seed) * len(args.major_strat))
    print('!! Full simulation block complete!\n')

##########################


if __name__ == '__main__':
    main()
