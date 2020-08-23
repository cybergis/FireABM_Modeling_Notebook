import os
import argparse
import csv
import glob

def main():
    parser = argparse.ArgumentParser(description='Combine results from FireABM simulation strategy results files.')
    parser.add_argument('-ifd',    '--in_folder',          required=True, type=str, dest='in_folder', 
                        help='a directory containing simulation results collection files')
    parser.add_argument('-ofd',    '--output_file',          required=True, type=str, dest='out_file', 
                        help='a file used to store combined simulation results')
    
    args = parser.parse_args()
    print('in_folder:', args.in_folder)
    print('out_file:', args.out_file)
    assert os.path.isdir(args.in_folder), "path to input folder not found"
    all_files = glob.glob(args.in_folder+"/*.txt")
    print('total number files:', len(all_files))

    fieldnames = ['Exp_no', 'NB_no',
                  'Treat_no', 'Rep_no', 'Seed',
                  'Elpsd_time_TS',
                  'Total_cars', 'Stuck_cars',
                  'Num_rds_in_bbox',
                  'veh_by_strat',
                  'Finish_time',
                  'Treat_desc', 'Exp_desc', 'RG_file',
                  'Veh_stat_by_time',
                  'Cong_by_time',
                  'Veh_by_edge',
                  'Init_Veh_coords']
    
    with open(args.out_file, 'w') as out_csvfile:
        writer = csv.DictWriter(out_csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for data_file in all_files:
            with open(data_file, 'r') as in_csvfile:
                reader = csv.DictReader(in_csvfile, delimiter='\t')
                for row in reader:
                    writer.writerow(row)

if __name__ == '__main__':
    main()
