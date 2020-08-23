import argparse
import os
import csv

def main():
    
    parser = argparse.ArgumentParser(description='Make output charts and image files from fire ABM simulation results.')
    parser.add_argument('-inf',    '--input_file', required=True, type=str, dest='input_file', help='simulation result file, tab seperated')
    args = parser.parse_args()
    assert os.path.isfile(args.input_file), "input file not found"
    
    seed_dict = {}
    
    with open(args.input_file, 'r') as input_csvfile:
        reader = csv.DictReader(input_csvfile, delimiter='\t')
        for row in reader:
            strat = row['veh_by_strat'].split("'")[1]
            seed = int(row['Seed'])
            if seed_dict.get(strat):
                seed_dict[strat].append(seed)
            else:
                seed_dict[strat] = [seed]
                
    all_strats = list(seed_dict.keys())
    
    for dstrat in all_strats:
        print(dstrat+':', len(set(seed_dict[dstrat])), 'seeds')
        
    common_seeds = None
    for di in range(len(all_strats)-1):
        if common_seeds == None:
            common_seeds = set(seed_dict[all_strats[di]]) & set(seed_dict[all_strats[di+1]])
        else:
            common_seeds = common_seeds & set(seed_dict[all_strats[di+1]])
            
    print('common seed len:', len(common_seeds))
    print('common seeds:')
    print(common_seeds)
    
if __name__ == '__main__':
    main()