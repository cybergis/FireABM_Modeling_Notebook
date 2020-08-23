#!/bin/tcsh


#SBATCH -n 1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=you@email.com

module purge
module add GNU
module add GPU
module use /data/cigi/common/cigi-modules
module load anaconda3
conda activate ox


echo $SEED
set y
cd ..
foreach x (`seq 0 5`)
        @ y = $x + $SEED
        echo $y
	python run_fireabm.py -nv 800 -sd $y -epath 'path_to_top_level_folder' -ofd summer_majrds -strat dist+road_type_weight -rg Sta_Rosa_8000.pkl -exdsc 'Quickest strat comp to mjrds and dist' -strd 1.0 -rfn 'sumer_result' -vfn 'summer_output'
end
