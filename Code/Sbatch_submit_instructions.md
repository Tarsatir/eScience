### Instructions on use of scripts with Sbatch (Snellius)

Manually calculate the number of combinatorical sample configurations, not counting ensembles. This value is <ncomb>.
The manually submit the jobs as below replacing <ncomb>.


sbatch --array=0-<ncomb> job_gen_vtess.sh

sbatch --array=0-<ncomb> job_att_vtess.sh



