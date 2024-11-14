#!/bin/bash
# Sample batchscript to run an OpenMP job on HPC
#SBATCH -N 1
#SBATCH -n 4
##SBATCH --account=guest
##SBATCH --partition=guest-compute               # queue to be used
#SBATCH --account=mrsec
#SBATCH -p mrsec-compute          # Queue (partition) name
#SBATCH --qos=medium
#SBATCH --time=71:59:59                         # Running time (in hours-minutes-seconds)
#SBATCH --mail-type=END,FAIL              # send an email when the job begins, ends or fails
#SBATCH --mail-user=saptorshighosh@brandeis.edu      # Email address to send the job status
##SBATCH --job-name=test_docker
##SBATCH --nodelist=compute-6-1
#SBATCh -o %x.o
#SBATCH -e %x.e
##SBATCH --exclude=compute-9-4,compute-9-3,compute-9-2,compute-9-1,compute-9-0,compute-3-26,compute-2-25,compute-5-1,compute-1-9,compute-2-14,compute-1-11, compute-2-16,compute-2-19,compute-1-18,compute-1-19,compute-3-13,compute-3-15,compute-3-16,compute-3-17,compute-3-21, compute-0-25,compute-1-22 


#export LD_LIBRARY_PATH=$HOME/libseccomp/lib:$LD_LIBRARY_PATH

# module load share_modules/ANACONDA/5.3_py3
module load share_modules/SINGULARITY/3.0.2

#module --ignore-cache load share_modules/APPTAINER/1.3.2
# module load intel mvapich2
module swap openmpi3 mvapich2/2.2

srun --mpi=pmi2 -n 4 singularity exec -H /home/saptorshighosh/docker_tmp -B /scratch0/saptorshighosh/subdomain:/home/fenics/shared fenics_stable.img python3 control_may29_subdomain.py  
