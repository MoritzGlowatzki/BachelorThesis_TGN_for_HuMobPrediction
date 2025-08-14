#!/bin/sh

#SBATCH --job-name=train2tgn
#SBATCH --output=./dgx_output/slurm-%j.out

# Run the container and forward all arguments
apptainer exec \
  -H "$PWD:/home" \
  --nv \
  --mount type=bind,src=/glob/g01-cache/bgdm/moritz/BachelorThesis_TGN_for_Human_Mobility_Prediction/data,dst=/home/data \
  pyg_24.05-py3.sif \
  /bin/bash -c "cd /home && ./jobscript_background_training.sh"
