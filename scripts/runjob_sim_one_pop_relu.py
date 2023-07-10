import os
import socket
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random


def runjobs():
    """
        Function to be run in a Sun Grid Engine queuing system. For testing the output, run it like
        python runjobs.py --test 1
        
    """
    
    #--------------------------------------------------------------------------
    # Test commands option
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--cluster_", help=" String", default='burg')
    parser.add_argument('--gbar', '-gb', type=float, default=-5.0)
    parser.add_argument('--g', '-g' type=float, default=2.0)
    parser.add_argument('--cbar', '-cb', type=float, default=5.0)
    parser.add_argument('--c', '-c' type=float, default=0.0)
    parser.add_argument('--tau', '-t' type=float, default=1.0)
    parser.add_argument('--N', '-N' type=int, default=10000)
    parser.add_argument('--seed', '-s' type=int, default=0)
    
    args2 = parser.parse_args()
    args = vars(args2)

    gbar = args['gbar']
    g = args['g']
    cbar = args['cbar']
    c = args['c']
    tau = args['tau']
    N = args['N']
    seed = args['seed']
    
    hostname = socket.gethostname()
    if 'ax' in hostname:
        cluster = 'axon'
    else:
        cluster = str(args["cluster_"])
    
    if (args2.test):
        print ("testing commands")
    
    #--------------------------------------------------------------------------
    # Which cluster to use

    
    if platform=='darwin':
        cluster='local'
    
    currwd = os.getcwd()
    print(currwd)
    #--------------------------------------------------------------------------
    # Ofiles folder
        
    if cluster=='haba':
        path_2_package="/rigel/theory/users/thn2112/MeanFieldCorrelations/scripts"
        ofilesdir = path_2_package + "/../Ofiles/"
        resultsdir = path_2_package + "/../results/"

    if cluster=='moto':
        path_2_package="/moto/theory/users/thn2112/MeanFieldCorrelations/scripts"
        ofilesdir = path_2_package + "/../Ofiles/"
        resultsdir = path_2_package + "/../results/"

    if cluster=='burg':
        path_2_package="/burg/theory/users/thn2112/MeanFieldCorrelations/scripts"
        ofilesdir = path_2_package + "/../Ofiles/"
        resultsdir = path_2_package + "/../results/"
        
    elif cluster=='axon':
        path_2_package="/home/thn2112/MeanFieldCorrelations/scripts"
        ofilesdir = path_2_package + "/../Ofiles/"
        resultsdir = path_2_package + "/../results/"
        
    elif cluster=='local':
        path_2_package="/Users/tuannguyen/MeanFieldCorrelations/scripts"
        ofilesdir = path_2_package+"/../Ofiles/"
        resultsdir = path_2_package + "/../results/"



    if not os.path.exists(ofilesdir):
        os.makedirs(ofilesdir)
    
    
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    
    
    #--------------------------------------------------------------------------
    # The array of hashes
    seed_vec=range(10)
    
    for seed in seed_vec:

        time.sleep(0.2)
        
        #--------------------------------------------------------------------------
        # Make SBTACH
        inpath = currwd + "/sim_one_pop_relu.py"
        c1 = "{:s} -gb {:.1f} -g {:.1f} -cb {:.1f} -c {:.1f} -t {:.1f} -N {:d} -s {:d}".format(inpath,gbar,g,cbar,c,tau,N,seed)
        
        jobname="sim_one_pop_relu"+"-s-{:d}".format(seed)
        
        if not args2.test:
            jobnameDir=os.path.join(ofilesdir, jobname)
            text_file=open(jobnameDir, "w");
            os.system("chmod u+x "+ jobnameDir)
            text_file.write("#!/bin/sh \n")
            if cluster=='haba' or cluster=='moto' or cluster=='burg':
                text_file.write("#SBATCH --account=theory \n")
            text_file.write("#SBATCH --job-name="+jobname+ "\n")
            text_file.write("#SBATCH -t 0-11:59  \n")
            text_file.write("#SBATCH --mem-per-cpu=8gb \n")
            text_file.write("#SBATCH --gres=gpu\n")
            text_file.write("#SBATCH -c 1 \n")
            text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
            text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
            text_file.write("python  -W ignore " + c1+" \n")
            text_file.write("echo $PATH  \n")
            text_file.write("exit 0  \n")
            text_file.close()

            if cluster=='axon':
                os.system("sbatch -p burst " +jobnameDir);
            else:
                os.system("sbatch " +jobnameDir);
        else:
            print (c1)



if __name__ == "__main__":
    runjobs()


