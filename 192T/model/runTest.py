import os
import time
import sys

def main():
    for i in range(6):

        
        x = 0
        while(x==0):
            time.sleep(2)
            os.system("squeue -u djohnston > jobs.txt")
            num_lines = sum(1 for line in open('jobs.txt'))-1
            if(num_lines<96):
                x = 1
        newpid = os.fork()
        time.sleep(1)
        if(newpid==0):
            child(i+1)

def child(j):
        
    print("start ", j)
    arg = "sbatch mimic3.sh " + str(j) + " tr/ " + sys.argv[1]
    #arg = "sbatch simpleJob.sh"
    os.system(arg) 
    sys.exit(0)

if __name__ == "__main__":

    main()
