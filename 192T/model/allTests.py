import os
import time
import sys

def main():

   imputes = ["data", "forward_time_series", "for_back_combination_time_series", "soft_time_series"]

   for i in imputes:

       arg = "sbatch runTest.sh " + i
       os.system(arg)
       time.sleep(10)
       sys.exit(0)

if __name__ == "__main__":

    main() 
