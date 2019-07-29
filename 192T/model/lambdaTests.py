import os
import time
import sys

def main():

   lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
   hard_code = [0, 0.5, .9]


   for l in hard_code:

       arg = "sbatch runTestMed.sh " + "data "+str(l)
       os.system(arg)
       time.sleep(10)

if __name__ == "__main__":

    main() 
