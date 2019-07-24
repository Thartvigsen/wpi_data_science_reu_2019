import os
import time
import sys

def main():

   imputes = ["data"]

   for i in imputes:

       arg = "sbatch runTest.sh " + i
       os.system(arg)
       time.sleep(10)

if __name__ == "__main__":

    main() 
