#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[40]:


timeSeries = np.array([[0, 0, 5],
    [3, 4, 0],
     [0, 0, 0],
    [7, 2, 0],
    [0, 0, 2],
    [2, 3, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [4, 0, 0]])

maskArray = np.array([[1, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
        [0, 1, 1]])

print("original \n", timeSeries)

testArr = np.array([[7,0,0,0,3,0,0,4]])

testMask = np.array([[0,1,1,1,0,1,1,0]])

testArr, testMask = np.transpose(testArr), np.transpose(testMask)
                       
 
def locfAndNocbTest(roTS, maskArray):  
    
    nColumns = len(roTS[0]) # len: 3
    nRows = len(roTS) # len: 12
    
    for col in range(nColumns): # 3
        row = 0
        while(row+1 < nRows): # 12
                                    
            # if the variable is the first row and is missing, keep looking one row ahead until
            # you have found first non-missing value and impute the first row with THAT value
            if (row==0) and maskArray[row][col] == 1:
                subIndex = 1
                while subIndex < nRows:
                    if maskArray[subIndex][col] == 0:
                        roTS[row][col] = roTS[subIndex][col]
                        break
                    subIndex +=1
                row+=1

            # else, if the missing variable is anywhere else in the list, search up and down the column for the 
            # observed values. say the column is [0, 5, 0, 4, 0, 0, 3, 0]. needs to turn into [5, 5, 4.5, 4, 4, 3, 3, 3]. 
            # if index of missing <= (low + high) / 2, impute with low. Else, impute with high. 
            

            elif(maskArray[row][col] == 0): # if a value is observed

                earlyValue = roTS[row][col]
                subIndex = row + 1
                nZeros = 0
                                
                rowCopy = row

                # search for the next observed value (mask = 0)

                while subIndex < nRows:
                    
                    row+=1

                    if maskArray[subIndex][col] == 0:

                        lateValue = roTS[subIndex][col]
                        midIndex = (int)((rowCopy + subIndex) / 2)
                        
                        # if the number of missing values between the two observed is odd
                        if (nZeros > 0 and nZeros % 2 == 1): # need to check that number of zeroes > 0 in case of something like [4, 2, 0, 0]
                            roTS[midIndex][col] = (earlyValue + lateValue) / 2.0
                            roTS[rowCopy : midIndex ,col] = earlyValue
                            roTS[(midIndex + 1): subIndex,col] = lateValue
                            break
                        elif nZeros > 0: 
                            # if the number of missing values between the two observed is even
                            roTS[rowCopy : midIndex, col] = earlyValue
                            roTS[midIndex : subIndex, col] = lateValue
                            break
                            
                    if(subIndex+1==nRows):
                        
                        roTS[rowCopy:nRows] = earlyValue
                            
                    nZeros += 1
                    subIndex += 1
                                                                            
            elif (maskArray[row][col] == 1):
                roTS[row][col] = roTS[row-1][col]
                row+=1
                
                                                
    return roTS, maskArray


# In[41]:



# newTS, newMask = locfAndNocbTest(roTS, maskArray)

newTS, newMask = locfAndNocbTest(timeSeries, maskArray)

print("original 2 \n", timeSeries)

print()
print(newMask)
#print(newTS, newMask)


# In[ ]:




