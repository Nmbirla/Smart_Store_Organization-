#!/usr/bin/env python
# coding: utf-8

# ## SMART STORE ORGANIZER

# ###### The Smart store Organizer is an application that analyzes basket data to derive interesting associations. This will help store managers, sales and marketing to place items and tune prices to increase sales revenue. The current solution uses A-Priori and PCY to find frequent item pairs. This solution can be extended to incorporate mutiple other algortihms that the users can take advantage of depending on their use case.

# ## Import necessary packages

# In[33]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ## Utility Functions for Algorithm

# In[62]:


# Given the basket data, generates all possible pairs

def GenerateAllPairs(basketData,itemCountLookup,labelItemLookup):
    
    candidatePairs = {}
    candidateNamePairs = {}
    
    for basketID, items in basketData.items():
        for i in range(len(items)):
            j = i + 1
            while j < len(items):
                if items[i] in itemCountLookup.keys() and items[j] in itemCountLookup.keys():
                    
                    t1 = items[i]
                    t2 = items[j]
                    
                    if t2 < t1:
                        t1 = items[j]
                        t2= items[i]
                        
                    pair = (t1,t2)
                    pairNames = (labelItemLookup[t1],labelItemLookup[t2])
                    if pair not in candidatePairs.keys():
                        candidatePairs[pair] = 1
                        candidateNamePairs[pairNames] = 1
                    else:
                        candidatePairs[pair] = candidatePairs[pair] + 1
                        candidateNamePairs[pairNames] = candidateNamePairs[pairNames] + 1
                j+=1
    
    print('Total number of candidate pairs: '+str(len(candidatePairs)))
    return candidatePairs, candidateNamePairs
                
    
def filterItems(itemCountLookup, supportThreshold):
    itemCountWithSupport = {}
    for label,count in itemCountLookup.items():
        if count >= supportThreshold:
            itemCountWithSupport[label] = count
    return itemCountWithSupport

# Bit Vector function for PCY
def GenerateBitvector(supportThreshold, hashTable):
    bitVector = hashTable
    for (hashBucket, bucketCount) in hashTable.items():
        if bucketCount >= supportThreshold:
            bitVector[hashBucket] = 1
        else:
            bitVector[hashBucket] = 0
    return bitVector


# Hash function for PCY
def hash(n1,n2, pcyBucketCount):
    return (n1 * n2) % pcyBucketCount


# ### Utility Functions for Basket Data Processing

# In[63]:


def LoadBasketData(basketDataSource):
    
    df = pd.read_csv(basketDataSource)
    df = df.groupby(['Transaction'])['Item'].apply(list).reset_index()
    df = ProcessBasketData(df)
    basketIDToItemsLookup, basketIDToLabelsLookup = GetFormattedBaskets(df)
    return df, basketIDToItemsLookup, basketIDToLabelsLookup

def ProcessBasketData(df):
    
    df['Item'][1] = list(dict.fromkeys(df['Item'][1]))
    index = 0
    for i in df['Item']:
        df['Item'][index] = list(dict.fromkeys(i))
        index+=1
    return df

def GetItemCounts(basketData,labelItemLookup):
    itemIDCountLookup = {}
    itemNameCountLookup = {}
    
    for basket, items in basketData.items():
        for item in items:
            if item not in itemIDCountLookup.keys():
                itemIDCountLookup[item] = 1
                itemNameCountLookup[labelItemLookup[item]] = 1
            else:
                itemIDCountLookup[item] = itemIDCountLookup[item] + 1
                itemNameCountLookup[labelItemLookup[item]] = itemNameCountLookup[labelItemLookup[item]] + 1
            
    return itemIDCountLookup, itemNameCountLookup

def BuildLabelledItems(basketData):
    
    itemLabelLookup = {}
    labelItemLookup = {}
    itemID = 0
    for basketID in basketData:
        for item in basketData[basketID]:
            if item not in itemLabelLookup.keys():
                itemID += 1
                itemLabelLookup[item] = itemID
                labelItemLookup[itemID] = item
    return itemLabelLookup, labelItemLookup

def GetFormattedBaskets(df):
    basketIDToLabelsLookup = {}
    basketIDToItemsLookup = dict(zip(df.Transaction, df.Item))
    itemLabelLookup, labelItemLookup = BuildLabelledItems(basketIDToItemsLookup)
    
    for basketID, items in basketIDToItemsLookup.items():
        basketIDToLabelsLookup[basketID] = []
        for item in items:
            itemLabel = itemLabelLookup[item]
            basketIDToLabelsLookup[basketID].append(itemLabel)
            
    return basketIDToItemsLookup, basketIDToLabelsLookup

def displayBasketSummary(df):
    print('Total Baskets: '+ str(df.shape[0]))
    print('**************** First 20 Transactions************')
    print(df.head(20))


# ## Step 1 : Load Data Set

# In[64]:


def GetBasketdata(filename):
    df, basketIDToItemsLookup, basketIDToLabelsLookup = LoadBasketData(filename)
    return basketIDToItemsLookup, basketIDToLabelsLookup


# ## Step 2: Find Candidate Pairs using A-Priori

# In[119]:


def Apriori_Pass1(filename):
    
    print('[Begin] Pass 1')
    
    basketIDToItemsLookup, basketIDToLabelsLookup = GetBasketdata(filename)
    itemLabelLookup, labelItemLookup = BuildLabelledItems(basketIDToItemsLookup) # label items
    itemIDCountLookup, itemNameCountLookup = GetItemCounts(basketIDToLabelsLookup, labelItemLookup) # store count of singleton item
    
    totalTransactions = len(basketIDToItemsLookup)
    
    print('Total Transactions: '+str(len(basketIDToItemsLookup)))
    print('Total distinct items: '+str(len(itemLabelLookup)))
    print('[End] Pass 1 \n')
    
    return itemIDCountLookup, itemNameCountLookup, totalTransactions
    

def Apriori_Pass2(filename, itemCountLookup, supportThreshold):
    
    print('[Begin] Pass 2')
    
    basketIDToItemsLookup, basketIDToLabelsLookup = GetBasketdata(filename)
    itemLabelLookup, labelItemLookup = BuildLabelledItems(basketIDToItemsLookup) # label items
    itemCountLookup = filterItems(itemCountLookup, supportThreshold)
    print('Total filtered Singletons honouring support threshold: '+str(len(itemCountLookup)))
    
    candidatePairs, candidateNamePairs = GenerateAllPairs(basketIDToLabelsLookup,itemCountLookup,labelItemLookup)
    frequentPairs = {}
    frequentNamePairs = {}
    
    for item,count in candidatePairs.items():
        if count > supportThreshold:
            pairNames = (labelItemLookup[item[0]],labelItemLookup[item[1]])
            frequentPairs[item] = count
            frequentNamePairs[pairNames] = count
    
    print('Total Frequent Pairs honouring support threshold: '+str(len(frequentPairs))) 
    print()
    print(frequentNamePairs)
    print('[End] Pass 2 \n')
    return frequentPairs, frequentNamePairs


# ## Step 2: Find Candidate Pairs using PCY

# In[120]:


def PCY_Pass1(filename):
    
    print('[Begin] Pass 1')
        
    basketIDToItemsLookup, basketIDToLabelsLookup = GetBasketdata(filename)
    itemLabelLookup, labelItemLookup = BuildLabelledItems(basketIDToItemsLookup) # label items
    itemIDCountLookup, itemNameCountLookup = GetItemCounts(basketIDToLabelsLookup, labelItemLookup) # store count of singleton item

    bucketCountLookup = {}

    for basketID, items in basketIDToLabelsLookup.items():
        for i in range(len(items)):
            j = i + 1
            while j < len(items):
                    t1 = items[i]
                    t2 = items[j]

                    if t2 < t1:
                        t1 = items[j]
                        t2= items[i]

                    bucketID = hash(t1,t2,pcyBucketCount)

                    if bucketID not in bucketCountLookup.keys():
                        bucketCountLookup[bucketID] = 1
                    else:
                        bucketCountLookup[bucketID] = bucketCountLookup[bucketID]+1

                    j+=1
    
    totalTransactions = len(basketIDToItemsLookup)
    
    print('Total Transactions: '+str(len(basketIDToItemsLookup)))
    print('Total distinct items: '+str(len(itemLabelLookup)))
    print('[End] Pass 1 \n')
    
    return itemIDCountLookup, itemNameCountLookup, bucketCountLookup, totalTransactions


def PCY_Pass2(filename, itemIDCountLookup, itemNameCountLookup, bucketCountLookup, supportThreshold):
    
    print('[Begin] Pass 2')
    bitVector = GenerateBitvector(supportThreshold, bucketCountLookup)
    basketIDToItemsLookup, basketIDToLabelsLookup = GetBasketdata(filename)
    itemLabelLookup, labelItemLookup = BuildLabelledItems(basketIDToItemsLookup) # label items
    itemIDCountLookup = filterItems(itemIDCountLookup, supportThreshold)
    candidatePairs = {}
    candidateNamePairs = {}
    
    for basketID, items in basketIDToLabelsLookup.items():
        for i in range(len(items)):
            j = i + 1
            while j < len(items):
                if items[i] in itemIDCountLookup.keys() and items[j] in itemIDCountLookup.keys():
                    
                    hashVal = hash(items[i],items[j],len(bucketCountLookup))                    
                    if bitVector[hashVal] is 1:
                    
                        t1 = items[i]
                        t2 = items[j]

                        if t2 < t1:
                            t1 = items[j]
                            t2= items[i]

                        pair = (t1,t2)
                        pairNames = (labelItemLookup[t1],labelItemLookup[t2])
                        if pair not in candidatePairs.keys():
                            candidatePairs[pair] = 1
                            candidateNamePairs[pairNames] = 1
                        else:
                            candidatePairs[pair] = candidatePairs[pair] + 1
                            candidateNamePairs[pairNames] = candidateNamePairs[pairNames] + 1
                j+=1
    
    frequentPairs = {}
    frequentNamePairs = {}
    
    for item,count in candidatePairs.items():
        if count > supportThreshold:
            pairNames = (labelItemLookup[item[0]],labelItemLookup[item[1]])
            frequentPairs[item] = count
            frequentNamePairs[pairNames] = count
    
    print('Total Candidate Pairs: '+str(len(candidatePairs)))
    print('Total Frequent Pairs honouring support threshold: '+str(len(frequentPairs)))
    print()
    print(frequentNamePairs)
    
    print('[End] Pass 2 \n')
    return frequentPairs, frequentNamePairs


# ## Step 3: Find Confidence for Association Rules

# In[40]:


def FindConfidence(Item1Count, Item2Count, PairCount):
    item1Item2Conf  = PairCount/Item1Count
    item2Item1Conf  = PairCount/Item2Count
    
    return item1Item2Conf, item2Item1Conf


# ## Step 4: Find Interesting Associations

# In[41]:


def FindInterest(Item1Count, Item2Count, basketCount, item1Item2Confidence, item2Item1Confidence):
    item1Item2Interest = abs(item1Item2Confidence - (Item2Count/basketCount))
    item2Item1Interest = abs(item2Item1Confidence - (Item1Count/basketCount))
    
    return item1Item2Interest, item2Item1Interest


# ## Step 5: Display Interesting Associations

# In[117]:


def DisplayInterestingAssocations(candidateNamePairs, itemNameCountLookup, totalTransactions, interestThreshold):
    
    print('Interesting Assocations:\n')

    totalInterestingAssociations = 0
    
    for pair,count in candidateNamePairs.items():
            
        t1 = pair[0]
        t2 = pair[1]
        
        item1Item2Conf, item2Item1Conf = FindConfidence(itemNameCountLookup[t1], itemNameCountLookup[t2], candidateNamePairs[pair])
        
        item1Item2Interest, item2Item1Interest = FindInterest(itemNameCountLookup[t1], itemNameCountLookup[t2],totalTransactions, item1Item2Conf, item2Item1Conf)
        
        
        if item1Item2Interest >= interestThreshold:
            print('Confidence of association:'+t1+'->'+t2+'='+str(item1Item2Conf))
            print('Interest of association:'+t1+'->'+t2+'='+str(item1Item2Interest))
            print()
            totalInterestingAssociations=totalInterestingAssociations+1
        
        if item2Item1Interest >= interestThreshold:
            print('Confidence of association:'+t2+'->'+t1+'='+str(item2Item1Conf))
            print('Interest of association:'+t2+'->'+t1+'='+str(item2Item1Interest))
            print()
            totalInterestingAssociations=totalInterestingAssociations+1
            
    print('Total Interesting associations: '+str(totalInterestingAssociations))


# ## Main Workflow

# In[121]:


supportThreshold = 200 # Chosen based on the mean value of item counts
basketDataSource = 'bread_basket.csv'
pcyBucketCount = 53 # chosen based on exeperiements with bucket counts. #53 yielded optimzed results.
interestThreshold = 0.06 #chosen based on the average of calculated interests of candidate pairs

def execute_APriori():
    itemIDCountLookup, itemNameCountLookup,totalTransactions = Apriori_Pass1(basketDataSource)
    candidatePairs,candidateNamePairs = Apriori_Pass2(basketDataSource, itemIDCountLookup, supportThreshold)
    DisplayInterestingAssocations(candidateNamePairs, itemNameCountLookup, totalTransactions, interestThreshold)
    
def execute_PCY():
    
    itemIDCountLookup, itemNameCountLookup, bucketCountLookup, totalTransactions = PCY_Pass1(basketDataSource)
    candidatePairs,candidateNamePairs = PCY_Pass2(basketDataSource, itemIDCountLookup, itemNameCountLookup, bucketCountLookup, supportThreshold)
    DisplayInterestingAssocations(candidateNamePairs, itemNameCountLookup, totalTransactions, interestThreshold)


def main():
    
    print('Data Source: '+basketDataSource)
    print('Support Threshold: '+str(supportThreshold))
    print('Bucket Count used for PCY: '+str(pcyBucketCount))
    print('Interest Threshold: '+str(interestThreshold))
    
    print('\n'+'*******Executing A-Priori Algorithm*******\n')
    
    execute_APriori()
    
    print('\n'+'*******Executing PCY Algorithm*******\n')
    execute_PCY()
         
if __name__ == "__main__":
    main()

