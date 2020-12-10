# Last update: 26 November
# 
# Author: Shaurya Shubham
# This code load the true graph and transform
# into train, valid and to_use datasets.
#
# inputs:
# true.csv

#
# outpus:
# train.csv
# valid.csv
# to_use.csv
# edges.csv
####################################################################################

# Load libraries
import pandas as pd # To handle dataframes
from sklearn.model_selection import train_test_split # to split the datasets


def Edges_Generation(true_graph):
    ''' Function to generate edges from the true graph '''

    edges = []
    for element in true_graph.iterrows():
        edges.append([element[1]['entity_a'],element[1]['entity_b'],element[1]['relation']])

    # Saving data
    df = pd.DataFrame(edges,columns=['entity_a', 'entity_b','relation'])
    df.to_csv('./Data/edges.csv',index= False)

def Generate_Train_Valid_Use(true_graph):
    ''' Function to generate train, valid and to_use datasets '''
    
    # Randomization of true
    true_graph = true_graph.sample(frac=1).reset_index(drop = True)
    
    # print some stats
    print 'Total unique entity_a : {}'.format(len(true_graph['entity_a'].unique().tolist()))
    print 'Total unique entity_b : {}'.format(len(true_graph['entity_b'].unique().tolist()))
    print 'Total unique entities : {}'.format(len(list(set(true_graph['entity_a'].unique().tolist() + true_graph['entity_b'].unique().tolist()))))
    print 'Total unique relations : {}'.format(len(true_graph['relation'].unique().tolist()))

	# Splitting true_graph into 3 pieces: true_train, true_valid, true_to_use
    train, valid_test = train_test_split(true_graph,test_size =0.2, random_state =1) # split the data into train and (test_vaidation)
    
    # print train dataset stats to see whether we are missing any entities or not
    print 'Total unique entity_a in train dataset : {}'.format(len(train['entity_a'].unique().tolist()))
    print 'Total unique entity_b in train dataset : {}'.format(len(train['entity_b'].unique().tolist()))
    print 'Total unique entities in train dataset : {}'.format(len(list(set(train['entity_a'].unique().tolist() + train['entity_b'].unique().tolist()))))
    
    unique_train_entities =list(set(train['entity_a'].unique().tolist() + train['entity_b'].unique().tolist()))
    unique_true_graph_entities = d =list(set(true_graph['entity_a'].unique().tolist() + true_graph['entity_b'].unique().tolist()))
    missing_entities_in_train =  set(unique_true_graph_entities) -set(unique_train_entities)
    
    train_extra = valid_test[valid_test.entity_b.isin(missing_entities_in_train)] # get the missing entities from entity_b to get embedding for all entities
    train = pd.concat([train,train_extra]) # making the complete train datset with all entities
    train = train.reset_index(drop = True)
    train.to_csv('./Data/train.csv',index=False)
    
    valid_test = valid_test[~valid_test.entity_b.isin(missing_entities_in_train)] # remove the train data which we added to train dataset by using train_extra
    
    valid,test = train_test_split(valid_test,test_size =0.75, random_state =1) # split the data into valid and test(to_use)
    valid.to_csv('./Data/valid.csv',index=False)
    test.to_csv('./Data/to_use.csv',index=False)


def main():
    ''' Principal function '''
    true_graph = pd.read_csv('./Data/true.csv')
    Generate_Train_Valid_Use(true_graph)
    Edges_Generation(true_graph)
    return 0

if __name__ == '__main__':
    main()