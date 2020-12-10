# Last update: 26 November 2018
# 
# This code create True MBSE dataset
# 
# Inputs: 
# 			None
# Outputs:
#			true.csv : List of true pair of nodes, 
#					   link type and target in format: [entity_a, entity_b, relation, target]
####################################################################################################

# Load libraries
import numpy as np
import pandas as pd

# Set parameters
Dimension = 1000

# Targe values
true = 1

# Big nodes
RequerimentType = 0
TestCaseType = 1000
SMT = 2000
ProjectType = 3000

#Relation type
instanceOf = 0
verifiedBy = 1
using = 2
belongsTo = 3

def Generate_True_Graph():
	''' Function to create all true links '''

	graph = []
	for requeriment in range(RequerimentType + 1, Dimension):
	    #[requirement[1-Dimension], RequirementType0, Relation = 0]
	    graph.append([requeriment, RequerimentType, instanceOf, true])
	    
	    #[requirement[1-Dimension], TestCase[1-Dimension], Relation = 1]
	    graph.append([requeriment, requeriment + TestCaseType, verifiedBy, true])
	    
	    #[requirement[1-Dimension], Proyect[1-Dimesion], Relation = 3]
	    graph.append([requeriment, requeriment + ProjectType, belongsTo, true])
	    
	for testcase in range(TestCaseType + 1, TestCaseType + Dimension):
	    #[testcase[1-Dimension], TestCaseType0, Relation = 0]
	    graph.append([testcase, TestCaseType, instanceOf, true])
	    
	    #[testcase[1-Dimension], SM[1-Dimension], Relation = 2]
	    graph.append([testcase, testcase + (SMT - Dimension), using, true])
	    
	    #[testcase[1-Dimension], TestCaseType0, Relation = 3]
	    graph.append([testcase, testcase + (ProjectType - Dimension), belongsTo, true])
	    
	for sm in range(SMT + 1, SMT + Dimension):
	    #[sm[1-Dimension], SMT, Relation = 0]
	    graph.append([sm, SMT, instanceOf, true])
	    
	    #[sm[1-Dimension], Proyect[1-Dimension], Relation = 3]
	    graph.append([sm, sm + Dimension, belongsTo, true])

	for project in range(ProjectType + 1, ProjectType + Dimension):
	    #[project[1-Dimension], ProjectType, Relation = 0]
	    graph.append([project, ProjectType, instanceOf, true])

	return graph


def main():
	''' Principal function '''

	# Call for Postive and Negative graph
	positive_graph = Generate_True_Graph()

	# Transform into a pandas dataframe
	true = pd.DataFrame(positive_graph, columns=['entity_a', 'entity_b', 'relation', 'target'])
	
	# Save data
	true.to_csv('./Data/true.csv',index = False)

	return 0

if __name__ == '__main__':
    main()