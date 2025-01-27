/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.OutputImpreciseLabelRanking;
import mulan.data.MultiLabelInstances;
import mulan.transformations.PairwiseTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.CredalDecisionTree;
import weka.core.*;

/**
 * <p>Class implementing the Ranking by Pairwise Comparisons (RPC) algorithm.
 * For more information, see <em>H&uuml;llermeier, E.; F&uuml;rnkranz, J.;
 * Cheng, W.; Brinker, K. (2008) Label ranking by learning pairwise preferences,
 * Artificial Intelligence 172(16-17):1897-1916</em></p>
 *
 * @author Grigorios Tsoumakas
 * @version 2012.11.1
 */

public class ImprecisePairwise{

	 /** Whether to consider predicted probability intervals or binary outputs */
    protected boolean probability_intervals_output = true;
    
    /** Array holding the one VS one models */
    protected CredalDecisionTree[] oneVSone_models;
    
    /** The base classifier */
    
    protected CredalDecisionTree base_classifier;
    
    /**
     * number of one vs one models
     */
    private int numModels;
    /**
     * whether no data exist for one-vs-one learning
     */
    private boolean[] nodata;
    
    /**
     * pairwise transformation
     */
    private PairwiseTransformation pt;
    
    /**
     * The number of labels
     */
    
    int num_labels;
    
    /**
     * Constructor that initializes the learner
     * By default, the prediction is in form of probability intervals
     */
    public ImprecisePairwise() {
    	base_classifier = new CredalDecisionTree();
        probability_intervals_output = true;
    }
    
    /**
     * Constructor that initializes the learner
     * @param intervals_output whether the output 
     * is in form of probability intervals
     */
    public ImprecisePairwise(boolean intervals_output) {
    	base_classifier = new CredalDecisionTree();
        probability_intervals_output = intervals_output;
    }
    
    /**
     * Gets the base classifier
     * @return the base classifier
     */
    
    CredalDecisionTree getBaseClassifier() {
    	return base_classifier;
    }
    
    /**
     * Gets the one VS one model associated with a given an index i
     * @param modelIndex the index of the one VS one model to obtain
     * @return the i-th one VS one model
     */
    
    public CredalDecisionTree getModel(int modelIndex) {
        return oneVSone_models[modelIndex];
    }
    
    /**
     * Checks whether there are no data for building the i-th one VS one model
     * @param index the index of the model to check
     * @return true if there are no data for the i-th model, false else
     */
    
    public boolean noData(int index) {        
        return nodata[index];
    }
    
    /**
     * Gets the pairwise transformation
     * @return the pairwise transform
     */
    
    public PairwiseTransformation getTransformation() {
        return pt;
    }
    
    /**
     * Gets the number of labels
     * @return the number of labels
     */
    
    public int getNumLabels(){
    	return num_labels;
    }
    
    /**
     * Sets whether the prediction is binary 
     * or in the form of probability intervals
     * @param probability_intervals
     */
    
    public void setFormOutput(boolean probability_intervals) {
    	probability_intervals_output = probability_intervals;
    }
    
    /**
     * Builds the binary Credal Decision trees given the Multi-Label training set
     * @param train the Multi-Label training instances
     * @throws Exception 
     */
    
    protected void buildInternal(MultiLabelInstances train) throws Exception {
    	num_labels = train.getNumLabels();
    	int counter_model;
    	Instances oneVSone_instances;
    	int num_instances_oneVSone;
    	
    	numModels = (num_labels*(num_labels - 1))/2;
    	
    	// Sets the base classifier for the one VS one models
    	oneVSone_models = new CredalDecisionTree[numModels];
    	
    	for(int i = 0; i < numModels; i++) {
    		oneVSone_models[i] = (CredalDecisionTree) AbstractClassifier.makeCopy(base_classifier);
    	}
    	
    	// oneVSone_models = (CredalDecisionTree[]) AbstractClassifier.makeCopies(base_classifier, numModels);
        
    	nodata = new boolean[numModels];
        pt = new PairwiseTransformation(train);

        counter_model = 0;
        
        // Creation of one-vs-one models

        for(int label1 = 0; label1 < num_labels - 1; label1++) {
        	for(int label2 = label1 + 1; label2 < num_labels; label2++) {
        		// Pairwise comparison of label1 and label2 
        		
        		// Prepare the training set
        		oneVSone_instances = pt.transformInstances(label1, label2);        		        		
        		num_instances_oneVSone = oneVSone_instances.numInstances();

        		/* If there are instances for  building the 
        		 * One VS One model, build it
        		 * Otherwise, save this information on the nodata vector
        		 */
        		
        		if(num_instances_oneVSone > 0) {
        			oneVSone_models[counter_model].buildClassifier(oneVSone_instances);
        		}
        		
        		else {
        			nodata[counter_model] = true;
        		}
        			
        		counter_model++;
        	}
        }
        
    }
    
    /**
     * It computes the lower and upper probabilities for each label
     * The lower probability is the sum of the lower probabilities in the pairwise classifiers
     * The upper probability is the sum of the upper probabilities in the pairwise classifiers
     * @param instance instance passed through the pairwise transformation filter
     * @param instance the array of upper probabilities to return
     * @return array of lower probabilities. 
     */
    
    public double[] computeLowerUpperProbabilities(Instance instance, double[] upper_probabilities){
    	 Instance transformed_instance = pt.transformInstance(instance);
    	 int counter_model;
    	 boolean model_built;
    	 double[] lower_probabilities;
    	 double[] lower_probabilities_pairwise_model, upper_probabilities_pairwise_model;
    	 CredalDecisionTree pairwise_model;
    	 
    	 lower_probabilities = new double[num_labels];
    	 counter_model = 0;
    	 
    	 for(int label1 = 0; label1 < num_labels-1; label1++) {
    		 for(int label2 = label1+1; label2 < num_labels; label2++) {
    			 /* Check whether the pairwise model corresponding to
    			  * label 1 and label 2 has been built */
    			 model_built = !nodata[counter_model];
    			 
    			 if(model_built) {
    				 pairwise_model = oneVSone_models[counter_model];
    				 lower_probabilities_pairwise_model = pairwise_model.getLowerProbabilities(transformed_instance);
    				 upper_probabilities_pairwise_model = pairwise_model.getUpperProbabilities(transformed_instance);
    				 
    				 lower_probabilities[label1]+=lower_probabilities_pairwise_model[1];
    				 lower_probabilities[label2]+=lower_probabilities_pairwise_model[0];
    				 upper_probabilities[label1]+=upper_probabilities_pairwise_model[1];
    				 upper_probabilities[label2]+=upper_probabilities_pairwise_model[0];

    			 }
    			 
    			 counter_model++;
    		 }
    	 }
    	 
    	 return lower_probabilities;
    }
    
    /**
     * Given an instance,it sums the lower and upper scores for each label
     * For each classifier y_i, y_j, 
     * it sums 1 to both lower and upper scores of y_i if y_i dominates y_j
     * sums 1 to both lower and upper scores of y_j if y_j dominates y_i
     * sums 1 to both lower and upper scores of y_j if y_j dominates y_i
     * y_i dominates y_j if the lower probability of y_i is >= upper probability of y_j
     * @param instance the test instance
     * @param upper scores the array of upper scores for each label
     * @return the array of lower scores for each label
     */
    
    public double[] getLowerUpperScores(Instance instance, double[] upper_scores) {
    	Instance transformed_instance = pt.transformInstance(instance);
    	int counter_model;
   	 	boolean model_built;
   	 	CredalDecisionTree pairwise_model;
        double[] lower_scores = new double[num_labels];
        double[] lower_probabilities, upper_probabilities;
        
   	 	counter_model = 0;
	 
   	 	for(int label1 = 0; label1 < num_labels-1; label1++) {
   	 		for(int label2 = label1+1; label2 < num_labels; label2++) {
			 /* Check whether the pairwise model corresponding to
			  * label 1 and label 2 has been built */
   	 			model_built = !nodata[counter_model];
			 
   	 			if(model_built) {
   	 				pairwise_model = oneVSone_models[counter_model];
   	 				lower_probabilities = pairwise_model.getLowerProbabilities(transformed_instance);
   	 				upper_probabilities = pairwise_model.getUpperProbabilities(transformed_instance);
				 
   	 				// Checks whether label1 dominates label2
				 
   	 				if(lower_probabilities[1] > upper_probabilities[0]) {
   	 					lower_scores[label1]++;
   	 					upper_scores[label1]++;
   	 				}
				 
   	 				// Checks whether label2 dominates label1
				 
   	 				else if(lower_probabilities[0] > upper_probabilities[1]) {
   	 					lower_scores[label2]++;
   	 					upper_scores[label2]++;
   	 				}
   	 				
   	 				else { // If neither label1 dominates label 2 nor vice-versa
   	 					upper_scores[label1]++;
   	 					upper_scores[label2]++;
   	 				}

   	 			}
			 
   	 			counter_model++;
   	 		}
   	 	}
   	 	
   	 	return lower_scores;
    }
    
    /**
     * It determines the probability intervals for the labels from the lower and upper scores
     * They are determined by the lower (upper) scores divided by the number of labels
     * @param lower_scores the lower scores for each label
     * @param upper_scores the upper scores for each label
     * @param upper_probabilities the upper probabilities estimated for each label
     * @return the lower probabilities estimated for each label
     */
    
    public double[] getLowerUpperProbabilitiesFromScores(double[] lower_scores, double[] upper_scores, double[] upper_probabilities){
   	 	double[] lower_probabilities = new double[num_labels];
   	 	   	 	
   	 	for(int i = 0; i < num_labels; i++){
   	 		lower_probabilities[i] = lower_scores[i]/num_labels;
   	 		upper_probabilities[i] = upper_scores[i]/num_labels;
   	 	}
   	 	
   	 	return lower_probabilities;
    }
    
  

}
