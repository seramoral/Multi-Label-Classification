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

import mulan.classifier.Experimentation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.OutputImpreciseLabelRanking;
import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.trees.*;

public class ImpreciseBinaryRelevance {
	/**
	 * The ensemble of Credal Decision Trees for Binary Relevance
	 * An instance of Credal Decision Tree
	 */
	
	protected CredalDecisionTree[] ensemble;
	
	/**The correspondence between ensemble models and labels */
    private String[] correspondence;
    
    /** The Binary Relevance transformation */
    private BinaryRelevanceTransformation brt;

   
    
    /** The base classifier */
    protected CredalDecisionTree base_classifier;

    /** The number of labels */
    
    protected int num_labels;
    
    /**
     * Constructor that initializes the learner
     * By default, the prediction is in form of probability intervals
     */
    public ImpreciseBinaryRelevance() {
    	base_classifier = new CredalDecisionTree();
    }
    
    
    /**
     * Gets the base classifier
     * @return the base classifier
     */
    
    CredalDecisionTree getBaseClassifier() {
    	return base_classifier;
    }
    
    /**
     * Gets the Binary Relevance transformation
     * @return the binary relevance transform
     */
    
    public BinaryRelevanceTransformation getTransformation() {
        return brt;
    }
    

    /**
     * Builds the Imprecise Binary Relevance Classifier 
     * from a given training set 
     * @param train the training set
     * @throws Exception if an exception occurs during the building process
     */
    
    protected void buildInternal(MultiLabelInstances train) throws Exception {
    	num_labels = train.getNumLabels();
    	int[] label_indices = train.getLabelIndices();
    	Instances transformed_instances;
    	
    	ensemble = new CredalDecisionTree[num_labels];
        correspondence = new String[num_labels];
        
        for (int i = 0; i < num_labels; i++) {
            correspondence[i] = train.getDataSet().attribute(label_indices[i]).name();
        }
        
        // Sets the Binary Relevance transformation for the instances
        brt = new BinaryRelevanceTransformation(train);

        // Build each one of the binary models
        
        for(int i = 0; i < num_labels; i++) {
        	ensemble[i] = (CredalDecisionTree) AbstractClassifier.makeCopy(base_classifier);
        	transformed_instances = brt.transformInstances(i);
        	ensemble[i].buildClassifier(transformed_instances);
        }

    }
    
    /**
     * Does a prediction for an unlabeled instance x
     * in the form of lower and upper probabilities 
     * @param instance the unlabeled instance
     * @param upper_probabilities the upper probabilities estimated for each label
     * @return the lower probabilities estimated for each label
     */
    
    public double[] computeLowerUpperProbabilities(Instance instance, double[] upper_probabilities){
    	double[] lower_probabilities = new double[num_labels];
    	double[] binary_lower_probabilities, binary_upper_probabilities;
    	Instance transformed_instance;
    	
    	for(int i = 0; i < num_labels; i++) {
    		transformed_instance = brt.transformInstance(instance, i);
    		binary_lower_probabilities = ensemble[i].getLowerProbabilities(transformed_instance);
    		lower_probabilities[i] = binary_lower_probabilities[1];
    		binary_upper_probabilities = ensemble[i].getUpperProbabilities(transformed_instance);
    		upper_probabilities[i] = binary_upper_probabilities[1];
    	}    	
    	
    	return lower_probabilities;
    }
	
}
