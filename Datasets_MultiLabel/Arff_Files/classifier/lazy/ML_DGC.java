package mulan.classifier.lazy;

import java.util.Hashtable;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
<!-- globalinfo-start -->
* Class implementing the Multi-Label KNN based on a Data Gravitational Model.<br>
* <br>
* For more information, see<br>
* <br>
* Oscar Reyes, Carlos Martorel and Sebastian Ventura (2016).Effective lazy learning algorithm based on a data gravitation model for multi-label learning. Information Sciences 40(7):159-174.
* <br>
<!-- globalinfo-end -->
*
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* &#64@article{REYES2016159,
title = "Effective lazy learning algorithm based on a data gravitation model for multi-label learning",
journal = "Information Sciences",
volume = "340-341",
pages = "159 - 174",
year = "2016",
issn = "0020-0255",
doi = "https://doi.org/10.1016/j.ins.2016.01.006",
author = "Oscar Reyes and Carlos Morell and Sebastián Ventura",
}
* </pre>
* <br>
*/

public class ML_DGC extends MultiLabelKNN{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * The training instances
	 */
	private Instance[] training_instances;
	
	/** 
	 * The gravitation coefficients for each training instance
	 */
	private double[] gravitational_coefficients;
		
	public ML_DGC(){
		super();
	}
	
	public ML_DGC(int num_of_neighbors) {
		super(num_of_neighbors);
	}
	
	public String globalInfo() {
	    return "Class implementing the Multi-Label KNN based on a Data Gravitational Model." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
	}

	/**
	 * Technical information about the method
	 * @return technical information 
	 */
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

	    result = new TechnicalInformation(Type.ARTICLE);
	    result.setValue(Field.AUTHOR, "Oscar Reyes, Carlos Morell and Sebastián Ventura");
	    result.setValue(Field.TITLE, "Effective lazy learning algorithm based on a data gravitationmodel for multi-label learning");
	    result.setValue(Field.JOURNAL, "Information Sciences.");
	    result.setValue(Field.VOLUME, "340");
	    result.setValue(Field.NUMBER, "");
	    result.setValue(Field.YEAR, "2016");
	    result.setValue(Field.ISSN, "0020-0255");
	    result.setValue(Field.PAGES, "159-174");
	    result.setValue(Field.PUBLISHER, "Elsevier Science Inc.");
	    result.setValue(Field.ADDRESS, "Cuba");

	    return result;
	 }
	
	/**
	 * Calculates the distance between two instances based on features
	 * @param instance1 the first instance
	 * @param instance2 the second instance
	 * @return feature distance the Euclidean distance
	 */

	private double calculateFeatureDistance(Instance instance1, Instance instance2) {
		double feature_distance = dfunc.distance(instance1, instance2);
		
		return feature_distance;
	}
	
	/**
	 * Calculates the distance between two instances based on labels
	 * @param instance1 the first instance
	 * @param instance2 the second instance
	 * @param label_indices indices of the labels
	 * @return label distance the label distance, based on symmetric difference
	 */
	
	private double calculateLabelDistance(Instance instance1, Instance instance2) {
		double symmetric_difference = 0;
		double label_distance;
		double value1, value2;
		int index;
		
		for(int i = 0; i < numLabels; i++) {
			index = labelIndices[i];
			value1 = instance1.value(index);
			value2 = instance2.value(index);

			if(value1 > value2 || value2 > value1)
				symmetric_difference+=1.0;
		}
		
		label_distance = symmetric_difference/numLabels;
		
		return label_distance;
	}
	
	/**
	 * The prior probability that nearest instances to i belong to different set of labels
	 * @param instance instance
	 * @param neighbors the set of k-nearest neighbors
	 * @return the prior probability
	 */
	
	private double priorProbabilityLabel(Instance instance, Instances neighbors){
		Instance neighbor;
		double sum_distances = 0;
		double label_distance;
		double prior_probability;
		
		for(int i = 0; i < numOfNeighbors; i++) {
			neighbor = neighbors.get(i);
			label_distance = calculateLabelDistance(instance, neighbor);
			sum_distances+=label_distance;
		}
		
		prior_probability = sum_distances/numOfNeighbors;
		
		return prior_probability;
	}
	
	/**
	 * The prior probability that nearest instances to i belong to different set of features
	 * @param instance instance
	 * @param neighbors the set of k-nearest neighbors
	 * @return the prior probability
	 */
	
	private double priorProbabilityFeature(Instance instance, Instances neighbors){
		Instance neighbor;
		double sum_distances = 0;
		double feature_distance;
		double prior_probability;
		
		for(int i = 0; i < numOfNeighbors; i++) {
			neighbor = neighbors.get(i);
			feature_distance = calculateFeatureDistance(instance, neighbor);
			sum_distances+=feature_distance;
		}
		
		prior_probability = sum_distances/numOfNeighbors;
		
		return prior_probability;
	}
	
	/**
	 * The probability that nearest instances to i have dissimilar labels set given that
	 * they are far in feature spacebelong to different set of features
	 * @param instance instance
	 * @param neighbors the set of k-nearest neighbors
	 * @return the prior probability
	 */
	
	private double priorProbabilityLabelFeature(Instance instance, Instances neighbors){
		Instance neighbor;
		double sum_distances = 0;
		double label_distance, feature_distance;
		double distance;
		double prior_probability;
		
		for(int i = 0; i < numOfNeighbors; i++) {
			neighbor = neighbors.get(i);
			label_distance = calculateLabelDistance(instance, neighbor);
			feature_distance = calculateFeatureDistance(instance, neighbor);
			distance = label_distance*feature_distance;
			sum_distances+=distance;
		}
		
		prior_probability = sum_distances/numOfNeighbors;
		
		return prior_probability;
	}
	
	/**
	 * Calculates neighborhood weight for an instance given its set of neighbors
	 * @param instance instance
	 * @param neighbors the k-Nearest neighbors of the instance
	 * @return the neighborhood weight
	 */
	
	private double calculateNeighboorhoodWeight(Instance instance, Instances neighbors) {
		double neighborhood_weight;
		double fraction1, fraction2;
		double numerator1, numerator2;
		double denominator2;
		double prior_label = priorProbabilityLabel(instance, neighbors);
		double prior_feature = priorProbabilityFeature(instance, neighbors);
		double prior_label_feature = priorProbabilityLabelFeature(instance, neighbors);
		double aux2 = 1 - prior_label_feature;
		
		numerator1 = prior_label_feature*prior_feature;
		fraction1 = numerator1/prior_label;
		
		numerator2 = aux2*prior_feature;
		denominator2 = 1 - prior_label;	
		fraction2 = numerator2/denominator2;
				
		neighborhood_weight = fraction1 - fraction2;
		
		return neighborhood_weight;
	}
	
	/**
	 * Normalize the neighborhood weights into the interval [0,1]
	 * @param initial_weights The neighborhood weights
	 * @return The normalized neighborhood weights
	 */
	
	private double[] normalizeNeighborhoodWeights(double [] initial_weights) {
		int size = initial_weights.length;
		double max_weight = Double.MIN_VALUE;
		double min_weight = Double.MAX_VALUE;
		double wide;
		double numerator;
		double weight, normalized_weight;
		double[] normalized_weights = new double[size];
		
		for(int i = 0; i < size; i++){
			weight = initial_weights[i];
			
			if(weight > max_weight)
				max_weight = weight;
			
			if(weight < min_weight)
				min_weight = weight;
		}
		
		wide = max_weight - min_weight;
		
		for(int i = 0; i < size; i++){
			weight = initial_weights[i];
			numerator = weight - min_weight;
			normalized_weight = numerator/wide;
			normalized_weights[i] = normalized_weight;
		}
		
		return normalized_weights; 
	}
	/**
	 * Gets the gravitational coefficient for a training instance given its set of K-Nearest neighbors and its neoghborhood weight
	 * @param instance The training instance
	 * @param neighbors its set of K-nearest neighbors
	 * @param neighbor_weight the neighborhood weight of x
	 * @return the gravitational coefficient
	 */
	
	private double getGravitationalCoefficient(Instance instance, Instances neighbors, double neighbor_weight) {
		double neighborhood_density;
		double gravitational_coefficient;
		double feature_distance, label_distance;
		Instance neighbor;
		double sum_densities = 0;
		double partial_density;
		double numerator, denominator;
		
		for(int i = 0; i < numOfNeighbors; i++) {
			neighbor = neighbors.get(i);
			feature_distance = calculateFeatureDistance(instance, neighbor);
			label_distance = calculateLabelDistance(instance, neighbor);
			numerator = 1 - label_distance;
			denominator = feature_distance*feature_distance;
			partial_density = numerator/denominator;
			sum_densities+=partial_density;
		}
		
		neighborhood_density = 1+sum_densities;
		
		gravitational_coefficient = Math.pow(neighborhood_density, neighbor_weight);
		
		return gravitational_coefficient;
	}
	
	/**
	 * Obtains the gravitational force between a test instance and a train instance
	 * @param test test instance
	 * @param instance training instance
	 * @return the gravitational force
	 */
	   
    private double gravitationalForce(Instance test, Instance instance) {
    	double gravitational_force;
    	double feature_distance = this.calculateFeatureDistance(test, instance);
    	double denominator = feature_distance*feature_distance;
    	double gravitational_coefficient = Double.MIN_VALUE;
    	boolean found = false;
    	double distance;
    	int num_training_instances = training_instances.length;
    	Instance training_instance;
    	
     	for(int i = 0; i < num_training_instances && !found; i++) {
     		training_instance = training_instances[i];
     		distance = this.calculateFeatureDistance(instance, training_instance);
     		
     		if(distance == 0) {
     			found = true;
     			gravitational_coefficient = gravitational_coefficients[i];
     		}
     	}    			
    	
    	gravitational_force = gravitational_coefficient/denominator;
    	
    	return gravitational_force;
    }
	
	public void buildInternal(MultiLabelInstances training_set) throws Exception {
		super.buildInternal(training_set);
		int num_instances = train.numInstances();
		double[] neighborhood_weights = new double[num_instances];
		Instance instance;
		Instances neighbors;
		double neighborhood_weight, normalized_weight;
		Instances[] neighbors_set = new Instances[num_instances];
		double[] normalized_neighborhood_weights;
		double gravitational_coefficient;
		Double coefficient;
		
		training_instances = new Instance[num_instances];
						
		for(int i = 0; i < num_instances; i++) {
			instance = train.get(i);
			neighbors = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));
			neighbors_set[i] = neighbors;
			neighborhood_weight =  calculateNeighboorhoodWeight(instance, neighbors);
			neighborhood_weights[i] = neighborhood_weight;
			training_instances[i] = instance;
		}
		
		normalized_neighborhood_weights = normalizeNeighborhoodWeights(neighborhood_weights);
		
		gravitational_coefficients = new double[num_instances];
		
		for(int i = 0; i < num_instances; i++){
			instance = training_instances[i];
			neighbors = neighbors_set[i];
			normalized_weight = normalized_neighborhood_weights[i];
			gravitational_coefficient = getGravitationalCoefficient(instance, neighbors, normalized_weight);
			coefficient = new Double(gravitational_coefficient);
			gravitational_coefficients[i] = coefficient;
		}
	}
	
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        double[] confidences = new double[numLabels];
        boolean[] predictions = new boolean[numLabels];
        MultiLabelOutput prediction;
        Instance neighbor;
        double[] scores_relevant = new double[numLabels];
        double[] scores_irrelevant = new double[numLabels];
        double confidence, score_relevant, score_irrelevant, denominator;
        double gravitational_force;
        boolean neighbor_relevant;
        int label_index;
        double neighbor_value;
        boolean relevant;
        
        Instances knn = null;
        try {
            knn = lnn.kNearestNeighbours(instance, numOfNeighbors);
        } catch (Exception ex) {
            Logger.getLogger(MLkNN.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        for(int i = 0; i < numOfNeighbors; i++){
        	neighbor = knn.instance(i);
        	gravitational_force = gravitationalForce(instance, neighbor);
        	
        	for(int j = 0; j < numLabels; j++) {
        		label_index = labelIndices[j];
        		neighbor_value = neighbor.value(label_index);
        		neighbor_relevant = neighbor_value > 0;
        		
        		if(neighbor_relevant)
        			scores_relevant[j] += gravitational_force;
        		
        		else
        			scores_irrelevant[j] += gravitational_force;
        	}
        }
        
        for(int j = 0; j < numLabels; j++) {
        	score_relevant = scores_relevant[j];
        	score_irrelevant = scores_irrelevant[j];
        	denominator = score_relevant+score_irrelevant;
        	confidence = score_relevant/denominator;
        	confidences[j] = confidence;
        	relevant = score_relevant > score_irrelevant;
        	predictions[j] = relevant;
        }
        
        prediction = new MultiLabelOutput(predictions, confidences);
        
        return prediction;
	}

	public static void main(String[] args) throws InvalidDataException, Exception {	    	
	    	String location = "C:/Proyecto/MultiLabel_Ligeros";
	    	String location_arff = location + "/Arff_Files/CAL500.arff";
	    	String location_xml = location + "/XML_Files/CAL500.xml";
	    	MultiLabelInstances ml_instances =  new MultiLabelInstances(location_arff, location_xml);
	    	ML_DGC dgc = new ML_DGC();
	    	dgc.build(ml_instances);
	    	Instances instances = ml_instances.getDataSet();
	    	MultiLabelOutput prediction = dgc.makePredictionInternal(instances.instance(0));// 	double[] frequencies_labels = getFrequenciesLabels(instances, label_index, label_index2);
	    	System.out.println("End of the checking");
	    	
	    }
	
}
