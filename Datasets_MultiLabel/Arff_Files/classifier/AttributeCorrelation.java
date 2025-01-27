package mulan.classifier;

import mulan.data.MultiLabelInstances;
//import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class AttributeCorrelation {

	/**
	 * Calculates the entropy of a probability distribution
	 * @param distribution array with the probability distribution
	 * @return the entropy of the probability distribution
	 */
		
	private static double calculateEntropy(double[] distribution) {
		double entropy = 0.0;
		double partial_entropy;
		double probability, log_probability;
		int num_values = distribution.length;
		
		for(int i = 0; i < num_values; i++) {
			probability = distribution[i];
			
			if(probability > 0) {
				log_probability = Math.log(probability)/Math.log(2);
				partial_entropy = -probability*log_probability;
				entropy += partial_entropy; 
			}
				
		}
				
		return entropy;
	}
	
	/**
	 * Obtains the probability distribution corresponding to an array of frequencies
	 * @param frequencies the array of frequencies
	 * @return the array with the probability distribution
	 */
	
	private static double[] getProbabilityDistribution(double[] frequencies){
		int num_values = frequencies.length;
		double sum = Utils.sum(frequencies);
		double probability;
		double[] probability_distribution = new double[num_values];
		
		for(int i = 0; i < num_values; i++) {
			probability = frequencies[i]/sum;
			probability_distribution[i] = probability;
		}
		
		return probability_distribution;
			
	}
	
	
	/**
	 * 	Obtains the probability distribution with maximum entropy according to the A-NPI-M from an array of frequencies 
	 * @param frequencies the array of frequencies
	 * @return the array of the distribution of maximum entropy
	 */
	
	private static double[] distributionMaxEntropy(double[] frequencies){
		int num_values = frequencies.length;
		double[] probability_distribution;
		double[] transformed_frequencies = new double[num_values];
		double frequency, max_frequency;
		double mass = 0;
		double i;
		double ki, ki1;
		double aux;
		
		max_frequency = Double.MIN_VALUE;
		
		for(int j = 0; j < num_values; j++){
			frequency = frequencies[j];
			
			if(frequency > 0) {
				transformed_frequencies[j] = frequency - 1;
				mass += 1.0;
				
				if(frequency > max_frequency)
					max_frequency = frequency;
			}
			
			else
				 transformed_frequencies[j] = 0;
		}
		
		i = 0;
		
		while(mass > 0 && i <= max_frequency) {
			ki=0;
			ki1=0;
			
			for(int j = 0; j < num_values; j++) {
				frequency =  frequencies[j];
						
				if(frequency == i)
					ki+=1;
				
				else if(frequency == i+1)
					ki1+=1;
			}
			
			aux = ki + ki1;
			
			if(aux < mass) {
				for(int j = 0; j < num_values; j++) {
					frequency =  frequencies[j];
					
					if(frequency == i || frequency == i+1) {
						transformed_frequencies[j] += 1.0;
						mass -=1.0;
					}
				}
			}
			
			else {
				for(int j = 0; j < num_values; j++) {
					frequency =  frequencies[j];
					
					if(frequency == i || frequency == i+1) 
						 transformed_frequencies[j] += mass/aux;
					
				}
				
				mass = 0.0;
			}
			
			i+=1.0;
		}
		
		probability_distribution = getProbabilityDistribution(transformed_frequencies);
		
		return probability_distribution;
	}
	
	/**
	 * Obtains the entropy associated with an array of frequencies
	 * @param frequencies array of frequencies
	 * @return the corresponding entropy
	 */
	
	public static double entropyFrequencies(double[] frequencies){
		double[] probability_distribution = getProbabilityDistribution(frequencies);
		double entropy = calculateEntropy(probability_distribution);
		
		return entropy;
	}
	
	/**
	 * Obtains the maximum of entropy according to the A-NPI-M associated with an array of frequencies
	 * @param frequencies array of frequencies
	 * @return the corresponding entropy
	 */
	
	
	public static double maxEntropy(double [] frequencies) {
		double[] distribution_max_entropy = distributionMaxEntropy(frequencies);
		double entropy = calculateEntropy(distribution_max_entropy);
		
		return entropy;
	}
	
	/**
	 * Obtains the cut points for a continuous attribute in a dataset, in such a way that
	 * the attribute is discretized in three intervals with the same width.  
	 * @param instances dataset
	 * @param attribute_index Index of the continuos attribute
	 * @return Array with the three cut points
	 */
	
	private static double[] getCutPoints(Instances instances, int attribute_index){
		double[] cut_points = new double[2];
		double point1, point2;
		double max_value = Double.MIN_VALUE;
		double min_value = Double.MAX_VALUE;
		double value;
		Instance instance;
		int num_instances = instances.numInstances();
		double range, wide;
		
		for(int i = 0; i < num_instances; i++){
			instance = instances.get(i);
			value = instance.value(attribute_index);
			
			if(value < min_value)
				min_value = value;
			
			if(value > max_value)
				max_value = value;
		}
		
		range = max_value - min_value;
		wide = range/3;
		
		point1 = min_value + wide;
		point2 = point1 + wide;
		
		cut_points[0] = point1;
		cut_points[1] = point2;
		
		return cut_points;
	}
	
	/**
	 * Obtains the array values of the instances for an attribute, taking into 
	 * account the discretization for continuous attributes
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute
	 * @return the array of values of the attribute for the instances
	 */
	
	private static int[] getArrayValues(Instances instances, int attribute_index) {
		Instance instance;
		int num_instances = instances.numInstances();
		Attribute attribute = instances.attribute(attribute_index);
		int[] array_values = new int[num_instances];
		double value;
		int value_instance;
		double[] cut_points = null;
		boolean discrete = attribute.isNominal();
		
		if(!discrete)
			cut_points = getCutPoints(instances, attribute_index);
		
		for(int i = 0; i < num_instances; i++) {
			instance = instances.get(i);
			value = instance.value(attribute);
			
			if(discrete)
				value_instance = (int)value;
			
			else {
				if(value < cut_points[0])
					value_instance = 0;
				
				else if (value < cut_points[1])
					value_instance = 1;
				
				else
					value_instance = 2;
			}
			
			array_values[i] = value_instance;
		}
		
		return array_values;
	}
	
	/**
	 * Obtains the array of frequencies of the instances for an attribute, taking into 
	 * account the discretization for continuous attributes
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute
	 * @return the array of frequencies
	 */
	
	private static double[] getFrequenciesAttribute(Instances instances, int attribute_index) {
		Attribute attribute = instances.attribute(attribute_index);
		boolean discrete = attribute.isNominal();
		int[] array_values = getArrayValues(instances, attribute_index);
		double[] array_frequencies;
		int num_values;
		int num_instances = instances.numInstances();
		int value;
		
		if(discrete)
			num_values = attribute.numValues();
			
		else
			num_values = 3;
		
		array_frequencies = new double[num_values];
		
		for(int i = 0; i < num_instances; i++) {
			value = array_values[i];
			array_frequencies[value]+=1.0;
		}
		
		return array_frequencies;
			
	}
	/**
	 * Obtains the entropy for an attribute corresponding to a dataset 
	 * account the discretization for continuous attributes
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute
	 * @return the entropy of the attribute
	 */
	
	public static double entropyAttribute(Instances instances, int attribute_index) {
		double[] frequencies = getFrequenciesAttribute(instances, attribute_index);
		double entropy = entropyFrequencies(frequencies);
		
		return entropy;
	}
	
	/**
	 * Obtains the maximum of entropy according to A-NPI-M for an attribute corresponding to a dataset 
	 * account the discretization for continuous attributes
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute
	 * @return the entropy of the attribute
	 */
	
	public static double maxEntropyAttribute(Instances instances, int attribute_index) {
		double[] frequencies = getFrequenciesAttribute(instances, attribute_index);
		double entropy = maxEntropy(frequencies);
		
		return entropy;
	}

	
	/**
	 * Contingency matrix of the attribute X given Y
	 * One row for each possible value of Y, and one column for each possible value of X
	 * m[j][i] = number of occurrences of Y = y_j and X = x_i
	 * @param instances instances of the dataset
	 * @param attribute_index1 index of the attribute X
	 * @param attribute_index2 index of the  attribute Y
	 * @return contingency table
	 */
	
	
	private static double[][] getContingencyTable(Instances instances, int attribute_index1, int attribute_index2){
		int num_instances = instances.numInstances();
		int value1, value2;
		Attribute attribute1 = instances.attribute(attribute_index1);
		Attribute attribute2 = instances.attribute(attribute_index2);
		double[][] contingency_table;
		boolean nominal1 = attribute1.isNominal();
		boolean nominal2 = attribute2.isNominal();
		int num_values1, num_values2;
		int[] values_attribute1 = getArrayValues(instances, attribute_index1); 
		int[] values_attribute2 = getArrayValues(instances, attribute_index2); 
		
		if(nominal1)
			num_values1 = attribute1.numValues();
		
		else
			num_values1 = 3;
		
		if(nominal2)
			num_values2 = attribute2.numValues();
		
		else
			num_values2 = 3;
		
		contingency_table = new double[num_values2][num_values1];	
		
		for(int i = 0; i< num_instances; i++) {
			value1 =  values_attribute1[i];
			value2 =  values_attribute2[i];
			contingency_table[value2][value1]+=1.0;
		}
		
		return contingency_table;
	}
	
	
	/**
	 * Conditioned entropy of X given Y
	 * Array of probability distribution of Y. Consider the contingency table. Entropy of each row
	 * Entropy of row i (H(X|Y = yj)) x P(Y = yj). Sum of the entropies of the rows. 
	 * @param instances instances of the dataset
	 * @param attribute_index1 index of the  attribute X
	 * @param attribute_index2 index of the  attribute Y
	 * @return  H(X|Y)
	 */
	
	public static double conditionedEntropy(Instances instances, int attribute_index1, int attribute_index2) {
		double conditioned_entropy = 0;
		double partial_conditioned_entropy, partial_entropy;
		double[][] contingency_table = getContingencyTable(instances, attribute_index1, attribute_index2);
		double[] frequencies2 = getFrequenciesAttribute(instances, attribute_index2);
		double[] probability_distribution2 = getProbabilityDistribution(frequencies2);
		int num_values2 = probability_distribution2.length;
		double[] partial_frequencies1;
		double probability2;
		
		for(int i = 0; i < num_values2; i++) {
			partial_frequencies1 = contingency_table[i];
			partial_entropy =  entropyFrequencies(partial_frequencies1);
			probability2 = probability_distribution2[i];
			partial_conditioned_entropy = probability2*partial_entropy;
			conditioned_entropy+=partial_conditioned_entropy;
		}
		
		return conditioned_entropy;
	}
	
	/**
	 * Maximum conditioned entropy of X given Y
	 * Array of probability distribution of Y. Consider the contingency table. Entropy of each row
	 * Entropy of row i (H*(X|Y = yj)) x P*(Y = yj). P* distribution of maximum of entropy for Y. Sum of the entropies of the rows. 
	 * @param instances instances of the dataset
	 * @param attribute_index1 attribute_index2 index of the  attribute X
	 * @param attribute_index2 attribute_index2 index of the  attribute Y
	 * @return  H*(X|Y)
	 */
	
	public static double conditionedMaxEntropy(Instances instances, int attribute_index1, int attribute_index2) {
		double conditioned_entropy = 0;
		double partial_conditioned_entropy, partial_entropy;
		double[][] contingency_table = getContingencyTable(instances, attribute_index1, attribute_index2);
		double[] frequencies2 = getFrequenciesAttribute(instances, attribute_index2);
		double[] probability_distribution2 = distributionMaxEntropy(frequencies2);
		int num_values2 = probability_distribution2.length;
		double[] partial_frequencies1;
		double probability2;
		
		for(int i = 0; i < num_values2; i++) {
			partial_frequencies1 = contingency_table[i];
			partial_entropy =  maxEntropy(partial_frequencies1);
			probability2 = probability_distribution2[i];
			partial_conditioned_entropy = probability2*partial_entropy;
			conditioned_entropy+=partial_conditioned_entropy;
		}
		
		return conditioned_entropy;
	}
	
	
	/**
	 * Mutual Information of X and Y
	 * I(X,Y) = H(X) - H(X|Y)
	 * @param instances instances of the dataset
	 * @param attribute_index1 attribute_index2 index of the  attribute X
	 * @param attribute_index2 attribute_index2 index of the  attribute Y
	 * @return mutual information Mutual Information I(X,Y)
	 */
	
	public static double mutualInformation(Instances instances, int attribute_index1, int attribute_index2){
		double mutual_information;
		double entropy, conditioned_entropy;
		
		entropy =  entropyAttribute(instances, attribute_index1);
		conditioned_entropy = conditionedEntropy(instances, attribute_index1, attribute_index2);
		mutual_information = entropy - conditioned_entropy;
		
		return mutual_information;
	}
	
	/**
	 * Imprecise Mutual Information of X and Y
	 * I*(X,Y) = H*(X) - H*(X|Y)
	 * @param instances instances of the dataset
	 * @param attribute_index1 attribute_index2 index of the  attribute X
	 * @param attribute_index2 attribute_index2 index of the  attribute Y
	 * @return mutual information Mutual Information I*(X,Y)
	 */
	
	public static double impreciseMutualInformation(Instances instances, int attribute_index1, int attribute_index2){
		double mutual_information;
		double entropy, conditioned_entropy;
		
		entropy =  maxEntropyAttribute(instances, attribute_index1);
		conditioned_entropy = conditionedMaxEntropy(instances, attribute_index1, attribute_index2);
		mutual_information = entropy - conditioned_entropy;
		
		return mutual_information;
	}
	
	/** Calculates the imprecise symmetrical uncertainty of two variables 
	 * SU*(X,Y) = I*(X,Y)/[H*(X)+H*(Y)]
	 * @param instances instances of the dataset
	 * @param attribute_index1 attribute_index2 index of the  attribute X
	 * @param attribute_index2 attribute_index2 index of the  attribute Y
	 * @return mutual information Imprecise Symmetrical Uncertainty SU*(X,Y)
	
	 */
	
	public static double impreciseSymmetricalUncertainty(Instances instances, int attribute_index1, int attribute_index2) {
		double imprecise_symmetrical_uncertainty;
		double entropy1 = AttributeCorrelation.maxEntropyAttribute(instances, attribute_index1);
		double entropy2 = AttributeCorrelation.maxEntropyAttribute(instances, attribute_index2);
		double numerator = impreciseMutualInformation(instances, attribute_index1, attribute_index2);
		double denominator = entropy1+entropy2;
		double fraction = numerator/denominator;
		
		imprecise_symmetrical_uncertainty = 2*fraction;
		
		return imprecise_symmetrical_uncertainty;
		


	}
	
	/**
	 * Array of frequencies corresponding to a pair of labels li, lj
	 * position 0: li = lj = 0, position1: li=0, lj=1, 
	 * position2: li = 1, lj = 0, position 3: li=lj=1
	 * @param instances instances of the dataset
	 * @param label_index1 index of label li
	 * @param label_index2 index of label lj
	 * @return array of frequencies of the labels
	 */
	
	private static double[] getFrequenciesLabels(Instances instances, int label_index1, int label_index2) {
		double[] frequencies_labels = new double[4];
		int value_label1, value_label2;
		int num_instances = instances.numInstances();
		Instance instance;
		
		for(int i = 0; i < num_instances; i++) {
			instance = instances.get(i);
			value_label1 = (int)instance.value(label_index1);
			value_label2 = (int)instance.value(label_index2);
			
			if(value_label1 == 0) {
				
				if(value_label2 == 0)
					frequencies_labels[0]+=1.0;
				
				else
					frequencies_labels[1]+=1.0;
			}
			
			else {
				
				if(value_label2 == 0)
					frequencies_labels[2]+=1.0;
				
				else
					frequencies_labels[3]+=1.0;
			}
		}
		
		return frequencies_labels;
	}
	
	/**
	 * Array of probabilities corresponding to a pair of labels li, lj
	 * position 0: li = lj = 0, position1: li=0, lj=1, 
	 * position2: li = 1, lj = 0, position 3: li=lj=1
	 * @param instances instances of the dataset
	 * @param label_index1 index of label li
	 * @param label_index2 index of label lj
	 * @return array of probability distribution of the labels
	 */
	
	private static double[] getProbabilitiesLabels(Instances instances, int label_index1, int label_index2){
		double[] frequencies_labels = getFrequenciesLabels(instances, label_index1, label_index2);
		double [] probabilities_labels = getProbabilityDistribution(frequencies_labels);
		
		return probabilities_labels; 
	}
	
	/**
	 * Array of probabilities that attains the maximum of entropy according to A-NPI-M
	 * corresponding to a pair of labels li, lj 
	 * position 0: li = lj = 0, position1: li=0, lj=1, 
	 * position2: li = 1, lj = 0, position 3: li=lj=1
	 * @param instances instances of the dataset
	 * @param label_index1 index of label li
	 * @param label_index2 index of label lj
	 * @return array of probability distribution of the labels
	 */
	
	private static double[] getProbabilitiesLabelsMaxEntropy(Instances instances, int label_index1, int label_index2){
		double[] frequencies_labels = getFrequenciesLabels(instances, label_index1, label_index2);
		double [] probabilities_labels = distributionMaxEntropy(frequencies_labels);
		
		return probabilities_labels; 
	}
	
	/**
	 * Conditioned entropy of an attribute X given labels li, lj
	 * Contingency matrix, each row corresponds to a combination of values li, lj
	 * One column for each possible value of X
	 * Calculation of each H(X|li = k, l_j = m)*P(li=k, l_j = m), k,m = 1,2.
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute X
	 * @param label_index1 index of the first label
	 * @param label_index2 index of the second label
	 * @return H(X|l_i,l_j)
	 */
	
	public static double conditionedEntropyLabels(Instances instances, int attribute_index, int label_index1, int label_index2) {
		int[] values_label1 = getArrayValues(instances, label_index1);
		int[] values_label2 = getArrayValues(instances, label_index1);
		int[] values_attribute = getArrayValues(instances, attribute_index);
		int num_instances = instances.numInstances();
		Attribute attribute = instances.attribute(attribute_index);
		boolean discrete = attribute.isNominal();
		int num_values;
		double[][] frequencies;
		int attribute_value, label_value1, label_value2;
		double[] probability_labels = getProbabilitiesLabels(instances, label_index1, label_index2);
		double[] partial_frequencies;
		double partial_entropy, entropy;
		double probability;
		double conditioned_entropy = 0.0;
		
		if(discrete)
			num_values = attribute.numValues();
		
		else
			num_values = 3;
		
		frequencies = new double[4][num_values];
		
		for(int i = 0; i < num_instances; i++) {
			attribute_value = values_attribute[i];
			label_value1 = values_label1[i];
			label_value2 = values_label2[i];
		
			if(label_value1 == 0) {
				if(label_value2 == 0) 
					frequencies[0][attribute_value]+=1.0;
				
				else
					frequencies[1][attribute_value]+=1.0;
				
			}
			
			else {
				if(label_value2 == 0) 
					frequencies[2][attribute_value]+=1.0;
				
				else
					frequencies[3][attribute_value]+=1.0;
			}
		}
		
		for(int i = 0; i < 4; i++) {
			partial_frequencies = frequencies[i];
			partial_entropy = entropyFrequencies(partial_frequencies);
			probability = probability_labels[i];
			entropy = partial_entropy*probability;
			conditioned_entropy+=entropy;
		}
		
		return conditioned_entropy;
	}
	
	/**
	 * Conditioned maximum of entropy (A-NPI-M) of an attribute X given labels li, lj
	 * Contingency matrix, each row corresponds to a combination of values li, lj
	 * One column for each possible value of X
	 * Calculation of each H*(X|li = k, l_j = m)*P*(li=k, l_j = m), k,m = 1,2.
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute X
	 * @param label_index1 index of the first label
	 * @param label_index2 index of the second label
	 * @return H*(X|l_i,l_j)
	 */
	
	
	public static double conditionedMaxEntropyLabels(Instances instances, int attribute_index, int label_index1, int label_index2) {
		int[] values_label1 = getArrayValues(instances, label_index1);
		int[] values_label2 = getArrayValues(instances, label_index2);
		int[] values_attribute = getArrayValues(instances, attribute_index);
		int num_instances = instances.numInstances();
		Attribute attribute = instances.attribute(attribute_index);
		boolean discrete = attribute.isNominal();
		int num_values;
		double[][] frequencies;
		int attribute_value, label_value1, label_value2;
		double[] probability_labels = getProbabilitiesLabelsMaxEntropy(instances, label_index1, label_index2);
		double[] partial_frequencies;
		double partial_entropy, entropy;
		double probability;
		double conditioned_entropy = 0.0;
		
		if(discrete)
			num_values = attribute.numValues();
		
		else
			num_values = 3;
		
		frequencies = new double[4][num_values];
		
		for(int i = 0; i < num_instances; i++) {
			attribute_value = values_attribute[i];
			label_value1 = values_label1[i];
			label_value2 = values_label2[i];
		
			if(label_value1 == 0) {
				if(label_value2 == 0) 
					frequencies[0][attribute_value]+=1.0;
				
				else
					frequencies[1][attribute_value]+=1.0;
				
			}
			
			else {
				if(label_value2 == 0) 
					frequencies[2][attribute_value]+=1.0;
				
				else
					frequencies[3][attribute_value]+=1.0;
			}
		}
		
		for(int i = 0; i < 4; i++) {
			partial_frequencies = frequencies[i];
			partial_entropy = maxEntropy(partial_frequencies);
			probability = probability_labels[i];
			entropy = partial_entropy*probability;
			conditioned_entropy+=entropy;
		}
		
		return conditioned_entropy;
	}
	
	/**
	 *  Mutual conditioned information of an attribute X given labels li, lj
	 * I(X; li|l_j) = H(f|yj) - H(f,li,lj)
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute X
	 * @param label_index1 index of the label li
	 * @param label_index2 index of the label lj
	 * @return I(X; li|lj)
	 */
	
	public static double mutualConditionedInformation(Instances instances, int attribute_index, int label_index1, int label_index2) {
		double conditioned_entropy1 = conditionedEntropy(instances, attribute_index, label_index2);
		double conditioned_entropy2 = conditionedEntropyLabels(instances, attribute_index, label_index1, label_index2);
		double mutual_conditioned_information = conditioned_entropy1 - conditioned_entropy2;
		
		return mutual_conditioned_information;
	}
	
	/**
	 *  Mutual conditioned information of an attribute X given labels li, lj
	 * I*(X; li|l_j) = H*(f|yj) - H*(f,li,lj)
	 * @param instances instances of the dataset
	 * @param attribute_index index of the attribute X
	 * @param label_index1 index of the label li
	 * @param label_index2 index of the label lj
	 * @return I*(X; li|lj)
	 */
	
	public static double impreciseMutualConditionedInformation(Instances instances, int attribute_index, int label_index1, int label_index2) {
		double conditioned_entropy1 = conditionedMaxEntropy(instances, attribute_index, label_index2);
		double conditioned_entropy2 = conditionedMaxEntropyLabels(instances, attribute_index, label_index1, label_index2);
		double mutual_conditioned_information = conditioned_entropy1 - conditioned_entropy2;
		
		return mutual_conditioned_information;
	}
	
	public static void main(String[] args) throws InvalidDataException, Exception {
    //	double[] frequencies = {7,2,1,0};
    //	double[] distribution_entropy = distributionMaxEntropy(frequencies);
    	String location = "C:/Proyecto/MultiLabel_Ligeros";
    	String location_arff = location + "/Arff_Files/emotions.arff";
    	String location_xml = location + "/XML_Files/emotions.xml";
    	MultiLabelInstances ml_instances =  new MultiLabelInstances(location_arff, location_xml);
    	Instances instances = ml_instances.getDataSet();
    	// int attribute_index = 80;
    	// int attribute_index2;
    	// double[] cut_points = getCutPoints(instances, attribute_index);
    	// int[] attribute_values = getArrayValues(instances, attribute_index);
    	// double[] frequencies =  getFrequenciesAttribute(instances, attribute_index);
    	int[] label_indices = ml_instances.getLabelIndices();
    	//double[][] contingency_table = getContingencyTable(instances, attribute_index,attribute_index2);
    //	double conditioned_entropy = conditionedEntropy(instances, attribute_index,attribute_index2);
    //	double mutual_information = impreciseMutualInformation(instances, attribute_index,attribute_index2);
    	int label_index = label_indices[0];
    	int label_index2 = label_indices[5];
    	double[][] contingency_table = getContingencyTable(instances, label_index,label_index2);
    	double conditioned_entropy = conditionedEntropy(instances, label_index2, label_index);
   // 	double[] frequencies_labels = getFrequenciesLabels(instances, label_index, label_index2);
    //	double conditioned_entropies_labels = conditionedMaxEntropyLabels(instances, attribute_index, label_index, label_index2);
    //	double mutual_information = mutualConditionedInformation(instances, attribute_index, label_index, label_index2);
    //	double imprecise_mutual_information = impreciseMutualConditionedInformation(instances, attribute_index, label_index, label_index2);
    	System.out.println("Contingency table " + contingency_table);
    	System.out.println("Conditioned entropy = " + conditioned_entropy);
    	
    }

}
