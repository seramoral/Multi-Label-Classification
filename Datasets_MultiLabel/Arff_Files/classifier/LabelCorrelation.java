package mulan.classifier;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class LabelCorrelation {
	
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
	
	/* Method to obtain the probability distribution array given the array of frequencies.
	 * 
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
	
	/* Method to obtain the distribution with the maximum of entropy given an array of frequencies
	 * According to the A-NPI-M
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
				frequency =  transformed_frequencies[j];
						
				if(frequency == i)
					ki+=1;
				
				else if(frequency == i+1)
					ki1+=1;
			}
			
			aux = ki + ki1;
			
			if(aux < mass) {
				for(int j = 0; j < num_values; j++) {
					frequency =  transformed_frequencies[j];
					
					if(frequency == i || frequency == i+1) {
						transformed_frequencies[j] += 1.0;
						mass -=1.0;
					}
				}
			}
			
			else {
				for(int j = 0; j < num_values; j++) {
					frequency =  transformed_frequencies[j];
					
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
	
	public static double entropy(double[] frequencies) {
		int num_values = frequencies.length;
		double[] probabilities = new double[num_values];
		double N = Utils.sum(frequencies);
		double probability;
		double entropy_probability;
		
		for(int i = 0; i < num_values; i++) {
			probability = frequencies[i]/N;
			probabilities[i] = probability;
		}
		
		entropy_probability = calculateEntropy(probabilities);
		
		return entropy_probability;
	}
	
	public static double entropyLabel(Instances dataset, int label_index) {
		double[] frequencies = new double[2];
		int num_instances = dataset.numInstances();
		Instance instance;
		int label_value;
		double entropy_label;
		
		frequencies[0] = 0.0;
		frequencies[1] = 0.0;
		
		for(int i = 0; i < num_instances; i++) {
			instance = dataset.instance(i);
			label_value = (int)instance.value(label_index);
			
			if(label_value == 0)
				frequencies[0]+=1.0;
			
			else
				frequencies[1]+=1.0;
		}
		
		entropy_label = entropy(frequencies);
		
		return entropy_label;
	}
	
	public static double maxEntropy(double[] frequencies) {
		double[] distribution_max_entropy = distributionMaxEntropy(frequencies);
		double max_entropy = calculateEntropy(distribution_max_entropy);
		
		return max_entropy;
	}
	
	public static double maxEntropyLabel(Instances dataset, int label_index) {
		double[] frequencies = new double[2];
		int num_instances = dataset.numInstances();
		Instance instance;
		int label_value;
		double max_entropy;
		
		frequencies[0] = 0.0;
		frequencies[1] = 0.0;
		
		for(int i = 0; i < num_instances; i++) {
			instance = dataset.instance(i);
			label_value = (int)instance.value(label_index);
			
			if(label_value == 0)
				frequencies[0]+=1.0;
			
			else
				frequencies[1]+=1.0;
		}
		
		max_entropy = maxEntropy(frequencies);
		
		return max_entropy;
	}
	public static double conditionedEntropy(Instances dataset, int label_index1, int label_index2){
		double conditioned_entropy;
		int num_instances = dataset.numInstances();
		Instance instance;
		int label_value1, label_value2;
		double[] frequencies_j = new double[2];
		double[] frequencies_i0 = new double[2];
		double[] frequencies_i1 = new double[2];
		double[] probabilities_j = new double[2];
		double entropy0, entropy1;
		
		for(int i = 0; i < num_instances; i++) {
			instance = dataset.instance(i);
			label_value1 = (int)instance.value(label_index1);
			label_value2 = (int)instance.value(label_index2);
			
			if(label_value2 == 0) {
				frequencies_j[0]+=1.0;
				
				if(label_value1 == 0)
					frequencies_i0[0]+=1.0;
				
				else
					frequencies_i0[1]+=1.0;
			}
			
			else {
				frequencies_j[1]+=1.0;
				
				if(label_value1 == 0)
					frequencies_i1[0]+=1.0;
				
				else
					frequencies_i1[1]+=1.0;
			}

		}
		
		entropy0 = entropy(frequencies_i0);
		entropy1 = entropy(frequencies_i1);
		
		probabilities_j[0] = frequencies_j[0]/num_instances;
		probabilities_j[1] = frequencies_j[1]/num_instances;
		
		conditioned_entropy = entropy0*probabilities_j[0] + entropy1*probabilities_j[1];

		return conditioned_entropy;
	}

	public static double conditionedMaxEntropy(Instances dataset, int label_index1, int label_index2){
		double conditioned_entropy;
		int num_instances = dataset.numInstances();
		Instance instance;
		int label_value1, label_value2;
		double[] frequencies_j = new double[2];
		double[] frequencies_i0 = new double[2];
		double[] frequencies_i1 = new double[2];
		double[] distribution_max_entropy_j;
		double max_entropy_i0, max_entropy_i1;
		
		for(int i = 0; i < num_instances; i++) {
			instance = dataset.instance(i);
			label_value1 = (int)instance.value(label_index1);
			label_value2 = (int)instance.value(label_index2);
			
			if(label_value2 == 0) {
				frequencies_j[0]+=1.0;
				
				if(label_value1 == 0)
					frequencies_i0[0]+=1.0;
				
				else
					frequencies_i0[1]+=1.0;
			}
			
			else {
				frequencies_j[1]+=1.0;
				
				if(label_value1 == 0)
					frequencies_i1[0]+=1.0;
				
				else
					frequencies_i1[1]+=1.0;
			}

		}
		distribution_max_entropy_j = distributionMaxEntropy(frequencies_j);
		max_entropy_i0 = maxEntropy(frequencies_i0);
		max_entropy_i1 = maxEntropy(frequencies_i1);

		conditioned_entropy = distribution_max_entropy_j[0]*max_entropy_i0 + distribution_max_entropy_j[1]*max_entropy_i1;
	
		return conditioned_entropy;
	}
	
	public static double mutualInformation(Instances dataset, int label_index1, int label_index2) {
		double max_entropy = maxEntropyLabel(dataset, label_index1);
		double conditioned_max_entropy = conditionedMaxEntropy(dataset, label_index1,label_index2);
		double mutual_information = max_entropy - conditioned_max_entropy;
		
		return mutual_information;
	}
	
	public static double conditionedPairEntropy(Instances dataset, int label_index1, int label_index2, int label_index3) {
		double[][] frequencies12 = new double[2][4];
		double[] frequencies3 = new double[2];
		double[] probabilities3= new double[2];
		double[] partial_frequencies;
		double partial_entropy;
		double conditioned_pair_entropy = 0;
		int num_instances = dataset.numInstances();
		Instance instance;
		int label_value1, label_value2, label_value3;
		
		for(int i = 0; i < num_instances; i++) {
			instance = dataset.instance(i);
			label_value1 = (int)instance.value(label_index1);
			label_value2 = (int)instance.value(label_index2);
			label_value3 = (int)instance.value(label_index3);
			
			if(label_value3 == 0) {
				frequencies3[0]+=1.0;
				
				if(label_value1 == 0) { 
					
					if(label_value2 == 0)
						frequencies12[0][0] += 1.0;
					
					else
						frequencies12[0][1] += 1.0;
				}
				
				else {
					
					if(label_value2 == 0)
						frequencies12[0][2] += 1.0;
					
					else
						frequencies12[0][3] += 1.0;
					
				}
				
			}
			
			else {			
				frequencies3[1]+=1.0;
				
				if(label_value1 == 0) { 
					
					if(label_value2 == 0)
						frequencies12[1][0] += 1.0;
					
					else
						frequencies12[1][1] += 1.0;
				}
				
				else {
					
					if(label_value2 == 0)
						frequencies12[1][2] += 1.0;
					
					else
						frequencies12[1][3] += 1.0;
					
				}
			}
		}
		
		probabilities3[0] = frequencies3[0]/num_instances;
		probabilities3[1] = frequencies3[1]/num_instances;
		
		for(int i = 0; i < 2; i++) {
			partial_frequencies = frequencies12[i];
			partial_entropy = entropy(partial_frequencies);
			conditioned_pair_entropy+= partial_entropy;
		}
		
		return conditioned_pair_entropy;
	}
	
    public static void main(String[] args) throws InvalidDataException, Exception {
    	double[] frequencies = {8,1,1};
    	double[] distribution_entropy = distributionMaxEntropy(frequencies);
    	double entropy = calculateEntropy(distribution_entropy);
    	String location = "C:/Proyecto/MultiLabel_Ligeros";
    	String location_arff = location + "/Arff_Files/emotions.arff";
    	String location_xml = location + "/XML_Files/emotions.xml";
    	MultiLabelInstances ml_instances =  new MultiLabelInstances(location_arff, location_xml);
    	int[] label_indices = ml_instances.getLabelIndices();
    	int label_index = label_indices[0];
    	int label_index2 = label_indices[5];
    	Instances instances = ml_instances.getDataSet();
    	double max_entropy = maxEntropyLabel(instances,label_index);
    	double conditioned_max_entropy = conditionedMaxEntropy(instances, label_index2, label_index);
    	
    	
    	System.out.println("Maxima entropia = " + max_entropy);
    }
}
