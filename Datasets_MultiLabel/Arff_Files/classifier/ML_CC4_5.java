package mulan.classifier;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;

@SuppressWarnings("serial")
public class ML_CC4_5 extends MultiLabelLearnerBase{
	private ML_CC4_5[] successors;
	private Attribute m_Attribute;
	private MultiLabelOutput output;
	private double s_value;
	
	public ML_CC4_5(){}
	
	public ML_CC4_5(double s) {
		s_value = s;
	}
	
	private double[] computeFrequencies(MultiLabelInstances ML_instances, Attribute attribute){
		int num_instances = ML_instances.getNumInstances();
		Instances instances = ML_instances.getDataSet();
		Instance instance;
		int num_values = attribute.numValues();
		double[] frequencies = new double[num_values];
		int corresponding_index;
		double value;
		
		for(int i = 0; i < num_instances; i++) {
			instance = instances.instance(i);
			value = instance.value(attribute);
			corresponding_index = (int) value;
			frequencies[corresponding_index] += 1.0;
		}
		
		return frequencies;
	}
	
	private double[] distributeFrecuencies(double[] frequencies, double s) {
		if(s == 0)
			return frequencies;
		
		double aux;
		int num_frequencies = frequencies.length;
		double [] distributed_frequencies = new double[num_frequencies];
		double frequency;
		int num_min = 0;
		double min_frequency = frequencies[0];
		
		if(s <= 1)
			aux = s;
		
		else
			aux = 1.0;
			
		for(int i = 0; i < num_frequencies; i++) {
			frequency = frequencies[i];
			distributed_frequencies[i] = frequency;
			
			if(frequency == min_frequency)
				num_min++;
			
			else if(frequency < min_frequency){
				num_min = 1;
				min_frequency = frequency;
			}
		}
		
		for(int i = 0; i < num_frequencies; i++) {
			frequency = frequencies[i];
			
			if(frequency == min_frequency) 
				distributed_frequencies[i] += aux/(double)num_min;
	
		}
		
		if(s <= 1)
			return distributed_frequencies;
		
		
		double s2 = s-1.0;
		double[] distributed_frequencies2 = distributeFrecuencies(distributed_frequencies, s2);
		
		return distributed_frequencies2;
		
	}
	
	private double computeAttributeEntropy(MultiLabelInstances ML_instances, int index){
		double attribute_entropy = 0.0;
		double s;
		boolean label_index = false;
		int index2;
		Attribute attribute = ML_instances.getDataSet().attribute(index);
		double[] frequencies = computeFrequencies(ML_instances, attribute);
		int num_values = attribute.numValues();
		double frequency, probability, logarithm;
		double partial_entropy;
		int num_instances = ML_instances.getNumInstances();
		double[] distributed_frequencies;
		
		for(int i = 0; i < numLabels && !label_index; i++) {
			index2 = labelIndices[i];
			
			if(index2 == index)
				label_index = true;
		}
		
		if(label_index){
			s = s_value;
			num_instances+=s;
			distributed_frequencies = distributeFrecuencies(frequencies, s);
		}
		
		else 
			distributed_frequencies = frequencies;
		
		for(int i = 0; i < num_values; i++) {
			frequency = distributed_frequencies[i];
			probability = frequency/num_instances;
			
			if(probability > 0) {
				logarithm = Utils.log2(probability);
				partial_entropy = -probability*logarithm;
				attribute_entropy += partial_entropy; 
			}
			
		}
		
		return attribute_entropy;
	}
	
	private double computeEntropy(MultiLabelInstances ML_instances){
		double total_entropy = 0.0;
		double partial_entropy;
		int index;
		
		for(int i = 0; i < numLabels; i++) {
			index = labelIndices[i];
			partial_entropy = computeAttributeEntropy(ML_instances, index);
			total_entropy+=partial_entropy;
		}
		
		return total_entropy;
	}
	
	private MultiLabelInstances[] splitData(MultiLabelInstances ML_instances, Attribute attribute) throws InvalidDataFormatException {
		int num_values = attribute.numValues();
		Instances[] previous_splits = new Instances[num_values];
		MultiLabelInstances[] splits = new MultiLabelInstances[num_values];
		Instances original_instances = ML_instances.getDataSet();
		int num_instances = original_instances.numInstances();
		Instance instance;
		double value;
		int corresponding_index;
		
		for(int i = 0; i < num_values; i++)
			previous_splits[i] = new Instances(original_instances,num_instances);
		
		for(int j = 0;  j < num_instances; j++) {
			instance = original_instances.instance(j);
			value = instance.value(attribute);
			corresponding_index = (int) value;
			previous_splits[corresponding_index].add(instance);
		}
		
		for(int i = 0; i < num_values; i++) {
			previous_splits[i].compactify();
			splits[i] = ML_instances.reintegrateModifiedDataSet(previous_splits[i]);
		}
		
		return splits;
	}
	
	private double computeInfoGainRatio(MultiLabelInstances ML_instances, int attribute_index, double entropy) throws InvalidDataFormatException{
		Attribute attribute = ML_instances.getDataSet().attribute(attribute_index);
		int num_values = attribute.numValues();
		int partition_size;
		double probability;
		double info_gain, info_gain_ratio;
		int num_instances = ML_instances.getNumInstances();
		double attribute_entropy = computeAttributeEntropy( ML_instances, attribute_index);
		MultiLabelInstances[] splits = splitData(ML_instances, attribute);
		MultiLabelInstances split;
		double sum_entropies = 0.0;
		double product, partial_entropy;
		
		for(int i = 0; i < num_values; i++) {
			split = splits[i];
			partition_size = split.getNumInstances();
			probability = (double)partition_size/(double)num_instances;
			partial_entropy = computeEntropy(split);
			product = partial_entropy*probability;
			sum_entropies+=product;
		}
		
		info_gain = entropy - sum_entropies;
		
		info_gain_ratio = info_gain/attribute_entropy;
		
		return info_gain_ratio;
		
	}
	
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception{
		int num_instances = trainingSet.getNumInstances();
		
		if(num_instances == 0) {
			m_Attribute = null;
			boolean bipartition[] = new boolean[numLabels];
			double confidences[] = new double[numLabels];
			output = new MultiLabelOutput(bipartition, confidences);
		}
		
		else {
			int feature_index;
			int num_features = featureIndices.length;
			double[] infoGainsRatios = new double[num_features];
			double info_gain_ratio, max_info_gain_ratio;
			int max_index, max_feature_index;
			double class_entropy = computeEntropy(trainingSet);
			
			for(int i = 0; i < num_features; i++) {
				feature_index = featureIndices[i];
				info_gain_ratio = computeInfoGainRatio(trainingSet, feature_index, class_entropy); 
				infoGainsRatios[i] = info_gain_ratio;
			}
			
			max_index = Utils.maxIndex(infoGainsRatios);
			max_info_gain_ratio = infoGainsRatios[max_index];
			max_feature_index = featureIndices[max_index];
			
			if(max_info_gain_ratio > 0) {
				m_Attribute = trainingSet.getDataSet().attribute(max_feature_index);
				int num_values = m_Attribute.numValues();
				successors = new ML_CC4_5[num_values];
				MultiLabelInstances[] splits = splitData(trainingSet, m_Attribute);
				MultiLabelInstances split;
				
				for(int i = 0; i < num_values; i++) {
					successors[i] = new ML_CC4_5();
					split = splits[i];
					successors[i].build(split);
				}
			}
			
			else {
				m_Attribute = null;
				boolean bipartition[] = new boolean[numLabels];
				double confidences[] = new double[numLabels];
				boolean relevant;
				int index;
				double[] frequencies;
				Attribute label_attribute;
				
				for(int i = 0; i < numLabels; i++) {
					index = labelIndices[i];
					label_attribute = trainingSet.getDataSet().attribute(index);
					frequencies = computeFrequencies(trainingSet, label_attribute);
					confidences[i] = frequencies[1]/(double)num_instances;
					relevant = confidences[i] >= 0.5;
					bipartition[i] = relevant;
				}
				
				output = new MultiLabelOutput(bipartition, confidences);
			}
		}
	}
	
	protected MultiLabelOutput makePredictionInternal(Instance instance){
		
		if(m_Attribute == null)
			return output;
		
		double value = instance.value(m_Attribute);
		int index = (int)value;
		ML_CC4_5 corresponding_successor = successors[index]; 
		MultiLabelOutput output2 = corresponding_successor.makePredictionInternal(instance);

		return output2;
	}
	
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}
	
}
