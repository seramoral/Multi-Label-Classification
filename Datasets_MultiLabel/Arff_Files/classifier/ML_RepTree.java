package mulan.classifier;

import java.io.Serializable;
import java.util.Random;

import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.ImpreciseEntropyContingencyTables;

public class ML_RepTree extends MultiLabelLearnerBase implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected double s_value;
	
	public ML_RepTree(double s) {
		s_value = s;
	}

	protected class Tree{
		 protected Tree[] m_Successors;
		    
		    /** The attribute to split on. */
		 protected int m_Attribute = -1;
		 
		 /** The split point. */
		 protected double m_SplitPoint = Double.NaN;
		    
		 /** The proportions of training instances going down each branch. */
	     protected double[] m_Prop = null;

	     protected MultiLabelOutput prediction = null;
	     
	     protected double[][] classProbs = null;
		    
	     /** Class distribution of hold-out set at node in the nominal case. */
		 protected double[] m_HoldOutDist = null;
		    
		 /** The hold-out error of the node. The number of miss-classified
		 instances */
		 protected double m_HoldOutError = 0;
		
		 
		 protected double distribution(double[][] props, double[][][][] dists, int att, int[] sortedIndices, double[] weights, double[][] subsetWeights, Instances data)  throws Exception {
			 double splitPoint = Double.NaN;
			 Attribute attribute = data.attribute(att);
			 int num_values;
		     double[][][] dist = null;
		     int num_instances = sortedIndices.length;
		     int i;
		     Instance instance;
		     boolean nominal = attribute.isNominal();
		     int index_instance;
		     boolean missing;
		     double value;
		     int index_value, label_index, label_value;
		     double proportion;
		     
		     if(nominal){
		    	 num_values  = attribute.numValues();
		    	 dist = new double[num_values][numLabels][2];
		    	 
		    	 for(i = 0; i < num_instances; i++) {
		    		 index_instance = sortedIndices[i];
		    		 instance = data.instance(index_instance);
		    		 missing = instance.isMissing(attribute);
		    		 
		    		 if(missing)
		    			 break;
		    		 
		    		 value = instance.value(attribute);
		    		 index_value = (int) value;
		    		 
		    		 for(int l = 0; l < numLabels; l++) {
		    			 label_index = labelIndices[l];
		    			 label_value = (int)instance.value(label_index);
		    			 dist[index_value][l][label_value]+=weights[i];
		    		 }
		    	 }
		     }
		     
		     else{
		    	 num_values = 2;
		    	 double[][][] currDist = new double[num_values][numLabels][2];
		    	 dist = new double[num_values][numLabels][2];
		    	 
		    	 for(int j = 0; j < num_instances; j++) {
		    		 index_instance = sortedIndices[j];
		    		 instance = data.instance(index_instance);
		    		 missing = instance.isMissing(attribute);
		    		 
		    		 if(missing)
		    			 break;
		    		 
		    		 for(int l = 0; l < numLabels; l++) {
		    			 label_index = labelIndices[l];
		    			 label_value = (int)instance.value(label_index);
		    			 currDist[1][l][label_value] += weights[j];
		    		 }
		    	 }
		    	 
		    	 double priorVal = priorVal(currDist);
		    	 
		    	 for(int k = 0; k < num_values; k++) {
		    		 for(int l = 0; l < numLabels; l++)
		    		 System.arraycopy(currDist[k][l], 0, dist[k][l], 0, 2);
		    	 }
		    		// Try all possible split points
		    	 double currSplit = data.instance(sortedIndices[0]).value(att);
		    	 double currVal, bestVal = -Double.MAX_VALUE;
		    	 
		    	 
		    	 for(i = 0; i < num_instances; i++) {
		    		 index_instance = sortedIndices[i];
		    		 instance = data.instance(index_instance);
		    		 missing = instance.isMissing(attribute);
		    		 
		    		 if(missing)
		    			 break;
		    		 
		    		 value = instance.value(attribute);
		    		 
		    		 if(value > currSplit) {
		    			    currVal = gain(currDist, priorVal);
		    			    
		    			    if (currVal > bestVal) {
		    			      bestVal = currVal;
		    			      splitPoint = (value + currSplit) / 2.0;	
		    			      
		    			      for(int k = 0; k < num_values; k++) {
		    			    		 for(int l = 0; l < numLabels; l++)
		    			    		 System.arraycopy(currDist[k][l], 0, dist[k][l], 0, 2);
		    			      }
		    			      
		    			    } 
		    		 }
		    		 
		    		 currSplit = value;
		    		  
		    		  for(int l = 0; l < numLabels; l++) {
			    			 label_index = labelIndices[l];
			    			 label_value = (int)instance.value(label_index);
			    			 
			    			 currDist[0][l][label_value] += weights[i];
			    			 currDist[1][l][label_value] -= weights[i];
		    		  }
		    	 }
		    	 
		     }
		     props[att] = new double[num_values];
		     double sum;
		     
		     for (int k = 0; k < num_values; k++) {
		    	 sum = Utils.sum(dist[k][0]);
		    	 props[att][k] = sum;
		     }
		     
		     if (!(Utils.sum(props[att]) > 0)) {
		    	for (int k = 0; k < num_values; k++) 
		    		props[att][k] = 1.0 / (double)num_values;
		     }
		    	
		     else {
		    		Utils.normalize(props[att]);
		     }
		     
		     while(i < num_instances){
		    	 index_instance = sortedIndices[i];
		    	 instance = data.instance(index_instance);
		    	 
		    	 for(int j = 0; j < num_values; j++){
		    		 for(int l = 0; l < numLabels; l++) {
		    			 label_index = labelIndices[l];
		    			 label_value = (int)instance.value(label_index);
		    			 proportion = props[att][j];
		    			 dist[j][l][label_value]+=weights[i]*proportion;
		    		 }
		    	 }
		    	 
		    	 i++;
		    }
		     
		    subsetWeights[att] = new double[num_values];
		    
		    for (int j = 0; j < num_values; j++) {
		    	sum = Utils.sum(dist[j][0]);		    	
		    	subsetWeights[att][j] += sum;	
		    }
			 
		    dists[att] = dist;
		    return splitPoint;
		 }
			 
		 public void buildTree(int [][] sortedIndices, double[][] weights, Instances train, double totalWeight, double[][] m_classProbs, int minNum) throws Exception{
			 int help = featureIndices[0];
			 int num_instances = sortedIndices[help].length;
			 
			 if(num_instances == 0) {
				double[] confidences = new double[numLabels];
				prediction = new MultiLabelOutput(confidences,0.5); 
				return;
			 }
			 
			 classProbs = new double[numLabels][2];
		     System.arraycopy(m_classProbs, 0, classProbs, 0, classProbs.length);
			 boolean min_instances = totalWeight >= 2 * (double)minNum;
			 boolean pure = true;
			 double[] partial_probs;
			  
			 for(int i = 0; i < numLabels && pure; i++) {
				 partial_probs = m_classProbs[i];
				 if(!Utils.eq(partial_probs[Utils.maxIndex(partial_probs)], Utils.sum(partial_probs)))
					 	pure = false;
			 }
			 
			 double[] confidences;
			 double[] probs;
			 double frequency;
			 double confidence;
			 double threshold = 0.5 - s_value/num_instances;
			 			 
			 if(pure || !min_instances) {
				 m_Attribute = -1;
				 confidences = new double[numLabels];
				 
				 for(int i = 0; i < numLabels; i++) {
					 probs = classProbs[i];
					 frequency = probs[1];
					 confidence = frequency/num_instances;
					 confidences[i] = confidence;
				 }
				 prediction = new MultiLabelOutput(confidences, threshold);
				 return;
			 }
			 
			 int num_attributes = featureIndices.length;
			 double[] vals = new double[num_attributes];
			 double[][][][] dists = new double[num_attributes][0][0][0];
			 double[][] props = new double[num_attributes][0];
			 double[][] totalSubsetWeights = new double[num_attributes][0];
			 double[] splits = new double[num_attributes];
			 int feature_index,max_index;
			 int num_values;
			 			 
			 for(int i = 0; i < num_attributes; i++) {
				 feature_index = featureIndices[i];
				 splits[i] = distribution(props, dists, feature_index, sortedIndices[i],  weights[i], totalSubsetWeights, train);
				 vals[i] = gain(dists[i], priorVal(dists[i]));
			 }
			 
			 max_index = Utils.maxIndex(vals);
			 m_Attribute = featureIndices[max_index];
			 num_values = dists[max_index].length;
			 
			 int count = 0;
		     
			 for (int i = 0; i < num_values; i++) {
				 if (totalSubsetWeights[max_index][i] >= minNum) {
					 count++;
				 }
				 
				 if (count > 1) 
					 break;
			
		     }
			 
			 boolean min_counts = count > 1;
			 boolean useful_split = vals[max_index] > 0;
			 
			 if(min_counts && useful_split){
				 m_SplitPoint = splits[max_index];
				 m_Prop = props[max_index];
				 int[][][] subsetIndices = new int[num_values][num_attributes][0];
				 double[][][] subsetWeights = new double[num_values][num_attributes][0];
				 splitData(subsetIndices, subsetWeights, m_Attribute, m_SplitPoint, sortedIndices, weights, train);
				 m_Successors = new Tree[num_values];
				 
				 for (int i = 0; i < num_values; i++) {
					 m_Successors[i] = new Tree();
					 m_Successors[i].buildTree(subsetIndices[i], subsetWeights[i],train, totalSubsetWeights[max_index][i], dists[max_index][i], minNum);
				 }
			 } 
			 
			 else{			
				m_Attribute = -1;
				confidences = new double[numLabels];
				 
				for(int i = 0; i < numLabels; i++) {
					probs = classProbs[i];
					frequency = probs[1];
					confidence = frequency/num_instances;
					confidences[i] = confidence;
				}
				prediction = new MultiLabelOutput(confidences, threshold);
			 }
			 
		 }
		 
		 protected int numNodes() {
			    
		      if (m_Attribute == -1) {
		    	  return 1;
		      } 
		      else {
		    	  int size = 1;
		    	  
		    	  for (int i = 0; i < m_Successors.length; i++) 
		    		  	size += m_Successors[i].numNodes();
		    	  
		    	  return size;
		      }
		    }
		 
		 protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException{
			 MultiLabelOutput output = null;
			 
			 if(m_Attribute > -1) {
				 boolean missing = instance.isMissing(m_Attribute);
				 int num_successors = m_Successors.length;
				 double proportion, confidence, partial_confidence;
				 MultiLabelOutput[] outputs_successors  = new MultiLabelOutput[num_successors];
				 MultiLabelOutput output2;
				 double[] partial_confidences;
				 boolean nominal;
				 
				 if(missing) {
					 double[] confidences = new double[numLabels];
					 
					 for(int j = 0; j < num_successors; j++) 
						 outputs_successors[j] = m_Successors[j].makePredictionInternal(instance); 
					 		 
					 for(int i = 0; i < numLabels; i++) {
						 confidence = 0.0;
						 
						 for(int j = 0; j < num_successors; j++) {
							 proportion = m_Prop[j];
							 output2 =  outputs_successors[j];
							 partial_confidences = output2.getConfidences();
							 partial_confidence = proportion*partial_confidences[i];
							 confidence+=partial_confidence;
						 }
						 
						 confidences[i] = confidence; 
					 }
					 
					 output = new MultiLabelOutput(confidences);
				 }
				
				 else {
					 nominal = instance.attribute(m_Attribute).isNominal();
					 double value = instance.value(m_Attribute); 
					 int index;
					 
					 if(nominal)
						 index = (int) value;
					 
					  
					 else{
						 if(value < m_SplitPoint)
						 	index = 0;
						 	
						 else
							index = 1; 
						 		
					 }
					 
					 output = m_Successors[index].makePredictionInternal(instance);
				}
			 }
			 
			 if ((m_Attribute == -1) || (output == null))
				 output = prediction;
				 
			 return output;
		 }
		 
		 protected void splitData(int[][][] subsetIndices, double[][][] subsetWeights, int att, double splitPoint,int[][] sortedIndices, double[][] weights,  Instances data) throws Exception {			    
		      int j;
		      int[] num;
		      Attribute attribute = data.attribute(att);
		      int num_attributes = featureIndices.length;
		      int feature_index;
		      int num_values;
		      boolean nominal = attribute.isNominal();
		      boolean missing;
		      int num_instances_attribute;
		      Instance instance;
		      int index_instance;
		      // For each attribute
		      for (int i = 0; i < num_attributes; i++) {
		    	  feature_index = featureIndices[i];
		    	  num_instances_attribute = sortedIndices[feature_index].length;
		    	  
		    	  if (nominal) {
		    		  num_values = attribute.numValues();
		    		  num = new int[num_values];
		    			  
		    		  for(int k = 0; k < num_values; k++) {
		    			  subsetIndices[k][i] = new int[num_instances_attribute];
		    			  subsetWeights[k][i] = new double[num_instances_attribute];
		    		  }
		            
		    		  for (j = 0; j < num_instances_attribute; j++) {
		    			  index_instance = sortedIndices[i][j];
		    			  instance = data.instance(index_instance);
		    			  missing = instance.isMissing(att);
		    				  
		    			  if(missing) {
		    				  for(int k = 0; k < num.length; k++) {	
		    					 if(m_Prop[k] > 0) {
		    						subsetIndices[k][i][num[k]] = sortedIndices[i][j];
		    						subsetWeights[k][i][num[k]] = m_Prop[k] * weights[i][j];
		    						num[k]++;
		    					 }
		    				  }
		    			  } 
			      
		    			  else {
		    				  int subset = (int)instance.value(att);
								subsetIndices[subset][i][num[subset]] = sortedIndices[i][j];
								subsetWeights[subset][i][num[subset]] = weights[i][j];
								num[subset]++;
		    				  }
		    			  }
		    		  } 
		    		  
		    		  else {
		    			  num = new int[2];
		    			  
		    			  for (int k = 0; k < 2; k++) {
		    				  subsetIndices[k][i] = new int[sortedIndices[i].length];
		    				  subsetWeights[k][i] = new double[weights[i].length];
		    			  }
		    			  
		    			  for (j = 0; j < sortedIndices[i].length; j++) {
		    				  index_instance = sortedIndices[i][j];
		    				  instance = data.instance(index_instance);
		    				  missing = instance.isMissing(att);
		    				  
		    				  if (missing){

		    					  for (int k = 0; k < num.length; k++) {
		    						  if (m_Prop[k] > 0) {
		    							  subsetIndices[k][i][num[k]] = sortedIndices[i][j];
		    							  subsetWeights[k][i][num[k]] = m_Prop[k] * weights[i][j];
		    							  num[k]++;
		    						  }
		    					  }
		    				  } 
		    				  
		    				  else {
		    					  int subset = (instance.value(att) < splitPoint) ? 0 : 1;
		    					  subsetIndices[subset][i][num[subset]] = sortedIndices[i][j];
		    					  subsetWeights[subset][i][num[subset]] = weights[i][j];
		    					  num[subset]++;
		    				  } 
		    			  }
		    		  }
			
			  // Trim arrays
					  for (int k = 0; k < num.length; k++) {
					    int[] copy = new int[num[k]];
					    System.arraycopy(subsetIndices[k][i], 0, copy, 0, num[k]);
					    subsetIndices[k][i] = copy;
					    double[] copyWeights = new double[num[k]];
					    System.arraycopy(subsetWeights[k][i], 0,copyWeights, 0, num[k]);
					    subsetWeights[k][i] = copyWeights;
					  }
		    	  }
		    
		 }
		 
		 protected double[][][] dispositionDist(double[][][] dist){
			 int num_values = dist.length;
			 double[][][] new_dist = new double[numLabels][num_values][];
			 
			 for(int i = 0; i < numLabels; i++){
				 for(int j = 0; j < num_values; j++){
					 new_dist[i][j] = dist[j][i];
				 }
			 }
			 
			 return new_dist;
		 }
		 
		 protected double priorVal(double[][][] dist){
			 double entropy = 0.0;
			 double[][][] new_dist = dispositionDist(dist);
			 double partial_entropy;
			 double[][] partial_dist;
			 
			 for(int i = 0; i < numLabels; i++) {
				 partial_dist = new_dist[i];
				 partial_entropy = ImpreciseEntropyContingencyTables.impreciseEntropyOverColumns(partial_dist, s_value);
				 entropy+= partial_entropy;
			 }
			 
			 return entropy;
		 }
		 
		 protected double gain(double[][][] dist, double priorVal) {
			 double info_gain;
			 double condicionated_entropy = 0.0;
			 double partial_entropy;
			 double[][][] new_dist = dispositionDist(dist);
			 double[][] partial_dist = new_dist[0];
			 
			 for(int i = 0; i < numLabels; i++) {
				 partial_dist = new_dist[i];
				 partial_entropy = ImpreciseEntropyContingencyTables.impreciseEntropyConditionedOnRows(partial_dist, s_value);
				 condicionated_entropy+= partial_entropy;
			 }
				 
			 info_gain = priorVal - condicionated_entropy;
			 
			 return info_gain;
		 }
		 
		 /*
		  protected void insertHoldOutSet(Instances data) throws Exception {
			  Instance instance;
			  double weight;

		      for (int i = 0; i < data.numInstances(); i++) {
		    	  instance = data.instance(i);
		    	  weight = instance.weight();
		    	  insertHoldOutInstance(instance, weight, this);
		      }
		   }
		  
		  protected void insertHoldOutInstance(Instance inst, double weight,  Tree parent) throws Exception {	
			m_HoldOutDist[(int)inst.classValue()] += weight;
			int predictedClass = 0;
			if (m_ClassProbs == null) {
			  predictedClass = Utils.maxIndex(parent.m_ClassProbs);
			} else {
			  predictedClass = Utils.maxIndex(m_ClassProbs);
			}
			if (predictedClass != (int)inst.classValue()) {
			  m_HoldOutError += weight;
			}
		   
		   
		   // The process is recursive
		   if (m_Attribute != -1) {
			
			// If node is not a leaf
			if (inst.isMissing(m_Attribute)) {
			  
			  // Distribute instance
			  for (int i = 0; i < m_Successors.length; i++) {
			    if (m_Prop[i] > 0) {
			      m_Successors[i].insertHoldOutInstance(inst, weight * 
								    m_Prop[i], this);
			    }
			  }
			} else {
			  
			  if (m_Info.attribute(m_Attribute).isNominal()) {
			    
			    // Treat nominal attributes
			    m_Successors[(int)inst.value(m_Attribute)].
			      insertHoldOutInstance(inst, weight, this);
			  } else {
			    
			    // Treat numeric attributes
			    if (inst.value(m_Attribute) < m_SplitPoint) {
			      m_Successors[0].insertHoldOutInstance(inst, weight, this);
			    } else {
			      m_Successors[1].insertHoldOutInstance(inst, weight, this);
			    }
			  }
			}
		   }
		 }
		  
		 protected double reducedErrorPrune() throws Exception {

		      // Is node leaf ? 
		      if (m_Attribute == -1) {
		    	  return m_HoldOutError;
		      }

		      // Prune all sub trees
		      double errorTree = 0;
		      
		      for (int i = 0; i < m_Successors.length; i++) {
		    	  errorTree += m_Successors[i].reducedErrorPrune();
		      }

		      // Replace sub tree with leaf if error doesn't get worse
		      if (errorTree >= m_HoldOutError) {
		    	  m_Attribute = -1;
		    	  m_Successors = null;
		    	  return m_HoldOutError;
		      } 
		      
		      else {
		    	  return errorTree;
		      }
		   }
	  */
	}
	
	protected Tree m_Tree = null;
	/** Number of folds for reduced error pruning. */
	protected int m_NumFolds = 3;
	    
	  /** Seed for random data shuffling. */
	protected int m_Seed = 1;
	    
	  /** Don't prune */
	protected boolean m_NoPruning = true;

	  /** The minimum number of instances per leaf. */
	protected int m_MinNum = 2;
	
	public void setSValue(double s) {
		s_value = s;
	}
	
	public double getSValue() {
		return s_value;
	}
	
	public boolean getNoPruning() {
		 return m_NoPruning;
	}
		  
	public void setNoPruning(boolean newNoPruning) {
		 m_NoPruning = newNoPruning;
	}
	 
	public int getMinNum() {	    
		return m_MinNum;
	}
		  
	public void setMinNum(int newMinNum) {
		      m_MinNum = newMinNum;
	}
	 
	public int getSeed() {
	    return m_Seed;
	}
	  
	public void setSeed(int newSeed) {
	    m_Seed = newSeed;
	} 

	public int getNumFolds() {
	   return m_NumFolds;
	}
	  
	public void setNumFolds(int newNumFolds) {
	   m_NumFolds = newNumFolds;
	}
	  
	public int numNodes() {
		return m_Tree.numNodes();
	}
	
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception{
		Instances data = trainingSet.getDataSet();
		int num_attributes = featureIndices.length;
		int feature_index;
		Instance instance;
		Random random = new Random(m_Seed);
		int num_instances = data.numInstances();
		
	    data.randomize(random);
	    	
	   // data.stratify(m_NumFolds);
	    
	//    MultiLabelInstances ML_data = trainingSet.reintegrateModifiedDataSet(data);
	    
	    //MultiLabelInstances train;
	    Instances aux_train;
	    
	  /*  if (!m_NoPruning) {
	        aux_train = data.trainCV(m_NumFolds, 0, random);
	        aux_prune = data.testCV(m_NumFolds, 0);
	        train = ML_data.reintegrateModifiedDataSet(aux_train);
	        prune = ML_data.reintegrateModifiedDataSet(aux_prune);
	    } 
	    
	    else{
	    */ 
//	        train = ML_data;
	        aux_train = data;
	   // }
	    
	    num_instances = aux_train.numInstances();
	    
	    int[][] sortedIndices = new int[num_attributes][0];
	    double[][] weights = new double[num_attributes][num_instances];
	    double[] vals = new double[num_instances];
	    int count;
	    boolean nominal, missing;
	    double weight, value;
	    
	    for(int j = 0; j < num_attributes; j++) {
	    	feature_index = featureIndices[j];
	    	nominal = aux_train.attribute(feature_index).isNominal();
	    	
	    	if(nominal){
	    		count = 0;
		    	sortedIndices[j] = new int[num_instances];
		    	
		    	for(int i = 0; i < num_instances; i++) {
		    		instance = aux_train.instance(i);
		    		missing = instance.isMissing(feature_index);
		    		
		    		if(!missing) {
		    			sortedIndices[j][count] = i;
		    			weight = instance.weight();
		    		    weights[j][count] = weight;
		    			count++;
		    		}
		    	}
		    	
		    	for(int i = 0; i < num_instances; i++) {
		    		instance = aux_train.instance(i);
		    		missing = instance.isMissing(feature_index);
		    		
		    		if(missing) {
		    			sortedIndices[j][count] = i;
		    			weight = instance.weight();
		    		    weights[j][count] = weight;
		    			count++;
		    		}
		    	}
	    	}
	    	
	    	else {
	    		for(int i = 0; i < num_instances; i++) {
	    			instance = aux_train.instance(i);
	    			value = instance.value(feature_index);
	    			vals[i] = value;
	    		}
	    		
	    		sortedIndices[j] = Utils.sort(vals);
	    		
	    		for (int i = 0; i < num_instances; i++) {
	    			instance = aux_train.instance(sortedIndices[j][i]);
	    			weight = instance.weight();
	    		    weights[j][i] = weight;
	    		}
	    	}
	    }
	    
	    double[][] classProbs = new double[numLabels][2];
	    double totalWeight = 0.0;
	    int label_index;
	    int label_value;
	    
	    for(int i = 0; i < num_instances; i++) {
	    	instance = aux_train.instance(i);
	    	weight = instance.weight();
	    	
	    	for(int l = 0; l < numLabels; l++) {
	    		label_index = labelIndices[l];
	    		label_value = (int)instance.value(label_index);
	    		classProbs[l][label_value]+=weight;
	    	}
	    	
	    	totalWeight+=weight;
	    }
	    
	    m_Tree = new Tree();
	    m_Tree.buildTree(sortedIndices, weights, aux_train, totalWeight, classProbs, m_MinNum);
	    /*
	    if (!m_NoPruning) {
	        m_Tree.insertHoldOutSet(prune);
	        m_Tree.reducedErrorPrune();
	        m_Tree.backfitHoldOutSet(prune);
	    }
	    */
	    
	}
	
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException{
		MultiLabelOutput output = m_Tree.makePredictionInternal(instance);;
		
		return output;
	}
	
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}
	
	public static void main(String[] args) throws InvalidDataException, Exception {
	
		Experimentation2 experimentation;
		int[] noise_levels = {0,5,10,20};
		String location = "C:/Proyecto/Datasets_MultiLabel";
		String location_arff = location + "/" + "Arff_Files";
		String location_xml = location + "/" + "XML_Files";
		String file_results = location + "/ML_RepTree";
		int num_folds = 5;
		int num_learners = 3;
		MultiLabelLearner[] learners = new MultiLabelLearner[num_learners];
		String[] names = new String[num_learners];
		int seed = 1;
		
		MultiLabelLearner ML_s0 = new ML_RepTree(0);
		MultiLabelLearner ML_s1 = new ML_RepTree(1.0);
		MultiLabelLearner ML_s2 = new ML_RepTree(2.0);
		
		learners[0] = ML_s0;
		learners[1] = ML_s1;
		learners[2] = ML_s2;
		
		names[0] = "S=0";
		names[1] = "S=1";
		names[2] = "S=2";
		
		experimentation = new Experimentation2(learners, num_folds, location_arff, location_xml, names, noise_levels, seed,file_results);
		 
		System.out.println("A punto de hacerlo");
		
		experimentation.computeResults();
	}

}
