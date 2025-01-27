package mulan.classifier.transformation;

import mulan.classifier.AttributeCorrelation;
import mulan.classifier.Experimentation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
// import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class ConditionalEntropy_ClassifierChain extends TransformationBasedMultiLabelLearner{
	/**
	 * For serialization
	 */
	private static final long serialVersionUID = 1L;

	/**
     * The chain ordering of the label indices
     */
    protected int[] chain;
    
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[] ensemble;
    
    /**
     * The conditional entropies for all the instances
     */
    
    protected double[][] conditional_entropies;
    
    /**
     * Creates a new instance using J48 as the underlying classifier
     */
    public ConditionalEntropy_ClassifierChain() {
        super(new J48());
    }


    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public ConditionalEntropy_ClassifierChain(Classifier classifier) {
        super(classifier);
    }
    
    /**
     * Calculate the conditioned entropy of each pair of labels
     * @param train the training instances
     */
    
    protected void computeConditionedEntropies(MultiLabelInstances train) {
    	Instances training_instances = train.getDataSet();
    	double conditioned_entropy;
    	int label_indexi, label_indexj;
    	
    	conditional_entropies = new double[numLabels][numLabels];
    	
    	for(int i = 0; i < numLabels; i++) {
    		label_indexi = labelIndices[i];
    		for(int j = 0; j < numLabels; j++) {
    			label_indexj = labelIndices[j];
    			conditioned_entropy = AttributeCorrelation.conditionedEntropy(training_instances, label_indexi, label_indexj);
    			conditional_entropies[i][j] = conditioned_entropy;
    		}
    	}
    }
    
    /**
     * Compute, for each i = 1,...,q \sum_{j \neq i}H(l_j|l_i)
     * @return the array with that sum of conditioned entropies 
     */
    
    protected double[] computeSumConditionedEntropies(){
    	double[] sums_conditioned_entropies = new double[numLabels];
    	double conditioned_entropy, sum_conditioned_entropies;
    	
    	for(int i = 0; i < numLabels; i++) {
    		sum_conditioned_entropies = 0;
    		
    		for(int j = 0; j < numLabels; j++) {
    			if(j != i) { 
    				conditioned_entropy = conditional_entropies[j][i];
    				sum_conditioned_entropies += conditioned_entropy;
    			}
    		}
    		
    		sums_conditioned_entropies[i] = sum_conditioned_entropies;
    		
    	}
    	
    	return sums_conditioned_entropies;
    }
    
    /**
     * Calculates the optimal order for the classifier chain 
     * In each step, the label l_i such that the sum of H(l_j|l_i) among the lj not belonging to the chain is minimum
     * @param train the training instances
     */
    
    protected void optimizeLaberingOrderChain(MultiLabelInstances train) {
    	double[] sums_conditioned_entropies;
    	double min_sum_entropies, sum_conditioned_entropies, conditioned_entropy;
    	int index_min_sum_entropies;
    	int temp;
    	
    	// Firstly, chain[i] = 1 \forall i = 1,...,q.
    	
    	chain = new int[numLabels];
    	
    	for(int i = 0; i < numLabels; i++) 
    		chain[i] = i;
    	
    	// Compute the conditional entropies
    	computeConditionedEntropies(train);
    	
    	// Calculate, for each i = 1,...,q, sum_{j \neq i} H(l_j|l_i)
    	sums_conditioned_entropies = computeSumConditionedEntropies();
    	
    	for(int i = numLabels - 1; i > 0; i--) {
    		/* Select the label which will be put in the position i of the chain (at the end)
    		Select the j = 1,..,i such that chain[j] = min_{1,..,i} sum_conditioned_entropies
    		*/
    		index_min_sum_entropies = 0;
    		min_sum_entropies = Double.MAX_VALUE;
    		
    		for(int j = 0; j <=i; j++) {
    			sum_conditioned_entropies = sums_conditioned_entropies[chain[j]];
    			
    			if(sum_conditioned_entropies < min_sum_entropies) {
    				index_min_sum_entropies = j;
    				min_sum_entropies = sum_conditioned_entropies;
    			}
    		}
    		
    		// change chain[i] by chain[index_min_sum_entropies]
    		
    		temp = chain[i];
    		chain[i] = chain[index_min_sum_entropies];
    		chain[index_min_sum_entropies] = temp;
    		
    		/* Update the sum of the conditioned entropies for the elements not yet inserted at the end of the chain
    		 * for j = 1,,,i-1, resting H(l_{chain(i)}|l_{chain(j)})
    		 */
    		
    		for(int j = 0; j < i; j++) {
    			conditioned_entropy = conditional_entropies[chain[i]][chain[j]];
    			sums_conditioned_entropies[chain[j]]-= conditioned_entropy;
    		}
    	}
    }
    
    /**
     * Builds the classifier chain
     * Select the order and built the binary classifier as CC
     * @param train the training instances
     */

    protected void buildInternal(MultiLabelInstances train) throws Exception {
        Instances trainDataset;
        numLabels = train.getNumLabels();
        ensemble = new FilteredClassifier[numLabels];
        trainDataset = train.getDataSet();
                
    	optimizeLaberingOrderChain(train);

        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

            // Indices of attributes to remove first removes numLabels attributes
            // the numLabels - 1 attributes and so on.
            // The loop starts from the last attribute.
            int[] indicesToRemove = new int[numLabels - 1 - i];
            int counter2 = 0;
            for (int counter1 = 0; counter1 < numLabels - i - 1; counter1++) {
                indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
                counter2++;
            }
            
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);
            ensemble[i].setFilter(remove);
            
            trainDataset.setClassIndex(labelIndices[chain[i]]);
            debug("Bulding model " + (i + 1) + "/" + numLabels);
            ensemble[i].buildClassifier(trainDataset);
        }
    }
    
    /**
     * Makes a multi-label prediction for an instance using the classifier chian
     * @param instance the instance
     * @return the multi-label prediction: boolean vector  + posterior probabilities
     */
    
     protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
         boolean[] bipartition = new boolean[numLabels];
         double[] confidences = new double[numLabels];

         Instance tempInstance = DataUtils.createInstance(instance, instance.weight(), instance.toDoubleArray());
        
         for (int counter = 0; counter < numLabels; counter++) {
             double distribution[];
             try {
                 distribution = ensemble[counter].distributionForInstance(tempInstance);
             } catch (Exception e) {
                 System.out.println(e);
                 return null;
             }
             int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

             // Ensure correct predictions both for class values {0,1} and {1,0}
             Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
             bipartition[chain[counter]] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

             // The confidence of the label being equal to 1
             confidences[chain[counter]] = distribution[classAttribute.indexOfValue("1")];
             
             // Put the prediction of the remaining classifiers
             tempInstance.setValue(labelIndices[chain[counter]], maxIndex);

         }

         MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
         return mlo;
     }
     
     public static void main(String[] args) throws InvalidDataException, Exception {
 		/*String location_xml, location_arff;
 		 MultiLabelLearner learner = new ConditionalEntropy_ClassifierChain();
 		 MultiLabelInstances dataset;
 		 location_xml = "C:/Proyecto/Datasets_MultiLabel/XML_Files/emotions.xml";
 		 location_arff = "C:/Proyecto/Datasets_MultiLabel/ARFF_Files/emotions.arff";
 		 
 		 dataset = new MultiLabelInstances(location_arff, location_xml);
 		 
 	//	 System.out.println("Building the model");
 		 
 		 learner.build(dataset);
 		 */

 		Experimentation experimentation;
     	int[] noise_levels = {0,10};
     	String location = "C:/Proyecto/MultiLabel_Ligeros";
     	int seed = 1;
     	int num_folds = 5;
     	String location_arff = location + "/" + "Arff_Files";
     	String location_xml = location + "/" + "XML_Files";
     	String folder_results = location + "/Prueba_ConditionalEntropy_CC";
     	SMO base_classifier = new SMO();
     	MultiLabelLearner classifier1 = new ClassifierChain(base_classifier);
     	MultiLabelLearner classifier2 = new LSF_ClassifierChain(base_classifier, 5);
     	MultiLabelLearner classifier3 = new ConditionalEntropy_ClassifierChain(base_classifier);
     	int num_classifiers = 3;
     	MultiLabelLearner[] classifiers = new MultiLabelLearner[num_classifiers];
     	String[] classifier_names = new String[num_classifiers];
     	
     	classifiers[0] = classifier1;
     	classifiers[1] = classifier2;
     	classifiers[2] = classifier3;
     	
     	classifier_names[0] = "ClassifierChain";
     	classifier_names[1] = "LSF_ClassifierChain";
     	classifier_names[2] = "ConditionalEntropy_ClassifierChain";
     	
        experimentation = new Experimentation(classifiers, num_folds, location_arff, location_xml, classifier_names, noise_levels, seed);

     	experimentation.computeResults();
     	experimentation.writeResults(folder_results);	
     	
 	}
    
}
