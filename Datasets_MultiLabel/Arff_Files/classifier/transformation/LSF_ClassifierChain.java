package mulan.classifier.transformation;

import mulan.classifier.Experimentation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.Remove;

public class LSF_ClassifierChain extends TransformationBasedMultiLabelLearner{
	 /**
	 * For serialization
	 */
	private static final long serialVersionUID = 1L;

	/**
     * The chain ordering of the label indices
     */
    private int[] chain;
    
    /**
     * The ensemble of binary relevance models. These are Weka
     * FilteredClassifier objects, where the filter corresponds to removing all
     * label apart from the one that serves as a target for the corresponding
     * model.
     */
    protected FilteredClassifier[] ensemble;
    
    /**
     * The labels vector for the training instances
     */
    
    protected int[][] labels_vectors;
    
    /**
     * The number of neighbors considered in the RelieF methods
     */
    int num_neighbors;
    
    /**
     * The number of labels that are supposed to be correlated with another one 
     */
    protected int number_labels_correlated;
    
    /**
     * The labels correlated with each label. 
     */
    protected int[][] correlated_labels;
    
    /**
     * Creates a new instance using J48 as the underlying classifier
     * @param p the number of labels correlated with each label
     * @param k the number of neighbors in the RelieF methods

     */
    
    public LSF_ClassifierChain(int p, int k) {
        super(new J48());
    	number_labels_correlated = p;
    	num_neighbors = k;
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param aChain contains the order of the label indexes [0..numLabels-1] 
     * @param p the number of labels correlated with each label
     * @param k the number of neighbors in the RelieF methods

     */
    public LSF_ClassifierChain(Classifier classifier, int[] aChain, int p, int k) {
        super(classifier);
        chain = aChain;
        number_labels_correlated = p;
        num_neighbors = k;
    }

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param p the number of labels correlated with each label
     * @param k the number of neighbors in the RelieF methods
     */
    public LSF_ClassifierChain(Classifier classifier, int p, int k) {
        super(classifier);
        number_labels_correlated = p;
        num_neighbors = k;
    }
    
    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     * @param k the number of neighbors in the RelieF methods
     */
    public LSF_ClassifierChain(Classifier classifier, int k) {
        super(classifier);
        num_neighbors = k;
    }
    
    /**
     * Compute the label vectors for the training instances
     * the ith component will be 1 if the i-th label is relevant for the instance and 0 otherwise
     * @param train the training instances
     */
    
    private void computeLabelVectors(MultiLabelInstances train) {
    	int num_instances = train.getNumInstances();
    	Instances instances = train.getDataSet();
    	Instance instance;
    	int label_index;
    	int label_value;
    	
    	labels_vectors = new int[num_instances][numLabels];
    	
    	for(int i = 0; i < num_instances; i++){
    		instance = instances.get(i);
    		
    		for(int j = 0; j < numLabels; j++) {
    			label_index = labelIndices[j];
    			label_value = (int) instance.value(label_index);
    			labels_vectors[i][j] = label_value;
    		}
    	}
    	
    }
    
    /**
     * Calculates the distance between two labels vector excluding 
     * @param label the label that does not influence the distance
     * @param label_vector1 the vector of the labels for the first instance
     * @param label_vector2 the vector of the labels for the first instance
     * @return the distance between the two vectors
     */
 
    private double distanceLabelVectors(int label, int[] label_vector1, int[] label_vector2){
    	double distance = 0.0;
    	
    	for(int i = 0; i < numLabels; i++) {
    		if(i !=label) {
    			if(label_vector1[i] != label_vector2[i])
    				distance+=1.0;
    		}
    	}
    	
    	return distance;
    }
    /**
     * Computes the difference in a label between two label vectors (diff)
     * @param label the label
     * @param label_vector1 the first label vector
     * @param label_vector2 the second label vector
     * @return 0 if both vectors coindice in the label, 1 otherwise
     */
    
    private double differenceLabel(int label, int[] label_vector1, int[] label_vector2) {
    	if(label_vector1[label] == label_vector2[label])
    		return 0;
    	
    	else
    		return 1;
    }
    
    /**
     *  Computes the optimal labels order for the chain
     * @param train the training instances
   	*/
    
    private void optimizeLaberingOrderChain(MultiLabelInstances train) {
    	int num_instances = train.getNumInstances();
    	int[] ordered_labels;
    	
    	/* 3-dimensional array with the distances of the labels vectors excluding a label
    	 * Component[i][j][k] = distance among the label vectors j k excluding label i. 
    	 */
    	double[][][] distances_labels_vectors = new double[numLabels][num_instances][num_instances];
    	int[] label_vector1, label_vector2;
    	
    	// The number of times that each label appears in 
    	double[] num_subsets_label = new double[numLabels];
    	
    	// The labels ordered by the number of times that appear in the label sets 
    	int[] labels_ordered;
    	
    	// the candidates for the order
    	int candidate1, candidate2;
    	int correlated_label;
    	boolean already_inserted;
    	boolean candidates_found;
    	int label_candidate;
    	
    	int[] correlated_candidate;
    	boolean empty_intersection;
    	
    	double difference_distance;
    	int[] label_ordered_distances;
    	int neighbors_same_label, neighbors_different_label;
    	int[] ordered_label_vector;
    	int order_label_index;
    	double total_instances = num_instances*num_neighbors;
    	
    	/*
    	 * Value estimated the correlation between each pair of labels 
    	 */
    	
    	double[][] estimated_correlations = new double[numLabels][numLabels];
    	
    	// Compute the label vectors for each instance
    	computeLabelVectors(train);
    	
    	// Calculate distance between label vectors
    	
    	for(int i = 0; i < numLabels; i++) {
    		for(int j = 0; j < num_instances; j++) {
    			label_vector1 = labels_vectors[j];
    			distances_labels_vectors[i][j][j] = 0;
    			
    			for(int k = 0; k < j; k++) {
    				label_vector2 = labels_vectors[k];
    				distances_labels_vectors[i][j][k] = distanceLabelVectors(i, label_vector1, label_vector2);
    				distances_labels_vectors[i][k][j] = distances_labels_vectors[i][j][k];
    			}
    		}
    	}
    	
    	/* Computing wle: for each label and each instances: Ordered the neighbors by distance
    	 *  Find the k nearest neighbors that coincide in label j and the k ones that not coincide
    	 *  in label j.
    	 */
    	
    	for(int j = 0; j < numLabels; j++) {
    		for(int i = 0; i < num_instances; i++) {
    			label_vector1 = labels_vectors[i];
    			label_ordered_distances = Utils.sort(distances_labels_vectors[j][i]);
    			
    			for(int l = 0; l < numLabels; l++) {
    				neighbors_same_label = 0;
        			neighbors_different_label = 0;
        			int k = 0;
        			
        			while((neighbors_same_label < num_neighbors || neighbors_different_label < num_neighbors) && k < num_instances) {
        				order_label_index = label_ordered_distances[k];
        				ordered_label_vector = labels_vectors[order_label_index];
        				difference_distance = differenceLabel(l, label_vector1, ordered_label_vector);
        				
        				if(label_vector1[j] == ordered_label_vector[j] && neighbors_same_label < num_neighbors){
        					estimated_correlations[j][l] -= difference_distance/total_instances;
        					neighbors_same_label++;
        				}
        				
        				if(label_vector1[j] != ordered_label_vector[j] && neighbors_different_label < num_neighbors){
        					estimated_correlations[j][l] += difference_distance/total_instances;
        					neighbors_different_label++;
        				}
        				
        				k++;
        			}
    			}
    		}
			estimated_correlations[j][j] = Double.MAX_VALUE;
    	}
    	
    	number_labels_correlated = numLabels-2;
    	
    	correlated_labels = new int[numLabels][number_labels_correlated - 1];
    	
    	/*
    	 * Order the labels for each label according to the correlation
    	 * Select the p first whenever a label does not coincide with itself
    	 */
    	
    	
    	for(int i = 0; i < numLabels; i++){
    		ordered_labels = Utils.sort(estimated_correlations[i]);
    		
    		for(int j = 0; j < number_labels_correlated - 1; j++) 
    			correlated_labels[i][j] = ordered_labels[numLabels - 2 - j]; 		
    	}
    	
    	/*
    	 * Compute the number of times that each label appears
    	 * in the set of correlated labels of another one
    	 */
    	
    	for(int i = 0; i < numLabels; i++) {
    		for(int j = 0; j < number_labels_correlated - 1; j++) {
    			correlated_label = correlated_labels[i][j];
    			num_subsets_label[correlated_label] +=1;
    		}
    	}
    	
    	labels_ordered = Utils.sort(num_subsets_label);
    	
    	chain = new int[numLabels];
    	
    	// The label that more frequently appears at the beginning of the chain
    	chain[0] = labels_ordered[numLabels - 1];
    	
    	for(int i = 1; i < numLabels; i++){
    		candidate1 = -1;
    		candidate2 = -1;
    		candidates_found = false;
    		int j = 0;
    		
    		while(j < numLabels && !candidates_found) {
    			label_candidate = labels_ordered[numLabels-1-j];
    			already_inserted = false;
    			int k = 0;
    			
    			while(k < i && !already_inserted) {
    				if(chain[k] == label_candidate)
    					already_inserted = true;
    				
    				else
    					k++;
    			}
    			
    			if(!already_inserted) {
    				if(candidate1 == -1)
    					candidate1 = label_candidate;
    				
    				empty_intersection = true;
    				k = 0;
    				correlated_candidate = correlated_labels[label_candidate];
    				
    				// Check if, for some l,k chain[k] == correlated_candidate[l]
    				
    				while(k < i && empty_intersection) {
        				int l = 0;
        				
        				while(l < correlated_candidate.length && empty_intersection) {
        					if(chain[k] == correlated_candidate[l]) {
        						empty_intersection = false;
        						candidate2 = label_candidate;
        						candidates_found = true;
        					}
        					
        					else
        						l++;
        				}
        				
        				k++;
    				}
    			}
    			j++;
    		}
    		
    		if(candidate2 == -1)
    			chain[i] = candidate2;
    		
    		else
    			chain[i] = candidate1;
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
        boolean label_correlated;
        int num_attributes_remove;
        
        number_labels_correlated = numLabels/2 + 1;
        
    	optimizeLaberingOrderChain(train);

        for (int i = 0; i < numLabels; i++) {
            ensemble[i] = new FilteredClassifier();
            ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));
            
            int num_labels_no_correlated = 0;
            int k;
            
            for(int j = 0; j < i; j++) {
            	label_correlated = false;
            	k = 0;
            	
            	while( k < number_labels_correlated - 1 && !label_correlated) {
            		if(correlated_labels[chain[i]][k] == chain[j])
            			label_correlated = true;
            			
            		else
            			k++;
            	}
            	
            	if(!label_correlated)
            		num_labels_no_correlated++;
            }

            // Indices of attributes to remove first removes numLabels attributes
            // the numLabels - 1 attributes and so on.
            // The loop starts from the last attribute.
            num_attributes_remove = numLabels - 1 - i + num_labels_no_correlated;
            int[] indicesToRemove = new int[num_attributes_remove];
            int counter1 = 0;
            
            for (int counter2 = 0; counter2 < numLabels - i - 1; counter2++) {
                indicesToRemove[counter1] = labelIndices[chain[numLabels - 1 - counter2]];
                counter1++;
            }
            
            counter1 = numLabels-i-1;
            
            for(int j = 0; j < i; j++) {
            	label_correlated = false;
            	k = 0;
            	
            	while(k < number_labels_correlated - 1 && !label_correlated) {
            		if(correlated_labels[chain[i]][k] == chain[j])
            			label_correlated = true;
            			
            		else
            			k++;
            	}
            	
            	if(!label_correlated) {
            		indicesToRemove[counter1] = labelIndices[chain[j]];
            		counter1++;
            	}
            }
            
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(indicesToRemove);
            remove.setInputFormat(trainDataset);
            remove.setInvertSelection(false);
            ensemble[i].setFilter(remove);
            
            /** Remove indices of the labels no correlated
            	For each one of the predecesor labels, check whether 
             * 	it is in the list of the labels correlated with that label
             */

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
		 MultiLabelLearner learner = new LSF_ClassifierChain(5, 5);
		 MultiLabelInstances dataset;
		 location_xml = "C:/Proyecto/Datasets_MultiLabel/XML_Files/emotions.xml";
		 location_arff = "C:/Proyecto/Datasets_MultiLabel/ARFF_Files/emotions.arff";
		 
		 dataset = new MultiLabelInstances(location_arff, location_xml);
		 Instances data = dataset.getDataSet();
		 int num_instances = data.numInstances();
		// Evaluator eval = new Evaluator();
		 // MultiLabelOutput outputs[] = new MultiLabelOutput[num_instances];
		 MultiLabelOutput output;
		 
	//	 System.out.println("Building the model");
		 
		 learner.build(dataset);
		 
*/
		Experimentation experimentation;
    	int[] noise_levels = {0};
    	String location = "C:/Proyecto/MultiLabel_Ligeros";
    	int seed = 1;
    	int num_folds = 5;
    	int num_neighbors = 5;
    	String location_arff = location + "/" + "Arff_Files";
    	String location_xml = location + "/" + "XML_Files";
    	String folder_results = location + "/PruebaLFS_CC";
    	Classifier base_classifier = new SMO();
    	MultiLabelLearner classifier1 = new ClassifierChain(base_classifier);
    	MultiLabelLearner classifier2 = new LSF_ClassifierChain(base_classifier, num_neighbors);
    	int num_classifiers = 2;
    	MultiLabelLearner[] classifiers = new MultiLabelLearner[num_classifiers];
    	String[] classifier_names = new String[num_classifiers];
    	
    	classifiers[0] = classifier1;
    	classifiers[1] = classifier2;
    	
    	classifier_names[0] = "ClassifierChain";
    	classifier_names[1] = "LSF_ClassifierChain";
    	
       	experimentation = new Experimentation(classifiers, num_folds, location_arff, location_xml, classifier_names, noise_levels, seed);

    	experimentation.computeResults();
    	experimentation.writeResults(folder_results);

    	
	}
    

}
