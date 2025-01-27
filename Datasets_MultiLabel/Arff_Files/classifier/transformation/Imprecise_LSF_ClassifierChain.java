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
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.unsupervised.attribute.Remove;

public class Imprecise_LSF_ClassifierChain extends TransformationBasedMultiLabelLearner{
	 /**
		 * For serialization
		 */
		private static final long serialVersionUID = 1L;
		
		// The matrix of mutual informations 
		
		protected double[][] mutual_informations;
		
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
	    
	    public Imprecise_LSF_ClassifierChain(int p) {
	        super(new J48());
	    	number_labels_correlated = p;
	    }

	    /**
	     * Creates a new instance
	     *
	     * @param classifier the base-level classification algorithm that will be
	     * used for training each of the binary models
	     * @param aChain contains the order of the label indexes [0..numLabels-1] 
	     * @param p the number of labels correlated with each label

	     */
	    public Imprecise_LSF_ClassifierChain(Classifier classifier, int[] aChain, int p) {
	        super(classifier);
	    	chain = aChain;
	        number_labels_correlated = p;
	    }
	    
	    /**
	     * Creates a new instance
	     *
	     * @param classifier the base-level classification algorithm that will be
	     * used for training each of the binary models
	     * @param k the number of neighbors in the RelieF methods
	     */
	    public Imprecise_LSF_ClassifierChain(Classifier classifier) {
	        super(classifier);
	    }
	    /**
	     * Computes the mutual informations of each pair of labels
	     * @param train the training instances
	     */
	    
	    private void ComputeMutualInformations(MultiLabelInstances train) {
	    	Instances instances = train.getDataSet();
	    	double mutual_information;
	    	int label_index_i, label_index_j;
	    	
	    	mutual_informations = new double[numLabels][numLabels];
	    	
	    	for(int i = 0; i < numLabels; i++) {
	    		label_index_i = labelIndices[i];
	    		mutual_informations[i][i] = Double.MAX_VALUE;
	    		
	    		for(int j = 0; j < i; j++) {
	    			label_index_j = labelIndices[j];
	    			mutual_information = AttributeCorrelation.impreciseMutualInformation(instances, label_index_i, label_index_j);
	    			mutual_informations[i][j] = mutual_information;
	    			mutual_informations[j][i] = mutual_information;
	    		}
	    	}
	    }
	    
	    /**
	     *  Computes the optimal labels order for the chain
	     * @param train the training instances
	   	*/
	    
	    private void optimizeLaberingOrderChain(MultiLabelInstances train){
	    	int[] ordered_labels;
	    	
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
	    	
	    	int order_label_index;
	    	double score, max_score;
	    	int index_max_score;
	    	double relevance, sum_redundancy, average_redundancy, redundancy;
	    	
	    	ComputeMutualInformations(train);
	    	
	    	number_labels_correlated = numLabels/2;
	    	
	    	correlated_labels = new int[numLabels][number_labels_correlated - 1];
	    	
	    	/** For each label, the p with highest mutual information, are correlated with it
	    	 * The label itself is not taken into account
	    	 */
	    	
	    	for(int i = 0; i < numLabels; i++) {
	    		// At the beginning, the label with highest label correlation
	    		ordered_labels = Utils.sort(mutual_informations[i]);
	    		order_label_index = ordered_labels[numLabels -2];
	    		correlated_labels[i][0] = order_label_index;
	    		
	    		// Insert in position j
	    		for(int j = 1; j < number_labels_correlated - 1; j++) {
	    			max_score = Double.NEGATIVE_INFINITY;
	    			index_max_score = -1;
	    			
	    			// candidate k
	    			
	    			for(int k = 0; k < numLabels; k++) {
	    				if(k!=i) {
	    					already_inserted = false;
	    					int l = 0;
	    				
	    				/* check if the label k has already been inserted in the list
	    				of labels correlated with i */
	    				
	    					while(l < j && !already_inserted) {
	    						if(correlated_labels[i][l] == k)
	    							already_inserted = true;
	    					
	    						else
	    							l++;
	    					}
	    				// If does not in the list yet I will consider it
	    					if(!already_inserted) {
	    						relevance = mutual_informations[i][k];
	    						sum_redundancy = 0;
	    					// Average of the redundancy with the labels already inserted. 
	    						for(int m = 0; m < j; m++) {
	    							order_label_index = correlated_labels[i][m];
	    							redundancy = mutual_informations[order_label_index][k];
	    							sum_redundancy+=redundancy;
	    						}
	    					
	    						average_redundancy = sum_redundancy/j;
	    						score = relevance - average_redundancy;
	    					
	    						if(score > max_score) {
	    							max_score = score;
	    							index_max_score = k;
	    						}
	    					}
	    				}
	    			}
	    			
	    			correlated_labels[i][j] = index_max_score;
	    		}
	    	}
	    	
	    	/*
	    	 * Compute the number of times that each label appears
	    	 * in the number of correlated labels of another one
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
	 		 MultiLabelLearner learner = new ConditionalEntropy_ClassifierChain();
	 		 MultiLabelInstances dataset;
	 		 location_xml = "C:/Proyecto/Datasets_MultiLabel/XML_Files/emotions.xml";
	 		 location_arff = "C:/Proyecto/Datasets_MultiLabel/ARFF_Files/emotions.arff";
	 		 
	 		 dataset = new MultiLabelInstances(location_arff, location_xml);
	 		 
	 	//	 System.out.println("Building the model");
	 		 
	 		 learner.build(dataset);
	 		 */

	 		Experimentation experimentation;
	     	int[] noise_levels = {0};
	     	String location = "C:/Proyecto/MultiLabel_Ligeros";
	     	int seed = 1;
	     	int num_folds = 5;
	     	String location_arff = location + "/" + "Arff_Files";
	     	String location_xml = location + "/" + "XML_Files";
	     	String folder_results = location + "/Prueba_ConditionalEntropy_CC";
	     	SMO base_classifier = new SMO();
	     	MultiLabelLearner classifier1 = new ClassifierChain(base_classifier);
	     	MultiLabelLearner classifier2 = new LSF_ClassifierChain(base_classifier, 5);
	     	MultiLabelLearner classifier3 = new Imprecise_LSF_ClassifierChain(base_classifier);
	     	MultiLabelLearner classifier4 = new ConditionalEntropy_ClassifierChain(base_classifier);
	     	int num_classifiers = 4;
	     	MultiLabelLearner[] classifiers = new MultiLabelLearner[num_classifiers];
	     	String[] classifier_names = new String[num_classifiers];
	     	
	     	classifiers[0] = classifier1;
	     	classifiers[1] = classifier2;
	     	classifiers[2] = classifier3;
	     	classifiers[3] = classifier4;
	     	
	     	classifier_names[0] = "ClassifierChain";
	     	classifier_names[1] = "LSF_ClassifierChain";
	     	classifier_names[2] = "Imprecise_LSF_ClassifierChain";
	     	classifier_names[3] = "ConditionalEntropy_ClassifierChain";
	     	
	        experimentation = new Experimentation(classifiers, num_folds, location_arff, location_xml, classifier_names, noise_levels, seed);

	     	experimentation.computeResults();
	     	experimentation.writeResults(folder_results);	
	     	
	 	}
	    

}
