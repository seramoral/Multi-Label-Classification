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
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class Imprecise_ConditionalEntropy_ClassifierChain extends TransformationBasedMultiLabelLearner{
	
	/**
	 * 
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
     * The imprecise symmetrical uncertainty of each pair of labels
     */
    
    protected double[][] symmetrical_uncertainties;
    
	
	/**
	 * Creates a new instance 
	 * @param classifier the base classifier used in CC
	 */
	
	
	public Imprecise_ConditionalEntropy_ClassifierChain(Classifier classifier) {
		super(classifier);
	}
	
	/**
     * Calculate the Imprecise Symmetrical Uncertainty (ISU) of each pair of labels
     * @param train the training instances
     */

	protected void computeSymmetricalUncertainties(MultiLabelInstances train) {
    	Instances training_instances = train.getDataSet();
    	double symmetrical_uncertainty;
    	int label_indexi, label_indexj;
    	
    	symmetrical_uncertainties = new double[numLabels][numLabels];
    	
    	for(int i = 0; i < numLabels; i++) {
    		label_indexi = labelIndices[i];
    		
    		for(int j = 0; j <= i; j++) {
    			label_indexj = labelIndices[j];
    			symmetrical_uncertainty = AttributeCorrelation.impreciseSymmetricalUncertainty(training_instances, label_indexi, label_indexj);
    			symmetrical_uncertainties[i][j] = symmetrical_uncertainty;
    			symmetrical_uncertainties[j][i] = symmetrical_uncertainty;
    		}
    	}
    	
    }
	/**
	 * Compute, for each i = 1,..,q \sum_{j \neq i}I(l_i,l_j)
	 * @return the array of \sum_{j \neq i}ISU(l_i,l_j), for all i = 1,...,q
	 */
	
	protected double[] computeSumSymmetricalUncertainties() {
		double[] sums_symmetrical_uncertainties = new double[numLabels];
		double sum_symmetrical_uncertainties, symmetrical_uncertainty;
		
		for(int i = 0; i < numLabels; i++) {
			sum_symmetrical_uncertainties = 0;
			
			for(int j = 0; j < numLabels; j++) {
				if(j != i) {
					symmetrical_uncertainty = symmetrical_uncertainties[i][j];
					sum_symmetrical_uncertainties+=symmetrical_uncertainty;
				}
			}
			sums_symmetrical_uncertainties[i] = sum_symmetrical_uncertainties;
		}
		
		return sums_symmetrical_uncertainties;
	}
    
    /* Calculates the optimal order for the classifier chain 
    * In each step, the label l_i such that the average of I(l_i|l_j) among the l_j not belonging to the chain 
    * and the average of I(l_i|l_j) among the l_j already inserted is minimum
    * @param train the training instances
    */
   
    protected void optimizeLaberingOrderChain(MultiLabelInstances train) {
	    double[] sums_symmetrical_uncertainties_non_inserted, sums_symmetrical_uncertainties_inserted;
	    double average_symmetrical_uncertainties_inserted, average_symmetrical_uncertainties_non_inserted;
	    double sum_symmetrical_uncertainties_inserted, sum_symmetrical_uncertainties_non_inserted;
	    double symmetrical_uncertainty;
   		double score, max_score;
   		int index_max_score;
   		int temp;
   		int num_labels_non_inserted;
   	
   		// Firstly, chain[i] = 1 \forall i = 1,...,q.
   	
   		chain = new int[numLabels];
   	
   		for(int i = 0; i < numLabels; i++) 
   			chain[i] = i;
   	
   		// Compute the ISU
   		computeSymmetricalUncertainties(train);
   	
   		// Calculate, for each i = 1,...,q, sum_{j \neq i} ISU(l_i|,l_j)
   		sums_symmetrical_uncertainties_non_inserted =  computeSumSymmetricalUncertainties();
   		
   		/* At the beginning, sum of the symmetrical uncertainties
   		with the elements inserted yet is equal to */
   		
   		sums_symmetrical_uncertainties_inserted = new double[numLabels];
   	
   	// In the position i. 
   	
   	for(int i = 0; i < numLabels; i++) {
   		/* Select the label which will be put in the position i of the chain (at the beginning)
   		Select the j = i,..,q-1 such that chain[j] = min_{i,..,q-1} average ISU(l_k,l_j), 
   		with l_k non inserted, + average ISU(l_k,l_j), with l_k inserted, is minimum  
   		*/
   		index_max_score = -1;
   		max_score = Double.NEGATIVE_INFINITY;
   		// Candidate j
   		for(int j = i; j < numLabels; j++){
   			// Average among the already inserted
   			if(i == 0)
   				average_symmetrical_uncertainties_inserted = 0;
   			
   			else {
   				sum_symmetrical_uncertainties_inserted = sums_symmetrical_uncertainties_inserted[chain[j]];
   				average_symmetrical_uncertainties_inserted = sum_symmetrical_uncertainties_inserted/i;
   			}
   			
   			// Average among the non-inserted
   			
   			if(i == numLabels-1) 
   				average_symmetrical_uncertainties_non_inserted = 0;
   			
   			else {
   				sum_symmetrical_uncertainties_non_inserted = sums_symmetrical_uncertainties_non_inserted[chain[j]];
   				num_labels_non_inserted = numLabels-1-i;
   				average_symmetrical_uncertainties_non_inserted = sum_symmetrical_uncertainties_non_inserted/num_labels_non_inserted;
   			}
   				
   			
   			score = average_symmetrical_uncertainties_inserted + average_symmetrical_uncertainties_non_inserted;
   			// Update the minimum score, as well as the corresponding index
   			
   			if(score > max_score) {
   				index_max_score = j;
   				max_score = score;
   			}
   		}
   		
   		// change chain[i] by chain[index_max_score]
   		
   		temp = chain[i];
   		chain[i] = chain[index_max_score];
   		chain[index_max_score] = temp;
   		
   		/* for j = i+1,...q, resting ISU(l_{chain(j)},l_{chain(i)}) to the sum of ISU with non inserted
   		 * Sum ISU(l_{chain(j)},l_{chain(i)}) to the sum of ISU with inserted
    	*/
    		
    	for(int j = i+1; j < numLabels; j++) {
    		symmetrical_uncertainty = symmetrical_uncertainties[chain[j]][chain[i]];
    		sums_symmetrical_uncertainties_non_inserted[chain[j]]-= symmetrical_uncertainty;
    		sums_symmetrical_uncertainties_inserted[chain[j]]+= symmetrical_uncertainty;
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
     	int[] noise_levels = {0, 10};
     	String location = "C:/Proyecto/MultiLabel_Ligeros";
     	int seed = 1;
     	int num_folds = 5;
     	String location_arff = location + "/" + "Arff_Files";
     	String location_xml = location + "/" + "XML_Files";
     	String folder_results = location + "/Prueba_Imprecise_ConditionalEntropy_CC";
     	SMO base_classifier = new SMO();
     	MultiLabelLearner classifier1 = new BinaryRelevance(base_classifier);
     	MultiLabelLearner classifier2 = new ClassifierChain(base_classifier);
     	MultiLabelLearner classifier3 = new LSF_ClassifierChain(base_classifier, 5);
     	MultiLabelLearner classifier4 = new ConditionalEntropy2_ClassifierChain(base_classifier);
     	MultiLabelLearner classifier5 = new Imprecise_ConditionalEntropy_ClassifierChain(base_classifier);
     	int num_classifiers = 5;
     	MultiLabelLearner[] classifiers = new MultiLabelLearner[num_classifiers];
     	String[] classifier_names = new String[num_classifiers];
     	
     	classifiers[0] = classifier1;
     	classifiers[1] = classifier2;
     	classifiers[2] = classifier3;
     	classifiers[3] = classifier4;
     	classifiers[4] = classifier5;
     	
     	classifier_names[0] = "BinaryRelevance";
     	classifier_names[1] = "ClassifierChain";
     	classifier_names[2] = "LSF_ClassifierChain";
     	classifier_names[3] = "ConditionalEntropy2_ClassifierChain";
     	classifier_names[4] = "Imprecise_ConditionalEntropy_ClassifierChain";
     	
        experimentation = new Experimentation(classifiers, num_folds, location_arff, location_xml, classifier_names, noise_levels, seed);

     	experimentation.computeResults();
     	experimentation.writeResults(folder_results);	
     	
 	}
    
	

}
