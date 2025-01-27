package mulan.classifier.transformation;

import java.util.Random;

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
import weka.core.Utils;
import weka.filters.unsupervised.attribute.Remove;

public class Genetic_ClassifierChain extends TransformationBasedMultiLabelLearner{
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
     * The conditional entropies for all the instances
     */
    
    protected double[][] conditional_entropies;
    
    /**
     * The population size for the genetic procedure
     */

    protected int population_size;
    
    /**
     * The number of generations for the genetic procedure
     */
    
    protected int number_of_generations;
    
    /**
     * The cross rate for the genetic algorithm;
     */
    
    protected double cross_rate;
    
    /**
     * The mutation rate in the genetic procedure
     */
    
    protected double mutation_rate;
    
    /**
     * Creates a new instance
     * @param pop size the population size for the genetic algorithm
     * @param n_generations the number of generations
     * @param prob_cross the cross rate for the genetic procedure
     * @param prob_mut the mutation rate for the genetic algorithm
     * @param classifier the base-level classification algorithm that will be
     * used for training each of the binary models
     */
    public Genetic_ClassifierChain(int pop_size, int n_generations, double prob_cross, double prob_mut, Classifier classifier) {
    	super(classifier);
    	population_size = pop_size;
    	cross_rate = prob_cross;
    	mutation_rate = prob_mut;
    	number_of_generations = n_generations;
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
    			conditioned_entropy = AttributeCorrelation.conditionedMaxEntropy(training_instances, label_indexi, label_indexj);
    			conditional_entropies[i][j] = conditioned_entropy;
    		}
    	}
    }
    
    /**
     * Evaluates the fitness of a given permutation sigma[1],...,sigma[q]
     * Consider, the sum of H(sigma(j)|sigma(i)) for all i, j = 1,..., such that 
     * sigma(j) is places before in the chain than sigma(i)
     * @param candidate_solution the permutation
     * @return the fitness of the permutation
     */
  /*  
    protected double evaluateCandidateSolution(int[] candidate_solution){
    	double sum_conditioned_entropies = 0;
    	double conditioned_entropy;
    	
    	for(int i = numLabels - 1; i > 0; i--) {
    		// Calculate \sum_{j < i}H(sigma[j]|sigma[i])
    		for(int j = 0; j < i; j++) {
    			conditioned_entropy = conditional_entropies[candidate_solution[j]][candidate_solution[i]];
    			sum_conditioned_entropies+=conditioned_entropy;
    		}

    	}
    	
    	return sum_conditioned_entropies;
    }
   */ 
    
    
    /**
     * Evaluates the fitness of a given permutation sigma[1],...,sigma[q]
     * Consider, the sum of the average conditioned entropies
     * The average conditioned entropy, for a 1 = 1,...,q is the average of H(sigma[j]|sigma[i]) for j < i. 
     * @param candidate_solution the permutation
     * @return the fitness of the permutation
     */
    protected double evaluateCandidateSolution(int[] candidate_solution){
    	double sum_average_conditioned_entropies = 0;
    	double sum_conditioned_entropies, conditioned_entropy, average_conditioned_entropy;
    	
    	for(int i = numLabels - 1; i > 0; i--) {
    		sum_conditioned_entropies = 0;
    		// Calculate \sum_{j < i}H(sigma[j]|sigma[i])
    		for(int j = 0; j < i; j++) {
    			conditioned_entropy = conditional_entropies[candidate_solution[j]][candidate_solution[i]];
    			sum_conditioned_entropies+=conditioned_entropy;
    		}
    		
    		average_conditioned_entropy = sum_conditioned_entropies/i;
    		sum_average_conditioned_entropies+=average_conditioned_entropy;
    	}
    	
    	return sum_average_conditioned_entropies;
    }
    
    protected void optimizeLaberingOrderChain(MultiLabelInstances train) {
    	int[][] previous_population = new int[population_size][];
    	int[][] new_population;
    	int[] random_permutation;
    	double[] fitness_previous_population, fitness_new_population;
    	double fitness;
    	int[] donor, receptor;
    	int[] mutated_permutation;
    	int index_candidate1, index_candidate2;
    	int[] child1, child2;
    	double number_cross, number_mutation;
    	Random random = new Random(1);
    	int[] index_sort_previous_population, index_sort_new_population;
    	int index_best, index_second_best, index_worst, index_second_worst;
    	
    	// Compute the conditional entropies
    	computeConditionedEntropies(train);
    	
    	/*Generate the initial population
    	 * Evaluate each random permutation generated
    	*/
    	
    	fitness_previous_population = new double[population_size];
    	
    	for(int i = 0; i < population_size; i++) {
    		random_permutation = GeneticOperators.generateRandomSolution(numLabels);
    		fitness = evaluateCandidateSolution(random_permutation);
    		previous_population[i] = random_permutation;
    		fitness_previous_population[i] = fitness;
    	}
    	
    	for(int gen = 1; gen < number_of_generations; gen++) {
    		// Generate new population
    		
    		new_population = new int[population_size][];
    		for(int i = 0; i < population_size/2; i++) {
    			/* Play a tournament for the donor
    			Random_selection of two permutations
    			The one with the better fitness is selected
    			Make sure that we select different candidates
    			*/
    			do{
    				index_candidate1 = random.nextInt(population_size);
    				index_candidate2 = random.nextInt(population_size);
    			}while(index_candidate1 == index_candidate2);
    		
    			if(fitness_previous_population[index_candidate1] < fitness_previous_population[index_candidate2])
    				donor = previous_population[index_candidate1];
    		
    			else
    				donor = previous_population[index_candidate2];
    		
    			//The same procedure for the receptor
    			do {
    				index_candidate1 = random.nextInt(population_size);
    				index_candidate2 = random.nextInt(population_size);
    			}while(index_candidate1 == index_candidate2);
    		
    			if(fitness_previous_population[index_candidate1] < fitness_previous_population[index_candidate2])
    				receptor = previous_population[index_candidate1];
    		
    			else
    				receptor = previous_population[index_candidate2];
    			
    			// decide if there is cross
    			number_cross = random.nextDouble();
    		
    			if(number_cross < cross_rate) {
    				child1 = GeneticOperators.crossOperator(donor, receptor);
    				// Change the roles of donor and receptor to generate the second child
    				child2 = GeneticOperators.crossOperator(receptor, donor);
    			}
    			
    			else {
    				child1 = donor;
    				child2 = receptor;
    			}
    			
    			// Mutate the child1 with a certain probability
    			number_mutation = random.nextDouble();
    			
    			if(number_mutation < mutation_rate) {
    				mutated_permutation = GeneticOperators.mutatePermutation(child1);
    				new_population[2*i] = mutated_permutation;
    			}
    			
    			else
    				new_population[2*i] = child1;
    			
    			// The same with child2
    			number_mutation = random.nextDouble();
    			
    			if(number_mutation < mutation_rate) {
    				mutated_permutation = GeneticOperators.mutatePermutation(child2);
    				new_population[2*i+1] = mutated_permutation;
    			}
    			
    			else
    				new_population[2*i+1] = child2;	
    		}
    		
    		// Evaluate the new population
    		
    		fitness_new_population = new double[population_size];
    		
    		for(int i = 0; i < population_size; i++) {
    			fitness = evaluateCandidateSolution(new_population[i]);
    			fitness_new_population[i] = fitness;
    		}
    		
    		/* Replace the two worst individual of the new population 
    		with the two best ones of the previous population
    		*/
    		
    		index_sort_new_population = Utils.sort(fitness_new_population);
    		index_sort_previous_population = Utils.sort(fitness_previous_population);		
        	index_best = index_sort_previous_population[0];
        	index_second_best = index_sort_previous_population[1];
        	index_worst = index_sort_new_population[population_size - 1];
        	index_second_worst = index_sort_new_population[population_size - 2];
        	new_population[index_worst] = previous_population[index_best];
        	fitness_new_population[index_worst] = fitness_previous_population[index_best];
        	new_population[index_second_worst] = previous_population[index_second_best];
        	fitness_new_population[index_second_worst] = fitness_previous_population[index_second_best];
        	
        	// Upate the previous population, as well as the fitness
        	
        	for(int i = 0; i < population_size; i++) {
        		previous_population[i] = new_population[i];
        		fitness_previous_population[i] = fitness_new_population[i];
        	}
    	}
    	
		index_sort_previous_population = Utils.sort(fitness_previous_population);
    	index_best = index_sort_previous_population[population_size - 1];
		
    	chain = previous_population[index_best];
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
     
     /**
      * Sets the population size
      * @param pop_size the population size
      */
    
    public void setPopulationSize(int pop_size) {
    	population_size = pop_size;
    }
    
    /**
     * Gets the population size
     * @return the population size
     */
    
    public int getPopulationSize() {
    	return population_size;
    }
    
    /**
     * Sets the number of generations
     * @param n_generations the number of generations
     */
    
    public void setNumberGenerations(int n_generations) {
    	number_of_generations = n_generations;
    }
    
    /**
     * Gets the number of generations
     * @return the number of generations
     */
    
    public int getNumberGenerations() {
    	return number_of_generations;
    }
    
    /**
     * Sets the cross rate
     * @param prob_cross the cross rate
     */
    
    public void setCrossRate(double prob_cross) {
    	cross_rate = prob_cross;
    }
    /**
     * Gets the cross rate
     * @return the cross rate
     */
    
    public double getCrossRate() {
    	return cross_rate;
    }
    
    /**
     * Sets the mutation rate
     * @param prob_mutation the mutation rate
     */
    
    public void setMutationRate(double prob_mutation) {
    	mutation_rate = prob_mutation;
    }
    /**
     * Gets the mutation rate
     * @return the mutation rate
     */
    
    public double getMutationRate() {
    	return mutation_rate;
    }
    
    public static void main(String[] args) throws InvalidDataException, Exception {
    	/*String location_xml, location_arff;
     	SMO base_classifier = new SMO();
     	int population_size = 10;
     	int num_generations = 6;
     	double prob_cross = 1.0;
     	double prob_mutation = 0.25;
    	MultiLabelLearner learner = new Genetic_ClassifierChain(population_size, num_generations, prob_cross, prob_mutation, base_classifier);
		MultiLabelInstances dataset;
		location_xml = "C:/Proyecto/Datasets_MultiLabel/XML_Files/emotions.xml";
		location_arff = "C:/Proyecto/Datasets_MultiLabel/ARFF_Files/emotions.arff";
		 
		dataset = new MultiLabelInstances(location_arff, location_xml);
		
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
     	int pop_size = 1000;
     	int num_generations = 100;
     	double prob_cross = 1.0;
     	double prob_mutation = 1.0;
     	MultiLabelLearner classifier1 = new ConditionalEntropy_ClassifierChain(base_classifier);
     	MultiLabelLearner classifier2 = new Genetic_ClassifierChain(pop_size, num_generations, prob_cross, prob_mutation, base_classifier);
     	int num_classifiers = 2;
     	MultiLabelLearner[] classifiers = new MultiLabelLearner[num_classifiers];
     	String[] classifier_names = new String[num_classifiers];
     	
     	classifiers[0] = classifier1;
     	classifiers[1] = classifier2;
     	
     	classifier_names[0] = "ConditionalEntropy_ClassifierChain";
     	classifier_names[1] = "Genetic_ClassifierChain";
     	
        experimentation = new Experimentation(classifiers, num_folds, location_arff, location_xml, classifier_names, noise_levels, seed);

     	experimentation.computeResults();
     	experimentation.writeResults(folder_results);
	}
}
