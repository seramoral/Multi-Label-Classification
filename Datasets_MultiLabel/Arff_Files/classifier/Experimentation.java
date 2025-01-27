package mulan.classifier;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mulan.classifier.lazy.*;
import mulan.classifier.lazy.BRkNN.ExtensionType;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.ConditionalEntropy2_ClassifierChain;
import mulan.classifier.transformation.Imprecise_ConditionalEntropy_ClassifierChain;
import mulan.classifier.transformation.LSF_ClassifierChain;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import mulan.data.NoiseFilter;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

public class Experimentation{
/** Number of folds considered in cross-validation**/
	
	protected int num_folds;
	
	
	/** Multi-label classifiers used in the experimentation **/
	
	private MultiLabelLearner[] classifiers;
	
	/** levels of noise considered in the experimentation **/
	
	protected int[] noise_levels; 
	
	/** seed for reproduction of cross-validation results **/
	
	private int seed;
	
	/** Names of the multi-label classifiers used in the experimentation **/
	
	private String[] classifiers_names;
	
/** Datasets considered in the experimentation. One array per level of noise **/
	
	private MultiLabelInstances[][] all_datasets;
	
	
/** Array 4-dimensional with the results of our experimentation **/
	
	private double[][][][] results;
	
	/**
	 * Constructs a new object
	 * @param learners Multi-label classifiers to use
	 * @param n_folds Number of folds for cross validation
	 * @param f_arff Path of the arff files of the original datasets
	 * @param f_xml Path of the arff files of the original datasets
	 * @param cl_names Names of the multi-label classifiers
	 * @param nois_lev Array with the noise levels considered in the experimentation
	 * @param s seed for the reproduction of cross validation results
	 * @throws Exception
	 */
	
	public Experimentation(MultiLabelLearner[] learners, int n_folds, String f_arff, String f_xml, String[] cl_names, int[] nois_lev, int s) throws Exception {
		setClassifiers(learners); 
		setClassifiersNames(cl_names);
		setNoiseLevels(nois_lev);
		
		loadAllDatasets(f_arff, f_xml);
		
		num_folds = n_folds;
		seed = s;
	}
	
	public Experimentation() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * Sets the multi-label classifiers
	 * @param learners The multi-label classifiers
	 * @throws Exception
	 */
	
	public void setClassifiers(MultiLabelLearner[] learners) throws Exception{
		int num_learners = learners.length;
		classifiers = new MultiLabelLearner[num_learners];
		MultiLabelLearner learner, learner2;
		
		for(int i = 0; i < num_learners; i++) {
			learner = learners[i];
			learner2 = learner.makeCopy();
			classifiers[i] = learner2;	
		}
	}
	
	/**
	 * @return multi-label classifiers
	 */
	
	public MultiLabelLearner[] getClassifiers() {
		return classifiers;
	}
	
	/**
	 * Sets the number of folds for cross validation
	 * @param n_folds the number of folds
	 */
	
	public void setNumFolds(int n_folds) {
		num_folds = n_folds;
	}
	/**
	 * @return the number of folds in cross validation
	 */
	
	public int getNumFolds() {
		return num_folds;
	}
	
	/**
	 * Sets the names of the multi-label classifiers
	 * @param names names of the multi-label classifiers
	 */
	
	public void setClassifiersNames(String [] names) {
		int num_learners = names.length;		
		
		classifiers_names = new String[num_learners];
		
		for(int i = 0; i < num_learners; i++) 
			classifiers_names[i] = names[i];
		
	}
	
	/**
	 * 
	 * @return names of the multi-label classifies
	 */
	
	public String[] getClassifiersNames() {
		return classifiers_names;
	}
	
	/**
	 * Set the levels of the noise for the experimentation
	 * @param nois_lev the array with the noise levels
	 */
	
	public void setNoiseLevels(int [] nois_lev) {
		int num_levels = nois_lev.length;
		
		noise_levels = new int[num_levels];
		
		for(int i = 0; i < num_levels; i++) 		
			noise_levels[i] = nois_lev[i];
	}
	
	/**
	 * Obtains a dataset given the path of the arff and xml files, respectively
	 * @param file_arff Path of the arff file
	 * @param file_xml Path of the XML file
	 * @return The corresponding multi-label dataset
	 * @throws InvalidDataFormatException
	 */
	
	protected MultiLabelInstances getDataSet(String file_arff, String file_xml) throws InvalidDataFormatException{
		MultiLabelInstances ml_instances = new MultiLabelInstances(file_arff, file_xml);
		
		return ml_instances;
	}
	
	/**
	 * Obtains the noisy datasets from the original one. One dataset per noise level. 
	 * @param dataset original dataset
	 * @return array with the noisy datasets
	 * @throws Exception
	 */
	
	private MultiLabelInstances[] getNoisyDataSets(MultiLabelInstances dataset) throws Exception {
		int num_levels = noise_levels.length;
		MultiLabelInstances[] noisy_data_sets= new MultiLabelInstances[num_levels];
		MultiLabelInstances noisy_data;
		int noise_level;
		NoiseFilter noise_filter;
		
		for(int i = 0; i < num_levels; i++) {
			noise_level = noise_levels[i];
			noise_filter = new NoiseFilter(noise_level,seed);
			noisy_data = noise_filter.AddNoise(dataset);
			noisy_data_sets[i] = noisy_data;
		}
		
		return noisy_data_sets;
	}	
	/**
	 * Obtains the sets of datasets with all the noise levels 
	 *  given the paths of the folder with the arff and xml files.
	 * @param folder_arff Folder with the arff files
	 * @param folder_xml Folder with the XML files
	 * @return The corresponding multi-label dataset
	 * @throws Exception 
	 */
	
	protected void loadAllDatasets(String folder_arff, String folder_xml) throws Exception {
		int num_datasets;
		String arff_file, xml_file;
		String path_arff, path_xml;
		File file_arff, file_xml;
		File[] arff_files;
		File[] xml_files;
		MultiLabelInstances original_dataset;
		MultiLabelInstances[] noisy_datasets;
		
		file_arff = new File(folder_arff);
		file_xml = new File(folder_xml);
		
		arff_files = file_arff.listFiles();
		xml_files = file_xml.listFiles();
		
		num_datasets = arff_files.length;
		
		all_datasets = new MultiLabelInstances[num_datasets][]; 
		
		for(int i = 0; i < num_datasets; i++) {
			arff_file = arff_files[i].getName();
			path_arff = folder_arff + "/" + arff_file;
			xml_file = xml_files[i].getName();
			path_xml = folder_xml + "/" + xml_file;
			
			original_dataset = getDataSet(path_arff,path_xml);
			
			noisy_datasets = getNoisyDataSets(original_dataset);
			all_datasets[i] = noisy_datasets;
		}
		
	}
	

	/**
	 * Obtains the evaluation measures given the number of labels
	 * @param num_labels the number of labels of the multi-label dataset
	 * @return the List with the evaluation measures
	 */
	
	private List<Measure> generateEvaluationMeasures(int num_labels){
		List<Measure> evaluation_measures = new ArrayList<Measure>();
		HammingLoss measure1 = new HammingLoss();
		SubsetAccuracy measure2 = new SubsetAccuracy();
		ExampleBasedAccuracy measure3 = new ExampleBasedAccuracy();
		ExampleBasedPrecision measure4 = new ExampleBasedPrecision();
		ExampleBasedRecall measure5 = new ExampleBasedRecall();
		ExampleBasedFMeasure measure6 = new ExampleBasedFMeasure();
		MicroPrecision measure7 = new MicroPrecision(num_labels);
		MicroRecall measure8 = new MicroRecall(num_labels);
		MicroFMeasure measure9 = new MicroFMeasure(num_labels);
		MacroPrecision measure10 = new MacroPrecision(num_labels);
		MacroRecall measure11 = new MacroRecall(num_labels);
		MacroFMeasure measure12 = new MacroFMeasure(num_labels);
		AveragePrecision measure13 = new AveragePrecision();
		Coverage measure14 = new Coverage();
		OneError measure15 = new OneError();
		RankingLoss measure16 = new RankingLoss();
		
		evaluation_measures.add(measure1);
		evaluation_measures.add(measure2);
		evaluation_measures.add(measure3);
		evaluation_measures.add(measure4);
		evaluation_measures.add(measure5);
		evaluation_measures.add(measure6);
		evaluation_measures.add(measure7);
		evaluation_measures.add(measure8);
		evaluation_measures.add(measure9);
		evaluation_measures.add(measure10);
		evaluation_measures.add(measure11);
		evaluation_measures.add(measure12);
		evaluation_measures.add(measure13);
		evaluation_measures.add(measure14);
		evaluation_measures.add(measure15);
		evaluation_measures.add(measure16);
		
		return evaluation_measures;
	}
	

	/**
	 * Runs cross validation for a dataset with a Multi-Laber classifier for a noise level.
	 * Training with noisy dataset and test with the original. Return the average value
	 * for each evaluation metric.
	 * @param classifier Multi-Label classifier
	 * @param original_dataset The clean dataset
	 * @param noisy_dataset the noisy dataset
	 * @return array with the average values for each evaluation metric. 
	 * @throws Exception 
	 */
	
	private double[] computeResultsCV(MultiLabelLearner classifier, MultiLabelInstances original_dataset, MultiLabelInstances noisy_dataset) throws Exception {
		Evaluation[] evaluations = new Evaluation[num_folds];
		Evaluation evaluation;
		double[] partial_results;
		Evaluator eval = new Evaluator();
		Instances noisy_instances = noisy_dataset.getDataSet();
		Instances original_instances = original_dataset.getDataSet();
		Instances train, test;
		MultiLabelLearner copy_classifier;
		MultipleEvaluation multiple_evaluation;
		MultiLabelInstances ml_train, ml_test;
		LabelsMetaData labels_meta_data = original_dataset.getLabelsMetaData();
		int num_labels = original_dataset.getNumLabels();
		Instances work_train = new Instances(noisy_instances);
		Instances work_test = new Instances(original_instances);
		work_train.randomize(new Random(seed));
        work_test.randomize(new Random(seed));
        List<Measure> evaluation_measures = generateEvaluationMeasures(num_labels);
        Measure evaluation_measure;
        int num_measures = evaluation_measures.size();
        String measure_name;
        double mean_value;
        double time, average_time;
        double sum_times = 0;
        double start_time, end_time;
        
        for (int i = 0; i < num_folds; i++) {
            System.out.println("Fold " + (i + 1) + "/" + num_folds);
            train = work_train.trainCV(num_folds, i);
            test = work_test.testCV(num_folds, i);
            ml_train = new MultiLabelInstances(train,labels_meta_data);
            ml_test = new MultiLabelInstances(test,labels_meta_data);
        	copy_classifier = classifier.makeCopy();
        	start_time = System.currentTimeMillis();
        	copy_classifier.build(ml_train);
        	evaluation = eval.evaluate(copy_classifier, ml_test, evaluation_measures);
        	end_time = System.currentTimeMillis();
        	time = end_time - start_time;
        	sum_times+=time;
        	evaluations[i] = evaluation;
        }
        
        multiple_evaluation = new MultipleEvaluation(evaluations, original_dataset);
        multiple_evaluation.calculateStatistics();
        
        partial_results = new double[num_measures+1];
        
        for(int i = 0; i < num_measures; i++) {
        	evaluation_measure = evaluation_measures.get(i);
        	measure_name = evaluation_measure.getName();
        	mean_value = multiple_evaluation.getMean(measure_name);
        	partial_results[i] = mean_value;
        }
		
        average_time = sum_times/num_folds;
        partial_results[num_measures] = average_time;
        
		return partial_results;
	}
	

	/**
	 * Run the experiments. A cross validation procedure per classifier, dataset, 
	 * noise level and multi-label feature selector.
	 * @throws Exception 
	 */
	
	public void computeResults() throws Exception {
		String classifier_name, dataset_name;
		int noise_level;
		int num_classifiers = classifiers.length;
		int num_noise_levels = noise_levels.length;
		int num_datasets = all_datasets.length;
		double[] partial_results;
		MultiLabelLearner classifier;
		MultiLabelInstances original_dataset, noisy_dataset;
		MultiLabelInstances[] noisy_datasets;
		
		results = new double[num_datasets][num_noise_levels][num_classifiers][];
		
		for(int index_dataset = 0; index_dataset < num_datasets; index_dataset++) {
			noisy_datasets = all_datasets[index_dataset];
			original_dataset = noisy_datasets[0];
			dataset_name = original_dataset.getDataSet().relationName();
			System.out.println("Dataset = " + dataset_name);
				
			for(int index_noise_level = 0; index_noise_level < num_noise_levels; index_noise_level++) {
				noise_level = noise_levels[index_noise_level];
				noisy_dataset = noisy_datasets[index_noise_level];
				System.out.println("Level of noise " + noise_level);
					
				for(int index_classifier = 0; index_classifier < num_classifiers; index_classifier++) {
					classifier = classifiers[index_classifier];
					classifier_name = classifiers_names[index_classifier];
					System.out.println("Classifier " + classifier_name);
					partial_results = computeResultsCV(classifier, original_dataset, noisy_dataset);
					results[index_dataset][index_noise_level][index_classifier] = partial_results;
				}
			}
		}
	}
	
	/**
	 * Obtains the names of the evaluation measured used in the experimentation
	 * @return Array with the names of the evaluation measures
	 */
	
	private String[] getMeasuresNames() {
		int num_measures = 17;
		String[] measures_names = new String[num_measures];
		
		measures_names[0] = "HammingLoss";
		measures_names[1] = "SubsetAccuracy";
		measures_names[2] = "Accuracy";
		measures_names[3] = "Precision";
		measures_names[4] = "Recall";
		measures_names[5] = "F1";
		measures_names[6] = "MicroPrecision";
		measures_names[7] = "MicroRecall";
		measures_names[8] = "MicroF1";
		measures_names[9] = "MacroPrecision";
		measures_names[10] = "MacroRecall";
		measures_names[11] = "MacroF1";
		measures_names[12] = "AveragePrecision";
		measures_names[13] = "Coverage";
		measures_names[14] = "OneError";
		measures_names[15] = "RankingLoss";
		measures_names[16] = "Time";
		
		return measures_names;
	}
	
	/**
	 * Write the Equalized Loss results for each measure. Creates a folder for these results.
	 * @param folder_results folder of the results Folder where we want to save the results
	 * @throws IOException 
	 */
	
	public void writeRoboustnessResults(String folder_results) throws IOException {
		String folder_name_results, file_name_measure;
		String[] measures_names = getMeasuresNames();
		String measure_name, classifier_name, dataset_name;
		FileWriter file_measure;
		BufferedWriter buffered_writer;
		File folder_results_noise;
		int num_classifiers = classifiers.length;
		int num_datasets = all_datasets.length;
		int num_measures = measures_names.length;
		String separator = ";";
		String header;
		String total_string_result = "";
		double result, result_noise, loss_result;
		double difference;
		String string_result;
		MultiLabelInstances dataset;
		
		folder_name_results = folder_results + "/EqualizedLosses";
		folder_results_noise = new File(folder_name_results);
		folder_results_noise.mkdirs();
		
		for(int index_ev_measure = 0; index_ev_measure < num_measures-1; index_ev_measure++) {
			measure_name = measures_names[index_ev_measure];
			file_name_measure = folder_name_results + "/" + measure_name + ".csv";
			file_measure = new FileWriter(file_name_measure);
			buffered_writer = new BufferedWriter(file_measure);
			header = "Dataset" + separator;
				
			for(int index_classifier = 0; index_classifier < num_classifiers; index_classifier++) {
				classifier_name = classifiers_names[index_classifier];
				header = header + classifier_name + separator;
			}
				
			total_string_result+=header+"\n";
			
			for(int index_dataset = 0; index_dataset < num_datasets; index_dataset++) {
				dataset = all_datasets[index_dataset][0];
				dataset_name = dataset.getDataSet().relationName();
				string_result = dataset_name + separator;

				for(int index_classifier = 0; index_classifier < num_classifiers; index_classifier++){
					result = results[index_dataset][0][index_classifier][index_ev_measure];
					result_noise = results[index_dataset][1][index_classifier][index_ev_measure];
					
					if(index_ev_measure > 0 && index_ev_measure < 13)
						difference = 1 - result_noise;
					
					else
						difference = result_noise;
					
					loss_result = difference/result;
					
					string_result += loss_result+separator;
				}
				
				total_string_result+=string_result + "\n";
			}
				
			buffered_writer.write(total_string_result);
			buffered_writer.flush();
			buffered_writer.close();
			total_string_result = "";
		}
	}
	
	/**
	 * Write the results of our experimentation
	 * Creates a Folder for each multi-label classifier and noise level
	 * Within that folder, it creates a file per each evaluation measure
	 * @param folder_results Folder where we want to save the results
	 * @throws IOException if an exception happens when writing 
	 */
	
	public void writeResults(String folder_results) throws IOException {
		String folder_name_noise, file_name_measure;
		String[] measures_names = getMeasuresNames();
		String measure_name, classifier_name, dataset_name;
		FileWriter file_measure;
		BufferedWriter buffered_writer;
		File folder_results_noise;
		int num_classifiers = classifiers.length;
		int num_noise_levels = noise_levels.length;
		int num_datasets = all_datasets.length;
		int num_measures = measures_names.length;
		int noise_level;
		String separator = ";";
		String header;
		String total_string_result = "";
		double result;
		String string_result;
		MultiLabelInstances dataset;
		
		for(int index_noise_level = 0; index_noise_level < num_noise_levels; index_noise_level++) {
			// Make a folder for the level of noise
			noise_level = noise_levels[index_noise_level];
			folder_name_noise = folder_results + "/" + noise_level + "_Noise";
			folder_results_noise = new File(folder_name_noise);
			folder_results_noise.mkdirs();
				
			for(int index_ev_measure = 0; index_ev_measure < num_measures; index_ev_measure++) {
				measure_name = measures_names[index_ev_measure];
				file_name_measure = folder_name_noise + "/" + measure_name + ".csv";
				file_measure = new FileWriter(file_name_measure);
				buffered_writer = new BufferedWriter(file_measure);
				header = "Dataset" + separator;
					
				for(int index_classifier = 0; index_classifier < num_classifiers; index_classifier++) {
					classifier_name = classifiers_names[index_classifier];
					header = header + classifier_name + separator;
				}
					
				total_string_result+=header+"\n";
									
				for(int index_dataset = 0; index_dataset < num_datasets; index_dataset++) {
					dataset = all_datasets[index_dataset][0];
					dataset_name = dataset.getDataSet().relationName();
					string_result = dataset_name + separator;

					for(int index_classifier = 0; index_classifier < num_classifiers; index_classifier++){
						result = results[index_dataset][index_noise_level][index_classifier][index_ev_measure];
						string_result += result+separator;
					}
					
					total_string_result+=string_result + "\n";
				}
					
				buffered_writer.write(total_string_result);
				buffered_writer.flush();
				buffered_writer.close();
				total_string_result = "";
			}
		}
		
		if(num_noise_levels > 1)
			writeRoboustnessResults(folder_results);
	}
	
	public static void main(String[] args) throws InvalidDataException, Exception {
    	/*Experimentation experimentation;
    	int[] noise_levels = {0,10};
    	String location = "C:/Proyecto/Datasets_MultiLabel";
    	int seed = 1;
    	int num_folds = 5;
    	String location_arff = location + "/" + "Arff_Files";
    	String location_xml = location + "/" + "XML_Files";
    	String folder_results = location + "/LazyCredal";
    	MultiLabelLearner classifier1 = new MLkNN(10, 1.0);
    	ExtensionType extension = ExtensionType.EXTA;
    	MultiLabelLearner classifier2 = new BRkNN(10, extension);
    	MultiLabelLearner classifier3 = new BRCredalkNN(10, extension);
    	MultiLabelLearner classifier4 = new DMLkNN();
    	MultiLabelLearner classifier5 = new ML_CredalkNN();
    	int num_classifiers = 5;
    	MultiLabelLearner[] classifiers = new MultiLabelLearner[num_classifiers];
    	String[] classifier_names = new String[num_classifiers];
    	
    	classifiers[0] = classifier1;
    	classifiers[1] = classifier2;
    	classifiers[2] = classifier3;
    	classifiers[3] = classifier4;
    	classifiers[4] = classifier5;

    	classifier_names[0] = "ML_KNN";
    	classifier_names[1] = "BR_KNN_alpha";
    	classifier_names[2] = "BR_Credal_KNN_alpha";
    	classifier_names[3] = "DML_KNN";
    	classifier_names[4] = "ML_Credal_KNN";

    	experimentation = new Experimentation(classifiers, num_folds, location_arff, location_xml, classifier_names, noise_levels, seed);

    	experimentation.computeResults();
    	experimentation.writeResults(folder_results);
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