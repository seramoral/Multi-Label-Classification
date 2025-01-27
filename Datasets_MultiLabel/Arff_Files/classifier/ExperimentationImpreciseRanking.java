package mulan.classifier;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.EvaluateImpreciseClassifier;

/**
 * Class for running experiments with methods for imprecise ranking
 * @author Serafin
 */

public class ExperimentationImpreciseRanking extends Experimentation{

	
	/** Datasets considered in the experimentation. **/
	
	private MultiLabelInstances[] datasets;
	
	/** Array with the names of the measures of imprecise rankings */
	
	private String measures_names[];
	

	/** Whether use interval outputs for the imprecise prediction */
	
	private boolean intervals_output;
	
	
	
/** Array 3-dimensional with the results of our experimentation **/
	
	private double[][][] results_imprecise_ranking;
	
	/** Creates a new object
	 * @param n_folds the number of folds employed for cross validation
	 * @param folder_arff  location of the folder with the Arff files  
	 * @param folder_xml location of the folder with the XML files 
	 * @param nois_lev the array with the levels of noise considered
	 * @param intervals whether the output of the pairwise classifiers 
	 * is in the form of probability intervals
	 * @throws Exception if an exception occurs when loading the datasets
	 */
	
	public ExperimentationImpreciseRanking(int n_folds, String folder_arff, String folder_xml, int[] nois_lev, boolean intervals) throws Exception {
		super();
		setNumFolds(n_folds);
		setMeasuresNames();
		setNoiseLevels(nois_lev);
		setPredictedIntervalsOutput(intervals);
		
		loadDatasets(folder_arff, folder_xml);
	}
	
	/**
	 * Sets whether use interval outputs for the imprecise prediction
	 * @param intervals true if use predicted intervals
	 * false otherwise
	 */
	
	public void setPredictedIntervalsOutput(boolean intervals) {
		intervals_output = intervals;
	}
	
	/**
	 * Gets whether use interval outputs for the imprecise prediction
	 * @return  true if use predicted intervals false otherwise
	 */
	
	public boolean getPredictedIntervalsOutput() {
		return intervals_output;
	}
	
	private void loadDatasets(String folder_arff, String folder_xml) throws InvalidDataFormatException {
		File file_arff, file_xml;
		int num_datasets;
		File[] arff_files;
		File[] xml_files;
		file_arff = new File(folder_arff);
		file_xml = new File(folder_xml);
		String arff_file, xml_file;
		String path_arff, path_xml;
		MultiLabelInstances dataset;
		
		// Obtain the list of files of the folder
		arff_files = file_arff.listFiles();
		xml_files = file_xml.listFiles();
		
		num_datasets = arff_files.length;
		datasets = new MultiLabelInstances[num_datasets];
		
		for(int i = 0; i < num_datasets; i++) {
			/* Obtain the complete path of the arff and xml files */
			arff_file = arff_files[i].getName();
			path_arff = folder_arff + "/" + arff_file;
			xml_file = xml_files[i].getName();
			path_xml = folder_xml + "/" + xml_file;
			
			dataset = getDataSet(path_arff,path_xml);
			datasets[i] = dataset;
		}

	}
	
	/**
	 * Sets the names of the measures utilized for evaluation
	 * of the imprecise ranking method considered
	 */
	
	public void setMeasuresNames() {
		measures_names = new String[5];
		
		measures_names[0] = "Correctness_Precise";
		measures_names[1] = "Correctness_Imprecise";
		measures_names[2] = "Completeness";
		measures_names[3] = "Rate_Error_Covered";
		measures_names[4] = "Rate_Partial_Incorrect";
	}
	
	/**
	 * Computes the results of our experimentation with 
	 * imprecise ranking algorithms
	 * @throws Exception if an exception occurs 
	 * during the cross validation procedure
	 */
	
	public void computeResultsImpreciseRanking() throws Exception {
		EvaluateImpreciseClassifier evaluation;
		String dataset_name;
		int num_noise_levels = noise_levels.length;
		int num_datasets = datasets.length;
		int num_measures = measures_names.length;
		MultiLabelInstances dataset;
		int noise_level;
		double correctness_precise, correctness_imprecise, completeness;
		double rate_error_covered, rate_partial_incorrect;
		
		results_imprecise_ranking = new double[num_datasets][num_noise_levels][num_measures];
	
		for(int index_dataset = 0; index_dataset < num_datasets; index_dataset++) {
			dataset = datasets[index_dataset];
			dataset_name = dataset.getDataSet().relationName();
			System.out.println("Dataset = " + dataset_name);

			for(int index_noise_level = 0; index_noise_level < num_noise_levels; index_noise_level++) {
				noise_level = noise_levels[index_noise_level];
				System.out.println("Level of noise " + noise_level);
			
				evaluation = new EvaluateImpreciseClassifier(dataset, noise_level, num_folds, intervals_output);
				evaluation.performCrossValidation();
				
				correctness_precise = evaluation.getCorrectnessPrecise();
				correctness_imprecise = evaluation.getCorrectnessImprecise();
				completeness = evaluation.getCompleteness();
				rate_error_covered = evaluation.getRateErrorCovered();
				rate_partial_incorrect = evaluation.getRatePartialIncorrect();
				
				results_imprecise_ranking[index_dataset][index_noise_level][0] = correctness_precise;
				results_imprecise_ranking[index_dataset][index_noise_level][1] = correctness_imprecise;
				results_imprecise_ranking[index_dataset][index_noise_level][2] = completeness;
				results_imprecise_ranking[index_dataset][index_noise_level][3] = rate_error_covered;
				results_imprecise_ranking[index_dataset][index_noise_level][4] = rate_partial_incorrect;

			}

		}
	}
	
	/**
	 * Write the results of our experimentation
	 * Creates a file for each noise level
	 * Within that folder, it creates a file per each evaluation measure
	 * @param folder_results Folder where we want to save the results
	 * @throws IOException if an exception happens when writing 
	 */
	
	public void writeResults(String folder_results) throws IOException {
		int num_noise_levels = noise_levels.length;
		int num_datasets = datasets.length;
		int num_measures = measures_names.length;
		int noise_level;
		String separator = ";";
		String header;
		double result;
		String string_result;
		String total_string_result = "";
		String dataset_name, measure_name;
		String name_file_results_noise_level;
		FileWriter file_noise_level;
		BufferedWriter buffered_writer;

		for(int index_noise_level = 0; index_noise_level < num_noise_levels; index_noise_level++) {
			// Create a file for the revel of noise
			noise_level = noise_levels[index_noise_level];
			name_file_results_noise_level = folder_results + "/results_" + noise_level + "_Noise.csv";
			file_noise_level = new FileWriter(name_file_results_noise_level);
			buffered_writer = new BufferedWriter(file_noise_level);
			header = "Dataset" + separator;
			
			for(int index_measure = 0; index_measure < num_measures; index_measure++){
				measure_name = measures_names[index_measure];
				header = header + measure_name + separator;
			}
			
			total_string_result+=header+"\n";
			
			for(int index_dataset = 0; index_dataset < num_datasets; index_dataset++) {
				dataset_name = datasets[index_dataset].getDataSet().relationName();
				string_result = dataset_name + separator;
				
				for(int index_measure = 0; index_measure < num_measures; index_measure++){
					result = results_imprecise_ranking[index_dataset][index_noise_level][index_measure];
					string_result = string_result+result+separator;
				}
				total_string_result+=string_result + "\n";
			}
			
			buffered_writer.write(total_string_result);
			buffered_writer.flush();
			buffered_writer.close();
			total_string_result = "";

		}

	}

	public static void main(String[] args) throws InvalidDataException, Exception {
		int[] noise_levels = {0,5,10};
		ExperimentationImpreciseRanking experimentation_imprecise_ranking;
    	int num_folds = 5;
    	String location = "C:/Proyecto/Datasets_MultiLabel_Imprecise_Ranking";
    	String location_arff = location + "/" + "Arff_Files";
    	String location_xml = location + "/" + "XML_Files";
    	boolean interval_output = false;
    	String folder_results = location + "/ImpreciseRankingIntervals";
    	
    	experimentation_imprecise_ranking = new ExperimentationImpreciseRanking(num_folds, location_arff, location_xml, noise_levels, interval_output);
    	
    	experimentation_imprecise_ranking.computeResultsImpreciseRanking();
    	experimentation_imprecise_ranking.writeResults(folder_results);
	}
	
	
}
