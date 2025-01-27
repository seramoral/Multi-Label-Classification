package mulan.classifier;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.data.NoiseFilter;
import mulan.evaluation.Evaluator2;
import mulan.evaluation.MultipleEvaluation;

public class Experimentation2 {
	private File[] arff_files;
	private File[] xml_files;
	private int num_folds;
	private MultiLabelLearner[] classifiers;
	private Evaluator2 eval;
	private String folder_arff;
	private String folder_xml;
	private NoiseFilter noise_filter;
	private int[] noise_levels; 
	private int seed;
	private String[] classifiers_names;
	private MultiLabelInstances[][] all_datasets;
	private String folder_results;
	
	public Experimentation2(MultiLabelLearner[] learners, int n_folds, String f_arff, String f_xml,String[] names, int[] nois_lev, int s, String f_res) throws Exception {			
		setFolders(f_arff, f_xml);
		setClassifiers(learners); 
		setClassifiersNames(names);
		setNoiseLevels(nois_lev);
				
		num_folds = n_folds;
		seed = s;
		folder_results = f_res;
		
		eval = new Evaluator2();
	}
	
	public void setFolderResults(String f_res){
		folder_results = f_res;
	}
	
	public String getFolderResults() {
		return folder_results;
	}
	
	public void setFolders(String f_arff, String f_xml) {
		folder_arff = f_arff;
		folder_xml = f_xml;
	}
	
	public String getArffFolder() {
		return folder_arff;
	}
	
	public String getXMLFolder() {
		return folder_xml;
	}	
	
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
	
	public MultiLabelLearner[] getClassifiers() {
		return classifiers;
	}
	
	public void setClassifiersNames(String [] names) {
		int num_learners = names.length;		
		String name;
		
		classifiers_names = new String[num_learners];
		
		for(int i = 0; i < num_learners; i++) {
			name = names[i];
			classifiers_names[i] = name;
		}
	}
	
	public String[] getClassifiersNames() {
		return classifiers_names;
	}
	
	public void setNoiseLevels(int [] nois_lev) {
		int num_levels = nois_lev.length;
		int noise_level;	
		
		noise_levels = new int[num_levels];
		
		for(int i = 0; i < num_levels; i++) {
			noise_level = nois_lev[i];
			noise_levels[i] = noise_level;
		}
	}
	
	private MultiLabelInstances getDataSet(String file_arff, String file_xml) throws InvalidDataFormatException{
		MultiLabelInstances ml_instances = new MultiLabelInstances(file_arff, file_xml);
		
		return ml_instances;
	}
	
	private MultiLabelInstances[] getNoisyDataSets(MultiLabelInstances dataset) throws Exception {
		int num_levels = noise_levels.length;
		MultiLabelInstances[] noisy_data_sets= new MultiLabelInstances[num_levels];
		MultiLabelInstances noisy_data;
		int noise_level;
		
		for(int i = 0; i < num_levels; i++) {
			noise_level = noise_levels[i];
			noise_filter = new NoiseFilter(noise_level,seed);
			noisy_data = noise_filter.AddNoise(dataset);
			noisy_data_sets[i] = noisy_data;
		}
		
		return noisy_data_sets;
	}
	
	private MultiLabelInstances[][] loadAllDatasets() throws Exception{
		int num_datasets;
		MultiLabelInstances original_dataset;
		MultiLabelInstances[] noisy_datasets;
		String arff_file, xml_file;
		String path_arff, path_xml;
		File file_arff, file_xml;
		System.out.println("Cargando datasets");
		
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
			System.out.println("Dataset cargado");
		}
		
		return all_datasets;
	}
	
	public void computeResults() throws Exception{
		MultiLabelInstances[][] datasets = loadAllDatasets();
		int num_datasets = datasets.length;
		int num_classifiers = classifiers.length;
		int num_levels = noise_levels.length;
		MultiLabelInstances original_dataset, noisy_dataset;
		MultiLabelInstances[] noisy_datasets;
		MultipleEvaluation results;
		MultiLabelLearner learner;
		String relation_name;
		int noise_level;
		String classifier_name;
		String string_results = "";
		String folder_name;
		File folder;
		String file_results_name;
		BufferedWriter bw;
		FileWriter file;
		
		for(int i = 0; i < num_datasets; i++){
			noisy_datasets = datasets[i];
			original_dataset = noisy_datasets[0];
			relation_name = original_dataset.getDataSet().relationName();
			System.out.println("Dataset = " + relation_name);
			
			for(int j = 0; j < num_levels;j++) {
				noisy_dataset = noisy_datasets[j];
				noise_level = noise_levels[j];
				folder_name = folder_results + "/" + relation_name + "/" + noise_level+"Noise";
				folder = new File(folder_name);
				folder.mkdirs();
				file_results_name = folder_name + "/results.txt";
				file = new FileWriter(file_results_name);
				bw = new BufferedWriter(file);
				
				for(int k = 0; k < num_classifiers;k++){
					classifier_name = classifiers_names[k];
					string_results = "\nClasificador " + classifier_name + ":\n";
					learner = classifiers[k];
					results = eval.crossValidate(learner, original_dataset,noisy_dataset, num_folds);
					string_results += results.toString();
					bw.write(string_results);
					bw.flush();
				}
				
				bw.close();
			}
		}
	}
	
	
}
