package mulan.classifier;

public class ML_RepTree2 {
	 public static void main(String[] args) throws InvalidDataException, Exception {
		 Experimentation experimentation;
		 int[] noise_levels = {0};
		 MultiLabelLearner ML_s0 = new ML_RepTree(0);
		 MultiLabelLearner ML_s1 = new ML_RepTree(1.0);
		 //MultiLabelLearner ML_s2 = new ML_RepTree(2.0);
		 //MultiLabelLearner ML_s3 = new ML_RepTree(3.0);
		 MultiLabelLearner[] learners = new MultiLabelLearner[2];
		 String[] names = new String[2];
		 int num_folds = 5;
		 int seed = 1;
		 String location = "C:/Proyecto/Datasets_MultiLabel2";
		 String write_folder = location + "ML_RepTree";
		 String location_arff = location + "Arff_Files";
		 String location_xml = location + "XML_Files";
		 
		 learners[0] = ML_s0;
		 learners[1] = ML_s1;
		 //learners[2] = ML_s2;
		 //learners[3] = ML_s3;
		 
		 names[0] = "S=0";
		 names[1] = "S=1";
		// names[2] = "S=2";
		// names[3] = "S=3";
		 
		 experimentation = new Experimentation(learners, num_folds, location_arff, location_xml, names, noise_levels, seed, write_folder);
		 
		 experimentation.computeResults();
	 }
}
