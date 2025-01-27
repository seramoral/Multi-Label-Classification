package mulan.classifier.lazy;


import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.Experimentation;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;

public class ML_CredalkNN extends MultiLabelKNN{
   
    /**
     *
     */
    private static final long serialVersionUID = 1L;

    /**
     * A table holding the prior probability for an instance to belong in each
     * class
     */
    private double[] PriorProbabilities;
   
    /** A table holding the prior probability for an instance not to belong in
    * each class
    */
   private double[] PriorNProbabilities;
   /**
    * A table holding the probability for an instance to belong in each
    * class<br> given that i:0..k of its neighbors belong to that class
    */
   private double[][] CondProbabilities;
   /**
    * A table holding the probability for an instance not to belong in each
    * class<br> given that i:0..k of its neighbors belong to that class
    */
   private double[][] CondNProbabilities;
    
   public ML_CredalkNN(int numOfNeighbors) {
       super(numOfNeighbors);
   }
  
   public ML_CredalkNN() {
       super();
   }
  
   public String globalInfo() {
       return "Class implementing the ML-kNN (Multi-Label k Nearest Neighbours) algorithm." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
   }

   @Override
   public TechnicalInformation getTechnicalInformation() {
       TechnicalInformation result;

       result = new TechnicalInformation(Type.ARTICLE);
       result.setValue(Field.AUTHOR, "Min-Ling Zhang and Zhi-Hua Zhou");
       result.setValue(Field.TITLE, "ML-KNN: A lazy learning approach to multi-label learning");
       result.setValue(Field.JOURNAL, "Pattern Recogn.");
       result.setValue(Field.VOLUME, "40");
       result.setValue(Field.NUMBER, "7");
       result.setValue(Field.YEAR, "2007");
       result.setValue(Field.ISSN, "0031-3203");
       result.setValue(Field.PAGES, "2038--2048");
       result.setValue(Field.PUBLISHER, "Elsevier Science Inc.");
       result.setValue(Field.ADDRESS, "New York, NY, USA");

       return result;
   }
  
   @Override
   protected void buildInternal(MultiLabelInstances train) throws Exception {
       super.buildInternal(train);
       PriorProbabilities = new double[numLabels];
       PriorNProbabilities = new double[numLabels];
       CondProbabilities = new double[numLabels][numOfNeighbors + 1];
       CondNProbabilities = new double[numLabels][numOfNeighbors + 1];
       ComputePrior();
       ComputeCond();

       if (getDebug()) {
           System.out.println("Computed Prior Probabilities");
           for (int i = 0; i < numLabels; i++) {
               System.out.println("Label " + (i + 1) + ": " + PriorProbabilities[i]);
           }
           System.out.println("Computed Posterior Probabilities");
           for (int i = 0; i < numLabels; i++) {
               System.out.println("Label " + (i + 1));
               for (int j = 0; j < numOfNeighbors + 1; j++) {
                   System.out.println(j + " neighbours: " + CondProbabilities[i][j]);
                   System.out.println(j + " neighbours: " + CondNProbabilities[i][j]);
               }
           }
       }
   }  
  
  public static double[] getProbabilityANPI(double[] frequencies){
	  int num_values = frequencies.length;
	  double[] probabilities = new double[num_values];
	  double epsilon = 0.001;
	  double mass = 0;
	  int i = 0;
	  double k_i, k_i1;
	  double aux;
	  double denominator;
	  double num_instances = Utils.sum(frequencies);
	  
	  for(int j = 0; j < num_values; j++) {
		  
		  if(frequencies[j] > epsilon) {
			  mass += 1.0;
			  probabilities[j] = (frequencies[j]-1)/num_instances;
		  }
		  
		  else
			  probabilities[j] = 0.0;
			  
	  }
	  
	  while(mass > epsilon) {
		  k_i = 0;
		  k_i1 = 0;
		  
		  for(int j = 0; j < num_values; j++) {
			  
			  if(frequencies[j] == i)
				  k_i = k_i+1;
			  
			  else if(frequencies[j] == i+1)
				  k_i1 = k_i1+1;
		  }
		  
		  aux = k_i+k_i1;
		  
		  if(aux < mass) {
			  for(int j = 0; j < num_values; j++) {
				  
				  if(frequencies[j] == i || frequencies[j] == i+1){
					  probabilities[j] = probabilities[j] + 1/num_instances;
					  mass = mass - 1;
				  }
			  }
			  
		  }
		  
		  else{
			  
			  for(int j = 0; j < num_values; j++) {
				  
				  if(frequencies[j] == i || frequencies[j] == i+1) {
					  denominator = aux*num_instances;
					  probabilities[j] = probabilities[j] + mass/denominator;
				  }
			  }
			  
			  mass = 0;
		  }
		  
		  i = i+1;
	  }
	  
	  return probabilities;
  }
   
   private void ComputeCond() throws Exception {
       double[][] temp_Ci = new double[numLabels][numOfNeighbors + 1];
       double[][] temp_NCi = new double[numLabels][numOfNeighbors + 1];
       double [] frequencies, n_frequencies;
       double[] n_probability_ANPI, probability_ANPI;
       int num_instances = train.numInstances();
       Instance instance;
       Instances knn;
       int aces;
       Instance knn_instance;
       double value, knn_value;
       int label_index;
       int label_value, knn_label_value;
      
       for (int i = 0; i < num_instances; i++) {
           instance = train.instance(i);
          
           knn = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));
          
           for (int j = 0; j < numLabels; j++) {
               aces = 0;
               label_index = labelIndices[j];
               label_value = (int)instance.value(label_index);
              
               for (int k = 0; k < numOfNeighbors; k++) {
                   knn_instance = knn.instance(k);
                   knn_label_value = (int) knn_instance.value(label_index);
                   knn_value = Double.parseDouble(train.attribute(label_index).value(knn_label_value));
                  
                   if (Utils.eq(knn_value, 1.0)) {
                       aces++;
                   }
               }
              
               value = Double.parseDouble(train.attribute(label_index).value(label_value));
              
               if (Utils.eq(value, 1.0)) 
                   temp_Ci[j][aces]++;
                  
               
               else 
                   temp_NCi[j][aces]++;
           }
          
       }
      
       for (int i = 0; i < numLabels; i++) {              
           frequencies = temp_Ci[i];
           n_frequencies = temp_NCi[i];
           probability_ANPI = getProbabilityANPI(frequencies);
           n_probability_ANPI = getProbabilityANPI(n_frequencies);

           for (int j = 0; j < numOfNeighbors + 1; j++) {
               CondProbabilities[i][j] = probability_ANPI[j];
               CondNProbabilities[i][j] = n_probability_ANPI[j];
           }
          
       }
   }
  
   /**
   * Computing Prior and PriorN Probabilities for each class of the training
   * set
   */
  private void ComputePrior() {
      double[] frequencies = new double[numLabels];
      double[] Nfrequencies = new double[numLabels];
      int num_instances = train.numInstances();
      int label_index;
      Instance instance;
      double value;
      int label_value;
      Attribute label_attribute;
      double aux;
      
      for (int j = 0; j < numLabels; j++) {
          label_index = labelIndices[j];
          label_attribute = train.attribute(label_index);
         
          for (int i = 0; i < num_instances; i++) {
              instance = train.instance(i);
              label_value =  (int) instance.value(label_index);
             
              value = Double.parseDouble(label_attribute.value(label_value));
             
              if (Utils.eq(value, 1.0))
                  frequencies[j]++;    
             
              else
                  Nfrequencies[j]++;                
          }
          
          aux = Math.abs(frequencies[j] - Nfrequencies[j]);
          
          if(aux <= 2) {
        	  PriorProbabilities[j] = 0.5;
        	  PriorNProbabilities[j] = 0.5;
          }
          
          else if (frequencies[j] > Nfrequencies[j]) {
        	  PriorProbabilities[j] = (frequencies[j] - 1)/num_instances;
        	  PriorNProbabilities[j] = (Nfrequencies[j] + 1)/num_instances;
          }
          
          else {
        	  PriorProbabilities[j] = (frequencies[j] + 1)/num_instances;
        	  PriorNProbabilities[j] = (Nfrequencies[j] - 1)/num_instances;
          }
      }
     
  }
 
  protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
      double[] confidences = new double[numLabels];
      boolean[] predictions = new boolean[numLabels];

      Instances knn = null;
      try {
          knn = new Instances(lnn.kNearestNeighbours(instance, numOfNeighbors));
      } catch (Exception ex) {
          Logger.getLogger(MLkNN.class.getName()).log(Level.SEVERE, null, ex);
      }

      for (int i = 0; i < numLabels; i++) {
          // compute sum of aces in KNN
          int aces = 0; // num of aces in Knn for i
         
          for (int k = 0; k < numOfNeighbors; k++) {
              double value = Double.parseDouble(train.attribute(labelIndices[i]).value(
                      (int) knn.instance(k).value(labelIndices[i])));
              if (Utils.eq(value, 1.0)) {
                  aces++;
              }
          }
         
          double Prob_in = PriorProbabilities[i] * CondProbabilities[i][aces];
          double Prob_out = PriorNProbabilities[i] * CondNProbabilities[i][aces];
         
          if (Prob_in > Prob_out)
              predictions[i] = true;
         
          else if (Prob_in < Prob_out)
              predictions[i] = false;
         
          else {
              Random rnd = new Random();
              predictions[i] = (rnd.nextInt(2) == 1) ? true : false;
          }
          // ranking function
          confidences[i] = Prob_in / (Prob_in + Prob_out);
      }
      MultiLabelOutput mlo = new MultiLabelOutput(predictions, confidences);
      return mlo;
  }
  
  public static void main(String[] args) throws InvalidDataException, Exception {
	/*  Experimentation experimentation;
	  int[] noise_levels = {0,5,10};
	  String location = "C:/Proyecto/Datasets_MultiLabel2";
	  String location_arff = location + "/" + "Arff_Files";
	  String location_xml = location + "/" + "XML_Files";
	  String file_results = location + "/ML_CredalKNN";
	  int num_folds = 5;
	  int num_learners = 2;
	  MultiLabelLearner[] learners = new MultiLabelLearner[num_learners];
	  String[] names = new String[num_learners];
	  int seed = 1;
	  int k = 10;
	  double smooth = 1.0;
	  
	  MultiLabelLearner learner1 = new MLkNN(k, smooth);
	  MultiLabelLearner learner2 = new ML_CredalkNN(k);
	  learners[0] = learner1;
	  learners[1] = learner2;
	  
	  names[0] = "CredalKNN";
	  names[1] = "ML_CredalKNN";
	  
	  experimentation = new Experimentation(learners, num_folds, location_arff, location_xml, names, noise_levels, seed,file_results);
		 
	  experimentation.computeResults();
*/
  }
}



	

